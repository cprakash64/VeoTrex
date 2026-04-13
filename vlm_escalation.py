#!/usr/bin/env python3
"""
VLM Escalation Module — Gemini 1.5 Flash safety-violation analyser.

Pipeline per event
──────────────────
  Event (list[io.BytesIO])
      │
      ▼
  Storyboard.build()
      • Select 9 frames evenly from the stream list (np.linspace)
      • Decode each JPEG stream → NumPy BGR array
      • Resize every cell to cfg.cell_w × cfg.cell_h
      • Annotate each cell: frame label (T1-T9), final-frame marker
      • np.hstack each row → np.vstack rows into 3×3 grid
      • cv2.imencode → JPEG bytes → base64 string
      │
      ▼
  GeminiClient.analyse()
      • POST to generativelanguage.googleapis.com (aiohttp)
      • response_schema enforces ViolationAnalysis JSON structure
      • Exponential back-off on 429 / 5xx; immediate fail on other 4xx
      │
      ▼
  ViolationAnalysis (Pydantic, frozen)
      • violation_detected: bool
      • violation_type:     str   ("No Hardhat", "None", …)
      • confidence_score:   float [0.0, 1.0]
      │
      ▼
  AnalysisResult dispatched to on_analysis callback

Non-blocking guarantee
──────────────────────
  VLMEscalationHandler.__call__ is an async coroutine.
  EventManager already awaits it on its own dedicated asyncio loop thread,
  so the main video-processing loop is never touched.

Usage
─────
  from vlm_escalation import VLMEscalationHandler, GeminiConfig, StoryboardConfig

  handler = VLMEscalationHandler(
      gemini_cfg=GeminiConfig(),  # uses ADC; project/location default to Vertex AI config
      on_analysis=lambda r: print(r.analysis),
  )
  # Pass as the on_event callback to rtsp_tracker.run()
  run(..., on_event=handler)
  # At shutdown (called automatically by rtsp_tracker.run()):
  handler.close_sync()

Dependencies
────────────
  pip install google-genai pydantic
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional

from google import genai
from google.genai import types
import cv2
import numpy as np
from pydantic import BaseModel, Field, field_validator

# rtsp_tracker is imported for the Event type only; no circular dependency
# because rtsp_tracker never imports from this module.
from rtsp_tracker import Event

logger = logging.getLogger("vlm_escalation")


# ─── Pydantic response schema ─────────────────────────────────────────────────


class ViolationAnalysis(BaseModel):
    """
    Strict schema for the Gemini structured-output response.

    Marked ``frozen=True`` so instances are hashable and safe to share
    across threads without mutation risk.
    """

    model_config = {"frozen": True}

    violation_detected: bool = Field(
        description="True when a safety violation is present in the storyboard."
    )
    violation_type: str = Field(
        description=(
            "A concise label for the most significant violation observed, or 'None'. "
            "Examples: 'No Hardhat', 'No Safety Vest', 'Carrying Heavy Load', "
            "'Restricted Area Breach', 'Unsafe Posture', 'Blocked Emergency Exit'."
        )
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Model confidence in the analysis, as a float in [0.0, 1.0].",
    )
    reasoning: str = Field(
        default="",
        description=(
            "One or two sentences explaining which visual evidence in the storyboard "
            "supports the conclusion.  Must be non-empty when violation_detected is true."
        ),
    )

    @field_validator("confidence_score", mode="before")
    @classmethod
    def _clamp_confidence(cls, v: object) -> float:
        """Clamp silently in case the model returns a fractionally out-of-range value."""
        return max(0.0, min(1.0, float(v)))  # type: ignore[arg-type]

    @field_validator("violation_type", mode="before")
    @classmethod
    def _normalise_none(cls, v: object) -> str:
        """Map empty / 'N/A' / 'no violation' variants → canonical 'None'."""
        cleaned = str(v).strip()
        if cleaned.lower() in ("none", "n/a", "no violation", "no violations", ""):
            return "None"
        return cleaned


# ─── Configuration dataclasses ────────────────────────────────────────────────


@dataclass(frozen=True)
class StoryboardConfig:
    """
    Controls the layout and quality of the 3×3 temporal storyboard image.

    Memory / token trade-off
    ────────────────────────
    Each cell is resized to ``cell_w × cell_h`` before stitching.
    The final grid is ``(cols * cell_w) × (rows * cell_h)`` pixels.
    Reducing ``storyboard_jpeg_quality`` shrinks the base64 payload and
    lowers Gemini token consumption at the cost of image fidelity.
    """

    rows: int = 3
    cols: int = 3
    cell_w: int = 320                  # pixels per cell, width
    cell_h: int = 240                  # pixels per cell, height
    storyboard_jpeg_quality: int = 82  # higher than event snapshots for VLM clarity
    label_font_scale: float = 0.65
    label_thickness: int = 2


@dataclass(frozen=True)
class GeminiConfig:
    """
    Gemini SDK connection and generation parameters — Vertex AI backend.

    Authentication is handled by Application Default Credentials (ADC).
    No API key is required.  Before running, ensure the environment is
    authenticated::

        gcloud auth application-default login

    Or set GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON path.
    """

    project: str = "luna-ai-490605"
    location: str = "us-central1"
    model: str = "gemini-2.5-flash"
    temperature: float = 0.10           # low = deterministic safety judgements
    max_output_tokens: int = 512        # extra headroom for reasoning field
    max_retries: int = 3
    retry_base_delay: float = 1.0       # seconds; doubles on each retry


# ─── Result container ─────────────────────────────────────────────────────────


@dataclass
class AnalysisResult:
    """
    Combines the original event with the VLM's structured analysis.

    ``storyboard_jpeg_bytes`` holds the raw (not base64) JPEG grid so
    downstream consumers can forward or store it without re-decoding.
    """

    event: Event
    analysis: ViolationAnalysis
    storyboard_jpeg_bytes: bytes       # raw JPEG of the 3×3 grid
    api_latency_ms: float
    model: str

    @property
    def track_id(self) -> int:
        return self.event.track_id

    @property
    def triggered_at_str(self) -> str:
        return self.event.triggered_at_str

    def summary(self) -> str:
        flag = "VIOLATION" if self.analysis.violation_detected else "clear"
        return (
            f"[{flag}] ID={self.track_id} | "
            f"type='{self.analysis.violation_type}' | "
            f"conf={self.analysis.confidence_score:.2f} | "
            f"latency={self.api_latency_ms:.0f}ms | "
            f"t={self.triggered_at_str}"
        )


# ─── System prompt ────────────────────────────────────────────────────────────

_ANALYSIS_PROMPT = """\
You are an enterprise safety AI. Analyze this temporal 3×3 grid of frames \
sequentially. Identify if any worker in the frame is NOT wearing a \
high-visibility safety vest (orange or yellow). Return violation_detected: \
true if a vest is missing, otherwise false. Be strict.

Reading order: left-to-right, top-to-bottom.
  T1 (top-left) → earliest frame
  T9 (bottom-right) → most recent frame

Decision rules:
  1. Examine each frame T1–T9 in order.
  2. For every visible worker, determine whether they are wearing a \
high-visibility vest (orange or yellow).
  3. Set violation_detected = true if ANY worker is clearly missing a hi-vis vest \
across any frame in the sequence.
  4. Set violation_type to "No Hi-Vis Vest" when a violation is found, \
or "None" when every visible worker is correctly equipped.
  5. Set confidence_score in [0.0, 1.0] reflecting your certainty.
  6. Set reasoning to one or two sentences citing the specific frame numbers \
(T1–T9) and visual evidence (clothing colour, worker position) that support \
your conclusion.

Respond with a single valid JSON object — no markdown fence, no explanation, \
no trailing text outside the JSON.
"""


# ─── Temporal storyboard builder ─────────────────────────────────────────────


class Storyboard:
    """
    Builds a 3×3 temporal grid image from up to N JPEG byte streams.

    Frame selection
    ───────────────
    If more than ``rows * cols`` streams are available, exactly
    ``rows * cols`` are selected using ``np.linspace`` so the chosen frames
    span the full temporal window uniformly.  If fewer are available, all
    are used and the remaining cells are filled with black padding frames.

    All I/O (stream decoding, NumPy ops, cv2.imencode) is synchronous and
    CPU-bound; callers should run this inside ``loop.run_in_executor``.
    """

    def __init__(self, cfg: StoryboardConfig = StoryboardConfig()) -> None:
        self._cfg = cfg
        self._n_cells = cfg.rows * cfg.cols

    # ── Public ────────────────────────────────────────────────────────────────

    def build(self, jpeg_streams: list[io.BytesIO]) -> tuple[np.ndarray, bytes, str]:
        """
        Construct the storyboard from *jpeg_streams*.

        Args:
            jpeg_streams: JPEG-encoded frames from ``EventManager``, oldest first.

        Returns:
            A 3-tuple of:
              - grid_bgr:   ``np.ndarray`` (H×W×3, uint8) — the stitched grid
              - jpeg_bytes: raw JPEG bytes of the grid (for storage / forwarding)
              - b64_str:    base64-encoded JPEG string (for Gemini inline payload)
        """
        selected = self._select_streams(jpeg_streams)
        cells = self._decode_cells(selected)
        grid = self._stitch(cells)
        jpeg_bytes = self._encode_jpeg(grid)
        b64_str = base64.b64encode(jpeg_bytes).decode("ascii")
        return grid, jpeg_bytes, b64_str

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _select_streams(self, streams: list[io.BytesIO]) -> list[Optional[io.BytesIO]]:
        """
        Choose ``n_cells`` streams via evenly-spaced linspace indices.
        Returns a list of length ``n_cells``; missing slots are ``None``.
        """
        n = len(streams)
        if n == 0:
            return [None] * self._n_cells

        if n >= self._n_cells:
            indices = np.linspace(0, n - 1, self._n_cells, dtype=int)
            return [streams[int(i)] for i in indices]

        # Fewer frames than cells: use all, pad the rest
        selected: list[Optional[io.BytesIO]] = list(streams)
        selected += [None] * (self._n_cells - n)
        return selected

    def _decode_cells(
        self, slots: list[Optional[io.BytesIO]]
    ) -> list[np.ndarray]:
        """
        Decode each slot into a resized BGR cell.
        ``None`` slots (padding) become solid-black cells.
        Frame labels (T1–T9) and a final-frame marker are drawn in-place.
        """
        cfg = self._cfg
        cells: list[np.ndarray] = []

        for idx, slot in enumerate(slots):
            cell_idx = idx + 1  # 1-based label

            if slot is not None:
                slot.seek(0)
                raw = np.frombuffer(slot.read(), dtype=np.uint8)
                frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                if frame is None:
                    logger.warning("Could not decode JPEG for cell T%d — using black.", cell_idx)
                    frame = self._black_cell()
                else:
                    frame = cv2.resize(
                        frame, (cfg.cell_w, cfg.cell_h), interpolation=cv2.INTER_AREA
                    )
            else:
                frame = self._black_cell()

            self._draw_label(frame, cell_idx, is_final=(cell_idx == self._n_cells and slot is not None))
            cells.append(frame)

        return cells

    def _black_cell(self) -> np.ndarray:
        return np.zeros((self._cfg.cell_h, self._cfg.cell_w, 3), dtype=np.uint8)

    def _draw_label(self, cell: np.ndarray, idx: int, is_final: bool) -> None:
        """Annotate cell with its temporal index and an optional 'LATEST' badge."""
        cfg = self._cfg
        label = f"T{idx}"
        color = (255, 220, 80)   # warm yellow for standard frames
        shadow = (0, 0, 0)

        # Shadow pass (readability over any background)
        cv2.putText(
            cell, label, (9, 25),
            cv2.FONT_HERSHEY_SIMPLEX, cfg.label_font_scale,
            shadow, cfg.label_thickness + 2, cv2.LINE_AA,
        )
        cv2.putText(
            cell, label, (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX, cfg.label_font_scale,
            color, cfg.label_thickness, cv2.LINE_AA,
        )

        if is_final:
            # Draw a small red "LATEST" badge in the bottom-right corner
            badge = "LATEST"
            bx = cfg.cell_w - 76
            by = cfg.cell_h - 8
            cv2.putText(cell, badge, (bx + 1, by + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(cell, badge, (bx, by),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 80, 255), 1, cv2.LINE_AA)

    def _stitch(self, cells: list[np.ndarray]) -> np.ndarray:
        """Stack cells into rows then stack rows into a full grid."""
        cfg = self._cfg
        rows_imgs: list[np.ndarray] = []
        for r in range(cfg.rows):
            row_cells = cells[r * cfg.cols : (r + 1) * cfg.cols]
            rows_imgs.append(np.hstack(row_cells))
        return np.vstack(rows_imgs)

    def _encode_jpeg(self, grid: np.ndarray) -> bytes:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), self._cfg.storyboard_jpeg_quality]
        ok, buf = cv2.imencode(".jpg", grid, params)
        if not ok:
            raise RuntimeError("cv2.imencode failed for storyboard grid.")
        return buf.tobytes()


# ─── Async Gemini API client ──────────────────────────────────────────────────


class GeminiClient:
    """
    Async Gemini client backed by the new ``google-genai`` SDK
    (``from google import genai``).

    Client lifecycle
    ────────────────
    ``genai.Client`` is a scoped object — no global state is mutated via
    ``configure()``.  Multiple clients can coexist safely in the same process
    (e.g. multiple cameras, unit tests).  The client is cheap to construct
    and reused across all ``analyse()`` calls on this instance.

    Async inference
    ───────────────
    Calls go through ``client.aio.models.generate_content`` — the ``aio``
    sub-namespace exposes native ``async/await`` coroutines, keeping the
    EventManager's dedicated loop fully non-blocking.

    Schema enforcement
    ──────────────────
    ``response_mime_type="application/json"`` plus
    ``response_schema=ViolationAnalysis`` inside ``types.GenerateContentConfig``
    constrains the model's output to a JSON object matching the Pydantic schema.
    The raw text is then validated a second time by Pydantic so all custom
    validators (confidence clamping, ``None`` normalisation) are always applied.

    Retry policy
    ────────────
    Any SDK-level exception (quota, transient server error, network failure)
    is retried up to ``GeminiConfig.max_retries`` times with non-blocking
    exponential back-off via ``asyncio.sleep``.
    """

    def __init__(self, cfg: GeminiConfig) -> None:
        self._cfg = cfg
        # Route all requests through Vertex AI.  Authentication is resolved
        # automatically from Application Default Credentials (ADC) — no API
        # key is passed or stored.
        self._client = genai.Client(
            vertexai=True,
            project=cfg.project,
            location=cfg.location,
        )

    # ── Public ────────────────────────────────────────────────────────────────

    async def analyse(self, jpeg_bytes: bytes) -> ViolationAnalysis:
        """
        Send the storyboard JPEG to Gemini and return a validated
        ``ViolationAnalysis``.

        Args:
            jpeg_bytes: Raw JPEG bytes of the 3×3 temporal storyboard grid.

        Returns:
            A validated ``ViolationAnalysis`` Pydantic model.

        Raises:
            GeminiAPIError: When all retries are exhausted.
        """
        # types.Part.from_bytes attaches the mime-type explicitly so the SDK
        # never has to guess the format from magic bytes.
        grid_image = types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")

        last_exc: Optional[Exception] = None
        for attempt in range(self._cfg.max_retries + 1):
            try:
                response = await self._client.aio.models.generate_content(
                    model=self._cfg.model,
                    contents=[grid_image, _ANALYSIS_PROMPT],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        # ViolationAnalysis is passed as the response_schema;
                        # the SDK derives a constrained JSON schema from the
                        # Pydantic model so the model must emit valid structure.
                        response_schema=ViolationAnalysis,
                        temperature=self._cfg.temperature,
                        max_output_tokens=self._cfg.max_output_tokens,
                    ),
                )
                # Double-validate through Pydantic to enforce custom validators
                # (confidence clamping, None normalisation) that live outside
                # the JSON schema itself.
                return ViolationAnalysis.model_validate_json(response.text)

            except Exception as exc:  # noqa: BLE001
                delay = self._cfg.retry_base_delay * (2 ** attempt)
                logger.warning(
                    "Gemini SDK error (attempt %d/%d): %s. Retrying in %.1f s.",
                    attempt + 1, self._cfg.max_retries + 1, exc, delay,
                )
                last_exc = exc
                if attempt < self._cfg.max_retries:
                    await asyncio.sleep(delay)

        raise GeminiAPIError(
            f"Gemini request failed after {self._cfg.max_retries + 1} attempts."
        ) from last_exc

    async def close(self) -> None:
        """No-op: the SDK manages its own transport lifecycle."""


class GeminiAPIError(RuntimeError):
    """Raised when the Gemini API returns an unrecoverable error."""


# ─── Slack alerting ───────────────────────────────────────────────────────────

_CAMERA_LOCATION = "Zone A - Packing Line"


def _send_slack_alert(analysis: ViolationAnalysis, timestamp: str) -> None:
    """
    Construct a Block Kit payload and POST it to the Slack incoming-webhook.

    Execution model
    ───────────────
    This function is synchronous and uses only stdlib ``urllib`` — no extra
    dependencies are required.  It is always invoked via
    ``asyncio.to_thread`` so the event loop is never blocked, even if Slack's
    endpoint is slow or unreachable.

    Failure handling
    ────────────────
    Any network or HTTP error is caught and logged as a warning so a Slack
    outage can never crash the VLM pipeline.  When ``SLACK_WEBHOOK_URL`` is
    absent (e.g. a development machine) the function returns immediately
    after logging a single warning.
    """
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        logger.warning("SLACK_WEBHOOK_URL not set — Slack alert skipped.")
        return

    payload = {
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "🚨 Safety Violation Detected",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Camera Location:*\n{_CAMERA_LOCATION}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Time:*\n{timestamp}",
                    },
                ],
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Violation Type:*\n{analysis.violation_type}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Confidence:*\n{analysis.confidence_score:.0%}",
                    },
                ],
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Reasoning:*\n{analysis.reasoning}",
                },
            },
            {"type": "divider"},
        ]
    }

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            logger.info(
                "Slack alert sent (HTTP %d) for violation type '%s'.",
                resp.status,
                analysis.violation_type,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Slack alert failed: %s", exc)


# ─── VLM Escalation Handler ───────────────────────────────────────────────────


class VLMEscalationHandler:
    """
    Async callable used as the ``on_event`` callback for ``EventManager``.

    Threading model
    ───────────────
    ``__call__`` is a coroutine.  ``EventManager`` detects this via
    ``asyncio.iscoroutinefunction`` and awaits it directly on its dedicated
    asyncio loop thread — the main video-processing loop is never touched.

    The storyboard build (NumPy + cv2, CPU-bound) is offloaded to a thread
    pool executor so the event loop stays responsive to concurrent events
    from multiple tracked IDs.

    Lifecycle
    ─────────
    Call ``close_sync()`` from the main thread after ``EventManager.shutdown()``
    to chain teardown to any downstream callbacks that own async resources.

    Example
    ───────
    ::

        handler = VLMEscalationHandler(
            gemini_cfg=GeminiConfig(),  # ADC auth; no API key required
            on_analysis=lambda r: print(r.summary()),
        )
        run(..., on_event=handler)
        # close_sync() is called automatically by rtsp_tracker.run()
    """

    def __init__(
        self,
        gemini_cfg: GeminiConfig,
        storyboard_cfg: StoryboardConfig = StoryboardConfig(),
        on_analysis: Optional[Callable[[AnalysisResult], None]] = None,
    ) -> None:
        self._gemini_cfg = gemini_cfg
        self._storyboard = Storyboard(storyboard_cfg)
        self._on_analysis = on_analysis or self._default_on_analysis
        # GeminiClient and the asyncio loop reference are set lazily on first call
        self._client: Optional[GeminiClient] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ── Async callable (EventManager calls this) ──────────────────────────────

    async def __call__(self, event: Event) -> None:
        """
        Full async pipeline: storyboard → Gemini SDK → ViolationAnalysis.

        Args:
            event: The fired ``Event`` from ``EventManager``, carrying
                   the list of ``io.BytesIO`` JPEG streams.
        """
        # Lazily bind client to this event loop on first invocation
        if self._client is None:
            self._loop = asyncio.get_running_loop()
            self._client = GeminiClient(self._gemini_cfg)

        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()

        try:
            # ── Build storyboard in thread pool (CPU-bound) ───────────────────
            grid_bgr, jpeg_bytes, b64_str = await loop.run_in_executor(
                None, self._storyboard.build, event.jpeg_streams
            )
            logger.debug(
                "Storyboard built for ID %d: grid=%s  JPEG=%.1f KB",
                event.track_id,
                f"{grid_bgr.shape[1]}×{grid_bgr.shape[0]}",
                len(jpeg_bytes) / 1024,
            )

            # ── Send to Gemini via SDK (async, non-blocking) ──────────────────
            analysis: ViolationAnalysis = await self._client.analyse(jpeg_bytes)

            # ── Slack alert (fire-and-forget) ─────────────────────────────────
            # Scheduled as a concurrent Task so the HTTP POST to Slack never
            # blocks AnalysisResult construction or the on_analysis dispatch.
            # _send_slack_alert runs in a thread pool via asyncio.to_thread,
            # keeping the event loop fully non-blocking.  All network errors
            # are caught inside _send_slack_alert and demoted to warnings.
            if analysis.violation_detected:
                asyncio.create_task(
                    asyncio.to_thread(
                        _send_slack_alert,
                        analysis,
                        event.triggered_at_str,
                    ),
                    name=f"slack-alert-{event.track_id}",
                )

            latency_ms = (time.perf_counter() - t0) * 1000
            result = AnalysisResult(
                event=event,
                analysis=analysis,
                storyboard_jpeg_bytes=jpeg_bytes,
                api_latency_ms=latency_ms,
                model=self._gemini_cfg.model,
            )

            # ── Dispatch result ───────────────────────────────────────────────
            if asyncio.iscoroutinefunction(self._on_analysis):
                await self._on_analysis(result)  # type: ignore[misc]
            else:
                self._on_analysis(result)

        except GeminiAPIError as exc:
            logger.error("Gemini API error for event ID %d: %s", event.track_id, exc)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Unexpected error in VLM pipeline for event ID %d: %s",
                event.track_id, exc,
            )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close_sync(self) -> None:
        """
        Thread-safe shutdown.

        The ``google-generativeai`` SDK manages its own transport lifecycle;
        there is no session object to drain here.  This method exists solely
        to chain teardown to any downstream ``on_analysis`` callback that
        owns async resources (e.g. ``CloudTelemetry``'s HTTP session).

        Safe to call from the main thread after ``EventManager.shutdown()``.
        Does nothing if the handler was never called.
        """
        if self._client is None:
            return
        logger.info("VLMEscalationHandler closed (SDK transport self-managed).")
        # Chain teardown: if the on_analysis callback (e.g. CloudTelemetry)
        # also owns async resources, let it clean up on the same loop.
        if hasattr(self._on_analysis, "close_sync"):
            self._on_analysis.close_sync()  # type: ignore[union-attr]

    # ── Default analysis dispatcher ───────────────────────────────────────────

    @staticmethod
    def _default_on_analysis(result: AnalysisResult) -> None:
        """
        Log a structured summary.  Replace with a webhook call, database write,
        alarm trigger, or any other downstream action.
        """
        flag_color = "\033[91m" if result.analysis.violation_detected else "\033[92m"
        reset = "\033[0m"
        logger.info(
            "%s┌─ VLM ANALYSIS ────────────────────────────────────────\n"
            "│  Track ID        : %d\n"
            "│  Time            : %s\n"
            "│  Violation       : %s\n"
            "│  Type            : %s\n"
            "│  Confidence      : %.0f%%\n"
            "│  Dwell at trigger: %d frames\n"
            "│  Storyboard      : %.1f KB JPEG\n"
            "│  API latency     : %.0f ms  [%s]\n"
            "└───────────────────────────────────────────────────────%s",
            flag_color,
            result.track_id,
            result.triggered_at_str,
            result.analysis.violation_detected,
            result.analysis.violation_type,
            result.analysis.confidence_score * 100,
            result.event.dwell_frames_at_trigger,
            len(result.storyboard_jpeg_bytes) / 1024,
            result.api_latency_ms,
            result.model,
            reset,
        )
