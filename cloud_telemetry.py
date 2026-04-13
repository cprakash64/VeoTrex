#!/usr/bin/env python3
"""
Cloud Telemetry Module — async Supabase uploader for verified safety violations.

Responsibility
──────────────
Receives an ``AnalysisResult`` from the VLM escalation layer and, when the
result passes the confidence gate (``violation_detected=True`` and
``confidence_score > threshold``), performs two concurrent cloud operations:

  1. Storage upload  — POST the raw storyboard JPEG to a Supabase Storage
                       bucket under a deterministic, content-addressed path.
  2. Database insert — POST a structured JSON record to a Supabase PostgREST
                       table, with the pre-computed storage URL already included.

Both operations share a single ``aiohttp.ClientSession`` and run as concurrent
``asyncio.Task`` objects via ``asyncio.gather``.  Neither operation touches the
main video-processing thread.

Pre-computed URL design
───────────────────────
The storage path is derived deterministically from ``camera_id``, date, and
``(triggered_at_ms, track_id)``:

    {camera_id}/{YYYYMMDD}/{triggered_at_ms}_{track_id}.jpg

Because the public URL is ``{supabase_url}/storage/v1/object/public/{bucket}/{path}``,
we can compute it *before* either operation starts and embed it in the DB row
without waiting for the upload to complete first — enabling true parallelism.

If the storage upload fails (network error, quota, etc.) the database row is
still inserted with the URL recorded so the failure is auditable.  The image
can be re-uploaded later by replaying from the event log.

Non-blocking guarantee
──────────────────────
``CloudTelemetry.__call__`` is a coroutine.  ``VLMEscalationHandler`` detects
this via ``asyncio.iscoroutinefunction`` and awaits it on the EventManager's
dedicated asyncio loop thread — the main video-processing loop is never blocked.

Supabase table schema (run once in your project SQL editor)
───────────────────────────────────────────────────────────
    CREATE TABLE violations (
        id               uuid        DEFAULT gen_random_uuid() PRIMARY KEY,
        camera_id        text        NOT NULL,
        track_id         integer     NOT NULL,
        triggered_at     timestamptz NOT NULL,
        violation_type   text        NOT NULL,
        confidence_score real        NOT NULL CHECK (confidence_score BETWEEN 0 AND 1),
        dwell_frames     integer     NOT NULL,
        reasoning        text        NOT NULL DEFAULT '',
        storyboard_url   text,
        model            text        NOT NULL,
        api_latency_ms   real,
        created_at       timestamptz DEFAULT now()
    );
    CREATE INDEX violations_camera_time_idx
        ON violations (camera_id, triggered_at DESC);
    ALTER TABLE violations ENABLE ROW LEVEL SECURITY;

Supabase Storage setup (run once)
──────────────────────────────────
    -- Via Supabase dashboard → Storage → New bucket
    -- Name: storyboards, Public: true
    -- Or via SQL:
    INSERT INTO storage.buckets (id, name, public)
    VALUES ('storyboards', 'storyboards', true);

Usage
─────
    import os
    from cloud_telemetry import CloudTelemetry, TelemetryConfig
    from vlm_escalation import VLMEscalationHandler, GeminiConfig

    telemetry = CloudTelemetry(
        TelemetryConfig(
            supabase_url=os.environ["SUPABASE_URL"],
            supabase_service_key=os.environ["SUPABASE_SERVICE_KEY"],
            camera_id="entrance_cam_01",
        )
    )
    handler = VLMEscalationHandler(
        gemini_cfg=GeminiConfig(api_key=os.environ["GEMINI_API_KEY"]),
        on_analysis=telemetry,      # async callable — awaited by VLMEscalationHandler
    )
    run(..., on_event=handler)
    # close_sync() is chained automatically:
    #   run() → handler.close_sync() → telemetry.close_sync()

Dependencies
────────────
    pip install aiohttp pydantic
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

import aiohttp

from vlm_escalation import AnalysisResult

logger = logging.getLogger("cloud_telemetry")


# ─── Configuration ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TelemetryConfig:
    """
    Supabase connection and upload policy parameters.

    Use the ``service_role`` key (not the ``anon`` key) so that row-level
    security policies do not block writes from this server-side process.
    Never embed key values in source code; read them from environment variables.
    """

    supabase_url: str               # e.g. "https://abcdefgh.supabase.co"
    supabase_service_key: str       # service_role JWT — has full write access
    camera_id: str                  # unique identifier for this edge device / camera
    db_table: str = "violations"    # PostgREST table name
    storage_bucket: str = "storyboards"
    confidence_threshold: float = 0.85   # gate: skip if score ≤ this value
    request_timeout_seconds: float = 15.0
    max_retries: int = 3
    retry_base_delay: float = 1.0        # seconds; doubles per retry (max ~8 s)
    connector_limit: int = 4             # max simultaneous TCP connections
    dns_cache_ttl: int = 300             # seconds to cache DNS resolutions


# ─── Domain types ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StoredViolation:
    """
    Represents the exact JSON object sent to the Supabase PostgREST endpoint.

    Field names match the SQL column names exactly so ``to_dict()`` can be
    serialised and POSTed without any further transformation.
    """

    camera_id: str
    track_id: int
    triggered_at: str           # ISO 8601 UTC, e.g. "2024-01-15T14:23:45.123Z"
    violation_type: str
    confidence_score: float
    dwell_frames: int
    reasoning: str
    storyboard_url: str         # pre-computed; may point to a not-yet-uploaded object
    model: str
    api_latency_ms: float

    def to_dict(self) -> dict:
        return {
            "camera_id":        self.camera_id,
            "track_id":         self.track_id,
            "triggered_at":     self.triggered_at,
            "violation_type":   self.violation_type,
            "confidence_score": round(self.confidence_score, 6),
            "dwell_frames":     self.dwell_frames,
            "reasoning":        self.reasoning,
            "storyboard_url":   self.storyboard_url,
            "model":            self.model,
            "api_latency_ms":   round(self.api_latency_ms, 2),
        }


@dataclass
class TelemetryOutcome:
    """
    Result of a single cloud telemetry push.  Passed to the ``on_outcome``
    callback; useful for dashboards, alerting, or audit logging.
    """

    camera_id: str
    track_id: int
    triggered_at: str
    storage_path: str
    public_url: str
    db_row_id: Optional[str]    # None if the DB insert failed
    upload_ok: bool
    insert_ok: bool
    total_latency_ms: float

    @property
    def fully_ok(self) -> bool:
        return self.upload_ok and self.insert_ok

    def summary(self) -> str:
        status = "OK" if self.fully_ok else (
            "PARTIAL" if (self.upload_ok or self.insert_ok) else "FAILED"
        )
        return (
            f"[TELEMETRY {status}] "
            f"cam={self.camera_id} id={self.track_id} "
            f"upload={'✓' if self.upload_ok else '✗'} "
            f"insert={'✓' if self.insert_ok else '✗'} "
            f"row={self.db_row_id or 'none'} "
            f"latency={self.total_latency_ms:.0f}ms"
        )


# ─── Low-level Supabase operations ────────────────────────────────────────────


class _SupabaseOps:
    """
    Internal helper that wraps raw ``aiohttp`` calls against the two Supabase
    APIs (Storage and PostgREST).  Uses a single shared ``ClientSession`` with
    keep-alive connections for all requests from this process.

    All methods are coroutines and contain retry loops using
    ``asyncio.sleep`` — they never block the event loop.
    """

    def __init__(self, cfg: TelemetryConfig, session: aiohttp.ClientSession) -> None:
        self._cfg = cfg
        self._session = session
        self._storage_base = f"{cfg.supabase_url}/storage/v1/object"
        self._rest_base = f"{cfg.supabase_url}/rest/v1"
        # Headers reused for every request to avoid repeated dict construction
        self._auth_headers = {
            "Authorization": f"Bearer {cfg.supabase_service_key}",
            "apikey": cfg.supabase_service_key,  # required by PostgREST gateway
        }

    # ── Storage ───────────────────────────────────────────────────────────────

    async def upload_jpeg(
        self,
        storage_path: str,
        jpeg_bytes: bytes,
    ) -> None:
        """
        Upload *jpeg_bytes* to ``{bucket}/{storage_path}`` via the Supabase
        Storage REST API.

        Uses ``x-upsert: true`` so re-uploads overwrite silently (idempotent).
        Content-Length is set explicitly to avoid chunked encoding, which some
        edge proxies reject.

        Raises:
            SupabaseStorageError: after exhausting all retries.
        """
        url = f"{self._storage_base}/{self._cfg.storage_bucket}/{storage_path}"
        headers = {
            **self._auth_headers,
            "Content-Type":   "image/jpeg",
            "Content-Length": str(len(jpeg_bytes)),
            "x-upsert":       "true",
            "Cache-Control":  "max-age=3600",
        }
        timeout = aiohttp.ClientTimeout(total=self._cfg.request_timeout_seconds)
        await self._post_with_retry(
            url=url,
            data=jpeg_bytes,
            headers=headers,
            timeout=timeout,
            op_name="storage-upload",
            error_cls=SupabaseStorageError,
        )
        logger.debug("Storage upload complete: %s (%d B)", storage_path, len(jpeg_bytes))

    # ── Database ──────────────────────────────────────────────────────────────

    async def insert_violation(self, record: StoredViolation) -> str:
        """
        Insert *record* into the violations table via PostgREST.

        Returns:
            The ``id`` (UUID string) of the newly created row.

        Raises:
            SupabaseDBError: after exhausting all retries.
        """
        url = f"{self._rest_base}/{self._cfg.db_table}"
        headers = {
            **self._auth_headers,
            "Content-Type": "application/json",
            # Ask PostgREST to return the full inserted row (including generated id)
            "Prefer": "return=representation",
        }
        timeout = aiohttp.ClientTimeout(total=self._cfg.request_timeout_seconds)
        response_body = await self._post_with_retry(
            url=url,
            json=record.to_dict(),
            headers=headers,
            timeout=timeout,
            op_name="db-insert",
            error_cls=SupabaseDBError,
        )
        # PostgREST returns an array of inserted rows
        try:
            row_id: str = response_body[0]["id"]
        except (IndexError, KeyError, TypeError) as exc:
            raise SupabaseDBError(
                f"Unexpected PostgREST response shape: {response_body}"
            ) from exc
        logger.debug("DB insert complete: row id=%s", row_id)
        return row_id

    # ── Shared retry kernel ───────────────────────────────────────────────────

    async def _post_with_retry(
        self,
        *,
        url: str,
        headers: dict,
        timeout: aiohttp.ClientTimeout,
        op_name: str,
        error_cls: type[Exception],
        data: Optional[bytes] = None,
        json: Optional[dict] = None,
    ) -> object:
        """
        POST to *url* with exponential back-off on transient failures.

        Retry conditions:
          - HTTP 429 (rate-limited)
          - HTTP 5xx (server / gateway error)
          - ``aiohttp.ClientConnectorError`` (TCP-level failure)
          - ``asyncio.TimeoutError``

        Immediate failure (no retry):
          - HTTP 4xx other than 429 (client error — retrying won't help)

        Returns:
            Parsed JSON body on HTTP 2xx.

        Raises:
            *error_cls*: after all retries are exhausted.
        """
        cfg = self._cfg
        last_exc: Optional[Exception] = None

        for attempt in range(cfg.max_retries + 1):
            try:
                async with self._session.post(
                    url, headers=headers, data=data, json=json, timeout=timeout
                ) as resp:

                    if resp.status in range(200, 300):
                        # Parse JSON only once, on success
                        return await resp.json(content_type=None)

                    body_text = await resp.text()

                    if resp.status == 429 or resp.status >= 500:
                        delay = cfg.retry_base_delay * (2 ** attempt)
                        logger.warning(
                            "[%s] HTTP %d (attempt %d/%d). Body: %r. Retrying in %.1f s.",
                            op_name, resp.status,
                            attempt + 1, cfg.max_retries + 1,
                            body_text[:160], delay,
                        )
                        await asyncio.sleep(delay)
                        continue

                    # Definitive 4xx client error — raise immediately
                    raise error_cls(
                        f"[{op_name}] Supabase client error {resp.status}: {body_text[:300]}"
                    )

            except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as exc:
                delay = cfg.retry_base_delay * (2 ** attempt)
                logger.warning(
                    "[%s] Network error (attempt %d/%d): %s. Retrying in %.1f s.",
                    op_name, attempt + 1, cfg.max_retries + 1, exc, delay,
                )
                last_exc = exc
                await asyncio.sleep(delay)

        raise error_cls(
            f"[{op_name}] All {cfg.max_retries + 1} attempts failed."
        ) from last_exc


class SupabaseStorageError(RuntimeError):
    """Raised when the Supabase Storage upload fails permanently."""


class SupabaseDBError(RuntimeError):
    """Raised when the Supabase PostgREST insert fails permanently."""


# ─── Cloud Telemetry orchestrator ─────────────────────────────────────────────


class CloudTelemetry:
    """
    Async callable that gates, packages, and dispatches verified violations
    to Supabase Storage (image) and PostgREST (structured record) concurrently.

    Gate conditions (both must hold to trigger any upload):
      • ``analysis.violation_detected is True``
      • ``analysis.confidence_score > cfg.confidence_threshold``   (default 0.85)

    Concurrency model
    ─────────────────
    Inside ``__call__``, two ``asyncio.Task`` objects are created and awaited
    together via ``asyncio.gather(..., return_exceptions=True)``:

        storage_task  ─┐
                        ├── asyncio.gather  ──► TelemetryOutcome
        db_task       ─┘

    ``return_exceptions=True`` prevents one failure from cancelling the other;
    each outcome is inspected individually and logged.

    The storage path is computed before either task starts, so the public URL
    is embedded in the DB record at insert time — no sequential dependency.

    Session lifecycle
    ─────────────────
    The ``aiohttp.ClientSession`` is created lazily on the first ``__call__``
    invocation (which always runs on the EventManager's asyncio loop thread).
    Call ``close_sync()`` from the main thread at shutdown.  This is wired
    automatically when this instance is passed as ``on_analysis`` to
    ``VLMEscalationHandler`` — its ``close_sync()`` chains here.
    """

    def __init__(
        self,
        cfg: TelemetryConfig,
        on_outcome: Optional[Callable[[TelemetryOutcome], None]] = None,
    ) -> None:
        self._cfg = cfg
        self._on_outcome = on_outcome or self._default_on_outcome
        # Lazy-initialised on first async call to ensure correct event loop binding
        self._ops: Optional[_SupabaseOps] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ── Async callable (VLMEscalationHandler calls this) ─────────────────────

    async def __call__(self, result: AnalysisResult) -> None:
        """
        Entry point called by ``VLMEscalationHandler`` after every VLM analysis.

        Silently skips results that do not meet the confidence gate.
        All upload work is non-blocking from the caller's perspective.
        """
        # ── Gate: only persist high-confidence, confirmed violations ──────────
        if not result.analysis.violation_detected:
            logger.debug(
                "Telemetry skipped for ID %d: violation_detected=False.",
                result.track_id,
            )
            return

        if result.analysis.confidence_score <= self._cfg.confidence_threshold:
            logger.info(
                "Telemetry skipped for ID %d: confidence %.3f ≤ threshold %.2f.",
                result.track_id,
                result.analysis.confidence_score,
                self._cfg.confidence_threshold,
            )
            return

        await self._push(result)

    # ── Core upload pipeline ──────────────────────────────────────────────────

    async def _push(self, result: AnalysisResult) -> None:
        """
        Compute the deterministic storage path, then fire the storage upload
        and DB insert as concurrent tasks.
        """
        ops = await self._get_ops()

        triggered_dt = datetime.fromtimestamp(result.event.triggered_at, tz=timezone.utc)
        triggered_ms = int(result.event.triggered_at * 1000)

        # Deterministic path: camera/YYYYMMDD/timestamp_trackid.jpg
        storage_path = (
            f"{self._cfg.camera_id}/"
            f"{triggered_dt.strftime('%Y%m%d')}/"
            f"{triggered_ms}_{result.track_id}.jpg"
        )
        # Public URL is known before either upload starts — enables true parallelism
        public_url = (
            f"{self._cfg.supabase_url}/storage/v1/object/public/"
            f"{self._cfg.storage_bucket}/{storage_path}"
        )

        record = StoredViolation(
            camera_id        = self._cfg.camera_id,
            track_id         = result.track_id,
            triggered_at     = triggered_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            violation_type   = result.analysis.violation_type,
            confidence_score = result.analysis.confidence_score,
            dwell_frames     = result.event.dwell_frames_at_trigger,
            reasoning        = result.analysis.reasoning,
            storyboard_url   = public_url,
            model            = result.model,
            api_latency_ms   = result.api_latency_ms,
        )

        # Keep a local reference to bytes so the AnalysisResult can be released
        # by the caller's scope immediately after this coroutine returns.
        jpeg_bytes = result.storyboard_jpeg_bytes

        t0 = time.perf_counter()

        # ── Concurrent tasks ──────────────────────────────────────────────────
        storage_task = asyncio.create_task(
            ops.upload_jpeg(storage_path, jpeg_bytes),
            name=f"storage-{result.track_id}-{triggered_ms}",
        )
        db_task = asyncio.create_task(
            ops.insert_violation(record),
            name=f"db-{result.track_id}-{triggered_ms}",
        )

        # gather with return_exceptions=True: one failure cannot cancel the other
        storage_res, db_res = await asyncio.gather(
            storage_task, db_task, return_exceptions=True
        )

        total_ms = (time.perf_counter() - t0) * 1000
        upload_ok = not isinstance(storage_res, Exception)
        insert_ok = not isinstance(db_res, Exception)

        if not upload_ok:
            logger.error(
                "Storage upload FAILED for cam=%s id=%d path=%s: %s",
                self._cfg.camera_id, result.track_id, storage_path, storage_res,
            )
        if not insert_ok:
            logger.error(
                "DB insert FAILED for cam=%s id=%d: %s",
                self._cfg.camera_id, result.track_id, db_res,
            )

        row_id: Optional[str] = db_res if insert_ok else None  # type: ignore[assignment]

        outcome = TelemetryOutcome(
            camera_id        = self._cfg.camera_id,
            track_id         = result.track_id,
            triggered_at     = record.triggered_at,
            storage_path     = storage_path,
            public_url       = public_url,
            db_row_id        = row_id,
            upload_ok        = upload_ok,
            insert_ok        = insert_ok,
            total_latency_ms = total_ms,
        )

        # Dispatch outcome to callback (sync or async)
        if asyncio.iscoroutinefunction(self._on_outcome):
            await self._on_outcome(outcome)  # type: ignore[misc]
        else:
            self._on_outcome(outcome)

    # ── Session / ops lifecycle ───────────────────────────────────────────────

    async def _get_ops(self) -> _SupabaseOps:
        """Lazily create the shared aiohttp session and ops helper."""
        if self._ops is None:
            self._loop = asyncio.get_running_loop()
            connector = aiohttp.TCPConnector(
                limit=self._cfg.connector_limit,
                ttl_dns_cache=self._cfg.dns_cache_ttl,
                enable_cleanup_closed=True,
            )
            self._session = aiohttp.ClientSession(connector=connector)
            self._ops = _SupabaseOps(self._cfg, self._session)
            logger.info(
                "CloudTelemetry session initialised (url=%s camera=%s).",
                self._cfg.supabase_url, self._cfg.camera_id,
            )
        return self._ops

    def close_sync(self) -> None:
        """
        Thread-safe shutdown.  Submits ``_close()`` to the event loop that
        owns the aiohttp session and blocks until it completes (max 5 s).

        Called automatically by ``VLMEscalationHandler.close_sync()`` when
        this instance is the ``on_analysis`` callback.
        """
        if self._session is None or self._loop is None:
            return  # never initialised — nothing to close
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._close(), self._loop
            )
            future.result(timeout=5.0)
            logger.info("CloudTelemetry aiohttp session closed.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error closing CloudTelemetry session: %s", exc)

    async def _close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ── Default outcome handler ───────────────────────────────────────────────

    @staticmethod
    def _default_on_outcome(outcome: TelemetryOutcome) -> None:
        """
        Emit a structured log line for every telemetry push attempt.

        Replace or supplement with: Slack alert, PagerDuty call, metrics
        increment, local SQLite audit log, etc.
        """
        if outcome.fully_ok:
            logger.info(
                "┌─ TELEMETRY PUSHED ─────────────────────────────────────\n"
                "│  Camera    : %s\n"
                "│  Track ID  : %d\n"
                "│  Time      : %s\n"
                "│  DB row    : %s\n"
                "│  Image URL : %s\n"
                "│  Latency   : %.0f ms\n"
                "└─────────────────────────────────────────────────────────",
                outcome.camera_id,
                outcome.track_id,
                outcome.triggered_at,
                outcome.db_row_id,
                outcome.public_url,
                outcome.total_latency_ms,
            )
        else:
            logger.warning(outcome.summary())
