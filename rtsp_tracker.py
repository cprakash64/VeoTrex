#!/usr/bin/env python3
"""
High-performance RTSP person tracker with Event Management System.
YOLOv8 Nano + ByteTrack + Polygon ROI Dwell Detection + Async Event Pipeline.

Architecture
────────────
  FrameReader (daemon thread)
      └── queue.Queue(maxsize=1)  ← always holds the LATEST frame only
  FrameBuffer
      └── deque(maxlen=60)        ← rolling window of raw frames for snapshots
  Main thread
      ├── PersonDetector  (YOLOv8 Nano)
      ├── PersonTracker   (ByteTrack)
      ├── EntityRegistry  (per-ID state: dwell frames, cooldown, ROI status)
      └── EventManager    (asyncio loop thread — JPEG extraction + dispatch)

Event lifecycle
───────────────
  1. A TrackedEntity's centroid enters the PolygonROI.
  2. consecutive_roi_frames increments each frame.
  3. At frame 45 (configurable) and cooldown elapsed → Event fires.
  4. FrameBuffer snapshot is taken synchronously (O(n) list copy).
  5. An asyncio coroutine is scheduled on the EventManager's dedicated loop
     thread; CPU-bound JPEG encoding runs in a ThreadPoolExecutor.
  6. Event is dispatched to the user-supplied callback (default: logger).
  7. Per-ID 15 s cooldown blocks re-triggering.

Usage
─────
  python rtsp_tracker.py rtsp://user:pass@192.168.1.100:554/stream1
  python rtsp_tracker.py rtsp://... --device cuda --imgsz 416 --conf 0.45
  python rtsp_tracker.py rtsp://... \\
      --roi '[[100,200],[800,200],[800,600],[100,600]]' \\
      --dwell-frames 45 --cooldown 15 --display

Dependencies
────────────
  pip install ultralytics opencv-python-headless
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import io
import json
import logging
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.trackers import BYTETracker
from ultralytics.utils import IterableSimpleNamespace

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rtsp_tracker")


# ─── Configuration dataclasses ────────────────────────────────────────────────


@dataclass(frozen=True)
class StreamConfig:
    """RTSP stream and reconnection parameters."""

    rtsp_url: str
    reconnect_delay: float = 2.0     # base back-off interval (seconds)
    max_reconnect_attempts: int = 0  # 0 = retry forever
    capture_backend: int = cv2.CAP_FFMPEG


@dataclass(frozen=True)
class DetectorConfig:
    """YOLOv8 inference parameters."""

    model_path: str = "yolov8s.pt"
    confidence: float = 0.40
    iou: float = 0.45
    device: str = "cpu"       # "cuda" | "mps" | "cpu"
    imgsz: int = 1280
    person_class_id: int = 0  # COCO class 0 = person


@dataclass(frozen=True)
class TrackerConfig:
    """ByteTrack hyper-parameters."""

    track_high_thresh: float = 0.50
    track_low_thresh: float = 0.10
    new_track_thresh: float = 0.60
    track_buffer: int = 30   # frames to keep a lost track alive
    match_thresh: float = 0.80
    frame_rate: int = 30


@dataclass(frozen=True)
class EventConfig:
    """
    ROI dwell detection and event pipeline parameters.

    roi_polygon is a list of (x, y) integer vertex tuples in pixel space
    that define the closed region of interest.  A sensible default draws a
    central rectangle — override this for your camera layout.
    """

    roi_polygon: list[tuple[int, int]] = field(
        default_factory=lambda: [(320, 180), (960, 180), (960, 540), (320, 540)]
    )
    dwell_frames: int = 45           # consecutive in-ROI frames before trigger
    cooldown_seconds: float = 15.0   # per-ID minimum gap between events
    buffer_maxlen: int = 60          # rolling frame buffer depth
    snapshot_count: int = 10         # evenly-spaced frames extracted per event
    jpeg_quality: int = 60           # 0–100; lower = smaller in-memory streams
    entity_ttl_seconds: float = 30.0 # prune entities unseen for this long


# ─── Internal data types ─────────────────────────────────────────────────────


@dataclass
class Detections:
    """
    Container for per-frame person detections that satisfies the full
    attribute-and-subscript contract required by ``BYTETracker.update()``.

    Attribute contract
    ──────────────────
    ``BYTETracker`` reads four named attributes on whatever object is passed
    as its first argument:

        results.xyxy   → (N, 4) float32  [x1, y1, x2, y2]
        results.xywh   → (N, 4) float32  [cx, cy, w, h]   used by init_track
        results.conf   → (N,)   float32
        results.cls    → (N,)   float32

    Subscript contract
    ──────────────────
    Internally, ``BYTETracker`` performs index-based sub-selection on the
    detection object, e.g.::

        remain_inds  = scores > self.track_thresh        # boolean mask
        detections   = results[remain_inds]              # calls __getitem__
        u_detection  = results[u_detection_ind]          # integer-array index

    ``__getitem__`` therefore applies the provided index (boolean mask,
    integer array/list, or slice) uniformly to all three internal arrays
    and returns a new ``Detections`` instance — keeping our dataclass as the
    single representation throughout the pipeline instead of converting to
    a raw ``(N, 6)`` array.
    """

    xyxy: np.ndarray  # (N, 4)  absolute pixel coords [x1 y1 x2 y2]
    conf: np.ndarray  # (N,)    confidence in [0, 1]
    cls:  np.ndarray  # (N,)    integer class id

    @classmethod
    def empty(cls) -> "Detections":
        return cls(
            xyxy=np.empty((0, 4), dtype=np.float32),
            conf=np.empty((0,),   dtype=np.float32),
            cls =np.empty((0,),   dtype=np.float32),
        )

    @property
    def xywh(self) -> np.ndarray:
        """
        Return bounding boxes in ``[cx, cy, w, h]`` format as a float32
        array of shape ``(N, 4)``.

        ``BYTETracker.init_track()`` passes this format directly into the
        Kalman filter's state vector, so the property must exist on whatever
        object is handed to ``BYTETracker.update()``.

        Derived on-the-fly from ``self.xyxy`` — no extra storage is needed::

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w  =  x2 - x1
            h  =  y2 - y1
        """
        x1, y1, x2, y2 = (self.xyxy[:, i] for i in range(4))
        return np.stack(
            [
                (x1 + x2) / 2.0,  # cx
                (y1 + y2) / 2.0,  # cy
                x2 - x1,          # w
                y2 - y1,          # h
            ],
            axis=1,
        ).astype(np.float32)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, index: "np.ndarray | list[int] | slice") -> "Detections":
        """
        Return a new ``Detections`` containing only the rows selected by
        *index*.

        NumPy advanced indexing handles all three flavours transparently:

        * **Boolean mask** (shape ``(N,)``): selects rows where the mask is
          ``True``.  Applied to ``xyxy`` (2-D) this selects matching *rows*;
          applied to ``conf`` / ``cls`` (1-D) it selects matching *elements*.
          Both yield arrays of the same length K.
        * **Integer array / list** (e.g. ``[0, 3, 5]``): selects the rows at
          those positions, in order.
        * **Slice** (e.g. ``slice(0, 4)``): selects a contiguous sub-range.

        The returned instance is a *view* where possible (slice) and a *copy*
        where NumPy must materialise a new array (fancy indexing).  In either
        case the caller owns the result independently of this object.
        """
        return Detections(
            xyxy=self.xyxy[index],  # (N,4)[index] → (K,4)
            conf=self.conf[index],  # (N,)[index]  → (K,)
            cls =self.cls[index],   # (N,)[index]  → (K,)
        )


@dataclass
class TrackedEntity:
    """
    Per-track mutable state maintained across frames.

    Only the main thread reads/writes this object; no locking needed.
    """

    track_id: int
    bbox: np.ndarray                            # current [x1, y1, x2, y2]
    centroid: tuple[float, float]               # (cx, cy) derived from bbox
    frames_visible: int = 0                     # total frames track was active
    consecutive_roi_frames: int = 0             # resets when centroid leaves ROI
    in_roi: bool = False                        # current ROI membership
    first_seen_timestamp: float = field(default_factory=time.time)
    last_seen_timestamp:  float = field(default_factory=time.time)
    last_event_timestamp: float = 0.0           # 0.0 = event never fired


@dataclass
class Event:
    """
    Immutable event record produced when a dwell threshold is crossed.

    ``jpeg_streams`` is a list of ``io.BytesIO`` objects, each containing
    a single highly-compressed JPEG frame.  Streams are seeked to position 0
    before delivery so callers can ``.read()`` immediately.
    """

    track_id: int
    triggered_at: float
    dwell_frames_at_trigger: int
    jpeg_streams: list[io.BytesIO]

    @property
    def triggered_at_str(self) -> str:
        return datetime.fromtimestamp(self.triggered_at).strftime("%H:%M:%S.%f")[:-3]

    @property
    def total_jpeg_bytes(self) -> int:
        total = 0
        for s in self.jpeg_streams:
            pos = s.tell()
            s.seek(0, 2)
            total += s.tell()
            s.seek(pos)
        return total


# ─── Rolling FPS counter ──────────────────────────────────────────────────────


class FPSCounter:
    """Thread-safe rolling-window FPS estimator."""

    def __init__(self, window: int = 30) -> None:
        self._window = window
        self._timestamps: list[float] = []
        self._lock = threading.Lock()

    def tick(self) -> float:
        """Record a new frame and return the current FPS estimate."""
        now = time.perf_counter()
        with self._lock:
            self._timestamps.append(now)
            if len(self._timestamps) > self._window:
                self._timestamps.pop(0)
            if len(self._timestamps) < 2:
                return 0.0
            elapsed = self._timestamps[-1] - self._timestamps[0]
            return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0


# ─── Thread-safe rolling frame buffer ────────────────────────────────────────


class FrameBuffer:
    """
    Thread-safe rolling buffer of raw BGR frames backed by ``collections.deque``.

    The main thread pushes frames; the asyncio event-handling coroutine (on a
    different thread) calls ``snapshot()`` to obtain a point-in-time copy.
    A ``threading.Lock`` guards both operations.

    Memory note
    ───────────
    At 1080p, 60 frames ≈ 360 MB.  If memory is constrained, reduce
    ``EventConfig.buffer_maxlen`` or add frame downscaling here.
    """

    def __init__(self, maxlen: int = 60) -> None:
        self._buf: deque[np.ndarray] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, frame: np.ndarray) -> None:
        """Append *frame* to the buffer, evicting the oldest if full."""
        with self._lock:
            self._buf.append(frame)

    def snapshot(self) -> list[np.ndarray]:
        """
        Return a shallow copy of all buffered frames, oldest first.

        The returned list is independent of the deque; callers may hold it
        arbitrarily long without blocking the main thread.
        """
        with self._lock:
            return list(self._buf)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)


# ─── Frame Reader (daemon thread) ────────────────────────────────────────────


class FrameReader:
    """
    Daemon thread that continuously reads frames from an RTSP stream and
    exposes only the *latest* frame via a Queue of size 1.

    Older frames are explicitly discarded so the main thread never processes
    a stale frame — eliminating latency build-up.

    Disconnections trigger an exponential back-off retry loop; the thread
    keeps running until ``stop()`` is called.
    """

    def __init__(self, config: StreamConfig) -> None:
        self._cfg = config
        # maxsize=1: the queue always holds at most one frame (the freshest).
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="FrameReader",
            daemon=True,
        )
        # True for any source that is NOT a network stream.
        # Used in two places:
        #   1. _open_capture() — skips RTSP-specific buffer/timeout tuning that
        #      breaks local-file metadata scanning (moov atom, index tables).
        #   2. inner read loop — injects a 1/30 s sleep to simulate camera pacing
        #      instead of decoding at full CPU speed.
        _url_lower = config.rtsp_url.lower()
        self._is_local: bool = not (
            _url_lower.startswith("rtsp://")
            or _url_lower.startswith("http://")
            or _url_lower.startswith("https://")
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> "FrameReader":
        self._thread.start()
        logger.info("FrameReader started → %s", self._cfg.rtsp_url)
        return self

    def stop(self) -> None:
        self._stop_event.set()

    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Block up to *timeout* seconds for a fresh frame; returns None on stall."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _open_capture(self) -> cv2.VideoCapture:
        if self._is_local:
            return self._open_local()
        return self._open_network()

    def _open_local(self) -> cv2.VideoCapture:
        """
        Open a local file (MP4, AVI, …) with zero network-oriented constraints.

        Why each restriction is lifted
        ───────────────────────────────
        • **No backend hint** — omitting ``cv2.CAP_FFMPEG`` lets OpenCV choose
          the best available backend.  Forcing ``CAP_FFMPEG`` can activate
          network-transport code paths that don't apply to files on disk.

        • **No ``CAP_PROP_BUFFERSIZE = 1``** — this cap tells FFmpeg to keep
          only one decoded frame in its internal queue.  For container formats
          (MP4, MOV, MKV) FFmpeg must first locate and parse the *moov atom*
          (or equivalent metadata/index block) before it can seek to any frame.
          A buffer size of 1 races against that initial scan and causes
          ``moov atom not found`` / ``Stream timeout`` errors on the very first
          ``cap.read()`` call.

        • **``OPENCV_FFMPEG_CAPTURE_OPTIONS`` cleared** — if this environment
          variable is set (e.g., by a parent process, a previous test run, or a
          system-wide profile) it injects FFmpeg AVOptions such as
          ``rtsp_transport`` or ``stimeout`` that are meaningless — and
          potentially fatal — for local file I/O.  We pop it for the duration
          of the constructor call only and restore it immediately afterwards so
          no other thread is affected.
        """
        saved = os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
        try:
            cap = cv2.VideoCapture(self._cfg.rtsp_url)
        finally:
            # Restore unconditionally so the env is always left consistent,
            # even if VideoCapture raises an unexpected exception.
            if saved is not None:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = saved
        return cap

    def _open_network(self) -> cv2.VideoCapture:
        """
        Open an RTSP / HTTP stream with low-latency network optimizations.

        • ``capture_backend`` (default ``cv2.CAP_FFMPEG``) is passed explicitly
          so OpenCV uses the FFmpeg demuxer, which handles RTSP negotiation and
          the full range of transport protocols (TCP/UDP/multicast).
        • ``CAP_PROP_BUFFERSIZE = 1`` minimises the internal decoded-frame
          queue so the main thread always receives the *freshest* frame rather
          than one that has been queued for several hundred milliseconds.
        """
        cap = cv2.VideoCapture(self._cfg.rtsp_url, self._cfg.capture_backend)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _publish(self, frame: np.ndarray) -> None:
        """Drop the stale queued frame (if any) and publish the new one."""
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            pass  # extremely unlikely race; ignore

    def _run(self) -> None:
        attempt = 0
        while not self._stop_event.is_set():
            cap = self._open_capture()
            if not cap.isOpened():
                attempt += 1
                wait = min(self._cfg.reconnect_delay * (2 ** min(attempt - 1, 5)), 60.0)
                logger.warning(
                    "Cannot open stream (attempt %d). Retrying in %.1f s …", attempt, wait
                )
                if (
                    self._cfg.max_reconnect_attempts > 0
                    and attempt >= self._cfg.max_reconnect_attempts
                ):
                    logger.error("Max reconnect attempts reached. Stopping reader.")
                    break
                self._stop_event.wait(wait)
                continue

            logger.info("Stream opened (attempt %d).", attempt + 1)
            attempt = 0

            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    if self._is_local:
                        logger.info("Local file ended — reopening for loop playback.")
                    else:
                        logger.warning("Frame read failed — stream disconnected.")
                    break
                self._publish(frame)
                # Throttle local file decoding to ≈30 FPS so downstream
                # processing sees the same inter-frame cadence as a live camera.
                # The sleep is skipped entirely for real RTSP streams where the
                # network itself provides natural pacing.
                if self._is_local:
                    time.sleep(1.0 / 30.0)

            cap.release()
            if not self._stop_event.is_set():
                logger.info("Reconnecting in %.1f s …", self._cfg.reconnect_delay)
                self._stop_event.wait(self._cfg.reconnect_delay)

        logger.info("FrameReader stopped.")


# ─── Polygon Region of Interest ──────────────────────────────────────────────


class PolygonROI:
    """
    Closed polygon region of interest defined by integer pixel vertices.

    Uses ``cv2.pointPolygonTest`` for sub-pixel accurate containment checks
    in O(n) time relative to the number of polygon vertices.
    """

    def __init__(self, vertices: list[tuple[int, int]]) -> None:
        if len(vertices) < 3:
            raise ValueError("A polygon ROI requires at least 3 vertices.")
        self._pts = np.array(vertices, dtype=np.int32)

    def contains_point(self, point: tuple[float, float]) -> bool:
        """
        Return True if *point* (x, y) lies inside or on the polygon boundary.
        """
        # measureDist=False returns +1 (inside), 0 (on boundary), -1 (outside)
        return cv2.pointPolygonTest(self._pts, point, measureDist=False) >= 0

    def draw(
        self,
        frame: np.ndarray,
        color: tuple[int, int, int] = (0, 255, 255),
        thickness: int = 2,
        fill_alpha: float = 0.10,
    ) -> None:
        """Overlay the polygon on *frame* in-place with an optional filled tint."""
        if fill_alpha > 0:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [self._pts], color)
            cv2.addWeighted(overlay, fill_alpha, frame, 1 - fill_alpha, 0, frame)
        cv2.polylines(frame, [self._pts], isClosed=True, color=color, thickness=thickness)

    @property
    def vertices(self) -> np.ndarray:
        return self._pts


# ─── Person Detector (YOLOv8 Nano) ───────────────────────────────────────────


class PersonDetector:
    """
    Wraps Ultralytics YOLOv8 Nano.

    Filters to the *person* class only and returns a ``Detections`` container
    whose attributes satisfy ``BYTETracker.update()``'s contract.
    """

    def __init__(self, config: DetectorConfig) -> None:
        self._cfg = config
        logger.info(
            "Loading model '%s' on device='%s' …", config.model_path, config.device
        )
        self._model = YOLO(config.model_path)
        self._warmup()

    def _warmup(self) -> None:
        """One dummy inference pass to absorb JIT / kernel-launch overhead."""
        dummy = np.zeros((self._cfg.imgsz, self._cfg.imgsz, 3), dtype=np.uint8)
        self._infer(dummy)
        logger.info("Model warm-up complete.")

    def _infer(self, frame: np.ndarray):  # type: ignore[return]
        return self._model.predict(
            source=frame,
            classes=[self._cfg.person_class_id],
            conf=self._cfg.confidence,
            iou=self._cfg.iou,
            device=self._cfg.device,
            imgsz=self._cfg.imgsz,
            verbose=False,
        )

    def detect(self, frame: np.ndarray) -> Detections:
        """
        Run inference on *frame* and return person detections.

        Returns:
            ``Detections`` with arrays of shape (N, 4), (N,), (N,).
        """
        results = self._infer(frame)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return Detections.empty()
        return Detections(
            xyxy=boxes.xyxy.cpu().numpy().astype(np.float32),
            conf=boxes.conf.cpu().numpy().astype(np.float32),
            cls =boxes.cls.cpu().numpy().astype(np.float32),
        )


# ─── ByteTrack wrapper ────────────────────────────────────────────────────────


class PersonTracker:
    """
    Thin wrapper around Ultralytics' ``BYTETracker``.

    Accepts a ``Detections`` object and returns ``(track_id, bbox_xyxy)`` pairs.
    """

    def __init__(self, config: TrackerConfig) -> None:
        args = IterableSimpleNamespace(
            track_high_thresh=config.track_high_thresh,
            track_low_thresh =config.track_low_thresh,
            new_track_thresh =config.new_track_thresh,
            track_buffer     =config.track_buffer,
            match_thresh     =config.match_thresh,
            fuse_score       =True,
        )
        self._tracker = BYTETracker(args, frame_rate=config.frame_rate)

    def update(
        self,
        detections: Detections,
        frame: np.ndarray,
    ) -> list[tuple[int, np.ndarray]]:
        """
        Feed detections into ByteTrack and return active tracks.

        ``BYTETracker`` accesses detections via named attributes (``.xyxy``,
        ``.conf``, ``.cls``) **and** via subscript (``detections[mask]``).
        ``Detections`` satisfies both contracts natively, so we pass it
        through without any conversion.

        Returns:
            List of ``(track_id, [x1, y1, x2, y2])`` for every live track.
        """
        tracks = self._tracker.update(detections, frame)
        if tracks is None or len(tracks) == 0:
            return []
        result: list[tuple[int, np.ndarray]] = []
        for t in tracks:
            if hasattr(t, "track_id"):
                # Standard STrack object returned by most Ultralytics versions
                result.append((int(t.track_id), t.tlbr))
            else:
                # Raw numpy row: [x1, y1, x2, y2, track_id, ...]
                result.append((int(t[4]), t[:4]))
        return result


# ─── Entity Registry ─────────────────────────────────────────────────────────


class EntityRegistry:
    """
    Maintains a ``TrackedEntity`` for every ByteTrack ID seen so far.

    Only the main thread accesses this object — no locking required.
    Entities are kept alive after a track is lost (for cooldown tracking)
    and pruned by the main loop via ``prune_stale()``.
    """

    def __init__(self) -> None:
        self._entities: dict[int, TrackedEntity] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        active_tracks: list[tuple[int, np.ndarray]],
        roi: PolygonROI,
        now: float,
    ) -> None:
        """
        Reconcile the registry with the current frame's active tracks.

        For each active track:
          - Create a new ``TrackedEntity`` if this ID is first seen.
          - Update position, timestamp, and ROI membership.
          - Increment ``consecutive_roi_frames`` if centroid is in ROI,
            otherwise reset it to 0.

        For IDs no longer in ``active_tracks`` the entity is left intact
        (to preserve cooldown state) but ``consecutive_roi_frames`` is reset.
        """
        active_ids: set[int] = set()

        for tid, bbox in active_tracks:
            active_ids.add(tid)
            cx = float((bbox[0] + bbox[2]) / 2.0)
            cy = float((bbox[1] + bbox[3]) / 2.0)
            centroid: tuple[float, float] = (cx, cy)
            in_roi = roi.contains_point(centroid)

            if tid not in self._entities:
                self._entities[tid] = TrackedEntity(
                    track_id=tid,
                    bbox=bbox.copy(),
                    centroid=centroid,
                    first_seen_timestamp=now,
                    last_seen_timestamp=now,
                )

            entity = self._entities[tid]
            entity.bbox = bbox.copy()
            entity.centroid = centroid
            entity.last_seen_timestamp = now
            entity.frames_visible += 1
            entity.in_roi = in_roi

            if in_roi:
                entity.consecutive_roi_frames += 1
            else:
                entity.consecutive_roi_frames = 0

        # Reset dwell counter for tracks that vanished this frame
        for tid, entity in self._entities.items():
            if tid not in active_ids:
                entity.in_roi = False
                entity.consecutive_roi_frames = 0

    def all_entities(self) -> list[TrackedEntity]:
        """Return all known entities (including recently lost ones)."""
        return list(self._entities.values())

    def active_entities(
        self, active_ids: set[int]
    ) -> list[TrackedEntity]:
        """Return only entities whose track ID is currently active."""
        return [e for e in self._entities.values() if e.track_id in active_ids]

    def prune_stale(self, ttl: float, now: float) -> int:
        """
        Remove entities not seen for more than *ttl* seconds.

        Returns:
            Number of entities pruned.
        """
        stale = [
            tid
            for tid, e in self._entities.items()
            if now - e.last_seen_timestamp > ttl
        ]
        for tid in stale:
            del self._entities[tid]
        return len(stale)

    def __len__(self) -> int:
        return len(self._entities)


# ─── Async Event Manager ─────────────────────────────────────────────────────


class EventManager:
    """
    Evaluates per-entity dwell conditions and fires ``Event`` objects
    asynchronously when all conditions are met.

    Threading model
    ───────────────
    A dedicated daemon thread runs a private ``asyncio`` event loop.
    The main thread calls ``maybe_trigger()`` (synchronous); if conditions
    are met it schedules a coroutine on the async loop via
    ``asyncio.run_coroutine_threadsafe()``.  Inside the coroutine,
    CPU-bound JPEG encoding is further off-loaded to the loop's default
    ``ThreadPoolExecutor`` via ``loop.run_in_executor()``, keeping the
    async loop free for other coroutines.

    Callback
    ────────
    Supply ``on_event`` to handle events (e.g., push to message queue,
    HTTP webhook, database insert).  The default handler logs a summary.
    The callback is invoked from the asyncio thread pool — it must be
    thread-safe.
    """

    def __init__(
        self,
        config: EventConfig,
        on_event: Optional[Callable[[Event], None]] = None,
    ) -> None:
        self._cfg = config
        self._on_event = on_event or self._default_dispatch

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever,
            name="EventLoop",
            daemon=True,
        )
        self._loop_thread.start()
        logger.info("EventManager async loop started.")

    # ── Public API (main thread) ──────────────────────────────────────────────

    def maybe_trigger(
        self,
        entity: TrackedEntity,
        frame_buffer: FrameBuffer,
        now: float,
    ) -> bool:
        """
        Check whether *entity* should fire an event this frame.

        Conditions (all must hold):
          1. ``consecutive_roi_frames`` ≥ ``dwell_frames``
          2. Cooldown has elapsed since the last event for this ID

        If triggered:
          - ``last_event_timestamp`` is updated immediately (main thread) to
            block concurrent triggers before the async task completes.
          - ``consecutive_roi_frames`` is reset to 0.
          - A ``FrameBuffer`` snapshot is taken synchronously.
          - An async coroutine is scheduled for JPEG extraction + dispatch.

        Returns:
            True if an event was triggered this call.
        """
        if entity.consecutive_roi_frames < self._cfg.dwell_frames:
            return False

        elapsed_since_last = now - entity.last_event_timestamp
        if elapsed_since_last < self._cfg.cooldown_seconds:
            remaining = self._cfg.cooldown_seconds - elapsed_since_last
            logger.debug(
                "ID %d cooldown active — %.1f s remaining.", entity.track_id, remaining
            )
            return False

        # ── Arm the event ──────────────────────────────────────────────────
        dwell_count = entity.consecutive_roi_frames  # capture before reset
        entity.last_event_timestamp = now            # block re-trigger NOW
        entity.consecutive_roi_frames = 0            # reset dwell counter

        # Snapshot must be taken on the main thread (O(n) deque copy, very fast)
        snapshot: list[np.ndarray] = frame_buffer.snapshot()

        asyncio.run_coroutine_threadsafe(
            self._process_event(entity.track_id, dwell_count, snapshot, now),
            self._loop,
        )
        logger.info(
            "Event queued for ID %d (dwell=%d frames, buffer=%d frames).",
            entity.track_id,
            dwell_count,
            len(snapshot),
        )
        return True

    def shutdown(self) -> None:
        """Signal the async loop to stop and wait for it to exit."""
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=5.0)
        logger.info("EventManager shut down.")

    # ── Async pipeline (event loop thread) ───────────────────────────────────

    async def _process_event(
        self,
        track_id: int,
        dwell_frames: int,
        snapshot: list[np.ndarray],
        triggered_at: float,
    ) -> None:
        """
        Async coroutine: encode JPEG snapshots then dispatch the Event.

        JPEG encoding is CPU-bound → runs in the default ThreadPoolExecutor
        so this coroutine yields the event loop while waiting.
        """
        loop = asyncio.get_running_loop()
        jpeg_streams: list[io.BytesIO] = await loop.run_in_executor(
            None,  # default ThreadPoolExecutor
            self._encode_jpegs,
            snapshot,
        )

        event = Event(
            track_id=track_id,
            triggered_at=triggered_at,
            dwell_frames_at_trigger=dwell_frames,
            jpeg_streams=jpeg_streams,
        )

        # Support both plain sync callbacks and async callables transparently.
        # Sync callbacks are run in the executor to avoid blocking the event loop.
        #
        # Why two checks instead of one
        # ───────────────────────────────
        # asyncio.iscoroutinefunction (and the inspect variant it wraps) only
        # returns True when the object itself is a function or bound method with
        # the CO_COROUTINE flag.  A *callable instance* — i.e. any object whose
        # class defines ``async def __call__`` — is neither a function nor a
        # method, so the first check alone silently returns False.  The runtime
        # then dispatches to run_in_executor, which calls __call__ in a thread
        # pool, receives a coroutine object (not a result), and discards it →
        # "RuntimeWarning: coroutine '...__call__' was never awaited".
        #
        # The second check, inspect.iscoroutinefunction(type(cb).__call__),
        # catches exactly that pattern: callable class instances whose __call__
        # is declared with ``async def``.  Together the two checks cover every
        # supported callback shape:
        #
        #   • async def fn(event): ...              → first check  ✓
        #   • async def method on bound obj         → first check  ✓
        #   • class with async def __call__(self, event) → second check ✓
        #   • plain sync lambda / function          → neither → executor ✓
        cb = self._on_event
        if inspect.iscoroutinefunction(cb) or inspect.iscoroutinefunction(
            getattr(type(cb), "__call__", None)
        ):
            await cb(event)
        else:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, cb, event)

    def _select_frames(self, snapshot: list[np.ndarray]) -> list[np.ndarray]:
        """
        Select ``snapshot_count`` evenly-spaced frames from *snapshot*.

        Uses ``np.linspace`` to compute indices so the selection spans the
        full temporal extent of the buffer regardless of its current length.
        """
        n = len(snapshot)
        if n == 0:
            return []
        count = min(self._cfg.snapshot_count, n)
        indices = np.linspace(0, n - 1, count, dtype=int)
        return [snapshot[int(i)] for i in indices]

    def _encode_jpegs(self, snapshot: list[np.ndarray]) -> list[io.BytesIO]:
        """
        Encode selected frames as highly-compressed JPEGs into ``io.BytesIO``.

        Runs synchronously inside a thread pool worker — never call from the
        async loop directly.

        Returns:
            List of ``io.BytesIO`` streams, each seeked to position 0.
        """
        frames = self._select_frames(snapshot)
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self._cfg.jpeg_quality]
        streams: list[io.BytesIO] = []

        for idx, frame in enumerate(frames):
            ok, buf = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                logger.warning("JPEG encode failed for snapshot frame %d — skipping.", idx)
                continue
            stream = io.BytesIO(buf.tobytes())
            stream.seek(0)
            streams.append(stream)

        return streams

    # ── Default event handler ─────────────────────────────────────────────────

    @staticmethod
    def _default_dispatch(event: Event) -> None:
        """Log a structured summary of the fired event."""
        size_kb = event.total_jpeg_bytes / 1024
        logger.info(
            "┌─ EVENT FIRED ─────────────────────────────────────────\n"
            "│  Track ID  : %d\n"
            "│  Time      : %s\n"
            "│  Dwell     : %d consecutive frames in ROI\n"
            "│  Snapshots : %d JPEG frames\n"
            "│  JPEG size : %.1f KB (quality=%s)\n"
            "└───────────────────────────────────────────────────────",
            event.track_id,
            event.triggered_at_str,
            event.dwell_frames_at_trigger,
            len(event.jpeg_streams),
            size_kb,
            "varies",  # quality comes from config; shown for info
        )


# ─── Annotation helper ───────────────────────────────────────────────────────


def _annotate_and_show(
    frame: np.ndarray,
    tracks: list[tuple[int, np.ndarray]],
    fps: float,
    roi: PolygonROI,
    registry: EntityRegistry,
) -> None:
    """
    Render bounding boxes, track IDs, ROI polygon, and dwell counters
    onto a copy of *frame* and display it in an OpenCV window.
    """
    vis = frame.copy()

    # Draw ROI polygon (semi-transparent tint + outline)
    roi.draw(vis, color=(0, 255, 255), thickness=2, fill_alpha=0.10)

    # Draw per-track annotations
    entity_map: dict[int, TrackedEntity] = {
        e.track_id: e for e in registry.all_entities()
    }

    for tid, box in tracks:
        x1, y1, x2, y2 = map(int, box)
        entity = entity_map.get(tid)

        # Green when in ROI, grey otherwise
        color = (0, 220, 0) if (entity and entity.in_roi) else (160, 160, 160)

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"ID {tid}"
        if entity:
            label += f"  dwell:{entity.consecutive_roi_frames}"
        cv2.putText(
            vis, label,
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
        )

    # HUD: FPS + active count
    cv2.putText(
        vis, f"FPS: {fps:.1f}  |  People: {len(tracks)}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 80, 255), 2,
    )
    cv2.imshow("RTSP Tracker", vis)


# ─── Main pipeline ────────────────────────────────────────────────────────────


def run(
    stream_cfg: StreamConfig,
    detector_cfg: DetectorConfig,
    tracker_cfg: TrackerConfig,
    event_cfg: EventConfig,
    on_event: Optional[Callable[[Event], None]] = None,
    display: bool = False,
) -> None:
    """
    Wire together all subsystems into a single real-time inference loop.

    Per-frame execution order
    ─────────────────────────
    1. Acquire latest frame from FrameReader queue (blocks up to 2 s).
    2. Push frame to FrameBuffer (rolling window for event snapshots).
    3. Run YOLOv8 person detection.
    4. Update ByteTrack; receive active (track_id, bbox) pairs.
    5. Update EntityRegistry: recompute ROI membership and dwell counters.
    6. For each entity, call EventManager.maybe_trigger() (synchronous check;
       async JPEG work is scheduled if triggered).
    7. Prune stale entities.
    8. Compute FPS; print console status line.
    9. (Optional) Render annotated frame.
    """
    reader    = FrameReader(stream_cfg).start()
    detector  = PersonDetector(detector_cfg)
    tracker   = PersonTracker(tracker_cfg)
    fps_ctr   = FPSCounter(window=30)
    buf       = FrameBuffer(maxlen=event_cfg.buffer_maxlen)
    roi       = PolygonROI(event_cfg.roi_polygon)
    registry  = EntityRegistry()
    event_mgr = EventManager(event_cfg, on_event=on_event)

    logger.info(
        "Pipeline running.  ROI vertices: %s  dwell=%d frames  cooldown=%.0f s",
        event_cfg.roi_polygon,
        event_cfg.dwell_frames,
        event_cfg.cooldown_seconds,
    )
    logger.info("Press Ctrl+C to stop.")

    try:
        while True:
            frame = reader.get_frame(timeout=2.0)
            if frame is None:
                logger.debug("Waiting for frame …")
                continue

            now = time.time()

            # ── 1. Buffer ─────────────────────────────────────────────────────
            buf.push(frame)

            # ── 2. Detect ─────────────────────────────────────────────────────
            detections = detector.detect(frame)

            # ── 3. Track ──────────────────────────────────────────────────────
            active_tracks = tracker.update(detections, frame)
            active_ids = {tid for tid, _ in active_tracks}

            # ── 4. Update entity states ───────────────────────────────────────
            registry.update(active_tracks, roi, now)

            # ── 5. Evaluate event conditions ──────────────────────────────────
            triggered: list[int] = []
            for entity in registry.active_entities(active_ids):
                if event_mgr.maybe_trigger(entity, buf, now):
                    triggered.append(entity.track_id)

            # ── 6. Prune old entities ─────────────────────────────────────────
            registry.prune_stale(ttl=event_cfg.entity_ttl_seconds, now=now)

            # ── 7. Console status ─────────────────────────────────────────────
            fps        = fps_ctr.tick()
            ids        = sorted(active_ids)
            roi_ids    = sorted(e.track_id for e in registry.active_entities(active_ids) if e.in_roi)
            dwell_info = {
                e.track_id: e.consecutive_roi_frames
                for e in registry.active_entities(active_ids)
                if e.in_roi
            }
            print(
                f"\rFPS:{fps:5.1f}  people:{len(ids):3d}  "
                f"IDs:{ids!s:<25}  "
                f"in-ROI:{roi_ids!s:<20}  "
                f"dwell:{dwell_info!s:<25}  "
                f"events:{triggered!s:<15}",
                end="",
                flush=True,
            )

            # ── 8. Optional display ───────────────────────────────────────────
            if display:
                _annotate_and_show(frame, active_tracks, fps, roi, registry)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print()
        logger.info("Interrupted by user.")
    finally:
        reader.stop()
        event_mgr.shutdown()
        # If the caller passed a VLMEscalationHandler (or any handler with a
        # close_sync() hook), give it a chance to drain in-flight async work
        # and close network connections before the process exits.
        if on_event is not None and hasattr(on_event, "close_sync"):
            on_event.close_sync()  # type: ignore[union-attr]
        if display:
            cv2.destroyAllWindows()
        logger.info("Pipeline shut down.")


# ─── CLI entry point ──────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RTSP Person Tracker — YOLOv8 Nano + ByteTrack + ROI Events",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Stream ────────────────────────────────────────────────────────────────
    p.add_argument("rtsp_url", help="RTSP stream URL (e.g. rtsp://user:pw@host/path)")
    p.add_argument(
        "--reconnect-delay", type=float, default=2.0, metavar="SEC",
        help="Base reconnect back-off (doubles on each failure, max 60 s)",
    )
    p.add_argument(
        "--max-retries", type=int, default=0, metavar="N",
        help="Max reconnect attempts (0 = infinite)",
    )

    # ── Detector ──────────────────────────────────────────────────────────────
    p.add_argument("--model",  default="yolov8n.pt")
    p.add_argument("--conf",   type=float, default=0.40, help="Detection confidence threshold")
    p.add_argument("--iou",    type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--device", default="cpu",            help="cuda | mps | cpu")
    p.add_argument("--imgsz",  type=int,   default=640)

    # ── Tracker ───────────────────────────────────────────────────────────────
    p.add_argument("--fps",          type=int,   default=30)
    p.add_argument("--track-high",   type=float, default=0.50)
    p.add_argument("--track-low",    type=float, default=0.10)
    p.add_argument("--new-track",    type=float, default=0.60)
    p.add_argument("--track-buffer", type=int,   default=30)
    p.add_argument("--match-thresh", type=float, default=0.80)

    # ── Event ─────────────────────────────────────────────────────────────────
    p.add_argument(
        "--roi",
        default=None,
        metavar="JSON",
        help=(
            "Polygon ROI as a JSON array of [x,y] pairs, e.g. "
            "'[[100,200],[800,200],[800,600],[100,600]]'.  "
            "Defaults to a centre-frame rectangle."
        ),
    )
    p.add_argument(
        "--dwell-frames", type=int, default=45,
        help="Consecutive in-ROI frames required to fire an event",
    )
    p.add_argument(
        "--cooldown", type=float, default=15.0,
        help="Per-ID minimum gap between events (seconds)",
    )
    p.add_argument(
        "--buffer-len", type=int, default=60,
        help="Rolling frame buffer depth (frames)",
    )
    p.add_argument(
        "--snapshots", type=int, default=10,
        help="Number of evenly-spaced JPEG snapshots extracted per event",
    )
    p.add_argument(
        "--jpeg-quality", type=int, default=60,
        help="JPEG quality for in-memory snapshots (0–100)",
    )

    # ── Misc ──────────────────────────────────────────────────────────────────
    p.add_argument("--display", action="store_true", help="Show annotated OpenCV window")
    p.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")

    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse optional ROI polygon from JSON
    roi_polygon: list[tuple[int, int]]
    if args.roi:
        try:
            raw = json.loads(args.roi)
            roi_polygon = [(int(v[0]), int(v[1])) for v in raw]
        except (json.JSONDecodeError, (IndexError, TypeError, ValueError)) as exc:
            raise SystemExit(f"Invalid --roi JSON: {exc}") from exc
    else:
        roi_polygon = [(320, 180), (960, 180), (960, 540), (320, 540)]

    run(
        stream_cfg=StreamConfig(
            rtsp_url=args.rtsp_url,
            reconnect_delay=args.reconnect_delay,
            max_reconnect_attempts=args.max_retries,
        ),
        detector_cfg=DetectorConfig(
            model_path=args.model,
            confidence=args.conf,
            iou=args.iou,
            device=args.device,
            imgsz=args.imgsz,
        ),
        tracker_cfg=TrackerConfig(
            frame_rate=args.fps,
            track_high_thresh=args.track_high,
            track_low_thresh=args.track_low,
            new_track_thresh=args.new_track,
            track_buffer=args.track_buffer,
            match_thresh=args.match_thresh,
        ),
        event_cfg=EventConfig(
            roi_polygon=roi_polygon,
            dwell_frames=args.dwell_frames,
            cooldown_seconds=args.cooldown,
            buffer_maxlen=args.buffer_len,
            snapshot_count=args.snapshots,
            jpeg_quality=args.jpeg_quality,
        ),
        display=args.display,
    )


if __name__ == "__main__":
    # ── Local test mode ───────────────────────────────────────────────────────
    # Run directly (`python rtsp_tracker.py`) to process the local test video.
    # FrameReader will throttle to 30 FPS automatically because the path does
    # not start with "rtsp://".  Pass CLI arguments to use production settings:
    #   python rtsp_tracker.py rtsp://user:pass@host/stream
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        # Deferred import — vlm_escalation imports Event from this module at its
        # top level, so we must not import it at rtsp_tracker's module level or
        # we'd create a circular dependency.  Importing here (after this module
        # is fully loaded) is the standard Python pattern for breaking cycles.
        from vlm_escalation import VLMEscalationHandler, GeminiConfig

        # ── ANSI colours ──────────────────────────────────────────────────────
        _CYAN  = "\033[96m"   # bright cyan — stands out from the FPS log line
        _BOLD  = "\033[1m"
        _RESET = "\033[0m"

        def print_vlm_verdict(result) -> None:
            """Print the VLM JSON verdict to the console in bright cyan."""
            flag = "⚠  VIOLATION DETECTED" if result.analysis.violation_detected \
                   else "✓  No violation"
            print(
                f"\n{_CYAN}{_BOLD}"
                f"╔══ VLM VERDICT ════════════════════════════════════════════╗\n"
                f"║  {flag}\n"
                f"║  violation_type   : {result.analysis.violation_type}\n"
                f"║  confidence_score : {result.analysis.confidence_score:.2f}\n"
                f"║  reasoning        : {result.analysis.reasoning}\n"
                f"╚═══════════════════════════════════════════════════════════╝"
                f"{_RESET}",
                flush=True,
            )

        _vlm_handler = VLMEscalationHandler(
            gemini_cfg=GeminiConfig(),  # project/location default to Vertex AI config
            on_analysis=print_vlm_verdict,
        )

        _TEST_VIDEO = (
            "/Users/cprakash/Documents/MY_AI/Luna_V2/"
            "Luna_V2_Test_Video/faststart_factory_cam.mp4"
        )
        run(
            stream_cfg=StreamConfig(rtsp_url=_TEST_VIDEO),
            detector_cfg=DetectorConfig(),
            tracker_cfg=TrackerConfig(),
            # ROI expanded to cover almost the entire 1920×1080 frame so that
            # any detected person immediately starts accumulating dwell frames,
            # forcing an event trigger for Phase 3 VLM testing.
            event_cfg=EventConfig(
                roi_polygon=[(320, 180), (960, 180), (960, 540), (320, 540)],
            ),
            on_event=_vlm_handler,
            display=True,
        )
