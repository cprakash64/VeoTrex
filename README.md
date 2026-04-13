# New-Luna

AI-powered workplace safety pipeline for real-time video detection, tracking, event triggering, VLM escalation, and cloud telemetry.

## What it does

This project processes a camera or video stream, detects people, tracks them across frames, checks whether they enter a defined ROI for long enough, escalates the event to Gemini for structured analysis, and can send alerts / telemetry to Slack and Supabase.

## Current status

The current codebase already includes:

* RTSP / local video ingestion
* YOLO-based person detection with Ultralytics
* ByteTrack multi-object tracking
* ROI-based dwell detection
* Async event pipeline
* Gemini-based VLM escalation
* Slack alert hook
* Supabase telemetry layer for storage and database inserts

## Main files

### `rtsp_tracker.py`

Handles the live video pipeline.

Contains:

* `StreamConfig`
* `DetectorConfig`
* `TrackerConfig`
* `EventConfig`
* `Detections`
* `TrackedEntity`
* `Event`
* `FPSCounter`
* `FrameBuffer`
* `FrameReader`
* `PolygonROI`
* `PersonDetector`
* `PersonTracker`
* `EntityRegistry`
* `EventManager`

### `vlm_escalation.py`

Handles storyboard creation and Gemini analysis.

Contains:

* `ViolationAnalysis`
* `StoryboardConfig`
* `GeminiConfig`
* `AnalysisResult`
* `Storyboard`
* `GeminiClient`
* `VLMEscalationHandler`
* Slack alert helper

### `cloud_telemetry.py`

Handles cloud storage and database logging.

Contains:

* `TelemetryConfig`
* `StoredViolation`
* `TelemetryOutcome`
* `_SupabaseOps`
* `SupabaseStorageError`
* `SupabaseDBError`
* `CloudTelemetry`

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Recommended packages:

* `ultralytics`
* `opencv-python`
* `aiohttp`
* `google-genai`
* `pydantic`
* `numpy`
* `python-dotenv`

## Environment variables

Create a `.env` file in the project root:

```env
SLACK_WEBHOOK_URL="your_slack_webhook_url"
GOOGLE_CLOUD_PROJECT="your_gcp_project_id"
GOOGLE_APPLICATION_CREDENTIALS="C:/path/to/service-account.json"
```

Notes:

* `SLACK_WEBHOOK_URL` is optional if you only want local testing.
* `GOOGLE_CLOUD_PROJECT` is required for Gemini / Vertex AI.
* `GOOGLE_APPLICATION_CREDENTIALS` is required unless you use `gcloud auth application-default login`.

## Load `.env`

If you use a `.env` file, make sure your main script loads it:

```python
from dotenv import load_dotenv
load_dotenv()
```

## How to run

### Test with a local video file

This is the way you are testing right now:

```bash
python rtsp_tracker.py ./tests/vid1.mp4
```

If your script supports display mode, use it to see the ROI and detections:

```bash
python rtsp_tracker.py ./tests/vid1.mp4 --display
```

### Test with a real RTSP stream

```bash
python rtsp_tracker.py rtsp://user:pass@camera-ip:554/stream
```

## What to expect when it works

You should see logs like:

* frame reader started
* model loaded
* stream opened
* FPS updates
* people detected
* track IDs assigned
* ROI / dwell counters updating
* event fired when dwell threshold is reached
* Gemini analysis result
* Slack alert sent, if configured
* telemetry upload, if configured

## How to test step by step

### 1. Verify the video opens

Run:

```bash
python rtsp_tracker.py ./tests/vid1.mp4
```

If the path is correct, you should see:

* `Stream opened`
* FPS lines in the console

If you see `Cannot open stream`, the path is wrong or the file is not a valid video.

### 2. Verify detection and tracking

Look for output like:

* `people: 1`
* `IDs:[1]`

This means YOLO and ByteTrack are working.

### 3. Verify ROI and dwell

If `in-ROI:[]` stays empty, the person is outside the ROI polygon.

For testing, lower `dwell_frames` to something small like `5` or `10`.

### 4. Verify event firing

Once a person stays inside the ROI long enough, you should see an event queued.

### 5. Verify Gemini escalation

When an event fires, the storyboard is sent to Gemini and you should get a structured analysis result.

### 6. Verify Slack alerting

To test Slack alerts:

* set `SLACK_WEBHOOK_URL`
* make sure Gemini returns `violation_detected: true`
* confirm the alert log appears

## Common issues

### `Cannot open stream`

* the video path is wrong
* the file does not exist
* the file is not a valid video
* the path is relative to the wrong folder

### People are detected but no events fire

* ROI polygon does not cover the person
* `dwell_frames` is too high
* the person does not stay in the ROI long enough

### Slack alert not sent

* `SLACK_WEBHOOK_URL` is missing
* Gemini did not mark the event as a violation
* the event never fired in the first place

### Gemini fails

* Google auth is not configured
* `GOOGLE_CLOUD_PROJECT` is wrong
* `GOOGLE_APPLICATION_CREDENTIALS` is missing

## Notes

* YOLO is used for object detection.
* ByteTrack is used for identity tracking.
* Gemini is used for the final safety analysis and reasoning.
* Supabase is used for cloud telemetry and storage.

## Suggested next improvements

* Add a simple dashboard
* Add event review and resolution workflow
* Add better ROI visualization in UI
* Add tests for event triggering and Slack callbacks
* Add a sample video in `./tests/` for quick local verification

## License

No License currently.
