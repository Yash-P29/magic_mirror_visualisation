# Magic Mirror — Milestone 2A.1.2: Hardened Capture & Telemetry

> **Low-latency, decoupled camera capture + MediaPipe Fast Perception pipeline.**  
> Magic Mirror is a real-time visual understanding system that watches a lecturer through a live camera feed, understands pose and gestures, and overlays visualisations into the video stream.  
> Milestone 2A.1.2 hardens camera capture telemetry, enforces latest-frame-wins invariants, decouples capture FPS from processing FPS, and establishes a baseline for future GStreamer comparison.

---

## What this milestone does

| Capability | Status |
|---|---|
| Open webcam & query actual driver settings (FOURCC, res, FPS, buffer) | ✅ |
| Decoupled threaded capture (`AsyncCameraReader` capacity=1) | ✅ |
| Latest-frame-wins policy (`captured = consumed + dropped + buffered`) | ✅ |
| MediaPipe HolisticLandmarker Fast Perception integration | ✅ |
| Separate Capture FPS vs Processing FPS telemetry | ✅ |
| Disentangled `capture_read_ms` vs `consumer_wait_ms` | ✅ |
| Precise `frame_age_ms` and `capture_to_processing_ms` deltas | ✅ |
| Hand rendering bug fix & `HandData.is_left`/`is_right` properties | ✅ |
| Non-fabricating confidence semantics (`score=None` when unavailable) | ✅ |
| Immutable perception snapshot dataclasses (`Tuple` collections) | ✅ |
| Clean async producer thread shutdown without deadlocks | ✅ |
| Anti-busy-spin camera matrix diagnostic script | ✅ |

---

## Project structure

```
MagicMirror_Visualisation/
│
├── src/
│   ├── __init__.py
│   ├── camera/
│   │   ├── __init__.py
│   │   └── capture.py      ← Camera abstraction & AsyncCameraReader (capacity=1)
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   └── metrics.py      ← FPSCounter & LatencyTracker
│   │
│   ├── perception/
│   │   ├── __init__.py
│   │   ├── models.py       ← Pure Python data models (PoseData, HandData, Landmark)
│   │   └── human.py        ← MediaPipe HolisticLandmarker pipeline
│   │
│   └── main.py             ← Decoupled live capture + perception entry point
│
├── tests/
│   ├── test_camera.py
│   ├── test_async_camera.py
│   ├── test_metrics.py
│   ├── test_perception.py
│   └── test_telemetry_hardening.py  ← Telemetry & invariant regression tests
│
├── scripts/
│   └── camera_matrix_diagnostic.py  ← Multi-backend camera benchmark
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## How to run

```bash
# Default: DSHOW backend, async latest-frame-wins capture, perception enabled
python src/main.py

# Specify backend (dshow, msmf, any)
python src/main.py --backend dshow

# Legacy synchronous mode (no async producer thread)
python src/main.py --sync

# Run full automated test suite
python -m pytest tests/ -v

# Run camera diagnostic matrix
python scripts/camera_matrix_diagnostic.py
```

---

## Telemetry Metrics Definition

### Capture FPS vs Processing FPS
- **Cap FPS**: Rate of frames successfully acquired by the hardware capture thread.
- **Proc FPS**: Rate of frames processed and rendered by the main application loop.
- **Drop Ratio**: Percentage of captured frames overwritten in the 1-element buffer before consumer retrieval.

### Latency Metrics
- **Cap Read (ms)**: Time spent inside `cv2.VideoCapture.read()` inside the camera driver backend.
- **Consumer Wait (ms)**: Time the processing loop waits for a new frame to become available in the latest-frame slot.
- **Frame Age (ms)**: Delta between frame capture timestamp and the moment consumer processing begins (`process_start - capture_timestamp`).
- **App Work (ms)**: Time spent in the application loop performing mirror-flip, perception inference, and HUD rendering.
- **Perception (ms)**: Time spent executing MediaPipe inference model pass.
- **Cap→Total (ms)**: Total end-to-end duration from capture timestamp to processing completion (`process_end - capture_timestamp`).

### FPS (rolling)
Frames per second computed over a **2-second rolling time window**.  
`FPS = (frames_in_window − 1) / (newest_timestamp − oldest_timestamp)`  
Using a window rather than a single frame delta produces a stable, readable number even with minor jitter.

### Frame
The monotonically increasing frame ID assigned to every captured frame.  Starts at 0 when the camera is opened.

### Processing (ms)
Time our application spent on this frame:
```
process_end − process_start
```
where `process_start` is recorded immediately after `Camera.read()` returns and `process_end` is recorded after the overlay is drawn.  This tells us how much latency **our code** adds.

### Cap→Proc (ms)
```
process_end − frame.capture_timestamp
```
This is the time from when OpenCV handed us a decoded frame until we finished processing it.  It is the best approximation of **application-side latency** at this stage.

---

## Known limitations of latency measurement

The numbers shown on screen are **application-side** measurements only.  The following components contribute to true photon-to-screen latency but are **not measured here**:

| Component | Why not measured |
|---|---|
| Webcam sensor / ISP | Happens before OpenCV sees the frame |
| USB or CSI transfer time | Inside the driver stack |
| Kernel / driver frame queue | Not exposed by standard OpenCV APIs |
| OpenCV internal frame buffer | Partially mitigated by `CAP_PROP_BUFFERSIZE = 1`, but not eliminated |
| OS thread scheduling jitter | Between `read()` unblocking and our timestamp |
| OpenCV `imshow` rendering | Not measured after the call |
| Display pipeline / vsync | GPU, monitor scan-out, and response time |

**Do not treat the Cap→Proc number as true end-to-end latency.**  It is a lower bound.  Typical additional unmeasured overhead ranges from tens to hundreds of milliseconds depending on hardware and OS.

---

## Expected output

When running with a connected webcam you should see:

- A live camera window titled **Magic Mirror — Live Camera**.
- A semi-transparent diagnostic panel in the top-left corner showing FPS, frame counter, processing time, and capture→processing latency.
- FPS displayed in green when ≥ 25, amber when lower.
- Terminal logs showing camera configuration on startup and a session summary on exit.

---

## Architecture: where this fits in the future pipeline

```
[Milestone 1]  Camera → reliable low-latency frame stream   ← YOU ARE HERE
[Future]       Fast Perception (pose, gaze, gesture)
[Future]       Temporal Processing (state tracking)
[Future]       Streaming Semantic Understanding
[Future]       Intent Understanding
[Future]       Visualisation Inference
[Future]       Generation
[Future]       AR Overlay
```

Milestone 1 produces a stable `CapturedFrame` object (frame_id + capture_timestamp + image) that downstream components can consume without knowing anything about the camera.  The camera subsystem will not need to be rewritten as later stages are added.

---

## Milestone 2 recommendations

See the implementation report for detailed recommendations.  In brief:

1. **Add a threaded frame producer** — put `Camera.read()` in a background thread with a `queue.Queue(maxsize=1)` to decouple capture latency from processing latency.
2. **Integrate a lightweight pose detector** (e.g. MediaPipe Hands/Pose) as the first Fast Perception stage.
3. **Instrument the queue depth** to detect when processing falls behind capture.
