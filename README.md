# Magic Mirror — Milestone 1: Live Camera Foundation

> **This is the foundation for a future real-time visual understanding pipeline.**  
> Magic Mirror will ultimately become a low-latency, multimodal system that watches a lecturer through a live camera feed, understands what is being explained, infers when a visualisation would help, and overlays that visualisation into the live video.  
> Milestone 1 builds only the stable, measurable camera layer that everything else will be built on top of.

---

## What this milestone does

| Capability | Status |
|---|---|
| Open webcam | ✅ |
| Continuous frame capture | ✅ |
| Live display window | ✅ |
| Monotonic timestamps per frame | ✅ |
| Monotonically increasing frame ID | ✅ |
| Rolling FPS measurement | ✅ |
| Per-frame processing time measurement | ✅ |
| Capture→processing latency measurement | ✅ |
| On-screen diagnostic overlay | ✅ |
| Camera failure / disconnection handling | ✅ |
| Clean shutdown (q / ESC / window-close / Ctrl-C) | ✅ |
| Low-latency camera buffer configuration | ✅ |
| Session summary on exit | ✅ |

**Not included (future milestones):** YOLO, MediaPipe, VLMs, LLMs, speech recognition, audio, AR, cloud services, databases.

---

## Project structure

```
magic-mirror/
│
├── src/
│   ├── __init__.py
│   ├── camera/
│   │   ├── __init__.py
│   │   └── capture.py      ← Camera abstraction + CapturedFrame
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   └── metrics.py      ← FPSCounter + LatencyTracker
│   │
│   └── main.py             ← Entry point
│
├── tests/
│   ├── __init__.py
│   ├── test_camera.py      ← Camera unit tests (no webcam required)
│   └── test_metrics.py     ← Metrics unit tests (no webcam required)
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Supported Python version

Python **3.10 or later** is recommended.  
The code uses `from __future__ import annotations` for compatibility back to Python 3.9 if needed, but this has not been tested below 3.10.

---

## Dependencies

| Package | Purpose |
|---|---|
| `opencv-python >= 4.8` | Camera capture and display |
| `numpy >= 1.24` | Frame array operations (transitively required by OpenCV) |
| `pytest >= 7.4` | Test runner (development only) |

No other runtime dependencies.  No web framework.  No Docker.

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd MagicMirror_Visualisation

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## How to run

```bash
# Default: camera device 0
python src/main.py

# Specify a different camera device
python src/main.py --device 1

# Disable horizontal mirror-flip (selfie mode is on by default)
python src/main.py --no-flip
```

**To quit:** press `q`, `ESC`, or close the window.

The application prints a session summary to the terminal when it exits:

```
[21:05:14] [INFO] =====================================================
[21:05:14] [INFO]   Session summary
[21:05:14] [INFO] -----------------------------------------------------
[21:05:14] [INFO] Total frames captured : 2317
[21:05:14] [INFO] Min / Max processing  : 0.18 ms / 6.42 ms
[21:05:14] [INFO] Min / Max Cap→Proc    : 0.44 ms / 11.20 ms
[21:05:14] [INFO] Avg processing        : 1.83 ms
[21:05:14] [INFO] Avg Cap→Proc          : 3.51 ms
[21:05:14] [INFO] =====================================================
```

---

## Running the tests

No webcam required for the test suite.

```bash
pytest tests/ -v
```

---

## Camera configuration

The application requests **1280 × 720 at 30 FPS**.

It then reads back the actual configuration the driver accepted and reports it to the terminal. The requested values are hints — the driver may silently ignore them and run at a different resolution or frame rate. The application adapts to whatever the camera reports.

```
[21:04:58] [INFO] Camera resolution : 1280x720
[21:04:58] [INFO] Camera FPS (driver): 30.0
```

### Buffer size

The application sets `CAP_PROP_BUFFERSIZE = 1` on the OpenCV capture object to reduce the number of pre-decoded frames queued internally. This is supported on some backends (V4L2 on Linux, MSMF on Windows) but not all. If the setting is ignored, the application continues normally; a debug log message is emitted.

---

## Explanation of each metric

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
