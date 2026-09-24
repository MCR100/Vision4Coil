# Vision4Coil

> ⚙️ Python-based coil inspection using FFT and YOLO object detection.

A vision-based system to detect and analyze frequency characteristics in steel coil manufacturing using real-time or video feed. The system uses FFT analysis and YOLO detection to monitor coil tail presence and save relevant visual and statistical data during significant coil motion.

The system can run either:

- **Standalone (local GUI)** using OpenCV windows and Matplotlib graphs
- **Web interface mode** using a browser-based dashboard

---

## 📦 Installation

Create a virtual environment (optional but recommended), then install dependencies:

```bash
pip install -r requirements.txt
```

### Requirements
```
opencv-python~=4.11.0.86
numpy~=2.2.6
plotly~=6.1.2
matplotlib~=3.10.3
ultralytics~=8.3.158
pandas~=2.3.0
flask
```

---

## 📹 Using RTSP Stream

To process a live camera feed via RTSP, edit the script (`FFT_RTSP.py`) to include:

```python
USERNAME = "your_username"
PASSWORD = "your_password"
CAMERA_IP = "camera_ip"  # e.g., 192.168.1.100
RTSP_URL = f"rtsp://{USERNAME}:{PASSWORD}@{CAMERA_IP}:554/cam/realmonitor?channel=1&subtype=1"
```

Then uncomment:
```python
process_rtsp_stream(RTSP_URL, roi_points)
```

---

## 📁 Output Folder Structure

When a coil motion segment is detected (lasting at least 10 seconds), a timestamped folder is created under 'output/'. For example:

```
2025_Jul_13-14-00-12_to_14-00-22/
├── 2025_Jul_13-14-00-12_to_14-00-22.txt        # Frequency intensity over time
├── 2025_Jul_13-14-00-12_to_14-00-22.html       # Interactive Plotly graph
├── tail_detected_0.87.jpg                      # Frame with detected tail and bounding box
└── tail_detected_0.87.json                     # YOLO detection metadata (class, confidence, bbox)
```

Multiple segments will result in multiple such folders.

Each saved segment also acts as a capture for manual labeling. By default, the capture stores the last 30 segment frames; adjust `SEGMENT_FRAME_BUFFER_SIZE` in `FFT_RTSP.py` if you need more or fewer frames per capture.

```
2025_Jul_13-14-00-12_to_14-00-22/
├── manifest.json                              # Capture metadata and frame index
├── frames/                                    # Raw frames from the FFT segment
│   ├── frame_000000.jpg
│   └── frame_000001.jpg
└── labels/
    └── true_ellipse.json                     # Manual source-of-truth ellipse label
```

Start the capture labeler with:

```bash
python label_capture_web.py
```

The labeler loads `planar_homography.npz` by default. Click **Measure distance**,
press at the first point, drag, and release at the second point to display the
calibrated distance in millimetres. Measurement mode and ellipse drawing are
mutually exclusive. To use a different mapping:

```bash
python label_capture_web.py --homography path/to/planar_homography.npz
```

Then open:

```
http://localhost:8050
```

---

## ⚙️ Threshold Settings

FFT intensity threshold varies by coil type and must be set manually in the script for now:

```python
THRESHOLD = 4264.8  # For DB16
# THRESHOLD = 3200  # For R5.5
# THRESHOLD = 3900  # For R8.5
```

> In future versions, a configuration file will support automatic mapping between coil thickness, threshold, and acceptable ranges.

---

## ▶️ Running the Script

### Local GUI Mode (OpenCV + Matplotlib)

To run the inspection locally with the classic OpenCV window display:

```bash
python FFT_RTSP.py
```

This launches:

- **ROI visualization using `cv2.imshow()`**
- **Live intensity graph using Matplotlib**
- Automatic segment detection and saving

To run on a saved video, edit the bottom of `FFT_RTSP.py`:

```python
video_path = "long_video.mov"
process_rtsp_stream(video_path, roi_points)
```

---

### Web Interface Mode

A browser-based interface is also available.

Start the web server:

```bash
python webserver.py
```

Then open:

```
http://localhost:8000
```

The web interface provides:

- Live ROI video stream
- Full frame preview
- Real-time FFT intensity graph
- Start/Stop processing controls
- RTSP or video file input

The web server runs the same processing pipeline internally but streams results to the browser instead of using OpenCV GUI windows.

---

### Saving Logs

Every input run now creates its own log automatically under `logs/`, whether
processing is started from `FFT_RTSP.py`, from another headless caller, or
from the web interface. The generated filename contains the UTC start time and
a short unique ID:

```text
logs/pipeline_20260924T141530_123456Z_a1b2c3d4.log
```

Each line has a UTC timestamp and severity. The log records:

- Pipeline and input start/end events, run mode, and total duration
- Segment detection and skipped short segments
- Capture processing start/end, status, frame count, and processing duration
- Missing tail masks, missing ellipse fits, recoverable warnings, and exceptions

For example:

```text
2026-09-24T14:15:30.123Z | INFO | pipeline_started mode=web source=video.mov log_file=logs/...
2026-09-24T14:16:02.456Z | INFO | capture_processing_finished capture_id=... status=completed duration_s=6.731
2026-09-24T14:16:04.789Z | INFO | pipeline_finished status=completed duration_s=34.666 frames_processed=900 captures_processed=1
```

Credentials in RTSP URLs are replaced with `***:***` in log messages.
In web mode, the current log path is also returned as `log_file` by
`/api/state`.

---

## ✨ Coming Soon

- Config file for different coil types
- Web interface improvements (ROI selection, live controls)
- Integration with industrial dashboard
