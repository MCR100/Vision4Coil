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

To run and save the log:

```
python FFT_RTSP.py > output_log.txt 2>&1         # if using Linux or Windows CMD
python FFT_RTSP.py *>&1 | Tee-Object -FilePath output_log.txt   # if using Windows PowerShell
```

If using the web server:

```
python webserver.py > output_log.txt 2>&1
```

---

## ✨ Coming Soon

- Config file for different coil types
- Web interface improvements (ROI selection, live controls)
- Integration with industrial dashboard
