import cv2
import numpy as np
import os
from datetime import timedelta, datetime
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
import json
from collections import deque

# ----- CONFIG -----
THRESHOLD = 4264.8  # 4264.8 for DB16 ; 3200 for R5.5 ; 3900 for R8.5
TARGET_SIZE = (480, 270)

USERNAME = "admin"
PASSWORD = "passkey"
CAMERA_IP = "cam_ip"  # 192.168.1.100
RTSP_URL = f"rtsp://{USERNAME}:{PASSWORD}@{CAMERA_IP}:554/cam/realmonitor?channel=1&subtype=1"

OUTPUT_DIR = Path("output")   # change to whatever you want
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO("best.pt")


# --------------
# Sink interface
# --------------
class BaseSink:
    """
    Processing output sink.
    - Webserver passes its own sink implementing these methods
    - Running script directly uses CvGuiSink
    """

    def on_roi(self, roi_view, t_s: float, intensity: float):
        pass

    def on_frame(self, frame, t_s: float, intensity: float):
        pass

    def on_series(self, t_list, y_list):
        pass

    def should_stop(self) -> bool:
        return False

    def close(self):
        pass


class CvGuiSink(BaseSink):
    """
    Local GUI: cv2.imshow + optional matplotlib live plot.
    Only use this when running locally with a display.
    """

    def __init__(self, show_plot=True, roi_window="ROI View"):
        self.roi_window = roi_window
        self.show_plot = show_plot
        self._stop = False

        self._graph_time = []
        self._graph_intensity = []

        if self.show_plot:
            plt.ion()
            self.fig, self.ax = plt.subplots()
            (self.line,) = self.ax.plot([], [], color="black")
            self.ax.set_title("Live Frequency Intensity")
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Intensity")
            self.ax.set_xlim(0, 30)

    def on_roi(self, roi_view, t_s: float, intensity: float):
        cv2.imshow(self.roi_window, roi_view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self._stop = True

    def on_frame(self, frame, t_s: float, intensity: float):
        # accumulate series for plot if enabled
        if not self.show_plot:
            return

        self._graph_time.append(t_s)
        self._graph_intensity.append(intensity)

        # update plot at a modest rate to reduce CPU
        if len(self._graph_time) % 5 == 0 and self._graph_time:
            self.line.set_xdata(self._graph_time)
            self.line.set_ydata(self._graph_intensity)
            self.ax.set_xlim(min(self._graph_time), max(self._graph_time) + 5)
            self.ax.set_ylim(min(self._graph_intensity) - 50, max(self._graph_intensity) + 50)
            self.ax.figure.canvas.draw()
            self.ax.figure.canvas.flush_events()

    def should_stop(self) -> bool:
        return self._stop

    def close(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        if self.show_plot:
            try:
                plt.ioff()
                plt.show()
            except Exception:
                pass


class NullSink(BaseSink):
    """Headless default sink (no GUI)"""
    pass


# -------------------------
# Your existing functions
# -------------------------
def compute_fft_spectrum(frame, roi_points):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray, dtype=np.uint8)
    roi_contour = np.array(roi_points, dtype=np.int32)
    cv2.fillPoly(mask, [roi_contour], 255)
    roi = cv2.bitwise_and(gray, gray, mask=mask)
    x, y, w, h = cv2.boundingRect(roi_contour)
    roi_cropped = roi[y:y + h, x:x + w]

    roi_float = np.float32(roi_cropped)
    dft = cv2.dft(roi_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft, axes=[0, 1])
    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

    return np.mean(magnitude), roi_cropped


def format_time(seconds):
    return str(timedelta(seconds=int(seconds))).replace(":", "-")


def create_timestamped_folder(start_dt, end_dt):
    folder_name = start_dt.strftime("%Y_%b_%d-%H-%M-%S") + "_to_" + end_dt.strftime("%H-%M-%S")
    folder_path = OUTPUT_DIR / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)
    return str(folder_path)


def save_results_txt(time_axis, intensity_values, save_path):
    txt_file = f"{save_path}.txt"
    with open(txt_file, "w") as f:
        f.write("Time (s)\tFrequency Intensity\n")
        for t, v in zip(time_axis, intensity_values):
            f.write(f"{t:.2f}\t{v:.2f}\n")
    print(f"Saved: {txt_file}")


def save_results_html(time_axis, intensity_values, save_path):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_axis, y=intensity_values, mode='lines', name='Intensity'))
    fig.update_layout(title="Frequency Intensity Over Time",
                      xaxis_title="Time (s)",
                      yaxis_title="Intensity",
                      template="simple_white")
    html_file = f"{save_path}.html"
    fig.write_html(html_file)
    print(f"Saved: {html_file}")

def detect_tail_and_save(frames, roi_points, save_path, conf_thresh=0.6):
    best_conf = 0
    best_frame = None
    best_box = None
    best_cls = None

    for frame in reversed(frames[-10:]):
        result = model.predict(frame, conf=conf_thresh, verbose=False)[0]
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf > best_conf:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                best_conf = conf
                best_cls = cls_id
                best_box = [int(x1), int(y1), int(x2), int(y2)]
                best_frame = frame.copy()
                cv2.rectangle(best_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(best_frame, f"{conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if best_conf > 0 and best_frame is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        img_path = os.path.join(save_path, f"tail_detected_{best_conf:.2f}.jpg")
        cv2.imwrite(img_path, best_frame)
        print(f"Saved tail image: {img_path}")

        # Save JSON label data
        import json
        label_info = {
            "class_id": best_cls,
            "confidence": round(best_conf, 4),
            "bbox": best_box
        }
        json_path = os.path.join(save_path, f"tail_detected_{best_conf:.2f}.json")
        with open(json_path, "w") as f:
            json.dump(label_info, f, indent=2)
        print(f" Saved label info: {json_path}")


# -------------------------
# Main processing loop (GUI-free)
# -------------------------
def process_rtsp_stream(rtsp_url, roi_points, sink: BaseSink | None = None, fps_assumed=30):
    """
    If sink is CvGuiSink -> behaves like your old script (imshow + optional plot).
    If sink is WebSink (from webserver.py) -> no GUI calls, stream data via sink.
    If sink is None -> headless.
    """
    if sink is None:
        sink = NullSink()

    print("Real-time stream started.")

    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Error: Cannot open RTSP/video stream.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = float(fps_assumed)

    frame_idx = 0
    in_segment = False
    segment_start = None

    # bounded buffer to avoid memory issues (last 10 secs used for yolo)
    segment_frames = deque(maxlen=60)  # keep last 2 seconds @30fps
    segment_time = []
    segment_intensities = []

    # for graph we keep bounded history too
    graph_time = deque(maxlen=3000)       # tune: 3000 pts
    graph_intensity = deque(maxlen=3000)  # tune: 3000 pts
    segment_starts = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Stream ended or not receiving frames.")
                break

            current_time = frame_idx / fps
            intensity, roi_view = compute_fft_spectrum(frame, roi_points)

            # GUI/Web output
            sink.on_roi(roi_view, current_time, float(intensity))
            sink.on_frame(frame, current_time, float(intensity))
            if sink.should_stop():
                break

            # update in-memory graph buffers
            graph_time.append(current_time)
            graph_intensity.append(float(intensity))

            # start segment
            if not in_segment and intensity > THRESHOLD:
                in_segment = True
                segment_start = round(current_time, 2)
                segment_frames.clear()
                segment_time = []
                segment_intensities = []
                print(f"Segment START at {segment_start:.2f}s")

            # end segment
            elif in_segment and intensity < THRESHOLD:
                segment_end = round(current_time, 2)
                in_segment = False
                segment_duration = segment_end - segment_start
                print(f"Segment END at {segment_end:.2f}s")

                if segment_duration >= 10:
                    segment_starts.append(segment_start)

                    start_dt = datetime.now() - timedelta(seconds=segment_duration)
                    end_dt = datetime.now()
                    folder = create_timestamped_folder(start_dt, end_dt)
                    base = os.path.join(folder, os.path.basename(folder))
                    save_results_txt(segment_time, segment_intensities, base)
                    save_results_html(segment_time, segment_intensities, base)

                    # only pass small buffered frames (bounded)
                    detect_tail_and_save(list(segment_frames), roi_points, folder)
                else:
                    print(f"Segment duration {segment_duration:.2f}s too short. Skipped.")

                segment_frames.clear()
                segment_time.clear()
                segment_intensities.clear()

            if in_segment:
                # copy to avoid OpenCV reusing buffer internally
                segment_frames.append(frame.copy())
                segment_time.append(current_time)
                segment_intensities.append(float(intensity))

            # feed series to sink occasionally (web can ignore)
            if frame_idx % 10 == 0 and graph_time:
                sink.on_series(list(graph_time), list(graph_intensity))

            frame_idx += 1

    finally:
        cap.release()
        sink.close()
        print("RTSP stream processing complete.")

if __name__ == "__main__":
    roi_points = [(677, 1288), (1325, 1418), (1425, 1171), (893, 1051)]
    video_path = "out.mp4"

    gui = CvGuiSink(show_plot=True)
    # process_rtsp_stream(RTSP_URL, roi_points, sink=gui)
    process_rtsp_stream(video_path, roi_points, sink=gui)