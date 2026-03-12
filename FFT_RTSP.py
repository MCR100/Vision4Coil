import cv2
import numpy as np
import os
from datetime import timedelta, datetime
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
from collections import deque
import json

# ----- CONFIG -----
THRESHOLD = 4264.8  # 4264.8 for DB16 ; 3200 for R5.5 ; 3900 for R8.5
TARGET_SIZE = (480, 270)

USERNAME = "admin"
PASSWORD = "passkey"
CAMERA_IP = "cam_ip"  # 192.168.1.100
RTSP_URL = f"rtsp://{USERNAME}:{PASSWORD}@{CAMERA_IP}:554/cam/realmonitor?channel=1&subtype=1"

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO("best.pt")

# Fallback body-offset tuning
LOCAL_BAND_HALF_HEIGHT = 180
MIN_BODY_PIXELS_FOR_MEDIAN = 40

# Hardcoded body reference polyline
# Points should run roughly along the body center and are sorted by y internally.
BODY_POLYLINE_POINTS = [
    (1386, 753),
    (1167, 1074),
    (916, 1432)
]


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
        if not self.show_plot:
            return

        self._graph_time.append(t_s)
        self._graph_intensity.append(intensity)

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


def build_body_mask(frame, exclude_box=None):
    """
    Build a rough mask for the coil body from the frame.
    Used only for local median-x fallback.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    if exclude_box is not None:
        x1, y1, x2, y2 = exclude_box
        pad = 10
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(mask.shape[1], x2 + pad)
        y2 = min(mask.shape[0], y2 + pad)
        mask[y1:y2, x1:x2] = 0

    return mask


def median_body_x_from_mask(mask, center_y, half_height):
    y1 = max(0, int(round(center_y)) - half_height)
    y2 = min(mask.shape[0], int(round(center_y)) + half_height)

    local = mask[y1:y2, :]
    ys, xs = np.where(local > 0)
    if len(xs) < MIN_BODY_PIXELS_FOR_MEDIAN:
        return None

    return float(np.median(xs))


def get_sorted_polyline(points):
    return sorted(points, key=lambda p: p[1])


def polyline_x_at_y(points, y):
    """
    Interpolate x along a hardcoded polyline at the requested y.
    Returns None if y is outside the polyline y-range or interpolation is not possible.
    """
    pts = get_sorted_polyline(points)
    if len(pts) < 2:
        return None

    y_min = pts[0][1]
    y_max = pts[-1][1]
    if y < y_min or y > y_max:
        return None

    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]

        if y1 <= y <= y2 or y2 <= y <= y1:
            if y2 == y1:
                return float((x1 + x2) / 2.0)

            t = (y - y1) / (y2 - y1)
            x = x1 + t * (x2 - x1)
            return float(x)

    return None


def draw_polyline(frame, points, color=(255, 0, 0), thickness=2):
    pts = get_sorted_polyline(points)
    if len(pts) < 2:
        return

    for i in range(len(pts) - 1):
        p1 = (int(round(pts[i][0])), int(round(pts[i][1])))
        p2 = (int(round(pts[i + 1][0])), int(round(pts[i + 1][1])))
        cv2.line(frame, p1, p2, color, thickness)

    for x, y in pts:
        cv2.circle(frame, (int(round(x)), int(round(y))), 4, color, -1)


def choose_body_reference(frame, best_box, tail_cy):
    """
    Primary: hardcoded polyline interpolation.
    Fallback: local median x from image-derived body mask.
    """
    body_x = polyline_x_at_y(BODY_POLYLINE_POINTS, tail_cy)
    if body_x is not None:
        return body_x, "hardcoded_polyline", None

    body_mask = build_body_mask(frame, exclude_box=best_box)
    body_x = median_body_x_from_mask(body_mask, tail_cy, LOCAL_BAND_HALF_HEIGHT)
    if body_x is not None:
        return body_x, "local_median_x", body_mask

    return None, None, None


def detect_tail_and_save(frames, roi_points, save_path, conf_thresh=0.6):
    best_conf = 0
    best_frame = None
    best_box = None
    best_cls = None

    for frame in reversed(frames[-10:]):
        result = model.predict(frame, conf=conf_thresh, verbose=False, save=False)[0]
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

    if best_conf > 0 and best_frame is not None and best_box is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)

        x1, y1, x2, y2 = best_box
        tail_cx = (x1 + x2) / 2.0
        tail_cy = (y1 + y2) / 2.0

        body_x, body_method, fallback_mask = choose_body_reference(best_frame, best_box, tail_cy)

        signed_offset_x_px = None
        abs_offset_x_px = None

        annotated = best_frame.copy()

        # Draw YOLO box and confidence
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{best_conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        tail_pt = (int(round(tail_cx)), int(round(tail_cy)))
        cv2.circle(annotated, tail_pt, 6, (0, 255, 255), -1)

        # Draw hardcoded polyline only when it is the method being used
        if body_method == "hardcoded_polyline":
            draw_polyline(annotated, BODY_POLYLINE_POINTS, color=(255, 0, 0), thickness=2)

        # Draw fallback local band only when median fallback is used
        if body_method == "local_median_x":
            band_y1 = max(0, int(round(tail_cy)) - LOCAL_BAND_HALF_HEIGHT)
            band_y2 = min(best_frame.shape[0], int(round(tail_cy)) + LOCAL_BAND_HALF_HEIGHT)
            cv2.line(annotated, (0, band_y1), (best_frame.shape[1] - 1, band_y1), (80, 80, 80), 1)
            cv2.line(annotated, (0, band_y2), (best_frame.shape[1] - 1, band_y2), (80, 80, 80), 1)

        if body_x is not None:
            signed_offset_x_px = float(tail_cx - body_x)
            abs_offset_x_px = float(abs(signed_offset_x_px))

            body_pt = (int(round(body_x)), int(round(tail_cy)))
            cv2.circle(annotated, body_pt, 6, (255, 0, 255), -1)
            cv2.line(annotated, body_pt, tail_pt, (0, 165, 255), 2)

            offset_text = f"offset_x={signed_offset_x_px:.1f}px ({body_method})"
            cv2.putText(
                annotated,
                offset_text,
                (tail_pt[0] + 10, tail_pt[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
            )

        img_path = os.path.join(save_path, f"tail_detected_{best_conf:.2f}.jpg")
        cv2.imwrite(img_path, annotated)
        print(f"Saved tail image: {img_path}")

        label_info = {
            "class_id": best_cls,
            "confidence": round(best_conf, 4),
            "bbox": best_box,
            "tail_center": [round(float(tail_cx), 2), round(float(tail_cy), 2)],
            "tail_point_used": "bbox_center",
            "body_reference_method": body_method,
            "body_polyline_points": BODY_POLYLINE_POINTS,
            "body_x_at_tail_y": None if body_x is None else round(float(body_x), 2),
            "signed_offset_x_px": None if signed_offset_x_px is None else round(float(signed_offset_x_px), 2),
            "abs_offset_x_px": None if abs_offset_x_px is None else round(float(abs_offset_x_px), 2),
            "offset_sign_convention": "positive means tail is to the right of the body reference",
            "frame_shape": list(best_frame.shape),
        }

        json_path = os.path.join(save_path, f"tail_detected_{best_conf:.2f}.json")
        with open(json_path, "w") as f:
            json.dump(label_info, f, indent=2)
        print(f"Saved label info: {json_path}")


# -------------------------
# Main processing loop
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

    segment_frames = deque(maxlen=60)
    segment_time = []
    segment_intensities = []

    graph_time = deque(maxlen=3000)
    graph_intensity = deque(maxlen=3000)
    segment_starts = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Stream ended or not receiving frames.")
                break

            current_time = frame_idx / fps
            intensity, roi_view = compute_fft_spectrum(frame, roi_points)

            sink.on_roi(roi_view, current_time, float(intensity))
            sink.on_frame(frame, current_time, float(intensity))
            if sink.should_stop():
                break

            graph_time.append(current_time)
            graph_intensity.append(float(intensity))

            if not in_segment and intensity > THRESHOLD:
                in_segment = True
                segment_start = round(current_time, 2)
                segment_frames.clear()
                segment_time = []
                segment_intensities = []
                print(f"Segment START at {segment_start:.2f}s")

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

                    detect_tail_and_save(list(segment_frames), roi_points, folder)
                else:
                    print(f"Segment duration {segment_duration:.2f}s too short. Skipped.")

                segment_frames.clear()
                segment_time.clear()
                segment_intensities.clear()

            if in_segment:
                segment_frames.append(frame.copy())
                segment_time.append(current_time)
                segment_intensities.append(float(intensity))

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