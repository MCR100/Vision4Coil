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
import math

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
BODY_POLYLINE_POINTS = [
    (1386, 753),
    (1167, 1074),
    (916, 1432)
]

# Loop-fit tuning
ELLIPSE_SEARCH_PAD_X = 420
ELLIPSE_SEARCH_PAD_Y_UP = 260
ELLIPSE_SEARCH_PAD_Y_DOWN = 380
MIN_CONTOUR_POINTS_FOR_ELLIPSE = 30
TAIL_BOX_EXCLUDE_PAD = 12

# Candidate ellipse constraints
ELLIPSE_MIN_MAJOR = 50
ELLIPSE_MAX_MAJOR = 1200
ELLIPSE_MIN_MINOR = 25
ELLIPSE_MAX_MINOR = 900

# The loop center is usually somewhat near the body path and near/below the tail.
MAX_CENTER_TO_POLYLINE_DIST = 350
MAX_CENTER_TO_TAIL_DIST = 950
MIN_CENTER_TO_TAIL_DIST = 40
MAX_TAIL_TO_ELLIPSE_BOUNDARY_DIST = 160
CENTER_ALLOWED_ABOVE_TAIL = 280
CENTER_ALLOWED_BELOW_TAIL = 650


# --------------
# Sink interface
# --------------
class BaseSink:
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
    pass

def point_to_segment_distance(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return float(np.hypot(px - x1, py - y1))

    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return float(np.hypot(px - proj_x, py - proj_y))


def point_to_polyline_distance(px, py, points):
    pts = get_sorted_polyline(points)
    if len(pts) < 2:
        return None

    best = None
    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        d = point_to_segment_distance(px, py, x1, y1, x2, y2)
        if best is None or d < best:
            best = d
    return best


def ellipse_boundary_distance(point, ellipse_info):
    """
    Approximate how far a point is from the fitted ellipse boundary.
    Returns absolute radial mismatch in pixels-ish image units.
    Lower is better.
    """
    px, py = point
    cx, cy = ellipse_info["center"]
    a, b = ellipse_info["axes"]
    rot_deg = ellipse_info["rotation_deg"]

    if a <= 1e-6 or b <= 1e-6:
        return None

    theta = math.radians(rot_deg)
    dx = px - cx
    dy = py - cy

    # Undo ellipse rotation
    xr = dx * math.cos(theta) + dy * math.sin(theta)
    yr = -dx * math.sin(theta) + dy * math.cos(theta)

    norm = math.sqrt((xr * xr) / (a * a) + (yr * yr) / (b * b))
    # norm ~= 1 means point lies on ellipse
    # convert mismatch to a rough pixel scale
    return float(abs(norm - 1.0) * max(a, b))


def crop_guided_loop_roi(mask, tail_cx, tail_cy, polyline_points):
    """
    Build a tighter ROI around the tail, biased by the hardcoded body path.
    """
    h, w = mask.shape[:2]

    body_x = polyline_x_at_y(polyline_points, tail_cy)
    if body_x is None:
        body_x = tail_cx

    x_center = int(round((tail_cx + body_x) / 2.0))

    x1 = max(0, x_center - ELLIPSE_SEARCH_PAD_X)
    x2 = min(w, x_center + ELLIPSE_SEARCH_PAD_X)

    y1 = max(0, int(round(tail_cy)) - ELLIPSE_SEARCH_PAD_Y_UP)
    y2 = min(h, int(round(tail_cy)) + ELLIPSE_SEARCH_PAD_Y_DOWN)

    return mask[y1:y2, x1:x2], (x1, y1, x2, y2)


def contour_support_score(contour):
    return float(cv2.arcLength(contour, closed=False))


def fit_ellipse_from_contour(contour, roi_offset):
    if len(contour) < 5:
        return None

    ellipse = cv2.fitEllipse(contour)
    (cx, cy), (d1, d2), rotation_deg = ellipse

    rx1, ry1 = roi_offset
    cx_full = cx + rx1
    cy_full = cy + ry1

    a = d1 / 2.0
    b = d2 / 2.0
    rot = rotation_deg

    if b > a:
        a, b = b, a
        rot = (rot + 90.0) % 180.0

    return {
        "center": (float(cx_full), float(cy_full)),
        "axes": (float(a), float(b)),  # semi-major, semi-minor
        "rotation_deg": float(rot),
    }


def score_ellipse_candidate(ellipse_info, contour, tail_point, polyline_points):
    """
    Lower score is better.
    Returns None if candidate should be rejected.
    """
    cx, cy = ellipse_info["center"]
    a, b = ellipse_info["axes"]
    tail_cx, tail_cy = tail_point

    # Size filters
    if not (ELLIPSE_MIN_MAJOR <= a <= ELLIPSE_MAX_MAJOR):
        return None
    if not (ELLIPSE_MIN_MINOR <= b <= ELLIPSE_MAX_MINOR):
        return None

    # Center vs tail constraints
    center_to_tail = float(np.hypot(cx - tail_cx, cy - tail_cy))
    if center_to_tail < MIN_CENTER_TO_TAIL_DIST or center_to_tail > MAX_CENTER_TO_TAIL_DIST:
        return None

    center_dy = cy - tail_cy
    if center_dy < -CENTER_ALLOWED_ABOVE_TAIL or center_dy > CENTER_ALLOWED_BELOW_TAIL:
        return None

    # Center should be reasonably close to the expected body path
    center_to_poly = point_to_polyline_distance(cx, cy, polyline_points)
    if center_to_poly is None or center_to_poly > MAX_CENTER_TO_POLYLINE_DIST:
        return None

    # Tail should lie near the ellipse boundary
    tail_to_boundary = ellipse_boundary_distance((tail_cx, tail_cy), ellipse_info)
    if tail_to_boundary is None or tail_to_boundary > MAX_TAIL_TO_ELLIPSE_BOUNDARY_DIST:
        return None

    # Prefer stronger contour support
    support = contour_support_score(contour)

    # Weighted score: lower is better
    score = (
        2.5 * tail_to_boundary +
        1.2 * center_to_poly +
        0.4 * abs(center_dy) +
        0.15 * center_to_tail -
        0.02 * support
    )

    return {
        "score": float(score),
        "center_to_polyline_dist": float(center_to_poly),
        "center_to_tail_dist": float(center_to_tail),
        "tail_to_ellipse_boundary_dist": float(tail_to_boundary),
        "contour_support": float(support),
    }

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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    if exclude_box is not None:
        x1, y1, x2, y2 = exclude_box
        pad = TAIL_BOX_EXCLUDE_PAD
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
    body_x = polyline_x_at_y(BODY_POLYLINE_POINTS, tail_cy)
    if body_x is not None:
        return body_x, "hardcoded_polyline"

    body_mask = build_body_mask(frame, exclude_box=best_box)
    body_x = median_body_x_from_mask(body_mask, tail_cy, LOCAL_BAND_HALF_HEIGHT)
    if body_x is not None:
        return body_x, "local_median_x"

    return None, None


def build_loop_edge_mask(frame, exclude_box=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 40, 120)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    if exclude_box is not None:
        x1, y1, x2, y2 = exclude_box
        pad = TAIL_BOX_EXCLUDE_PAD
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(mask.shape[1], x2 + pad)
        y2 = min(mask.shape[0], y2 + pad)
        mask[y1:y2, x1:x2] = 0

    return mask


def score_contour_for_tail(contour, tail_point_local):
    if len(contour) < MIN_CONTOUR_POINTS_FOR_ELLIPSE:
        return None

    pts = contour.reshape(-1, 2).astype(np.float32)
    tx, ty = tail_point_local

    dists = np.sqrt((pts[:, 0] - tx) ** 2 + (pts[:, 1] - ty) ** 2)
    min_dist = float(np.min(dists))
    contour_len = float(cv2.arcLength(contour, closed=False))

    # Prefer contours close to the tail and reasonably long
    score = contour_len - 2.0 * min_dist
    return score


def fit_final_loop_ellipse(frame, tail_cx, tail_cy, exclude_box=None):
    """
    Fit the final loop ellipse using:
    - tight ROI around tail guided by the hardcoded polyline
    - multiple contour candidates
    - geometric scoring and rejection
    """
    mask = build_loop_edge_mask(frame, exclude_box=exclude_box)
    local_mask, roi_box = crop_guided_loop_roi(mask, tail_cx, tail_cy, BODY_POLYLINE_POINTS)
    rx1, ry1, rx2, ry2 = roi_box

    contours, _ = cv2.findContours(local_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, mask, roi_box, None

    tail_point = (float(tail_cx), float(tail_cy))

    best_candidate = None

    for cnt in contours:
        if len(cnt) < MIN_CONTOUR_POINTS_FOR_ELLIPSE:
            continue

        ellipse_info = fit_ellipse_from_contour(cnt, (rx1, ry1))
        if ellipse_info is None:
            continue

        metrics = score_ellipse_candidate(
            ellipse_info=ellipse_info,
            contour=cnt,
            tail_point=tail_point,
            polyline_points=BODY_POLYLINE_POINTS
        )
        if metrics is None:
            continue

        candidate = {
            "ellipse_info": ellipse_info,
            "metrics": metrics,
        }

        if best_candidate is None or metrics["score"] < best_candidate["metrics"]["score"]:
            best_candidate = candidate

    if best_candidate is None:
        return None, mask, roi_box, None

    ellipse_info = best_candidate["ellipse_info"]
    ellipse_info["roi_box"] = roi_box

    return ellipse_info, mask, roi_box, best_candidate["metrics"]

def point_to_ellipse_angle_deg(point, ellipse_info):
    """
    Image-plane ellipse angle.
    0° is along ellipse local +x axis before rotation.
    Returned angle is 0..360.
    """
    px, py = point
    cx, cy = ellipse_info["center"]
    a, b = ellipse_info["axes"]
    rot_deg = ellipse_info["rotation_deg"]

    theta = math.radians(rot_deg)
    dx = px - cx
    dy = py - cy

    # Undo ellipse rotation
    xr = dx * math.cos(theta) + dy * math.sin(theta)
    yr = -dx * math.sin(theta) + dy * math.cos(theta)

    if a <= 1e-6 or b <= 1e-6:
        return None

    # Ellipse parameter angle
    t = math.atan2(yr / b, xr / a)
    deg = math.degrees(t)
    if deg < 0:
        deg += 360.0
    return float(deg)


def draw_loop_ellipse(frame, ellipse_info, color=(255, 255, 0), thickness=2):
    cx, cy = ellipse_info["center"]
    a, b = ellipse_info["axes"]
    rot = ellipse_info["rotation_deg"]

    center_i = (int(round(cx)), int(round(cy)))
    axes_i = (int(round(a)), int(round(b)))

    cv2.ellipse(frame, center_i, axes_i, rot, 0, 360, color, thickness)
    cv2.circle(frame, center_i, 6, color, -1)


def draw_loop_search_roi(frame, roi_box, color=(120, 120, 120), thickness=1):
    x1, y1, x2, y2 = roi_box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


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
        tail_pt = (int(round(tail_cx)), int(round(tail_cy)))

        # Existing linear reference
        body_x, body_method = choose_body_reference(best_frame, best_box, tail_cy)
        signed_offset_x_px = None
        abs_offset_x_px = None

        # Improved loop/ellipse reference
        ellipse_info, loop_mask, loop_roi_box, ellipse_metrics = fit_final_loop_ellipse(
            best_frame, tail_cx, tail_cy, exclude_box=None
        )

        tail_loop_angle_deg = None

        annotated = best_frame.copy()

        # YOLO box
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
        cv2.circle(annotated, tail_pt, 6, (0, 255, 255), -1)

        # Linear body reference
        if body_method == "hardcoded_polyline":
            draw_polyline(annotated, BODY_POLYLINE_POINTS, color=(255, 0, 0), thickness=2)

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

        if loop_roi_box is not None:
            draw_loop_search_roi(annotated, loop_roi_box)

        # Ellipse reference
        if ellipse_info is not None:
            draw_loop_ellipse(annotated, ellipse_info, color=(255, 255, 0), thickness=2)

            cx, cy = ellipse_info["center"]
            center_pt = (int(round(cx)), int(round(cy)))
            cv2.line(annotated, center_pt, tail_pt, (255, 255, 0), 2)

            tail_loop_angle_deg = point_to_ellipse_angle_deg((tail_cx, tail_cy), ellipse_info)

            if tail_loop_angle_deg is not None:
                angle_text = f"loop_angle={tail_loop_angle_deg:.1f} deg"
                cv2.putText(
                    annotated,
                    angle_text,
                    (tail_pt[0] + 10, tail_pt[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )

        img_path = os.path.join(save_path, f"tail_detected_{best_conf:.2f}.jpg")
        cv2.imwrite(img_path, annotated)
        print(f"Saved tail image: {img_path}")

        loop_json = None
        if ellipse_info is not None:
            loop_json = {
                "ellipse_center": [
                    round(float(ellipse_info["center"][0]), 2),
                    round(float(ellipse_info["center"][1]), 2),
                ],
                "ellipse_axes_semi": [
                    round(float(ellipse_info["axes"][0]), 2),
                    round(float(ellipse_info["axes"][1]), 2),
                ],
                "ellipse_rotation_deg": round(float(ellipse_info["rotation_deg"]), 2),
                "tail_loop_angle_deg_image_plane": None if tail_loop_angle_deg is None else round(float(tail_loop_angle_deg), 2),
                "loop_search_roi": list(map(int, ellipse_info["roi_box"])),
                "fit_metrics": None if ellipse_metrics is None else {
                    "score": round(float(ellipse_metrics["score"]), 3),
                    "center_to_polyline_dist": round(float(ellipse_metrics["center_to_polyline_dist"]), 2),
                    "center_to_tail_dist": round(float(ellipse_metrics["center_to_tail_dist"]), 2),
                    "tail_to_ellipse_boundary_dist": round(float(ellipse_metrics["tail_to_ellipse_boundary_dist"]), 2),
                    "contour_support": round(float(ellipse_metrics["contour_support"]), 2),
                }
            }

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

            "final_loop_fit": loop_json,

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
    video_path = "temp/21_08_2025_P1-00.00.00.000-01.05.57.296.mov"

    gui = CvGuiSink(show_plot=True)
    # process_rtsp_stream(RTSP_URL, roi_points, sink=gui)
    process_rtsp_stream(video_path, roi_points, sink=gui)