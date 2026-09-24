import cv2
import numpy as np
import os
import json
import shutil
import time
from datetime import datetime, timedelta
from functools import lru_cache

import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque

from pipeline_logging import PipelineRunLog

from coil_cv import (
    BODY_POLYLINE_POINTS,
    CONTOUR_MERGE_ITERATIONS,
    CONTOUR_MERGE_KERNEL,
    LOOP_ACCUMULATION_MAX_FRAMES,
    LOOP_ACCUMULATION_MIN_VOTES,
    LOOP_ACCUMULATION_STRIDE,
    MIN_CONTOUR_SPAN_X,
    MIN_CONTOUR_SPAN_Y,
    choose_body_reference,
    extract_best_tail_segment,
    draw_loop_ellipse,
    draw_loop_search_roi,
    draw_polyline,
    draw_segment_outline,
    find_tail_tip_from_segment,
    point_to_ellipse_angle_deg,
    polygon_to_mask,
    save_loop_fit_diagnostics,
    segment_centroid,
    select_final_loop_model,
    select_loop_accumulation_frames,
    compute_ellipse_polyline_distance,
)

# ----- CONFIG -----
THRESHOLD = 4264.8  # 4264.8 for DB16 ; 3200 for R5.5 ; 3900 for R8.5

USERNAME = "admin"
PASSWORD = "passkey"
CAMERA_IP = "cam_ip"  # 192.168.1.100
RTSP_URL = f"rtsp://{USERNAME}:{PASSWORD}@{CAMERA_IP}:554/cam/realmonitor?channel=1&subtype=1"

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PIPELINE_LOG_DIR = Path("logs")

DEBUG_SAVE_INTERMEDIATE = True
MIN_SEGMENT_DURATION_SECONDS = 10.0
SEGMENT_FRAME_BUFFER_SIZE = 30  # Keep the last N frames for CV and label capture output.
CAPTURE_JPEG_QUALITY = 90

# --------------
# Sink interface
# --------------
class BaseSink:
    def on_pipeline_started(self, log_path: str):
        pass

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

        self._graph_time = deque(maxlen=3000)
        self._graph_intensity = deque(maxlen=3000)
        self._graph_update_count = 0

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
        self._graph_update_count += 1

        if self._graph_update_count % 5 == 0 and self._graph_time:
            self.line.set_xdata(list(self._graph_time))
            self.line.set_ydata(list(self._graph_intensity))
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


# -------------------------
# FFT / logging helpers
# -------------------------
@lru_cache(maxsize=8)
def fft_roi_geometry(frame_height, frame_width, roi_points):
    """Build the cropped ROI mask once for a frame geometry and point set."""
    roi_contour = np.asarray(roi_points, dtype=np.int32)
    x, y, width, height = cv2.boundingRect(roi_contour)
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(int(frame_width), x + width)
    y2 = min(int(frame_height), y + height)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("ROI does not overlap the frame.")

    local_contour = roi_contour - np.array([x1, y1], dtype=np.int32)
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    cv2.fillPoly(mask, [local_contour], 255)
    return (x1, y1, x2, y2), mask


def compute_fft_spectrum(frame, roi_points):
    frame_height, frame_width = frame.shape[:2]
    points_key = tuple((int(x), int(y)) for x, y in roi_points)
    (x1, y1, x2, y2), mask = fft_roi_geometry(frame_height, frame_width, points_key)
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    roi_cropped = cv2.bitwise_and(gray, gray, mask=mask)

    roi_float = np.float32(roi_cropped)
    dft = cv2.dft(roi_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft, axes=[0, 1])
    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

    return np.mean(magnitude), roi_cropped


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
    fig.add_trace(go.Scatter(x=time_axis, y=intensity_values, mode="lines", name="Intensity"))
    fig.update_layout(
        title="Frequency Intensity Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Intensity",
        template="simple_white",
    )
    html_file = f"{save_path}.html"
    fig.write_html(html_file)
    print(f"Saved: {html_file}")


class ActiveCaptureWriter:
    def __init__(
        self,
        source,
        roi_points,
        fps,
        segment_start,
        max_frames=SEGMENT_FRAME_BUFFER_SIZE,
    ):
        self.source = source
        self.roi_points = roi_points
        self.fps = float(fps)
        self.segment_start = float(segment_start)
        self.max_frames = max(1, int(max_frames))
        self.capture_frames = deque(maxlen=self.max_frames)
        self.detection_frames = deque(maxlen=self.max_frames)
        self.image_shape = None

    def append(self, frame, t_s, intensity, source_frame_index):
        if self.image_shape is None:
            self.image_shape = list(frame.shape)

        self.detection_frames.append(frame.copy())

        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(CAPTURE_JPEG_QUALITY)])
        if not ok:
            return

        self.capture_frames.append({
            "source_frame_index": int(source_frame_index),
            "t_s": round(float(t_s), 4),
            "fft_intensity": round(float(intensity), 4),
            "jpg_bytes": jpg.tobytes(),
        })

    def detection_frame_list(self):
        return list(self.detection_frames)

    def finalize(self, save_path, segment_end):
        capture_dir = Path(save_path)
        capture_dir.mkdir(parents=True, exist_ok=True)
        labels_dir = capture_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        final_frames_dir = capture_dir / "frames"
        if final_frames_dir.exists():
            shutil.rmtree(final_frames_dir)
        final_frames_dir.mkdir(parents=True, exist_ok=True)

        records = []
        for index, item in enumerate(self.capture_frames):
            filename = f"frame_{index:06d}.jpg"
            rel_path = f"frames/{filename}"
            (final_frames_dir / filename).write_bytes(item["jpg_bytes"])
            records.append({
                "index": index,
                "source_frame_index": item["source_frame_index"],
                "path": rel_path,
                "t_s": item["t_s"],
                "fft_intensity": item["fft_intensity"],
            })

        manifest = {
            "schema_version": 1,
            "capture_id": capture_dir.name,
            "capture_dir": str(capture_dir),
            "source": self.source,
            "threshold": float(THRESHOLD),
            "fps": self.fps,
            "segment_frame_buffer_size": self.max_frames,
            "roi_points": [[int(x), int(y)] for x, y in self.roi_points],
            "segment_start_s": round(float(self.segment_start), 4),
            "segment_end_s": round(float(segment_end), 4),
            "segment_duration_s": round(float(segment_end - self.segment_start), 4),
            "frame_count": len(records),
            "frame_shape": self.image_shape,
            "frames": records,
            "label_source_of_truth": "labels/true_ellipse.json",
        }

        manifest_path = capture_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"Saved capture manifest: {manifest_path}")

    def discard(self):
        self.capture_frames.clear()
        self.detection_frames.clear()


# -------------------------
# Detection + save
# -------------------------
def detect_tail_and_save(frames, roi_points, save_path, conf_thresh=0.6, run_log=None):
    capture_id = Path(save_path).name
    best = extract_best_tail_segment(frames, conf_thresh=conf_thresh)
    if best is None:
        if run_log is not None:
            run_log.warning(
                "tail_mask_not_found capture_id=%s frame_count=%d confidence_threshold=%.3f",
                capture_id,
                len(frames),
                conf_thresh,
            )
        return {"status": "no_tail_mask", "capture_id": capture_id}

    best_frame_index = best.get("frame_index", max(0, len(frames) - 1))
    best_frame = best["frame"]
    best_cls = best["class_id"]
    best_conf = best["confidence"]
    best_box = best["bbox"]
    segment_xy = best["segment_xy"]

    accumulation_frames = select_loop_accumulation_frames(
        frames,
        best_frame_index,
        max_frames=LOOP_ACCUMULATION_MAX_FRAMES,
        stride=LOOP_ACCUMULATION_STRIDE,
    )

    Path(save_path).mkdir(parents=True, exist_ok=True)

    tail_tip, tail_base = find_tail_tip_from_segment(segment_xy, BODY_POLYLINE_POINTS)
    seg_cent = segment_centroid(segment_xy)

    if tail_tip is None:
        if run_log is not None:
            run_log.warning(
                "tail_tip_not_found_using_bbox_center capture_id=%s",
                capture_id,
            )
        x1, y1, x2, y2 = best_box
        tail_tip = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    tail_cx, tail_cy = tail_tip
    tail_pt = (int(round(tail_cx)), int(round(tail_cy)))

    body_x, body_method = choose_body_reference(best_frame, best_box, tail_cy)
    signed_offset_x_px = None
    abs_offset_x_px = None

    dist_info = None

    ellipse_info, ellipse_metrics, loop_diagnostics = select_final_loop_model(
        best_frame,
        segment_xy=segment_xy,
        tail_tip=tail_tip,
        tail_base=tail_base,
        bbox=best_box,
        accumulation_frames=accumulation_frames,
    )
    loop_roi_box = None if loop_diagnostics is None else loop_diagnostics.get("roi_box")
    if ellipse_info is None and run_log is not None:
        selection_reason = None if loop_diagnostics is None else loop_diagnostics.get("selection_reason")
        run_log.warning(
            "ellipse_fit_not_found capture_id=%s selection_reason=%s",
            capture_id,
            selection_reason,
        )

    tail_loop_angle_deg = None
    annotated = best_frame.copy()

    draw_segment_outline(annotated, segment_xy, color=(0, 255, 0), thickness=2)

    if body_method == "hardcoded_polyline":
        draw_polyline(annotated, BODY_POLYLINE_POINTS, color=(255, 0, 0), thickness=2)

    if body_x is not None:
        signed_offset_x_px = float(tail_cx - body_x)
        abs_offset_x_px = float(abs(signed_offset_x_px))

    if loop_roi_box is not None:
        draw_loop_search_roi(annotated, loop_roi_box)
        annotation_x = int(loop_roi_box[0]) + 18
        annotation_y = int(loop_roi_box[1]) + 30
    else:
        annotation_x = tail_pt[0] + 18
        annotation_y = max(30, tail_pt[1] - 70)

    if ellipse_info is not None:
        draw_loop_ellipse(annotated, ellipse_info, color=(255, 255, 0), thickness=2, draw_center=False)

        cx, cy = ellipse_info["center"]
        center_pt = (int(round(cx)), int(round(cy)))
        cv2.line(annotated, center_pt, tail_pt, (255, 0, 255), 2)

        tail_loop_angle_deg = point_to_ellipse_angle_deg((tail_cx, tail_cy), ellipse_info)
        if tail_loop_angle_deg is not None:
            angle_text = f"loop_angle={tail_loop_angle_deg:.1f} deg"
            cv2.putText(
                annotated,
                angle_text,
                (annotation_x, annotation_y + 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255),
                2,
            )

        # compute distance from ellipse boundary (nearest to polyline) to segment
        try:
            dist_info = compute_ellipse_polyline_distance(
                ellipse_info, segment_xy, best_frame.shape, BODY_POLYLINE_POINTS
            )
        except Exception:
            if run_log is not None:
                run_log.warning(
                    "ellipse_distance_failed capture_id=%s",
                    capture_id,
                    exc_info=True,
                )
            dist_info = None

        if dist_info is not None:
            ep = dist_info.get("ellipse_point")
            nearest = dist_info.get("euclidean_segment_point")
            euclid = dist_info.get("euclidean_distance_px")
            if ep is not None and nearest is not None and euclid is not None:
                epx, epy = int(round(ep[0])), int(round(ep[1]))
                npx, npy = int(round(nearest[0])), int(round(nearest[1]))
                cv2.line(annotated, (epx, epy), (npx, npy), (0, 255, 255), 3)
                txt = f"euclidean_distance_px={euclid:.1f}"
                cv2.putText(
                    annotated, txt, (annotation_x, annotation_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 255), 2,
                )
        else:
            dist_info = None

    img_path = os.path.join(save_path, f"tail_detected_{best_conf:.2f}.jpg")
    if not cv2.imwrite(img_path, annotated) and run_log is not None:
        run_log.error(
            "capture_image_write_failed capture_id=%s path=%s",
            capture_id,
            img_path,
        )
    print(f"Saved tail image: {img_path}")

    seg_mask = polygon_to_mask(segment_xy, best_frame.shape)
    loop_json = None
    if ellipse_info is not None:
        loop_json = {
            "method": ellipse_info.get("method"),
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
            "loop_search_roi": None if ellipse_info.get("roi_box") is None else list(map(int, ellipse_info["roi_box"])),
            "fit_metrics": None if ellipse_metrics is None else {
                key: (round(float(value), 4) if isinstance(value, (int, float, np.floating)) else value)
                for key, value in ellipse_metrics.items()
            },
            "direct_fit_metrics": None if loop_diagnostics is None or loop_diagnostics.get("direct", {}).get("metrics") is None else {
                key: (round(float(value), 4) if isinstance(value, (int, float, np.floating)) else value)
                for key, value in loop_diagnostics["direct"]["metrics"].items()
            },
            "thermal_fit_metrics": None if loop_diagnostics is None or loop_diagnostics.get("thermal", {}).get("metrics") is None else {
                key: (round(float(value), 4) if isinstance(value, (int, float, np.floating)) else value)
                for key, value in loop_diagnostics["thermal"]["metrics"].items()
            },
            "anchor_guided_fit_metrics": None if loop_diagnostics is None or loop_diagnostics.get("anchor_guided", {}).get("metrics") is None else {
                key: (round(float(value), 4) if isinstance(value, (int, float, np.floating)) else value)
                for key, value in loop_diagnostics["anchor_guided"]["metrics"].items()
            },
            "angle_note": "Image-plane ellipse parameter angle only; not corrected for camera/view geometry.",
        }

    label_info = {
        "class_id": best_cls,
        "confidence": round(best_conf, 4),
        "bbox": best_box,
        "segment_xy": [[round(float(x), 2), round(float(y), 2)] for x, y in segment_xy],
        "segment_point_count": int(len(segment_xy)),
        "segment_area_px": int(np.count_nonzero(seg_mask)),
        "tail_tip": [round(float(tail_cx), 2), round(float(tail_cy), 2)],
        "tail_base": None if tail_base is None else [round(float(tail_base[0]), 2), round(float(tail_base[1]), 2)],
        "segment_centroid": None if seg_cent is None else [round(float(seg_cent[0]), 2), round(float(seg_cent[1]), 2)],
        "tail_point_used": "segment_tip_endpoint_from_major_axis",
        "body_reference_method": body_method,
        "body_polyline_points": BODY_POLYLINE_POINTS,
        "body_x_at_tail_y": None if body_x is None else round(float(body_x), 2),
        "signed_offset_x_px": None if signed_offset_x_px is None else round(float(signed_offset_x_px), 2),
        "abs_offset_x_px": None if abs_offset_x_px is None else round(float(abs_offset_x_px), 2),
        "offset_sign_convention": "positive means tail is to the right of the body reference",
        "final_loop_fit": loop_json,
        "loop_fit_diagnostics": None if loop_diagnostics is None else {
            "chosen_method": loop_diagnostics.get("chosen_method"),
            "selection_reason": loop_diagnostics.get("selection_reason"),
            "roi_box": None if loop_diagnostics.get("roi_box") is None else list(map(int, loop_diagnostics.get("roi_box"))),
            "direct_candidate_count": len(loop_diagnostics.get("direct", {}).get("candidates", [])),
            "thermal_candidate_count": None if loop_diagnostics.get("thermal", {}).get("debug_masks") is None else loop_diagnostics["thermal"]["debug_masks"].get("candidates_scored"),
            "thermal_roi_box": None if loop_diagnostics.get("thermal", {}).get("roi_box") is None else list(map(int, loop_diagnostics["thermal"]["roi_box"])),
            "component_count_total": len(loop_diagnostics.get("debug_masks", {}).get("all_contours_full", [])),
            "component_count_kept": len(loop_diagnostics.get("debug_masks", {}).get("kept_contours_full", [])),
            "min_contour_span_x": MIN_CONTOUR_SPAN_X,
            "min_contour_span_y": MIN_CONTOUR_SPAN_Y,
            "contour_merge_kernel": CONTOUR_MERGE_KERNEL,
            "contour_merge_iterations": CONTOUR_MERGE_ITERATIONS,
            "loop_accumulation_max_frames": LOOP_ACCUMULATION_MAX_FRAMES,
            "loop_accumulation_stride": LOOP_ACCUMULATION_STRIDE,
            "loop_accumulation_min_votes": LOOP_ACCUMULATION_MIN_VOTES,
            "best_frame_index": best_frame_index,
            "accumulated_frame_indices": (
                None if loop_diagnostics.get("debug_masks", {}).get("accumulation_debug") is None
                else loop_diagnostics["debug_masks"]["accumulation_debug"].get("frame_indices")
            ),
        },
        "frame_shape": list(best_frame.shape),
    }

    # attach polyline-based ellipse->tail distance info if available
    if dist_info is None:
        label_info["tail_to_ellipse_polyline"] = None
    else:
        label_info["tail_to_ellipse_polyline"] = {
            "ellipse_point": None if dist_info.get("ellipse_point") is None else [
                round(float(dist_info["ellipse_point"][0]), 2),
                round(float(dist_info["ellipse_point"][1]), 2),
            ],
            "euclidean_segment_point": None if dist_info.get("euclidean_segment_point") is None else [
                round(float(dist_info["euclidean_segment_point"][0]), 2),
                round(float(dist_info["euclidean_segment_point"][1]), 2),
            ],
            "distance_method": "euclidean",
            "distance_value_px": None if dist_info.get("euclidean_distance_px") is None else round(float(dist_info["euclidean_distance_px"]), 2),

            "intersection_point": None if dist_info.get("intersection_point") is None else [
                round(float(dist_info["intersection_point"][0]), 2),
                round(float(dist_info["intersection_point"][1]), 2),
            ],
            "distance_along_polyline_px": None if dist_info.get("distance_along_polyline_px") is None else round(float(dist_info["distance_along_polyline_px"]), 2),
            "euclidean_distance_px": None if dist_info.get("euclidean_distance_px") is None else round(float(dist_info["euclidean_distance_px"]), 2),
            "polyline_distance_px": None if dist_info.get("polyline_distance_px") is None else round(float(dist_info["polyline_distance_px"]), 2),
        }

    json_path = os.path.join(save_path, f"tail_detected_{best_conf:.2f}.json")
    with open(json_path, "w") as f:
        json.dump(label_info, f, indent=2)
    print(f"Saved label info: {json_path}")

    if DEBUG_SAVE_INTERMEDIATE:
        save_loop_fit_diagnostics(
            Path(save_path) / "debug",
            frame=best_frame,
            segment_xy=segment_xy,
            tail_tip=tail_tip,
            tail_base=tail_base,
            annotated_final=annotated,
            loop_diagnostics=loop_diagnostics,
            final_label_info=label_info,
        )

    return {
        "status": "completed" if ellipse_info is not None else "no_ellipse_fit",
        "capture_id": capture_id,
        "ellipse_method": None if ellipse_info is None else ellipse_info.get("method"),
    }


# -------------------------
# Main processing loop
# -------------------------
def infer_run_mode(sink):
    if isinstance(sink, CvGuiSink):
        return "local_gui"
    if isinstance(sink, NullSink):
        return "headless"
    return sink.__class__.__name__.removesuffix("Sink").lower() or "headless"


def process_rtsp_stream(
    rtsp_url,
    roi_points,
    sink: BaseSink | None = None,
    fps_assumed=30,
    run_mode=None,
    log_dir=PIPELINE_LOG_DIR,
):
    if sink is None:
        sink = NullSink()

    run_mode = run_mode or infer_run_mode(sink)
    run_log = PipelineRunLog(rtsp_url, run_mode, log_dir=log_dir)
    cap = None
    status = "starting"
    frame_idx = 0
    frames_processed = 0
    captures_processed = 0
    in_segment = False
    segment_start = None
    active_capture = None
    segment_time = []
    segment_intensities = []
    graph_time = deque(maxlen=3000)
    graph_intensity = deque(maxlen=3000)

    print(f"Data input processing started. Log: {run_log.path}")

    try:
        sink.on_pipeline_started(str(run_log.path))
        run_log.info("input_opening")
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            status = "input_open_failed"
            run_log.error("input_open_failed source=%s", run_log.source)
            print("Error: Cannot open RTSP/video stream.")
            return {"status": status, "log_file": str(run_log.path)}

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = float(fps_assumed)
            run_log.warning("input_fps_unavailable using_assumed_fps=%.3f", fps)

        status = "running"
        run_log.info("input_opened fps=%.3f threshold=%.3f", fps, THRESHOLD)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Stream ended or not receiving frames.")
                run_log.info("input_exhausted frame_count=%d", frames_processed)
                status = "completed"
                break

            current_time = frame_idx / fps
            frames_processed += 1
            intensity, roi_view = compute_fft_spectrum(frame, roi_points)

            sink.on_roi(roi_view, current_time, float(intensity))
            sink.on_frame(frame, current_time, float(intensity))
            if sink.should_stop():
                status = "stopped"
                run_log.info("stop_requested frame_index=%d input_time_s=%.3f", frame_idx, current_time)
                break

            graph_time.append(current_time)
            graph_intensity.append(float(intensity))

            if not in_segment and intensity > THRESHOLD:
                in_segment = True
                segment_start = round(current_time, 2)
                if active_capture is not None:
                    active_capture.discard()
                active_capture = ActiveCaptureWriter(rtsp_url, roi_points, fps, segment_start)
                segment_time = []
                segment_intensities = []
                print(f"Segment START at {segment_start:.2f}s")
                run_log.info(
                    "segment_started input_time_s=%.2f intensity=%.3f",
                    segment_start,
                    intensity,
                )

            elif in_segment and intensity < THRESHOLD:
                segment_end = round(current_time, 2)
                in_segment = False
                segment_duration = segment_end - segment_start
                print(f"Segment END at {segment_end:.2f}s")
                run_log.info(
                    "segment_finished input_time_s=%.2f duration_s=%.2f intensity=%.3f",
                    segment_end,
                    segment_duration,
                    intensity,
                )

                if segment_duration >= MIN_SEGMENT_DURATION_SECONDS:
                    start_dt = datetime.now() - timedelta(seconds=segment_duration)
                    end_dt = datetime.now()
                    folder = create_timestamped_folder(start_dt, end_dt)
                    base = os.path.join(folder, os.path.basename(folder))
                    save_results_txt(segment_time, segment_intensities, base)
                    save_results_html(segment_time, segment_intensities, base)
                    if active_capture is not None:
                        capture_id = Path(folder).name
                        capture_started = time.perf_counter()
                        capture_status = "failed"
                        run_log.info(
                            "capture_processing_started capture_id=%s frame_count=%d",
                            capture_id,
                            len(active_capture.detection_frames),
                        )
                        try:
                            active_capture.finalize(folder, segment_end)
                            result = detect_tail_and_save(
                                active_capture.detection_frame_list(),
                                roi_points,
                                folder,
                                run_log=run_log,
                            )
                            capture_status = (result or {}).get("status", "completed")
                        finally:
                            captures_processed += 1
                            run_log.info(
                                "capture_processing_finished capture_id=%s status=%s duration_s=%.3f",
                                capture_id,
                                capture_status,
                                time.perf_counter() - capture_started,
                            )
                else:
                    print(f"Segment duration {segment_duration:.2f}s too short. Skipped.")
                    run_log.info(
                        "segment_skipped reason=too_short duration_s=%.2f minimum_duration_s=%.2f",
                        segment_duration,
                        MIN_SEGMENT_DURATION_SECONDS,
                    )
                    if active_capture is not None:
                        active_capture.discard()

                active_capture = None
                segment_time.clear()
                segment_intensities.clear()

            if in_segment and active_capture is not None:
                active_capture.append(frame, current_time, float(intensity), frame_idx)
                segment_time.append(current_time)
                segment_intensities.append(float(intensity))

            if frame_idx % 10 == 0 and graph_time:
                sink.on_series(list(graph_time), list(graph_intensity))

            frame_idx += 1

        if status == "running":
            status = "completed"
    except Exception:
        status = "failed"
        run_log.exception("pipeline_error frame_index=%d", frame_idx)
        raise
    finally:
        if active_capture is not None:
            active_capture.discard()
            run_log.info(
                "active_capture_discarded reason=pipeline_finished segment_start_s=%s",
                segment_start,
            )
        if cap is not None:
            cap.release()
        try:
            sink.close()
        except Exception:
            status = "failed"
            run_log.exception("sink_close_failed")
        run_log.finish(
            status,
            frames_processed=frames_processed,
            captures_processed=captures_processed,
        )
        print(f"Data input processing complete. Log: {run_log.path}")

    return {"status": status, "log_file": str(run_log.path)}

if __name__ == "__main__":
    roi_points = [(677, 1288), (1325, 1418), (1425, 1171), (893, 1051)]
    video_path = "out.mp4"

    gui = CvGuiSink(show_plot=True)
    # process_rtsp_stream(RTSP_URL, roi_points, sink=gui)
    process_rtsp_stream(video_path, roi_points, sink=gui)
