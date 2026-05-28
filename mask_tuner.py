import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from coil_cv import BODY_POLYLINE_POINTS, draw_polyline, get_mask_tuning_defaults
from mask_tuner_common import (
    ELLIPSE_COLOR,
    ELLIPSE_THICKNESS,
    ellipse_from_points,
    ellipse_to_mask,
    build_last_loop_mask,
    search_mask_parameters,
)


WINDOW_NAME = "Mask Tuner"
CONTROL_WINDOW = "Mask Controls"
SETTINGS_SAVE_PATH = Path("mask_tuner_settings.json")
PANEL_W = 640
PANEL_H = 360

TRACKBAR_SPECS = [
    {"name": "COIL_RED_MIN", "source_key": "COIL_RED_MIN", "max": 255, "scale": 1.0, "offset": 0.0},
    {"name": "COIL_EXCESS_RED_MIN", "source_key": "COIL_EXCESS_RED_MIN", "max": 255, "scale": 1.0, "offset": 0.0},
    {"name": "COIL_LAB_A_MIN", "source_key": "COIL_LAB_A_MIN", "max": 255, "scale": 1.0, "offset": 0.0},
    {"name": "COIL_SAT_MIN", "source_key": "COIL_SAT_MIN", "max": 255, "scale": 1.0, "offset": 0.0},
    {"name": "COIL_VALUE_MIN", "source_key": "COIL_VALUE_MIN", "max": 255, "scale": 1.0, "offset": 0.0},
    {"name": "COIL_WHITEHOT_RED_MIN", "source_key": "COIL_WHITEHOT_RED_MIN", "max": 255, "scale": 1.0, "offset": 0.0},
    {"name": "COIL_ADAPTIVE_RED_FLOOR", "source_key": "COIL_ADAPTIVE_RED_FLOOR", "max": 255, "scale": 1.0, "offset": 0.0},
    {"name": "COIL_ADAPTIVE_EXCESS_FLOOR", "source_key": "COIL_ADAPTIVE_EXCESS_FLOOR", "max": 255, "scale": 1.0, "offset": 0.0},
    {"name": "COIL_ADAPTIVE_A_FLOOR", "source_key": "COIL_ADAPTIVE_A_FLOOR", "max": 255, "scale": 1.0, "offset": 0.0},
    {"name": "COIL_RELAXED_RED_FLOOR", "source_key": "COIL_RELAXED_RED_FLOOR", "max": 255, "scale": 1.0, "offset": 0.0},
    {"name": "COIL_RELAXED_EXCESS_FLOOR", "source_key": "COIL_RELAXED_EXCESS_FLOOR", "max": 255, "scale": 1.0, "offset": -80.0},
    {"name": "COIL_RELAXED_A_FLOOR", "source_key": "COIL_RELAXED_A_FLOOR", "max": 255, "scale": 1.0, "offset": 0.0},
    {"name": "COIL_RELAXED_VALUE_FLOOR", "source_key": "COIL_RELAXED_VALUE_FLOOR", "max": 255, "scale": 1.0, "offset": 0.0},
    {"name": "COIL_RELAXED_SAT_FLOOR", "source_key": "COIL_RELAXED_SAT_FLOOR", "max": 255, "scale": 1.0, "offset": 0.0},
    {"name": "COIL_RELAXED_SEGMENT_COVERAGE", "source_key": "COIL_RELAXED_SEGMENT_COVERAGE", "max": 400, "scale": 0.01, "offset": 0.0},
    {"name": "COIL_RELAXED_SEGMENT_INTERSECTION", "source_key": "COIL_RELAXED_SEGMENT_INTERSECTION", "max": 100, "scale": 0.01, "offset": 0.0},
    {"name": "COIL_RELAXED_PAD_X", "source_key": "COIL_RELAXED_PAD_X", "max": 1200, "scale": 1.0, "offset": 0.0},
    {"name": "COIL_RELAXED_PAD_Y", "source_key": "COIL_RELAXED_PAD_Y", "max": 800, "scale": 1.0, "offset": 0.0},
    {"name": "COIL_MASK_CLOSE_KERNEL", "source_key": "COIL_MASK_CLOSE_KERNEL", "max": 61, "scale": 1.0, "offset": 0.0},
    {"name": "COIL_MASK_OPEN_KERNEL", "source_key": "COIL_MASK_OPEN_KERNEL", "max": 61, "scale": 1.0, "offset": 0.0},
    {"name": "COIL_SUPPORT_DILATE_KERNEL", "source_key": "COIL_SUPPORT_DILATE_KERNEL", "max": 61, "scale": 1.0, "offset": 0.0},
    {"name": "COIL_ENVELOPE_CLOSE_KERNEL", "source_key": "COIL_ENVELOPE_CLOSE_KERNEL", "max": 121, "scale": 1.0, "offset": 0.0},
    {"name": "COIL_ENVELOPE_DILATE_KERNEL", "source_key": "COIL_ENVELOPE_DILATE_KERNEL", "max": 61, "scale": 1.0, "offset": 0.0},
    {"name": "COIL_ENVELOPE_MIN_AREA", "source_key": "COIL_ENVELOPE_MIN_AREA", "max": 200, "scale": 1000.0, "offset": 0.0},
]

DRAW_STATE = {
    "dragging": False,
    "start_pt": None,
    "end_pt": None,
    "ellipse_params": None,
}


def noop(_value):
    return None


def normalize_case_path(case_path):
    path = Path(case_path)
    if path.is_file():
        return path.parent
    return path


def find_latest_case():
    case_dirs = sorted([p for p in Path("output").iterdir() if p.is_dir()], key=lambda p: p.name)
    if not case_dirs:
        raise FileNotFoundError("No output case folders found under output/.")
    return case_dirs[-1]


def pick_case_assets(case_dir):
    json_files = sorted(case_dir.glob("tail_detected_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No tail_detected_*.json found in {case_dir}")

    json_path = json_files[-1]
    debug_frame = case_dir / "debug" / "01_frame.jpg"
    if debug_frame.exists():
        image_path = debug_frame
    else:
        image_files = sorted(case_dir.glob("tail_detected_*.jpg"))
        if not image_files:
            raise FileNotFoundError(f"No usable frame image found in {case_dir}")
        image_path = image_files[-1]
    return json_path, image_path


def load_case(case_path):
    case_dir = normalize_case_path(case_path)
    json_path, image_path = pick_case_assets(case_dir)

    data = json.loads(json_path.read_text())
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    segment_xy = data.get("segment_xy")
    if segment_xy:
        segment_xy = np.asarray(segment_xy, dtype=np.float32)
    else:
        bbox = data.get("bbox")
        if bbox is None or len(bbox) != 4:
            raise ValueError(f"No segment geometry available in {json_path}")
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        segment_xy = np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

    tail_tip = tuple(float(v) for v in data["tail_tip"])
    tail_base = None if data.get("tail_base") is None else tuple(float(v) for v in data["tail_base"])
    bbox = None if data.get("bbox") is None else [int(round(v)) for v in data["bbox"]]
    polyline = data.get("body_polyline_points") or BODY_POLYLINE_POINTS
    polyline = [(int(round(x)), int(round(y))) for x, y in polyline]

    return {
        "case_dir": case_dir,
        "json_path": json_path,
        "image_path": image_path,
        "frame": frame,
        "segment_xy": segment_xy,
        "tail_tip": tail_tip,
        "tail_base": tail_base,
        "bbox": bbox,
        "polyline": polyline,
    }


def trackbar_raw_from_value(spec, value):
    raw = int(round((float(value) - spec["offset"]) / spec["scale"]))
    return max(0, min(raw, spec["max"]))


def trackbar_value_from_raw(spec, raw):
    return (float(raw) * spec["scale"]) + spec["offset"]


def create_trackbars(defaults):
    cv2.namedWindow(CONTROL_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROL_WINDOW, 700, 1000)
    for spec in TRACKBAR_SPECS:
        raw = trackbar_raw_from_value(spec, defaults[spec["source_key"]])
        cv2.createTrackbar(spec["name"], CONTROL_WINDOW, raw, spec["max"], noop)


def read_settings_from_trackbars():
    settings = {}
    for spec in TRACKBAR_SPECS:
        raw = cv2.getTrackbarPos(spec["name"], CONTROL_WINDOW)
        value = trackbar_value_from_raw(spec, raw)
        if spec["scale"] == 1.0:
            value = int(round(value))
        settings[spec["source_key"]] = value
    return settings


def reset_trackbars():
    defaults = get_mask_tuning_defaults()
    for spec in TRACKBAR_SPECS:
        raw = trackbar_raw_from_value(spec, defaults[spec["source_key"]])
        cv2.setTrackbarPos(spec["name"], CONTROL_WINDOW, raw)


def get_current_ellipse_params():
    if DRAW_STATE["dragging"] and DRAW_STATE["start_pt"] and DRAW_STATE["end_pt"]:
        return ellipse_from_points(DRAW_STATE["start_pt"], DRAW_STATE["end_pt"])
    return DRAW_STATE["ellipse_params"]


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        DRAW_STATE["dragging"] = True
        DRAW_STATE["start_pt"] = (x, y)
        DRAW_STATE["end_pt"] = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and DRAW_STATE["dragging"]:
        DRAW_STATE["end_pt"] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        DRAW_STATE["dragging"] = False
        DRAW_STATE["end_pt"] = (x, y)
        DRAW_STATE["ellipse_params"] = ellipse_from_points(DRAW_STATE["start_pt"], DRAW_STATE["end_pt"])
    elif event == cv2.EVENT_RBUTTONUP:
        DRAW_STATE["dragging"] = False
        DRAW_STATE["start_pt"] = None
        DRAW_STATE["end_pt"] = None
        DRAW_STATE["ellipse_params"] = None


def mask_to_bgr(mask):
    if mask is None:
        return np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def fit_panel_image(image, width=PANEL_W, height=PANEL_H):
    if image is None or image.size == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    h, w = image.shape[:2]
    scale = min(width / max(1, w), height / max(1, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y0 = (height - new_h) // 2
    x0 = (width - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def label_panel(image, text):
    labeled = image.copy()
    cv2.putText(labeled, text, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return labeled


def build_preview(case_data, settings, ellipse_params=None):
    frame = case_data["frame"]
    segment_xy = case_data["segment_xy"]
    tail_tip = case_data["tail_tip"]
    tail_base = case_data["tail_base"]

    candidate_mask, debug = build_last_loop_mask(case_data, settings=settings, ellipse_params=ellipse_params)
    if candidate_mask is None:
        candidate_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    overlay = frame.copy()
    cv2.polylines(overlay, [np.round(segment_xy).astype(np.int32)], True, (0, 255, 0), 2)
    draw_polyline(overlay, case_data["polyline"], color=(255, 0, 0), thickness=2)
    cv2.circle(overlay, (int(round(tail_tip[0])), int(round(tail_tip[1]))), 6, (0, 255, 255), -1)
    if tail_base is not None:
        cv2.circle(overlay, (int(round(tail_base[0])), int(round(tail_base[1]))), 6, (255, 255, 0), -1)

    selected_ellipse = debug.get("best_ellipse")
    if selected_ellipse is not None:
        center, axes, angle = selected_ellipse
        cv2.ellipse(overlay, center, axes, angle, 0, 360, (255, 0, 255), 3)
    if ellipse_params is not None:
        center, axes, angle = ellipse_params
        cv2.ellipse(overlay, center, axes, angle, 0, 360, ELLIPSE_COLOR, ELLIPSE_THICKNESS)

    candidate_overlay = overlay.copy()
    candidate_overlay[candidate_mask > 0] = (255, 0, 255)

    panels = [
        label_panel(fit_panel_image(overlay), "Frame + Geometry"),
        label_panel(fit_panel_image(mask_to_bgr(debug.get("heat_mask"))), "Heat Score"),
        label_panel(fit_panel_image(mask_to_bgr(debug.get("material_mask"))), "Hot Material"),
        label_panel(fit_panel_image(mask_to_bgr(debug.get("geometry_roi_mask"))), "Geometry ROI"),
        label_panel(fit_panel_image(mask_to_bgr(debug.get("exclusion_mask"))), "Tail Exclusion"),
        label_panel(fit_panel_image(mask_to_bgr(debug.get("selected_loop_ring_mask"))), "Selected Ring"),
        label_panel(fit_panel_image(mask_to_bgr(debug.get("selected_loop_fill_mask"))), "Selected Fill"),
        label_panel(fit_panel_image(candidate_overlay), "Observed Loop"),
    ]
    preview = np.vstack([np.hstack(panels[:4]), np.hstack(panels[4:])])

    metrics = debug.get("best_metrics") or {}
    stats_text = (
        f"material={int(np.count_nonzero(debug.get('material_mask', 0)))}  "
        f"observed={int(np.count_nonzero(candidate_mask))}  "
        f"candidates={debug.get('candidates_scored', 0)}"
    )
    if metrics:
        stats_text += f"  score={metrics.get('score', 0.0):.3f}  poly={metrics.get('center_to_polyline_dist', metrics.get('polyline_distance', 0.0)):.1f}"
    if ellipse_params is not None:
        ellipse_mask = ellipse_to_mask(frame.shape, ellipse_params)
        overlap = int(np.count_nonzero((candidate_mask > 0) & (ellipse_mask > 0)))
        stats_text += f"  guide_overlap={overlap}"
    cv2.putText(preview, stats_text, (20, preview.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return preview


def save_settings(settings):
    SETTINGS_SAVE_PATH.write_text(json.dumps(settings, indent=2))
    print(json.dumps(settings, indent=2))
    print(f"Saved settings to {SETTINGS_SAVE_PATH}")


def parse_args():
    parser = argparse.ArgumentParser(description="Live tune coil masking on a saved detection case.")
    parser.add_argument(
        "case",
        nargs="?",
        default=None,
        help="Case folder, detection JSON, or detection JPG. Defaults to the latest output case.",
    )
    parser.add_argument("--web", action="store_true", help="Open the web-based mask tuner instead of the local OpenCV GUI.")
    parser.add_argument("--port", type=int, default=8001, help="Port to use for the web-based mask tuner.")
    return parser.parse_args()


def run_web_mode(case_path, port):
    command = [sys.executable, "mask_tuner_web.py"]
    if case_path is not None:
        command.append(str(case_path))
    command.extend(["--port", str(port)])
    subprocess.run(command)


def main():
    args = parse_args()
    case_path = args.case or str(find_latest_case())

    if args.web:
        print("Launching web-based mask tuner...")
        run_web_mode(case_path, args.port)
        return

    try:
        case_data = load_case(case_path)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, PANEL_W * 3, PANEL_H * 2)
        create_trackbars(get_mask_tuning_defaults())
    except cv2.error as exc:
        print("OpenCV GUI unavailable. Falling back to web-based interface.")
        print(str(exc))
        run_web_mode(case_path, args.port)
        return

    print(f"Loaded case: {case_data['case_dir']}")
    print(f"Frame source: {case_data['image_path']}")
    print(f"Detection JSON: {case_data['json_path']}")
    if "segment_xy" not in json.loads(case_data["json_path"].read_text()):
        print("Using bbox fallback for segment seed because this JSON predates saved segment_xy.")
    print("Keys: q quit | s save settings | r reset defaults | c clear ellipse | a auto-tune to ellipse")
    print("Draw ellipse: left drag, right click to clear")

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    last_signature = None
    last_ellipse_state = None
    while True:
        settings = read_settings_from_trackbars()
        signature = tuple(sorted(settings.items()))
        ellipse_params = get_current_ellipse_params()
        ellipse_state = (
            DRAW_STATE["dragging"],
            DRAW_STATE["start_pt"],
            DRAW_STATE["end_pt"],
            ellipse_params,
        )
        if signature != last_signature or ellipse_state != last_ellipse_state:
            preview = build_preview(case_data, settings, ellipse_params=ellipse_params)
            cv2.imshow(WINDOW_NAME, preview)
            last_signature = signature
            last_ellipse_state = ellipse_state

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            save_settings(settings)
        if key == ord("r"):
            reset_trackbars()
        if key == ord("c"):
            DRAW_STATE["dragging"] = False
            DRAW_STATE["start_pt"] = None
            DRAW_STATE["end_pt"] = None
            DRAW_STATE["ellipse_params"] = None
        if key == ord("a"):
            if ellipse_params is None:
                print("Draw an ellipse first before attempting auto-tune.")
            else:
                print("Auto-tuning mask parameters to ellipse...")
                tuned_settings, score = search_mask_parameters(case_data, settings, ellipse_params)
                if tuned_settings != settings:
                    for spec in TRACKBAR_SPECS:
                        raw = trackbar_raw_from_value(spec, tuned_settings[spec["source_key"]])
                        cv2.setTrackbarPos(spec["name"], CONTROL_WINDOW, raw)
                    print(f"Auto-tune complete: score={score:.4f}")
                else:
                    print(f"Auto-tune found no better parameters (score={score:.4f}).")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
