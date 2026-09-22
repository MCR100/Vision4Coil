"""Computer-vision modelling utilities for coil-tail detection.

This module owns the YOLO segmentation, body/loop geometry, ellipse fitting,
and optional diagnostic image generation. FFT_RTSP.py keeps the stream/FFT
orchestration and result persistence.
"""

from functools import lru_cache

import cv2
import json
import math
import numpy as np
from pathlib import Path

# Multi-frame loop accumulation
DEBUG_SAVE_TOP_K_DIRECT = 5

LOOP_FIT_FRAME_WINDOW = 24
LOOP_ACCUMULATION_MAX_FRAMES = 12
LOOP_ACCUMULATION_STRIDE = 1
LOOP_ACCUMULATION_MIN_VOTES = 3
LOOP_ACCUMULATION_HISTORY_BIAS = 0.75

@lru_cache(maxsize=1)
def get_segmentation_model():
    """Load the segmentation model only when tail detection is requested."""
    from ultralytics import YOLO

    return YOLO("YOLO11/weights/best.pt")

# Hardcoded body reference polyline
BODY_POLYLINE_POINTS = [
    (1386, 753),
    (1167, 1074),
    (916, 1432),
]

# Median-x fallback tuning
LOCAL_BAND_HALF_HEIGHT = 180
MIN_BODY_PIXELS_FOR_MEDIAN = 40

# Ellipse / loop-fit tuning
ELLIPSE_SEARCH_PAD_X = 420
ELLIPSE_SEARCH_PAD_Y_UP = 180
ELLIPSE_SEARCH_PAD_Y_DOWN = 380
MIN_CONTOUR_POINTS_FOR_ELLIPSE = 30
TAIL_BOX_EXCLUDE_PAD = 12
SEGMENT_BOUNDARY_THICKNESS = 3
ELLIPSE_OVERLAP_THICKNESS = 3
COIL_RED_MIN = 0
COIL_EXCESS_RED_MIN = 0
COIL_LAB_A_MIN = 185
COIL_SAT_MIN = 0
COIL_VALUE_MIN = 0
COIL_WHITEHOT_RED_MIN = 255
COIL_ADAPTIVE_RED_FLOOR = 255
COIL_ADAPTIVE_EXCESS_FLOOR = 0
COIL_ADAPTIVE_A_FLOOR = 1
COIL_RELAXED_RED_FLOOR = 35
COIL_RELAXED_EXCESS_FLOOR = -12
COIL_RELAXED_A_FLOOR = 118
COIL_RELAXED_VALUE_FLOOR = 28
COIL_RELAXED_SAT_FLOOR = 8
COIL_RELAXED_SEGMENT_COVERAGE = 1.35
COIL_RELAXED_SEGMENT_INTERSECTION = 0.55
COIL_RELAXED_PAD_X = 520
COIL_RELAXED_PAD_Y = 260
COIL_MASK_CLOSE_KERNEL = 9
COIL_MASK_OPEN_KERNEL = 25
COIL_SUPPORT_DILATE_KERNEL = 1
COIL_ENVELOPE_CLOSE_KERNEL = 9
COIL_ENVELOPE_DILATE_KERNEL = 1
COIL_ENVELOPE_MIN_AREA = 1
LOOP_OPENING_COLOR_DILATE_KERNEL = 5
LOOP_OPENING_OPEN_KERNEL = 5
LOOP_OPENING_CLOSE_KERNEL = 17
LOOP_OPENING_DILATE_KERNEL = 11
LAST_LOOP_OPENING_MIN_AREA = 1800
LAST_LOOP_OPENING_MIN_SPAN_X = 120
LAST_LOOP_OPENING_MIN_SPAN_Y = 45
LAST_LOOP_OPENING_MAX_BOUNDARY_TO_TAIL_DIST = 180.0
LAST_LOOP_OPENING_MAX_CENTER_TO_POLYLINE_DIST = 220.0

# Shape prior used only to keep direct fits from widening unrealistically.
ELLIPSE_SHAPE_PRIOR = {
    "a": 300.0,
    "b": 120.0,
    "rotation_deg": 168.0,
}

# Candidate ellipse constraints
ELLIPSE_MIN_MAJOR = 180
ELLIPSE_MAX_MAJOR = 1200
ELLIPSE_MIN_MINOR = 60
ELLIPSE_MAX_MINOR = 900
MAX_ELLIPSE_ASPECT_RATIO = 8.0

# Reject tiny local contour fragments before ellipse fitting
MIN_CONTOUR_SPAN_X = 180
MIN_CONTOUR_SPAN_Y = 60

# Merge nearby edge fragments before contour extraction
CONTOUR_MERGE_KERNEL = 5
CONTOUR_MERGE_ITERATIONS = 1

MAX_CENTER_TO_POLYLINE_DIST = 350
MAX_CENTER_TO_TAIL_DIST = 950
MIN_CENTER_TO_TAIL_DIST = 40
MAX_TAIL_TO_ELLIPSE_BOUNDARY_DIST = 160
CENTER_ALLOWED_ABOVE_TAIL = 280
CENTER_ALLOWED_BELOW_TAIL = 650

# Direct-fit support thresholds
ELLIPSE_CENTER_ROI_MARGIN = 30
MIN_ELLIPSE_MASK_OVERLAP_RATIO = 0.08
MIN_CONTOUR_SUPPORT_RATIO = 0.18
MIN_TAIL_SIDE_SUPPORT_RATIO = 0.06
DIRECT_VISIBLE_MAX_MASK_DIST = 55.0
DIRECT_TOP_VISIBLE_MAX_MASK_DIST = 65.0
MAX_DIRECT_FIT_SCORE = 260.0

# Last-loop direct extraction. We primarily need the center of the final
# visible loop nearest the tail, so we rank enclosed loop interiors first and
# only fall back to edge-fragment ellipse fitting if no stable interior exists.
LAST_LOOP_MIN_HOLE_AREA = 5000
LAST_LOOP_MIN_HOLE_SPAN_X = 120
LAST_LOOP_MIN_HOLE_SPAN_Y = 70
LAST_LOOP_MAX_BOUNDARY_TO_TAIL_DIST = 260.0
LAST_LOOP_MAX_CENTER_TO_TAIL_DIST = 950.0
LAST_LOOP_MAX_CENTER_TO_POLYLINE_DIST = 420.0
LAST_LOOP_MAX_CENTER_LEFT_OF_TAIL = 40.0
LAST_LOOP_MAX_LEFT_EDGE_LEFT_OF_TAIL = 35.0
LAST_LOOP_ARC_MIN_SPAN_X = 220
LAST_LOOP_ARC_MIN_SPAN_Y = 45
LAST_LOOP_ARC_MAX_SPAN_Y = 220
LAST_LOOP_ARC_MAX_NEAREST_TAIL_DIST = 60.0
LAST_LOOP_ARC_MAX_CENTER_TO_POLYLINE_DIST = 140.0

# Partial-arc fitting guards. When the lower loop is outside the usable ROI,
# a free ellipse fit tends to over-explain the visible top arc by widening.
DIRECT_MIN_ANGLE_COVERAGE_DEG = 145.0
PARTIAL_ARC_COVERAGE_DEG = 210.0
MAX_DIRECT_HORIZONTAL_EXTENT_SCALE = 1.25
DIRECT_AXIS_PRIOR_SCORE_WEIGHT = 1.4
PARTIAL_ARC_AXIS_PRIOR_SCORE_WEIGHT = 3.2
PARTIAL_ARC_DIRECT_SCORE_PENALTY = 140.0

# Anchor-guided fallback. This starts from the direct fit upper arc and reshapes
# the ellipse so left/right anchors land on visible coil edges.
ANCHOR_VISIBLE_MAX_MASK_DIST = 70.0
ANCHOR_MIN_MASK_OVERLAP_RATIO = 0.035
ANCHOR_AXIS_SCALE_FACTORS = [0.85, 0.9, 0.95, 1.0, 1.05]
ANCHOR_HORIZONTAL_SCALE_FACTORS = [0.65, 0.75, 0.85, 0.95, 1.0]
ANCHOR_MAX_TOP_DRIFT_PX = 2.0

# Thermal/polyline last-loop extraction. This is the production version of the
# mask-tuner experiment: look near the conveyor centerline, score hot material
# on a thin ellipse centerline, and explicitly exclude the YOLO tail segment.
THERMAL_LOOP_OUTPUT_RING_THICKNESS = 22
THERMAL_LOOP_CENTERLINE_THICKNESS = 3
THERMAL_LOOP_MIN_OUTPUT_THICKNESS = 10
THERMAL_LOOP_MAX_OUTPUT_THICKNESS = 28
THERMAL_LOOP_MIN_SCORE_SIGMA = 5.0
THERMAL_LOOP_MAX_SCORE_SIGMA = 14.0
THERMAL_LOOP_ROI_HALF_WIDTH = 560
THERMAL_LOOP_ROI_UP_FROM_TAIL = 130
THERMAL_LOOP_ROI_DOWN_FROM_TAIL = 380
THERMAL_LOOP_CENTER_Y_START = 70
THERMAL_LOOP_CENTER_Y_STOP = 340
THERMAL_LOOP_CENTER_Y_STEP = 18
THERMAL_LOOP_CENTER_X_OFFSETS = (-100, -50, 0, 50, 100)
THERMAL_LOOP_AXIS_A_VALUES = (240, 280, 320, 380)
THERMAL_LOOP_AXIS_B_VALUES = (90, 115, 140, 170)
THERMAL_LOOP_AXIS_Y_SCALE = 0.18
THERMAL_LOOP_ANGLE_OFFSETS = (-12.0, -6.0, 0.0, 6.0, 12.0)
THERMAL_LOOP_LONG_TAIL_ROI_MARGIN_PX = 60
THERMAL_LOOP_UPPER_SPAN_BLEND_CENTER_PX = 135.0
THERMAL_LOOP_UPPER_SPAN_BLEND_SCALE_PX = 10.0
THERMAL_LOOP_ROI_SMOOTHNESS_PX = 12.0
THERMAL_LOOP_UPPER_CENTER_Y_START = -80
THERMAL_LOOP_MIN_CENTERLINE_PIXELS = 90
THERMAL_LOOP_MIN_OBSERVED_PIXELS = 180
THERMAL_LOOP_MIN_CLOSE_SUPPORT_RATIO = 0.18
THERMAL_LOOP_MIN_VISIBLE_RATIO = 0.52
THERMAL_LOOP_SECTOR_COUNT = 12
THERMAL_LOOP_MIN_SUPPORTED_SECTORS = 5
THERMAL_LOOP_MIN_ACCEPTED_SCORE = 6.0
THERMAL_LOOP_PREFERRED_AXES = (270.0, 145.0)
THERMAL_LOOP_PREFERRED_ANGLE_DEG = 0.0
THERMAL_LOOP_ANGLE_PRIOR_SIGMA_DEG = 10.0
THERMAL_LOOP_REFINEMENT_STEPS = (
    (12, 8, 12, 8, 2.0),
    (4, 3, 4, 3, 0.75),
)
THERMAL_LOOP_PCA_MIN_PIXELS = 800
THERMAL_LOOP_METHOD = "thermal_polyline_last_loop_ellipse_v2"


# -------------------------
# Polyline / geometry helpers
# -------------------------
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


# -------------------------
# Body-mask / fallback helpers
# -------------------------
def build_body_mask(frame, exclude_box=None):
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


def choose_body_reference(frame, best_box, tail_cy):
    body_x = polyline_x_at_y(BODY_POLYLINE_POINTS, tail_cy)
    if body_x is not None:
        return body_x, "hardcoded_polyline"

    body_mask = build_body_mask(frame, exclude_box=best_box)
    body_x = median_body_x_from_mask(body_mask, tail_cy, LOCAL_BAND_HALF_HEIGHT)
    if body_x is not None:
        return body_x, "local_median_x"

    return None, None


# -------------------------
# Segmentation helpers
# -------------------------
def get_mask_tuning_defaults():
    return {
        "COIL_RED_MIN": COIL_RED_MIN,
        "COIL_EXCESS_RED_MIN": COIL_EXCESS_RED_MIN,
        "COIL_LAB_A_MIN": COIL_LAB_A_MIN,
        "COIL_SAT_MIN": COIL_SAT_MIN,
        "COIL_VALUE_MIN": COIL_VALUE_MIN,
        "COIL_WHITEHOT_RED_MIN": COIL_WHITEHOT_RED_MIN,
        "COIL_ADAPTIVE_RED_FLOOR": COIL_ADAPTIVE_RED_FLOOR,
        "COIL_ADAPTIVE_EXCESS_FLOOR": COIL_ADAPTIVE_EXCESS_FLOOR,
        "COIL_ADAPTIVE_A_FLOOR": COIL_ADAPTIVE_A_FLOOR,
        "COIL_RELAXED_RED_FLOOR": COIL_RELAXED_RED_FLOOR,
        "COIL_RELAXED_EXCESS_FLOOR": COIL_RELAXED_EXCESS_FLOOR,
        "COIL_RELAXED_A_FLOOR": COIL_RELAXED_A_FLOOR,
        "COIL_RELAXED_VALUE_FLOOR": COIL_RELAXED_VALUE_FLOOR,
        "COIL_RELAXED_SAT_FLOOR": COIL_RELAXED_SAT_FLOOR,
        "COIL_RELAXED_SEGMENT_COVERAGE": COIL_RELAXED_SEGMENT_COVERAGE,
        "COIL_RELAXED_SEGMENT_INTERSECTION": COIL_RELAXED_SEGMENT_INTERSECTION,
        "COIL_RELAXED_PAD_X": COIL_RELAXED_PAD_X,
        "COIL_RELAXED_PAD_Y": COIL_RELAXED_PAD_Y,
        "COIL_MASK_CLOSE_KERNEL": COIL_MASK_CLOSE_KERNEL,
        "COIL_MASK_OPEN_KERNEL": COIL_MASK_OPEN_KERNEL,
        "COIL_SUPPORT_DILATE_KERNEL": COIL_SUPPORT_DILATE_KERNEL,
        "COIL_ENVELOPE_CLOSE_KERNEL": COIL_ENVELOPE_CLOSE_KERNEL,
        "COIL_ENVELOPE_DILATE_KERNEL": COIL_ENVELOPE_DILATE_KERNEL,
        "COIL_ENVELOPE_MIN_AREA": COIL_ENVELOPE_MIN_AREA,
    }


def resolve_mask_tuning_settings(settings=None):
    cfg = get_mask_tuning_defaults()
    if settings:
        cfg.update(settings)

    for key in (
        "COIL_MASK_CLOSE_KERNEL",
        "COIL_MASK_OPEN_KERNEL",
        "COIL_SUPPORT_DILATE_KERNEL",
        "COIL_ENVELOPE_CLOSE_KERNEL",
        "COIL_ENVELOPE_DILATE_KERNEL",
    ):
        value = max(1, int(round(cfg[key])))
        if value % 2 == 0:
            value += 1
        cfg[key] = value

    cfg["COIL_ENVELOPE_MIN_AREA"] = max(1, int(round(cfg["COIL_ENVELOPE_MIN_AREA"])))
    cfg["COIL_RELAXED_PAD_X"] = max(0, int(round(cfg["COIL_RELAXED_PAD_X"])))
    cfg["COIL_RELAXED_PAD_Y"] = max(0, int(round(cfg["COIL_RELAXED_PAD_Y"])))
    cfg["COIL_RELAXED_SEGMENT_COVERAGE"] = max(0.05, float(cfg["COIL_RELAXED_SEGMENT_COVERAGE"]))
    cfg["COIL_RELAXED_SEGMENT_INTERSECTION"] = max(0.0, min(1.0, float(cfg["COIL_RELAXED_SEGMENT_INTERSECTION"])))
    return cfg


def polygon_to_mask(poly_xy, image_shape):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if poly_xy is None or len(poly_xy) < 3:
        return mask

    poly = np.round(poly_xy).astype(np.int32)
    cv2.fillPoly(mask, [poly], 255)
    return mask


def dilate_mask(mask, ksize=9, iterations=1):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.dilate(mask, kernel, iterations=iterations)


def extract_best_tail_segment(frames, conf_thresh=0.6):
    best = None
    best_conf = -1.0

    segmentation_model = get_segmentation_model()
    start_idx = max(0, len(frames) - LOOP_FIT_FRAME_WINDOW)

    for frame_idx in range(len(frames) - 1, start_idx - 1, -1):
        frame = frames[frame_idx]

        result = segmentation_model.predict(
            frame,
            conf=conf_thresh,
            verbose=False,
            save=False,
            retina_masks=True,
        )[0]

        boxes = result.boxes
        masks = result.masks

        if boxes is None or masks is None:
            continue

        segs = masks.xy
        for i, box in enumerate(boxes):
            if i >= len(segs):
                continue

            poly_xy = segs[i]
            if poly_xy is None or len(poly_xy) < 3:
                continue

            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if conf > best_conf:
                best_conf = conf
                best = {
                    "frame_index": frame_idx,
                    "frame": frame.copy(),
                    "class_id": cls_id,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "segment_xy": poly_xy.astype(np.float32),
                }

    return best


def select_loop_accumulation_indices(frame_count, best_frame_index, max_frames=LOOP_ACCUMULATION_MAX_FRAMES, stride=LOOP_ACCUMULATION_STRIDE):
    if int(frame_count) <= 0:
        return []

    n = int(frame_count)
    best_frame_index = max(0, min(int(best_frame_index), n - 1))
    stride = max(1, int(stride))
    max_frames = max(1, int(max_frames))

    if max_frames == 1:
        return [best_frame_index]

    usable_slots = max_frames - 1
    preferred_history = int(math.ceil(usable_slots * LOOP_ACCUMULATION_HISTORY_BIAS))

    history_candidates = list(range(best_frame_index - stride, -1, -stride))
    future_candidates = list(range(best_frame_index + stride, n, stride))

    history_take = min(len(history_candidates), preferred_history)
    future_take = min(len(future_candidates), usable_slots - history_take)

    selected_indices = [best_frame_index]
    selected_indices.extend(history_candidates[:history_take])
    selected_indices.extend(future_candidates[:future_take])

    remaining_slots = max_frames - len(selected_indices)
    if remaining_slots > 0:
        extra_history = history_candidates[history_take:]
        extra_future = future_candidates[future_take:]
        for idx in extra_history:
            if remaining_slots <= 0:
                break
            selected_indices.append(idx)
            remaining_slots -= 1
        for idx in extra_future:
            if remaining_slots <= 0:
                break
            selected_indices.append(idx)
            remaining_slots -= 1

    return sorted(set(selected_indices))


def select_loop_accumulation_frames(frames, best_frame_index, max_frames=LOOP_ACCUMULATION_MAX_FRAMES, stride=LOOP_ACCUMULATION_STRIDE):
    selected_indices = select_loop_accumulation_indices(
        len(frames), best_frame_index, max_frames=max_frames, stride=stride
    )
    return [(idx, frames[idx]) for idx in selected_indices]


def draw_segment_outline(frame, poly_xy, color=(0, 255, 255), thickness=2):
    if poly_xy is None or len(poly_xy) < 2:
        return
    pts = np.round(poly_xy).astype(np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)


def segment_centroid(poly_xy):
    pts = np.asarray(poly_xy, dtype=np.float32)
    if len(pts) == 0:
        return None
    return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))


def find_tail_tip_from_segment(poly_xy, polyline_points):
    """
    Find two endpoints from the segment major axis.
    For current footage, the endpoint closer to the polyline is used as the tip.
    """
    pts = np.asarray(poly_xy, dtype=np.float32)
    if len(pts) < 2:
        return None, None

    mean = np.mean(pts, axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]

    proj = centered @ axis
    idx_min = int(np.argmin(proj))
    idx_max = int(np.argmax(proj))

    end_a = (float(pts[idx_min, 0]), float(pts[idx_min, 1]))
    end_b = (float(pts[idx_max, 0]), float(pts[idx_max, 1]))

    dist_a = point_to_polyline_distance(end_a[0], end_a[1], polyline_points)
    dist_b = point_to_polyline_distance(end_b[0], end_b[1], polyline_points)

    if dist_a is None or dist_b is None:
        return end_a, end_b

    # In this camera view, the endpoint nearer the conveyor polyline is the tail tip.
    if dist_a <= dist_b:
        tip = end_a
        base = end_b
    else:
        tip = end_b
        base = end_a

    return tip, base


# -------------------------
# Ellipse-fit helpers
# -------------------------
def build_segment_boundary_mask(segment_mask, thickness=SEGMENT_BOUNDARY_THICKNESS):
    if segment_mask is None or np.count_nonzero(segment_mask) == 0:
        return np.zeros_like(segment_mask if segment_mask is not None else np.zeros((1, 1), dtype=np.uint8))

    kernel = np.ones((3, 3), np.uint8)
    boundary = cv2.morphologyEx(segment_mask, cv2.MORPH_GRADIENT, kernel)
    if thickness > 1:
        boundary = dilate_mask(boundary, ksize=max(3, int(thickness)), iterations=1)
    return boundary


def build_tail_exclusion_mask(
    image_shape,
    tail_tip,
    tail_base=None,
    bbox=None,
    segment_xy=None,
    tip_radius=90,
    dilate_ksize=9,
):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if tail_tip is None:
        return mask

    tx, ty = float(tail_tip[0]), float(tail_tip[1])

    if segment_xy is not None and len(segment_xy) >= 3:
        seg_mask = polygon_to_mask(segment_xy, image_shape)

        ys, xs = np.where(seg_mask > 0)
        if len(xs) > 0:
            dist2 = (xs.astype(np.float32) - tx) ** 2 + (ys.astype(np.float32) - ty) ** 2
            keep = dist2 <= float(tip_radius * tip_radius)

            local_mask = np.zeros_like(seg_mask)
            local_mask[ys[keep], xs[keep]] = 255

            if dilate_ksize and dilate_ksize > 1:
                kernel = np.ones((dilate_ksize, dilate_ksize), np.uint8)
                local_mask = cv2.dilate(local_mask, kernel, iterations=1)

            return local_mask

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - TAIL_BOX_EXCLUDE_PAD)
        y1 = max(0, y1 - TAIL_BOX_EXCLUDE_PAD)
        x2 = min(w, x2 + TAIL_BOX_EXCLUDE_PAD)
        y2 = min(h, y2 + TAIL_BOX_EXCLUDE_PAD)
        mask[y1:y2, x1:x2] = 255
        return mask

    cv2.circle(mask, (int(round(tx)), int(round(ty))), int(tip_radius * 0.5), 255, -1)
    return mask


def build_coil_color_mask(frame, segment_mask=None, settings=None):
    cfg = resolve_mask_tuning_settings(settings)
    b, g, r = cv2.split(frame)
    r16 = r.astype(np.int16)
    g16 = g.astype(np.int16)
    b16 = b.astype(np.int16)

    excess_red = (2 * r16) - g16 - b16
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    a_chan = lab[:, :, 1]

    hot_red = (
        (r >= cfg["COIL_RED_MIN"]) &
        (excess_red >= cfg["COIL_EXCESS_RED_MIN"]) &
        (a_chan >= cfg["COIL_LAB_A_MIN"]) &
        ((sat >= cfg["COIL_SAT_MIN"]) | (val >= cfg["COIL_VALUE_MIN"]))
    )
    white_hot = (
        (r >= cfg["COIL_WHITEHOT_RED_MIN"]) &
        (r16 >= g16 + 8) &
        (g16 >= b16 - 5)
    )

    adaptive_red = hot_red.copy()
    relaxed_red = np.zeros_like(hot_red, dtype=bool)
    if segment_mask is not None and np.count_nonzero(segment_mask) > 40:
        seg_pixels = segment_mask > 0
        seg_r = r[seg_pixels].astype(np.float32)
        seg_excess = excess_red[seg_pixels].astype(np.float32)
        seg_a = a_chan[seg_pixels].astype(np.float32)
        seg_sat = sat[seg_pixels].astype(np.float32)
        seg_val = val[seg_pixels].astype(np.float32)

        red_thr = max(cfg["COIL_ADAPTIVE_RED_FLOOR"], float(np.percentile(seg_r, 12)) - 35.0)
        excess_thr = max(cfg["COIL_ADAPTIVE_EXCESS_FLOOR"], float(np.percentile(seg_excess, 18)) - 28.0)
        a_thr = max(cfg["COIL_ADAPTIVE_A_FLOOR"], float(np.percentile(seg_a, 10)) - 10.0)
        sat_thr = max(10.0, float(np.percentile(seg_sat, 10)) - 18.0)
        val_thr = max(40.0, float(np.percentile(seg_val, 8)) - 25.0)

        adaptive_red = (
            (r.astype(np.float32) >= red_thr) &
            (excess_red.astype(np.float32) >= excess_thr) &
            (a_chan.astype(np.float32) >= a_thr) &
            ((sat.astype(np.float32) >= sat_thr) | (val.astype(np.float32) >= val_thr))
        )

        ys, xs = np.where(seg_pixels)
        if len(xs) > 0:
            x1 = max(0, int(xs.min()) - cfg["COIL_RELAXED_PAD_X"])
            x2 = min(frame.shape[1], int(xs.max()) + cfg["COIL_RELAXED_PAD_X"])
            y1 = max(0, int(ys.min()) - cfg["COIL_RELAXED_PAD_Y"])
            y2 = min(frame.shape[0], int(ys.max()) + cfg["COIL_RELAXED_PAD_Y"])

            relaxed_red_thr = max(cfg["COIL_RELAXED_RED_FLOOR"], float(np.percentile(seg_r, 5)) - 50.0)
            relaxed_excess_thr = max(cfg["COIL_RELAXED_EXCESS_FLOOR"], float(np.percentile(seg_excess, 5)) - 35.0)
            relaxed_a_thr = max(cfg["COIL_RELAXED_A_FLOOR"], float(np.percentile(seg_a, 5)) - 14.0)
            relaxed_sat_thr = max(cfg["COIL_RELAXED_SAT_FLOOR"], float(np.percentile(seg_sat, 5)) - 18.0)
            relaxed_val_thr = max(cfg["COIL_RELAXED_VALUE_FLOOR"], float(np.percentile(seg_val, 5)) - 30.0)

            local_relaxed = (
                (r[y1:y2, x1:x2].astype(np.float32) >= relaxed_red_thr) &
                (excess_red[y1:y2, x1:x2].astype(np.float32) >= relaxed_excess_thr) &
                (a_chan[y1:y2, x1:x2].astype(np.float32) >= relaxed_a_thr) &
                ((sat[y1:y2, x1:x2].astype(np.float32) >= relaxed_sat_thr) |
                 (val[y1:y2, x1:x2].astype(np.float32) >= relaxed_val_thr)) &
                (
                    (r16[y1:y2, x1:x2] >= g16[y1:y2, x1:x2] + 3) |
                    (a_chan[y1:y2, x1:x2].astype(np.float32) >= relaxed_a_thr + 6.0)
                )
            )
            relaxed_red[y1:y2, x1:x2] = local_relaxed

    mask_bool = hot_red | adaptive_red | white_hot
    mask = np.where(mask_bool, 255, 0).astype(np.uint8)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg["COIL_MASK_CLOSE_KERNEL"], cfg["COIL_MASK_CLOSE_KERNEL"]))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg["COIL_MASK_OPEN_KERNEL"], cfg["COIL_MASK_OPEN_KERNEL"]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

    if segment_mask is not None and np.count_nonzero(segment_mask) > 40:
        seg_area = float(np.count_nonzero(segment_mask))
        seg_overlap = float(np.count_nonzero((mask > 0) & (segment_mask > 0)))
        if (
            np.count_nonzero(mask) < cfg["COIL_RELAXED_SEGMENT_COVERAGE"] * seg_area
            or seg_overlap < cfg["COIL_RELAXED_SEGMENT_INTERSECTION"] * seg_area
        ):
            relaxed_mask = np.where(mask_bool | relaxed_red, 255, 0).astype(np.uint8)
            relaxed_mask = cv2.morphologyEx(relaxed_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
            relaxed_mask = cv2.morphologyEx(relaxed_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
            mask = relaxed_mask

    return mask


def fill_mask_holes(mask):
    if mask is None or mask.size == 0:
        return mask

    h, w = mask.shape[:2]
    flood = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)


def build_loop_opening_band_mask(coil_envelope_mask, coil_color_mask, tail_exclusion=None):
    if coil_envelope_mask is None or coil_color_mask is None:
        return (
            np.zeros_like(coil_envelope_mask if coil_envelope_mask is not None else coil_color_mask),
            np.zeros_like(coil_envelope_mask if coil_envelope_mask is not None else coil_color_mask),
        )

    color_dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (LOOP_OPENING_COLOR_DILATE_KERNEL, LOOP_OPENING_COLOR_DILATE_KERNEL),
    )
    color_keep = cv2.dilate(coil_color_mask, color_dilate_kernel, iterations=1)
    opening_mask = cv2.bitwise_and(coil_envelope_mask, cv2.bitwise_not(color_keep))

    if tail_exclusion is not None:
        opening_mask[tail_exclusion > 0] = 0

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (LOOP_OPENING_OPEN_KERNEL, LOOP_OPENING_OPEN_KERNEL),
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (LOOP_OPENING_CLOSE_KERNEL, LOOP_OPENING_CLOSE_KERNEL),
    )
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (LOOP_OPENING_DILATE_KERNEL, LOOP_OPENING_DILATE_KERNEL),
    )

    opening_mask = cv2.morphologyEx(opening_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    opening_band_mask = cv2.morphologyEx(opening_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    opening_band_mask = cv2.dilate(opening_band_mask, dilate_kernel, iterations=1)
    opening_band_mask = cv2.bitwise_and(opening_band_mask, coil_envelope_mask)

    return opening_mask, opening_band_mask


def build_boundary_candidate_mask(coil_boundary_mask, segment_mask, tail_tip=None, trim_half_window=22):
    if coil_boundary_mask is None:
        return None

    candidate = coil_boundary_mask.copy()
    if segment_mask is None or np.count_nonzero(segment_mask) == 0:
        return candidate

    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return candidate

    working = np.zeros_like(candidate)
    tx = None if tail_tip is None else float(tail_tip[0])
    ty = None if tail_tip is None else float(tail_tip[1])

    for contour in contours:
        pts = contour.reshape(-1, 2)
        if len(pts) < 2:
            continue

        if tx is None or ty is None:
            trimmed = pts
        else:
            dist2 = (
                (pts[:, 0].astype(np.float32) - tx) ** 2 +
                (pts[:, 1].astype(np.float32) - ty) ** 2
            )
            nearest_idx = int(np.argmin(dist2))
            n = len(pts)
            keep = np.ones(n, dtype=bool)

            # Remove only a short arc around the boundary point nearest the tail.
            for offset in range(-int(trim_half_window), int(trim_half_window) + 1):
                keep[(nearest_idx + offset) % n] = False

            trimmed = pts[keep]
            if len(trimmed) < max(10, n // 3):
                trimmed = pts

        trimmed_contour = trimmed.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(working, [trimmed_contour], False, 255, 1)

    return working


def select_tail_adjacent_component(mask, tail_tip, min_area=COIL_ENVELOPE_MIN_AREA):
    if mask is None or np.count_nonzero(mask) == 0:
        return np.zeros_like(mask if mask is not None else np.zeros((1, 1), dtype=np.uint8))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    tx, ty = float(tail_tip[0]), float(tail_tip[1])

    best_label = None
    best_score = None
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < int(min_area):
            continue

        component = (labels == label)
        ys, xs = np.where(component)
        if len(xs) == 0:
            continue

        dist2 = (xs.astype(np.float32) - tx) ** 2 + (ys.astype(np.float32) - ty) ** 2
        min_dist = float(np.sqrt(np.min(dist2)))
        score = min_dist - 0.0005 * float(area)
        if best_score is None or score < best_score:
            best_score = score
            best_label = label

    if best_label is None:
        return mask.copy()

    return np.where(labels == best_label, 255, 0).astype(np.uint8)


def build_coil_envelope_mask(frame, segment_mask, tail_tip, settings=None):
    cfg = resolve_mask_tuning_settings(settings)
    coil_color_mask = build_coil_color_mask(frame, segment_mask=segment_mask, settings=cfg)
    seed_mask = cv2.bitwise_or(coil_color_mask, segment_mask)

    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (cfg["COIL_ENVELOPE_DILATE_KERNEL"], cfg["COIL_ENVELOPE_DILATE_KERNEL"]),
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (cfg["COIL_ENVELOPE_CLOSE_KERNEL"], cfg["COIL_ENVELOPE_CLOSE_KERNEL"]),
    )

    envelope = cv2.dilate(seed_mask, dilate_kernel, iterations=1)
    envelope = cv2.morphologyEx(envelope, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    envelope = fill_mask_holes(envelope)
    envelope = select_tail_adjacent_component(envelope, tail_tip, min_area=cfg["COIL_ENVELOPE_MIN_AREA"])
    envelope = cv2.morphologyEx(envelope, cv2.MORPH_OPEN, dilate_kernel, iterations=1)
    envelope = fill_mask_holes(envelope)
    return coil_color_mask, envelope


def build_loop_candidate_mask(frame, segment_xy, tail_tip, tail_base=None, bbox=None, remove_segment_fill=True, settings=None):
    cfg = resolve_mask_tuning_settings(settings)
    segment_mask = polygon_to_mask(segment_xy, frame.shape)
    segment_boundary = build_segment_boundary_mask(segment_mask)
    coil_color_mask, coil_envelope_mask = build_coil_envelope_mask(frame, segment_mask, tail_tip, settings=cfg)
    coil_support_mask = dilate_mask(coil_envelope_mask, ksize=cfg["COIL_SUPPORT_DILATE_KERNEL"], iterations=1)
    coil_boundary = build_segment_boundary_mask(coil_envelope_mask)
    boundary_candidate_mask = build_boundary_candidate_mask(
        coil_boundary,
        segment_mask,
        tail_tip=tail_tip,
        trim_half_window=18,
    )
    kernel = np.ones((3, 3), np.uint8)

    candidate = coil_boundary.copy()

    tail_exclusion = build_tail_exclusion_mask(
        frame.shape,
        tail_tip,
        tail_base=tail_base,
        bbox=bbox,
        segment_xy=segment_xy,
        tip_radius=90,
        dilate_ksize=9,
    )
    candidate[tail_exclusion > 0] = 0
    coil_support_mask[tail_exclusion > 0] = 0
    coil_envelope_mask[tail_exclusion > 0] = 0

    loop_opening_mask, loop_opening_band_mask = build_loop_opening_band_mask(
        coil_envelope_mask,
        coil_color_mask,
        tail_exclusion=tail_exclusion,
    )

    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=1)
    candidate = cv2.bitwise_and(candidate, coil_support_mask)

    debug = {
        "segment_mask": segment_mask,
        "segment_boundary": segment_boundary,
        "coil_color_mask": coil_color_mask,
        "coil_envelope_mask": coil_envelope_mask,
        "coil_support_mask": coil_support_mask,
        "coil_boundary_mask": coil_boundary,
        "boundary_candidate_mask": boundary_candidate_mask,
        "loop_opening_mask": loop_opening_mask,
        "loop_opening_band_mask": loop_opening_band_mask,
        "edge_mask": candidate.copy(),
        "tail_exclusion_mask": tail_exclusion,
    }
    return candidate, debug



def ellipse_params_to_mask(image_shape, ellipse_params):
    mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    if ellipse_params is None:
        return mask

    center, axes, angle = ellipse_params
    if axes[0] <= 0 or axes[1] <= 0:
        return mask

    cv2.ellipse(mask, center, axes, angle, 0.0, 360.0, 255, -1)
    return mask


def ellipse_params_outline_to_mask(image_shape, ellipse_params, thickness=THERMAL_LOOP_OUTPUT_RING_THICKNESS):
    mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    if ellipse_params is None:
        return mask

    center, axes, angle = ellipse_params
    if axes[0] <= 0 or axes[1] <= 0:
        return mask

    cv2.ellipse(mask, center, axes, angle, 0.0, 360.0, 255, max(1, int(round(thickness))))
    return mask


def normalize_ellipse_angle_deg(angle_deg):
    angle = float(angle_deg) % 180.0
    if angle < 0.0:
        angle += 180.0
    return angle


def ellipse_params_to_info(ellipse_params, method=THERMAL_LOOP_METHOD):
    center, axes, angle = ellipse_params
    return {
        "center": (float(center[0]), float(center[1])),
        "axes": (float(axes[0]), float(axes[1])),
        "rotation_deg": float(normalize_ellipse_angle_deg(angle)),
        "method": method,
    }


def build_last_loop_heat_score(frame):
    b, g, r = cv2.split(frame)
    r32 = r.astype(np.float32)
    g32 = g.astype(np.float32)
    b32 = b.astype(np.float32)
    excess_red = np.clip((2.0 * r32 - g32 - b32 + 255.0) / 510.0, 0.0, 1.0)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    a_score = lab[:, :, 1].astype(np.float32) / 255.0
    value_score = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32) / 255.0
    red_score = r32 / 255.0
    heat = (0.42 * red_score) + (0.32 * excess_red) + (0.18 * a_score) + (0.08 * value_score)
    return np.clip(heat, 0.0, 1.0).astype(np.float32)


def build_last_loop_polyline_roi_mask(
    image_shape,
    tail_tip,
    polyline_points=BODY_POLYLINE_POINTS,
    roi_up_from_tail=None,
):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if tail_tip is None:
        return mask, None

    tail_x, tail_y = float(tail_tip[0]), float(tail_tip[1])
    roi_up = THERMAL_LOOP_ROI_UP_FROM_TAIL if roi_up_from_tail is None else float(roi_up_from_tail)
    y1 = max(0, int(round(tail_y - roi_up)))
    y2 = min(h, int(round(tail_y + THERMAL_LOOP_ROI_DOWN_FROM_TAIL)))
    if y2 <= y1:
        return mask, None

    x_min = w
    x_max = 0
    for y in range(y1, y2):
        body_x = polyline_x_at_y(polyline_points, y)
        if body_x is None:
            body_x = tail_x
        x1 = max(0, int(round(body_x - THERMAL_LOOP_ROI_HALF_WIDTH)))
        x2 = min(w, int(round(body_x + THERMAL_LOOP_ROI_HALF_WIDTH)))
        mask[y, x1:x2] = 255
        x_min = min(x_min, x1)
        x_max = max(x_max, x2)

    roi_box = None if x_max <= x_min else (int(x_min), int(y1), int(x_max), int(y2))
    return mask, roi_box


def build_last_loop_material_mask(frame, roi_mask, segment_mask=None, settings=None):
    heat = build_last_loop_heat_score(frame)
    color_mask = build_coil_color_mask(frame, segment_mask=segment_mask, settings=settings)

    roi_pixels = roi_mask > 0
    if np.count_nonzero(roi_pixels) == 0:
        return color_mask, heat, color_mask

    heat_values = heat[roi_pixels]
    heat_floor = max(0.40, float(np.percentile(heat_values, 76)))
    heat_mask = np.where((heat >= heat_floor) & roi_pixels, 255, 0).astype(np.uint8)
    color_roi_mask = cv2.bitwise_and(color_mask, roi_mask)
    material = cv2.bitwise_or(color_roi_mask, heat_mask)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    material = cv2.morphologyEx(material, cv2.MORPH_OPEN, open_kernel, iterations=1)
    material = cv2.morphologyEx(material, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    return material, heat, color_roi_mask


def polyline_cross_axis_angle_deg(polyline_points, y):
    pts = get_sorted_polyline(polyline_points)
    if len(pts) < 2:
        return 0.0

    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        if y1 <= y <= y2 or y2 <= y <= y1:
            tangent = math.degrees(math.atan2(float(y2 - y1), float(x2 - x1)))
            return normalize_ellipse_angle_deg(tangent + 90.0)

    # Clamp to the nearest segment outside the polyline y-range.
    if y < pts[0][1]:
        x1, y1 = pts[0]
        x2, y2 = pts[1]
    else:
        x1, y1 = pts[-2]
        x2, y2 = pts[-1]
    tangent = math.degrees(math.atan2(float(y2 - y1), float(x2 - x1)))
    return normalize_ellipse_angle_deg(tangent + 90.0)


def estimate_material_axis_angle_deg(material_mask, roi_box, polyline_points, tail_tip):
    fallback = polyline_cross_axis_angle_deg(polyline_points, float(tail_tip[1]))
    if material_mask is None or roi_box is None:
        return fallback, "polyline_cross_axis"

    x1, y1, x2, y2 = roi_box
    local = material_mask[y1:y2, x1:x2]
    ys, xs = np.where(local > 0)
    if len(xs) < THERMAL_LOOP_PCA_MIN_PIXELS:
        return fallback, "polyline_cross_axis"

    points = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    mean = np.mean(points, axis=0)
    centered = points - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    angle = math.degrees(math.atan2(float(axis[1]), float(axis[0])))
    return normalize_ellipse_angle_deg(angle), "material_pca"


def angle_candidates_from_footage(material_mask, roi_box, polyline_points, tail_tip):
    base_angle, source = estimate_material_axis_angle_deg(material_mask, roi_box, polyline_points, tail_tip)
    candidates = []
    for offset in THERMAL_LOOP_ANGLE_OFFSETS:
        angle = normalize_ellipse_angle_deg(base_angle + offset)
        if all(angular_difference_deg(angle, existing) > 1.0 for existing in candidates):
            candidates.append(angle)
    return candidates, base_angle, source


def estimate_material_width_px(material_mask, roi_box):
    if material_mask is None or roi_box is None:
        return THERMAL_LOOP_OUTPUT_RING_THICKNESS, 0.5 * THERMAL_LOOP_OUTPUT_RING_THICKNESS

    x1, y1, x2, y2 = roi_box
    local = np.where(material_mask[y1:y2, x1:x2] > 0, 255, 0).astype(np.uint8)
    if np.count_nonzero(local) < THERMAL_LOOP_MIN_OBSERVED_PIXELS:
        return THERMAL_LOOP_OUTPUT_RING_THICKNESS, 0.5 * THERMAL_LOOP_OUTPUT_RING_THICKNESS

    dist_inside = cv2.distanceTransform(local, cv2.DIST_L2, 3)
    values = dist_inside[local > 0]
    if values.size == 0:
        return THERMAL_LOOP_OUTPUT_RING_THICKNESS, 0.5 * THERMAL_LOOP_OUTPUT_RING_THICKNESS

    half_width = float(np.percentile(values, 60))
    thickness = int(round(2.0 * half_width + 4.0))
    thickness = max(THERMAL_LOOP_MIN_OUTPUT_THICKNESS, min(THERMAL_LOOP_MAX_OUTPUT_THICKNESS, thickness))
    sigma = max(THERMAL_LOOP_MIN_SCORE_SIGMA, min(THERMAL_LOOP_MAX_SCORE_SIGMA, 0.50 * float(thickness)))
    return thickness, sigma


def axis_candidates_for_y(
    center_y,
    tail_y,
    center_y_start=THERMAL_LOOP_CENTER_Y_START,
    center_y_stop=THERMAL_LOOP_CENTER_Y_STOP,
    axis_y_scale=THERMAL_LOOP_AXIS_Y_SCALE,
):
    rel = np.clip(
        (float(center_y) - (float(tail_y) + center_y_start)) /
        max(1.0, center_y_stop - center_y_start),
        0.0,
        1.0,
    )
    scale = 1.0 + axis_y_scale * (rel - 0.5)
    candidates = []
    seen = set()
    for a in THERMAL_LOOP_AXIS_A_VALUES:
        for b in THERMAL_LOOP_AXIS_B_VALUES:
            axes = (max(1, int(round(a * scale))), max(1, int(round(b * scale))))
            if axes not in seen:
                seen.add(axes)
                candidates.append(axes)
    return candidates


def thermal_loop_search_geometry(tail_upper_span):
    """Continuously adapt the search envelope to the observed upper-tail extent."""
    span = max(0.0, float(tail_upper_span))
    blend_argument = (span - THERMAL_LOOP_UPPER_SPAN_BLEND_CENTER_PX) / THERMAL_LOOP_UPPER_SPAN_BLEND_SCALE_PX
    adaptation_weight = 1.0 / (1.0 + math.exp(-blend_argument))

    desired_roi_up = span + THERMAL_LOOP_LONG_TAIL_ROI_MARGIN_PX
    smooth_argument = (desired_roi_up - THERMAL_LOOP_ROI_UP_FROM_TAIL) / THERMAL_LOOP_ROI_SMOOTHNESS_PX
    softplus = smooth_argument if smooth_argument > 50.0 else math.log1p(math.exp(smooth_argument))
    roi_up_from_tail = THERMAL_LOOP_ROI_UP_FROM_TAIL + THERMAL_LOOP_ROI_SMOOTHNESS_PX * softplus
    center_y_start = int(round(THERMAL_LOOP_CENTER_Y_START + adaptation_weight * (THERMAL_LOOP_UPPER_CENTER_Y_START - THERMAL_LOOP_CENTER_Y_START)))
    axis_y_scale = THERMAL_LOOP_AXIS_Y_SCALE * (1.0 - adaptation_weight)
    return {
        "adaptation_weight": float(adaptation_weight),
        "roi_up_from_tail": float(roi_up_from_tail),
        "center_y_start": int(center_y_start),
        "axis_y_scale": float(axis_y_scale),
    }


@lru_cache(maxsize=512)
def ellipse_centerline_samples(
    local_height,
    local_width,
    center_x,
    center_y,
    axis_a,
    axis_b,
    angle,
    thickness=THERMAL_LOOP_CENTERLINE_THICKNESS,
    sector_count=THERMAL_LOOP_SECTOR_COUNT,
):
    """Cache centerline coordinates and angular sectors for repeated ellipse shapes."""
    local_shape = (int(local_height), int(local_width))
    local_ellipse = (
        (int(center_x), int(center_y)),
        (axis_a, axis_b),
        float(angle),
    )
    centerline = ellipse_params_outline_to_mask(
        local_shape, local_ellipse, thickness=int(thickness)
    )
    ys, xs = np.where(centerline > 0)
    if len(xs) == 0:
        return ys, xs, np.empty(0, dtype=np.int32)

    theta = math.radians(float(angle))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    dx = xs.astype(np.float32) - float(center_x)
    dy = ys.astype(np.float32) - float(center_y)
    local_x = cos_t * dx + sin_t * dy
    local_y = -sin_t * dx + cos_t * dy
    params = np.mod(
        np.arctan2(
            local_y / max(1.0, float(axis_b)),
            local_x / max(1.0, float(axis_a)),
        ),
        2.0 * math.pi,
    )
    sector_ids = np.minimum(
        int(sector_count) - 1,
        (params * float(sector_count) / (2.0 * math.pi)).astype(np.int32),
    )
    return ys, xs, sector_ids


def ellipse_sector_support(
    sector_ids,
    valid_values,
    close_support,
    sector_count=THERMAL_LOOP_SECTOR_COUNT,
):
    close_values = np.zeros(len(sector_ids), dtype=bool)
    close_values[valid_values] = close_support
    sector_valid_pixels = np.bincount(
        sector_ids[valid_values], minlength=int(sector_count)
    )
    sector_close_pixels = np.bincount(
        sector_ids[close_values], minlength=int(sector_count)
    )
    sector_ratios = np.divide(
        sector_close_pixels,
        np.maximum(1, sector_valid_pixels),
        dtype=np.float64,
    )
    supported = np.count_nonzero(
        (sector_valid_pixels >= 12) & (sector_ratios >= 0.35)
    )
    return (
        int(supported),
        sector_ratios.astype(float).tolist(),
        sector_valid_pixels.astype(int).tolist(),
    )


def ellipse_offset_contrast(local_shape, local_ellipse, valid_region, local_material, offset_px):
    (cx, cy), (a, b), angle = local_ellipse
    occupancies = []
    for direction in (-1.0, 1.0):
        offset_axes = (
            max(1, int(round(float(a) + direction * float(offset_px)))),
            max(1, int(round(float(b) + direction * float(offset_px)))),
        )
        offset_mask = ellipse_params_outline_to_mask(
            local_shape,
            ((cx, cy), offset_axes, angle),
            thickness=THERMAL_LOOP_CENTERLINE_THICKNESS,
        ) > 0
        valid_offset = offset_mask & valid_region
        valid_count = int(np.count_nonzero(valid_offset))
        if valid_count >= THERMAL_LOOP_MIN_CENTERLINE_PIXELS:
            occupancies.append(float(np.count_nonzero(valid_offset & local_material)) / valid_count)
    return 0.0 if not occupancies else float(1.0 - np.mean(occupancies))


def score_thermal_last_loop_ellipse(
    image_shape,
    heat,
    material_mask,
    material_distance,
    exclusion_mask,
    ellipse_params,
    tail_tip,
    output_thickness,
    score_sigma,
    polyline_points=BODY_POLYLINE_POINTS,
    guide_mask=None,
    minimum_score=None,
):
    center, axes, angle = ellipse_params
    cx_i, cy_i = int(center[0]), int(center[1])
    pad = int(max(axes) + max(output_thickness, score_sigma) + 8)
    h, w = image_shape[:2]
    x1 = max(0, cx_i - pad)
    x2 = min(w, cx_i + pad + 1)
    y1 = max(0, cy_i - pad)
    y2 = min(h, cy_i + pad + 1)
    if x2 <= x1 or y2 <= y1:
        return None

    local_shape = (y2 - y1, x2 - x1)
    local_ellipse = ((cx_i - x1, cy_i - y1), axes, angle)
    centerline_ys, centerline_xs, sector_ids = ellipse_centerline_samples(
        local_shape[0],
        local_shape[1],
        local_ellipse[0][0],
        local_ellipse[0][1],
        axes[0],
        axes[1],
        angle,
    )
    centerline_count = len(centerline_xs)
    if centerline_count < THERMAL_LOOP_MIN_CENTERLINE_PIXELS:
        return None

    local_material = material_mask[y1:y2, x1:x2] > 0
    local_exclusion = exclusion_mask[y1:y2, x1:x2] > 0
    if np.count_nonzero(local_material) == 0:
        return None

    valid_centerline_values = ~local_exclusion[centerline_ys, centerline_xs]
    valid_centerline_count = int(np.count_nonzero(valid_centerline_values))
    visible_ratio = valid_centerline_count / float(centerline_count)
    if (
        valid_centerline_count < THERMAL_LOOP_MIN_CENTERLINE_PIXELS
        or visible_ratio < THERMAL_LOOP_MIN_VISIBLE_RATIO
    ):
        return None

    valid_centerline_ys = centerline_ys[valid_centerline_values]
    valid_centerline_xs = centerline_xs[valid_centerline_values]
    local_distance = material_distance[y1:y2, x1:x2]
    line_distances = local_distance[valid_centerline_ys, valid_centerline_xs]
    close_support = line_distances <= float(score_sigma)
    close_support_count = int(np.count_nonzero(close_support))
    close_support_ratio = close_support_count / float(valid_centerline_count)
    if close_support_count < THERMAL_LOOP_MIN_OBSERVED_PIXELS or close_support_ratio < THERMAL_LOOP_MIN_CLOSE_SUPPORT_RATIO:
        return None

    support_values = np.exp(-0.5 * (line_distances / max(1e-6, float(score_sigma))) ** 2)
    support_score = float(np.mean(support_values))
    mean_distance = float(np.mean(line_distances))

    output_ring_mask = ellipse_params_outline_to_mask(local_shape, local_ellipse, thickness=output_thickness)
    output_ring_bool = output_ring_mask > 0
    valid_ring_bool = output_ring_bool & ~local_exclusion
    observed_bool = valid_ring_bool & local_material
    observed_count = int(np.count_nonzero(observed_bool))
    if observed_count < THERMAL_LOOP_MIN_OBSERVED_PIXELS:
        return None

    fill_mask = ellipse_params_to_mask(local_shape, local_ellipse)
    fill_bool = fill_mask > 0

    cx, cy = float(center[0]), float(center[1])
    tail_y = float(tail_tip[1])
    topness = 1.0 - np.clip(
        (cy - (tail_y + THERMAL_LOOP_CENTER_Y_START)) /
        max(1.0, THERMAL_LOOP_CENTER_Y_STOP - THERMAL_LOOP_CENTER_Y_START),
        0.0,
        1.0,
    )
    poly_dist = point_to_polyline_distance(cx, cy, polyline_points)
    if poly_dist is None:
        poly_score = 0.45
    else:
        poly_score = max(0.0, 1.0 - (float(poly_dist) / 260.0))

    local_heat = heat[y1:y2, x1:x2]
    supported_line_heat = float(np.mean(
        local_heat[
            valid_centerline_ys[close_support],
            valid_centerline_xs[close_support],
        ]
    ))
    valid_fill = fill_bool & ~local_exclusion
    valid_fill_count = max(1, int(np.count_nonzero(valid_fill)))
    fill_material = int(np.count_nonzero(valid_fill & local_material))
    fill_density = fill_material / float(valid_fill_count)
    exclusion_overlap = int(np.count_nonzero(output_ring_bool & local_exclusion)) / float(max(1, np.count_nonzero(output_ring_bool)))

    guide_score = 0.0
    if guide_mask is not None:
        local_guide = guide_mask[y1:y2, x1:x2] > 0
        union = np.count_nonzero(fill_bool | local_guide)
        if union > 0:
            guide_score = np.count_nonzero(fill_bool & local_guide) / float(union)

    a, b = axes
    preferred_a, preferred_b = THERMAL_LOOP_PREFERRED_AXES
    aspect = float(a) / max(1.0, float(b))
    preferred_aspect = preferred_a / preferred_b
    aspect_score = max(0.0, 1.0 - abs(aspect - preferred_aspect) / 1.5)
    axis_prior_score = math.exp(
        -0.5 * (((float(a) - preferred_a) / 90.0) ** 2 + ((float(b) - preferred_b) / 55.0) ** 2)
    )
    angle_error_deg = angular_difference_deg(float(angle), THERMAL_LOOP_PREFERRED_ANGLE_DEG)
    angle_prior_score = math.exp(-0.5 * (angle_error_deg / THERMAL_LOOP_ANGLE_PRIOR_SIGMA_DEG) ** 2)

    if minimum_score is not None:
        score_upper_bound = (
            3.4 * support_score +
            1.8 * close_support_ratio +
            1.5 * supported_line_heat +
            0.8 * min(1.0, fill_density * 4.0) +
            0.3 * poly_score +
            0.5 * topness +
            0.35 * aspect_score +
            0.8 * axis_prior_score +
            0.5 * angle_prior_score +
            0.9 +
            0.55 +
            1.4 * guide_score -
            0.04 * mean_distance +
            1e-9
        )
        if score_upper_bound <= float(minimum_score):
            return {"pruned_by_score_bound": True}

    supported_sectors, sector_support_ratios, sector_valid_pixels = ellipse_sector_support(
        sector_ids,
        valid_centerline_values,
        close_support,
    )
    sector_score = supported_sectors / float(THERMAL_LOOP_SECTOR_COUNT)
    offset_contrast = ellipse_offset_contrast(
        local_shape,
        local_ellipse,
        ~local_exclusion,
        local_material,
        offset_px=max(12.0, float(output_thickness)),
    )

    score = (
        3.4 * support_score +
        1.8 * close_support_ratio +
        1.5 * supported_line_heat +
        0.8 * min(1.0, fill_density * 4.0) +
        0.3 * poly_score +
        0.5 * topness +
        0.35 * aspect_score +
        0.8 * axis_prior_score +
        0.5 * angle_prior_score +
        0.9 * sector_score +
        0.55 * offset_contrast +
        1.4 * guide_score -
        0.04 * mean_distance
    )

    accepted = bool(
        score >= THERMAL_LOOP_MIN_ACCEPTED_SCORE
        and supported_sectors >= THERMAL_LOOP_MIN_SUPPORTED_SECTORS
        and visible_ratio >= THERMAL_LOOP_MIN_VISIBLE_RATIO
    )
    return {
        "score": float(score),
        "accepted": accepted,
        "candidate_type": "thermal_polyline_last_loop",
        "centerline_support_score": float(support_score),
        "centerline_close_ratio": float(close_support_ratio),
        "centerline_mean_dist_px": float(mean_distance),
        "ring_coverage": float(close_support_ratio),
        "ring_heat": float(supported_line_heat),
        "fill_density": float(fill_density),
        "center_to_polyline_dist": None if poly_dist is None else float(poly_dist),
        "polyline_score": float(poly_score),
        "topness": float(topness),
        "exclusion_overlap": float(exclusion_overlap),
        "visible_centerline_ratio": float(visible_ratio),
        "supported_sector_count": int(supported_sectors),
        "sector_support_ratios": sector_support_ratios,
        "sector_valid_pixels": sector_valid_pixels,
        "offset_contrast": float(offset_contrast),
        "axis_prior_score": float(axis_prior_score),
        "angle_prior_score": float(angle_prior_score),
        "angle_error_from_prior_deg": float(angle_error_deg),
        "guide_iou": float(guide_score),
        "observed_pixels": int(observed_count),
        "centerline_pixels": int(centerline_count),
        "valid_centerline_pixels": int(valid_centerline_count),
        "ring_pixels": int(np.count_nonzero(output_ring_bool)),
        "output_ring_thickness_px": int(output_thickness),
        "score_sigma_px": float(score_sigma),
    }


def refine_thermal_last_loop_ellipse(best, score_candidate):
    if best is None:
        return None, 0

    refinement_candidates = 0
    for center_dx, center_dy, axis_da, axis_db, angle_delta in THERMAL_LOOP_REFINEMENT_STEPS:
        improved = True
        while improved:
            improved = False
            (cx, cy), (a, b), angle = best["ellipse"]
            neighbors = (
                ((cx - center_dx, cy), (a, b), angle),
                ((cx + center_dx, cy), (a, b), angle),
                ((cx, cy - center_dy), (a, b), angle),
                ((cx, cy + center_dy), (a, b), angle),
                ((cx, cy), (a - axis_da, b), angle),
                ((cx, cy), (a + axis_da, b), angle),
                ((cx, cy), (a, b - axis_db), angle),
                ((cx, cy), (a, b + axis_db), angle),
                ((cx, cy), (a, b), normalize_ellipse_angle_deg(angle - angle_delta)),
                ((cx, cy), (a, b), normalize_ellipse_angle_deg(angle + angle_delta)),
            )
            for ellipse in neighbors:
                if ellipse[1][0] <= 1 or ellipse[1][1] <= 1:
                    continue
                metrics = score_candidate(
                    ellipse, best["metrics"]["score"] + 1e-6
                )
                refinement_candidates += 1
                if (
                    metrics is not None
                    and not metrics.get("pruned_by_score_bound", False)
                    and metrics["score"] > best["metrics"]["score"] + 1e-6
                ):
                    best = {"ellipse": ellipse, "metrics": metrics}
                    improved = True
                    break
    return best, refinement_candidates


def build_last_loop_mask_from_geometry(
    frame,
    segment_xy,
    tail_tip,
    tail_base=None,
    bbox=None,
    settings=None,
    polyline_points=BODY_POLYLINE_POINTS,
    guide_ellipse=None,
):
    if frame is None or tail_tip is None:
        return None, {}

    if segment_xy is not None:
        segment_mask = polygon_to_mask(segment_xy, frame.shape)
    else:
        segment_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    tail_segment_aspect_ratio = 0.0
    tail_upper_span = 0.0
    if segment_xy is not None and len(segment_xy) > 0:
        segment_xs = [float(point[0]) for point in segment_xy]
        segment_ys = [float(point[1]) for point in segment_xy]
        segment_width = max(segment_xs) - min(segment_xs)
        segment_height = max(segment_ys) - min(segment_ys)
        tail_segment_aspect_ratio = segment_width / max(1.0, segment_height)
        tail_upper_span = max(
            0.0,
            float(tail_tip[1]) - min(float(point[1]) for point in segment_xy),
        )
    search_geometry = thermal_loop_search_geometry(tail_upper_span)
    roi_up_from_tail = search_geometry["roi_up_from_tail"]
    center_y_start = search_geometry["center_y_start"]
    axis_y_scale = search_geometry["axis_y_scale"]

    roi_mask, roi_box = build_last_loop_polyline_roi_mask(
        frame.shape,
        tail_tip,
        polyline_points,
        roi_up_from_tail=roi_up_from_tail,
    )
    material_mask, heat, color_roi_mask = build_last_loop_material_mask(
        frame,
        roi_mask,
        segment_mask=segment_mask,
        settings=settings,
    )

    tail_exclusion = build_tail_exclusion_mask(
        frame.shape,
        tail_tip,
        tail_base=tail_base,
        bbox=bbox,
        segment_xy=segment_xy,
        tip_radius=85,
        dilate_ksize=11,
    )
    segment_exclusion = cv2.dilate(segment_mask, np.ones((3, 3), np.uint8), iterations=1)
    exclusion_mask = cv2.bitwise_or(tail_exclusion, segment_exclusion)
    material_mask = material_mask.copy()
    material_mask[exclusion_mask > 0] = 0
    material_distance = cv2.distanceTransform(
        np.where(material_mask > 0, 0, 255).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )

    output_thickness, score_sigma = estimate_material_width_px(material_mask, roi_box)
    angle_candidates, base_angle, angle_source = angle_candidates_from_footage(
        material_mask,
        roi_box,
        polyline_points,
        tail_tip,
    )

    guide_mask = None if guide_ellipse is None else ellipse_params_to_mask(frame.shape, guide_ellipse)
    tail_x, tail_y = float(tail_tip[0]), float(tail_tip[1])
    best = None
    candidates_scored = 0
    candidates_pruned_by_bound = 0

    def score_candidate(ellipse, minimum_score=None):
        nonlocal candidates_pruned_by_bound
        metrics = score_thermal_last_loop_ellipse(
            frame.shape,
            heat,
            material_mask,
            material_distance,
            exclusion_mask,
            ellipse,
            tail_tip,
            output_thickness=output_thickness,
            score_sigma=score_sigma,
            polyline_points=polyline_points,
            guide_mask=guide_mask,
            minimum_score=minimum_score,
        )
        if metrics is not None and metrics.get("pruned_by_score_bound", False):
            candidates_pruned_by_bound += 1
        return metrics

    for dy in range(center_y_start, THERMAL_LOOP_CENTER_Y_STOP + 1, THERMAL_LOOP_CENTER_Y_STEP):
        cy = int(round(tail_y + dy))
        if cy < 0 or cy >= frame.shape[0]:
            continue

        body_x = polyline_x_at_y(polyline_points, cy)
        if body_x is None:
            body_x = tail_x

        axis_candidates = axis_candidates_for_y(
            cy,
            tail_y,
            center_y_start=center_y_start,
            center_y_stop=THERMAL_LOOP_CENTER_Y_STOP,
            axis_y_scale=axis_y_scale,
        )
        for x_offset in THERMAL_LOOP_CENTER_X_OFFSETS:
            cx = int(round(body_x + x_offset))
            if cx < 0 or cx >= frame.shape[1]:
                continue

            for axes in axis_candidates:
                for angle in angle_candidates:
                    ellipse = ((cx, cy), axes, float(angle))
                    minimum_score = None if best is None else best["metrics"]["score"]
                    metrics = score_candidate(ellipse, minimum_score)
                    if metrics is None:
                        continue

                    candidates_scored += 1
                    if metrics.get("pruned_by_score_bound", False):
                        continue
                    if best is None or metrics["score"] > best["metrics"]["score"]:
                        best = {"ellipse": ellipse, "metrics": metrics}

    best, refinement_candidates = refine_thermal_last_loop_ellipse(best, score_candidate)
    candidates_scored += refinement_candidates

    heat_vis = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
    debug = {
        "heat_mask": heat_vis,
        "coil_color_mask": color_roi_mask,
        "geometry_roi_mask": roi_mask,
        "geometry_roi_box": roi_box,
        "material_mask": material_mask,
        "tail_exclusion_mask": tail_exclusion,
        "exclusion_mask": exclusion_mask,
        "selected_loop_centerline_mask": np.zeros(frame.shape[:2], dtype=np.uint8),
        "selected_loop_ring_mask": np.zeros(frame.shape[:2], dtype=np.uint8),
        "selected_loop_fill_mask": np.zeros(frame.shape[:2], dtype=np.uint8),
        "observed_loop_mask": np.zeros(frame.shape[:2], dtype=np.uint8),
        "best_ellipse": None,
        "best_ellipse_info": None,
        "best_metrics": None,
        "candidates_scored": int(candidates_scored),
        "candidates_pruned_by_score_bound": int(candidates_pruned_by_bound),
        "dynamic_ring_thickness_px": int(output_thickness),
        "score_sigma_px": float(score_sigma),
        "angle_base_deg": float(base_angle),
        "angle_source": angle_source,
        "angle_candidates_deg": [float(a) for a in angle_candidates],
        "tail_upper_span_px": float(tail_upper_span),
        "tail_segment_aspect_ratio": float(tail_segment_aspect_ratio),
        "geometry_adaptation_weight": search_geometry["adaptation_weight"],
        "roi_up_from_tail_px": float(roi_up_from_tail),
        "center_y_start_from_tail_px": int(center_y_start),
        "axis_y_scale": float(axis_y_scale),
    }

    if best is None:
        return None, debug

    best["metrics"].update({
        "method": THERMAL_LOOP_METHOD,
        "roi_box": roi_box,
        "candidate_count": int(candidates_scored),
        "dynamic_ring_thickness_px": int(output_thickness),
        "angle_base_deg": float(base_angle),
        "angle_source": angle_source,
        "angle_candidates_deg": [float(a) for a in angle_candidates],
    })

    centerline_mask = ellipse_params_outline_to_mask(
        frame.shape,
        best["ellipse"],
        thickness=THERMAL_LOOP_CENTERLINE_THICKNESS,
    )
    ring_mask = ellipse_params_outline_to_mask(frame.shape, best["ellipse"], thickness=output_thickness)
    fill_mask = ellipse_params_to_mask(frame.shape, best["ellipse"])
    observed_mask = cv2.bitwise_and(ring_mask, material_mask)
    observed_mask[exclusion_mask > 0] = 0
    observed_mask = dilate_mask(observed_mask, ksize=5, iterations=1)
    ellipse_info = ellipse_params_to_info(best["ellipse"])
    ellipse_info["roi_box"] = roi_box

    debug["selected_loop_centerline_mask"] = centerline_mask
    debug["selected_loop_ring_mask"] = ring_mask
    debug["selected_loop_fill_mask"] = fill_mask
    debug["observed_loop_mask"] = observed_mask
    debug["best_ellipse"] = best["ellipse"]
    debug["best_ellipse_info"] = ellipse_info
    debug["best_metrics"] = best["metrics"]
    return observed_mask, debug


def build_last_loop_mask(case_data, settings=None, ellipse_params=None):
    return build_last_loop_mask_from_geometry(
        case_data.get("frame"),
        case_data.get("segment_xy"),
        case_data.get("tail_tip"),
        tail_base=case_data.get("tail_base"),
        bbox=case_data.get("bbox"),
        settings=settings,
        polyline_points=case_data.get("polyline", BODY_POLYLINE_POINTS),
        guide_ellipse=ellipse_params,
    )


def find_last_loop_mask(case_data, settings=None, ellipse_params=None, return_debug=False):
    mask, debug = build_last_loop_mask(case_data, settings=settings, ellipse_params=ellipse_params)
    if return_debug:
        return mask, debug
    return mask


def fit_thermal_last_loop_ellipse(
    frame,
    segment_xy,
    tail_tip,
    tail_base=None,
    bbox=None,
    settings=None,
    polyline_points=BODY_POLYLINE_POINTS,
):
    candidate_mask, debug = build_last_loop_mask_from_geometry(
        frame,
        segment_xy,
        tail_tip,
        tail_base=tail_base,
        bbox=bbox,
        settings=settings,
        polyline_points=polyline_points,
    )
    roi_box = debug.get("geometry_roi_box") if debug else None
    ellipse_info = debug.get("best_ellipse_info") if debug else None
    metrics = debug.get("best_metrics") if debug else None
    return ellipse_info, candidate_mask, roi_box, metrics, debug

def crop_guided_loop_roi_from_segment(mask, tail_tip, polyline_points):
    tail_x, tail_y = tail_tip
    h, w = mask.shape[:2]

    body_x = polyline_x_at_y(polyline_points, tail_y)
    if body_x is None:
        body_x = tail_x

    x_center = int(round((tail_x + body_x) / 2.0))

    x1 = max(0, x_center - ELLIPSE_SEARCH_PAD_X)
    x2 = min(w, x_center + ELLIPSE_SEARCH_PAD_X)

    y1 = max(0, int(round(tail_y)) - ELLIPSE_SEARCH_PAD_Y_UP)
    y2 = min(h, int(round(tail_y)) + ELLIPSE_SEARCH_PAD_Y_DOWN)

    return mask[y1:y2, x1:x2], (x1, y1, x2, y2)


def merge_local_candidate_mask(local_mask, ksize=CONTOUR_MERGE_KERNEL, iterations=CONTOUR_MERGE_ITERATIONS):
    if local_mask is None:
        return None

    kernel = np.ones((ksize, ksize), np.uint8)
    merged = cv2.dilate(local_mask, kernel, iterations=iterations)
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel, iterations=1)
    return merged


def accumulate_loop_candidate_masks(
    frames_with_indices,
    segment_xy,
    tail_tip,
    tail_base,
    bbox,
    roi_box,
    remove_segment_from_edges=True,
    min_votes=LOOP_ACCUMULATION_MIN_VOTES,
):
    if not frames_with_indices:
        return None, {
            "accumulated_mask": None,
            "accumulated_local_mask": None,
            "per_frame_local_masks": [],
            "frame_indices": [],
            "vote_count_mask": None,
        }

    x1, y1, x2, y2 = roi_box
    roi_h = max(0, y2 - y1)
    roi_w = max(0, x2 - x1)

    vote_accum = np.zeros((roi_h, roi_w), dtype=np.uint16)
    per_frame_local_masks = []
    used_indices = []

    last_debug = None
    frame_shape = None

    for frame_idx, frame in frames_with_indices:
        candidate_mask, debug_masks = build_loop_candidate_mask(
            frame,
            segment_xy,
            tail_tip,
            tail_base=tail_base,
            bbox=bbox,
            remove_segment_fill=remove_segment_from_edges,
        )

        frame_shape = candidate_mask.shape
        local_mask = candidate_mask[y1:y2, x1:x2]
        if local_mask.shape[:2] != (roi_h, roi_w):
            padded = np.zeros((roi_h, roi_w), dtype=np.uint8)
            hh = min(roi_h, local_mask.shape[0])
            ww = min(roi_w, local_mask.shape[1])
            padded[:hh, :ww] = local_mask[:hh, :ww]
            local_mask = padded

        vote_accum += (local_mask > 0).astype(np.uint16)
        per_frame_local_masks.append(local_mask.copy())
        used_indices.append(int(frame_idx))
        last_debug = debug_masks

    accumulated_local_mask = np.where(vote_accum >= int(min_votes), 255, 0).astype(np.uint8)

    if last_debug is None:
        last_debug = {}

    if frame_shape is None:
        frame_shape = (1, 1)

    accumulated_full_mask = np.zeros(frame_shape, dtype=np.uint8)
    accumulated_full_mask[y1:y2, x1:x2] = accumulated_local_mask

    return accumulated_full_mask, {
        "accumulated_mask": accumulated_full_mask,
        "accumulated_local_mask": accumulated_local_mask,
        "per_frame_local_masks": per_frame_local_masks,
        "frame_indices": used_indices,
        "vote_count_mask": vote_accum,
        "last_debug_masks": last_debug,
    }


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
        "axes": (float(a), float(b)),
        "rotation_deg": float(rot),
        "method": "direct_observed_ellipse",
    }


def estimate_loop_ellipse_from_contour(contour, roi_offset, method_name="last_loop_direct_hole_ellipse"):
    ellipse_info = fit_ellipse_from_contour(contour, roi_offset)
    if ellipse_info is not None:
        ellipse_info["method"] = method_name
        return ellipse_info

    if contour is None or len(contour) < 3:
        return None

    (cx, cy), (w, h), rotation_deg = cv2.minAreaRect(contour)
    if w <= 1e-6 or h <= 1e-6:
        return None

    a = float(max(w, h) * 0.5)
    b = float(min(w, h) * 0.5)
    rot = float(rotation_deg)
    if h > w:
        rot = (rot + 90.0) % 180.0

    rx1, ry1 = roi_offset
    return {
        "center": (float(cx + rx1), float(cy + ry1)),
        "axes": (a, b),
        "rotation_deg": rot,
        "method": method_name,
    }


def component_touches_border(x, y, w, h, image_shape, margin=0):
    hh, ww = image_shape[:2]
    return (
        x <= margin or
        y <= margin or
        (x + w) >= (ww - margin) or
        (y + h) >= (hh - margin)
    )


def score_last_loop_hole_candidate(component_mask, contour, roi_box, tail_tip, polyline_points):
    if component_mask is None or contour is None or len(contour) < 3:
        return None, None

    rx1, ry1, _, _ = roi_box
    x, y, w, h = cv2.boundingRect(contour)
    area = float(np.count_nonzero(component_mask))
    if area < LAST_LOOP_MIN_HOLE_AREA:
        return None, None
    if w < LAST_LOOP_MIN_HOLE_SPAN_X or h < LAST_LOOP_MIN_HOLE_SPAN_Y:
        return None, None
    if component_touches_border(x, y, w, h, component_mask.shape, margin=1):
        return None, None

    ellipse_info = estimate_loop_ellipse_from_contour(contour, (rx1, ry1))
    if ellipse_info is None:
        return None, None

    moments = cv2.moments(component_mask, binaryImage=True)
    if moments["m00"] <= 1e-6:
        return None, None

    center_local = (
        float(moments["m10"] / moments["m00"]),
        float(moments["m01"] / moments["m00"]),
    )
    center = (center_local[0] + rx1, center_local[1] + ry1)
    ellipse_info["center"] = center

    tail_x, tail_y = float(tail_tip[0]), float(tail_tip[1])
    center_to_tail = float(np.hypot(center[0] - tail_x, center[1] - tail_y))
    if center_to_tail > LAST_LOOP_MAX_CENTER_TO_TAIL_DIST:
        return None, None
    if center[0] < tail_x - LAST_LOOP_MAX_CENTER_LEFT_OF_TAIL:
        return None, None

    center_to_poly = point_to_polyline_distance(center[0], center[1], polyline_points)
    if center_to_poly is None or center_to_poly > LAST_LOOP_MAX_CENTER_TO_POLYLINE_DIST:
        return None, None

    contour_points = contour_points_full(contour, (rx1, ry1))
    if contour_points is None or len(contour_points) == 0:
        return None, None

    deltas = contour_points - np.array([[tail_x, tail_y]], dtype=np.float32)
    dist2 = np.sum(deltas * deltas, axis=1)
    nearest_idx = int(np.argmin(dist2))
    nearest_pt = contour_points[nearest_idx]
    boundary_to_tail = float(np.sqrt(max(0.0, float(dist2[nearest_idx]))))
    if boundary_to_tail > LAST_LOOP_MAX_BOUNDARY_TO_TAIL_DIST:
        return None, None

    leftmost_idx = int(np.argmin(contour_points[:, 0]))
    leftmost_pt = contour_points[leftmost_idx]
    left_gap = float(max(0.0, float(leftmost_pt[0]) - tail_x))
    left_undershoot = float(max(0.0, tail_x - float(leftmost_pt[0])))
    if left_undershoot > LAST_LOOP_MAX_LEFT_EDGE_LEFT_OF_TAIL:
        return None, None

    nearest_y_gap = float(abs(float(nearest_pt[1]) - tail_y))
    tail_to_boundary_ellipse = ellipse_boundary_distance((tail_x, tail_y), ellipse_info)

    score = (
        1.5 * boundary_to_tail +
        0.20 * center_to_tail +
        0.9 * center_to_poly +
        4.0 * left_gap +
        3.5 * left_undershoot +
        0.45 * nearest_y_gap -
        0.0015 * area
    )

    metrics = {
        "score": float(score),
        "accepted": True,
        "inner_hole_area_px": float(area),
        "inner_hole_bbox": [int(x + rx1), int(y + ry1), int(x + rx1 + w), int(y + ry1 + h)],
        "center_to_tail_dist": float(center_to_tail),
        "center_to_polyline_dist": float(center_to_poly),
        "boundary_to_tail_dist": float(boundary_to_tail),
        "left_gap_from_tail_px": float(left_gap),
        "left_edge_undershoot_px": float(left_undershoot),
        "nearest_boundary_y_gap_px": float(nearest_y_gap),
        "tail_to_ellipse_boundary_dist": None if tail_to_boundary_ellipse is None else float(tail_to_boundary_ellipse),
        "candidate_type": "last_loop_hole",
    }
    return ellipse_info, metrics


def evaluate_last_loop_hole_candidates(merged_local_mask, roi_box, tail_tip, polyline_points):
    if merged_local_mask is None or merged_local_mask.size == 0:
        return {
            "all_contours_full": [],
            "kept_contours_full": [],
            "direct_candidates": [],
            "best_candidate": None,
        }

    rx1, ry1, _, _ = roi_box
    binary = np.where(merged_local_mask > 0, 255, 0).astype(np.uint8)
    inverse = cv2.bitwise_not(binary)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=8)

    evaluation = {
        "all_contours_full": [],
        "kept_contours_full": [],
        "direct_candidates": [],
        "best_candidate": None,
    }

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if component_touches_border(int(x), int(y), int(w), int(h), inverse.shape, margin=1):
            continue

        component_mask = np.where(labels == label, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        cnt_full = contour.copy().astype(np.int32)
        cnt_full[:, 0, 0] += int(rx1)
        cnt_full[:, 0, 1] += int(ry1)
        evaluation["all_contours_full"].append(cnt_full)

        ellipse_info, metrics = score_last_loop_hole_candidate(
            component_mask=component_mask,
            contour=contour,
            roi_box=roi_box,
            tail_tip=tail_tip,
            polyline_points=polyline_points,
        )
        if ellipse_info is None or metrics is None:
            continue

        candidate_record = {
            "ellipse_info": {
                "center": tuple(ellipse_info["center"]),
                "axes": tuple(ellipse_info["axes"]),
                "rotation_deg": float(ellipse_info["rotation_deg"]),
                "method": ellipse_info.get("method"),
            },
            "metrics": dict(metrics),
            "contour_bbox": metrics["inner_hole_bbox"],
            "contour_point_count": int(len(contour)),
            "contour_span_x": int(w),
            "contour_span_y": int(h),
            "contour_full": cnt_full,
        }

        evaluation["kept_contours_full"].append(cnt_full)
        evaluation["direct_candidates"].append(candidate_record)

        candidate = {
            "ellipse_info": ellipse_info,
            "metrics": metrics,
        }
        if evaluation["best_candidate"] is None or metrics["score"] < evaluation["best_candidate"]["metrics"]["score"]:
            evaluation["best_candidate"] = candidate

    return evaluation


def score_last_loop_arc_candidate(contour, roi_box, tail_tip, polyline_points):
    if contour is None or len(contour) < max(30, MIN_CONTOUR_POINTS_FOR_ELLIPSE):
        return None, None

    rx1, ry1, _, _ = roi_box
    x, y, w, h = cv2.boundingRect(contour)
    if w < LAST_LOOP_ARC_MIN_SPAN_X or h < LAST_LOOP_ARC_MIN_SPAN_Y or h > LAST_LOOP_ARC_MAX_SPAN_Y:
        return None, None

    ellipse_info = fit_ellipse_from_contour(contour, (rx1, ry1))
    if ellipse_info is None:
        return None, None

    contour_points = contour_points_full(contour, (rx1, ry1))
    if contour_points is None or len(contour_points) == 0:
        return None, None

    tail_x, tail_y = float(tail_tip[0]), float(tail_tip[1])
    deltas = contour_points - np.array([[tail_x, tail_y]], dtype=np.float32)
    dist2 = np.sum(deltas * deltas, axis=1)
    nearest_idx = int(np.argmin(dist2))
    nearest_pt = contour_points[nearest_idx]
    boundary_to_tail = float(np.sqrt(max(0.0, float(dist2[nearest_idx]))))
    if boundary_to_tail > LAST_LOOP_ARC_MAX_NEAREST_TAIL_DIST:
        return None, None

    center = ellipse_info["center"]
    center_to_tail = float(np.hypot(center[0] - tail_x, center[1] - tail_y))
    center_to_poly = point_to_polyline_distance(center[0], center[1], polyline_points)
    if center_to_poly is None or center_to_poly > LAST_LOOP_ARC_MAX_CENTER_TO_POLYLINE_DIST:
        return None, None

    leftmost_x = float(np.min(contour_points[:, 0]))
    left_gap = float(max(0.0, leftmost_x - tail_x))
    left_undershoot = float(max(0.0, tail_x - leftmost_x))
    if left_undershoot > LAST_LOOP_MAX_LEFT_EDGE_LEFT_OF_TAIL:
        return None, None

    nearest_y_gap = float(abs(float(nearest_pt[1]) - tail_y))
    right_span = float(np.max(contour_points[:, 0]) - leftmost_x)
    score = (
        5.0 * boundary_to_tail +
        0.25 * center_to_tail +
        1.2 * center_to_poly +
        1.3 * left_gap +
        3.0 * left_undershoot +
        0.8 * nearest_y_gap +
        0.15 * h +
        0.03 * max(0.0, right_span - 500.0)
    )

    metrics = {
        "score": float(score),
        "accepted": True,
        "boundary_to_tail_dist": float(boundary_to_tail),
        "center_to_tail_dist": float(center_to_tail),
        "center_to_polyline_dist": float(center_to_poly),
        "left_gap_from_tail_px": float(left_gap),
        "left_edge_undershoot_px": float(left_undershoot),
        "nearest_boundary_y_gap_px": float(nearest_y_gap),
        "contour_span_x": int(w),
        "contour_span_y": int(h),
        "candidate_type": "last_loop_arc",
    }
    ellipse_info["method"] = "last_loop_direct_arc_ellipse"
    return ellipse_info, metrics


def evaluate_last_loop_arc_candidates(merged_local_mask, roi_box, tail_tip, polyline_points):
    if merged_local_mask is None or merged_local_mask.size == 0:
        return {
            "all_contours_full": [],
            "kept_contours_full": [],
            "direct_candidates": [],
            "best_candidate": None,
        }

    rx1, ry1, _, _ = roi_box
    contours, _ = cv2.findContours(merged_local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    evaluation = {
        "all_contours_full": [],
        "kept_contours_full": [],
        "direct_candidates": [],
        "best_candidate": None,
    }

    for contour in contours:
        cnt_full = contour.copy().astype(np.int32)
        cnt_full[:, 0, 0] += int(rx1)
        cnt_full[:, 0, 1] += int(ry1)
        evaluation["all_contours_full"].append(cnt_full)

        ellipse_info, metrics = score_last_loop_arc_candidate(
            contour=contour,
            roi_box=roi_box,
            tail_tip=tail_tip,
            polyline_points=polyline_points,
        )
        if ellipse_info is None or metrics is None:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        candidate_record = {
            "ellipse_info": {
                "center": tuple(ellipse_info["center"]),
                "axes": tuple(ellipse_info["axes"]),
                "rotation_deg": float(ellipse_info["rotation_deg"]),
                "method": ellipse_info.get("method"),
            },
            "metrics": dict(metrics),
            "contour_bbox": [int(x + rx1), int(y + ry1), int(x + rx1 + w), int(y + ry1 + h)],
            "contour_point_count": int(len(contour)),
            "contour_span_x": int(w),
            "contour_span_y": int(h),
            "contour_full": cnt_full,
        }
        evaluation["kept_contours_full"].append(cnt_full)
        evaluation["direct_candidates"].append(candidate_record)

        candidate = {
            "ellipse_info": ellipse_info,
            "metrics": metrics,
        }
        if evaluation["best_candidate"] is None or metrics["score"] < evaluation["best_candidate"]["metrics"]["score"]:
            evaluation["best_candidate"] = candidate

    return evaluation


def evaluate_boundary_candidate_mask(boundary_candidate_local_mask, roi_box):
    if boundary_candidate_local_mask is None or boundary_candidate_local_mask.size == 0:
        return {
            "all_contours_full": [],
            "kept_contours_full": [],
            "direct_candidates": [],
            "best_candidate": None,
        }

    rx1, ry1, _, _ = roi_box
    contours, _ = cv2.findContours(boundary_candidate_local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    evaluation = {
        "all_contours_full": [],
        "kept_contours_full": [],
        "direct_candidates": [],
        "best_candidate": None,
    }

    for contour in contours:
        cnt_full = contour.copy().astype(np.int32)
        cnt_full[:, 0, 0] += int(rx1)
        cnt_full[:, 0, 1] += int(ry1)
        evaluation["all_contours_full"].append(cnt_full)

        if len(contour) < 5:
            continue

        ellipse_info = estimate_loop_ellipse_from_contour(
            contour,
            (rx1, ry1),
            method_name="coil_boundary_mask_ellipse",
        )
        if ellipse_info is None:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        metrics = {
            "score": float(-(w * h)),
            "accepted": True,
            "candidate_type": "coil_boundary_mask",
            "contour_span_x": int(w),
            "contour_span_y": int(h),
            "contour_bbox": [int(x + rx1), int(y + ry1), int(x + rx1 + w), int(y + ry1 + h)],
        }

        candidate_record = {
            "ellipse_info": {
                "center": tuple(ellipse_info["center"]),
                "axes": tuple(ellipse_info["axes"]),
                "rotation_deg": float(ellipse_info["rotation_deg"]),
                "method": ellipse_info.get("method"),
            },
            "metrics": dict(metrics),
            "contour_bbox": metrics["contour_bbox"],
            "contour_point_count": int(len(contour)),
            "contour_span_x": int(w),
            "contour_span_y": int(h),
            "contour_full": cnt_full,
        }
        evaluation["kept_contours_full"].append(cnt_full)
        evaluation["direct_candidates"].append(candidate_record)

        candidate = {
            "ellipse_info": ellipse_info,
            "metrics": metrics,
        }
        if evaluation["best_candidate"] is None or metrics["score"] < evaluation["best_candidate"]["metrics"]["score"]:
            evaluation["best_candidate"] = candidate

    return evaluation


def score_last_loop_opening_band_candidate(component_mask, contour, roi_box, tail_tip, polyline_points):
    if component_mask is None or contour is None or len(contour) < 3:
        return None, None

    rx1, ry1, _, _ = roi_box
    x, y, w, h = cv2.boundingRect(contour)
    area = float(np.count_nonzero(component_mask))
    if area < LAST_LOOP_OPENING_MIN_AREA:
        return None, None
    if w < LAST_LOOP_OPENING_MIN_SPAN_X or h < LAST_LOOP_OPENING_MIN_SPAN_Y:
        return None, None

    hull = cv2.convexHull(contour)
    ellipse_info = estimate_loop_ellipse_from_contour(hull, (rx1, ry1), method_name="last_loop_opening_band_ellipse")
    if ellipse_info is None:
        return None, None

    moments = cv2.moments(component_mask, binaryImage=True)
    if moments["m00"] <= 1e-6:
        return None, None

    center_local = (
        float(moments["m10"] / moments["m00"]),
        float(moments["m01"] / moments["m00"]),
    )
    center = (center_local[0] + rx1, center_local[1] + ry1)
    ellipse_info["center"] = center

    tail_x, tail_y = float(tail_tip[0]), float(tail_tip[1])
    center_to_tail = float(np.hypot(center[0] - tail_x, center[1] - tail_y))
    center_to_poly = point_to_polyline_distance(center[0], center[1], polyline_points)
    if center_to_poly is None or center_to_poly > LAST_LOOP_OPENING_MAX_CENTER_TO_POLYLINE_DIST:
        return None, None

    contour_points = contour_points_full(hull, (rx1, ry1))
    if contour_points is None or len(contour_points) == 0:
        return None, None

    deltas = contour_points - np.array([[tail_x, tail_y]], dtype=np.float32)
    dist2 = np.sum(deltas * deltas, axis=1)
    nearest_idx = int(np.argmin(dist2))
    nearest_pt = contour_points[nearest_idx]
    boundary_to_tail = float(np.sqrt(max(0.0, float(dist2[nearest_idx]))))
    if boundary_to_tail > LAST_LOOP_OPENING_MAX_BOUNDARY_TO_TAIL_DIST:
        return None, None

    top_y = float(np.min(contour_points[:, 1]))
    vertical_bias = float(max(0.0, center[1] - tail_y))
    top_clearance = float(max(0.0, tail_y - top_y))

    score = (
        2.5 * boundary_to_tail +
        0.85 * center_to_poly +
        0.10 * center_to_tail +
        1.8 * vertical_bias -
        0.035 * float(w) -
        0.010 * float(h) -
        0.0010 * float(area) -
        0.08 * top_clearance
    )

    metrics = {
        "score": float(score),
        "accepted": True,
        "opening_band_area_px": float(area),
        "opening_band_bbox": [int(x + rx1), int(y + ry1), int(x + rx1 + w), int(y + ry1 + h)],
        "center_to_tail_dist": float(center_to_tail),
        "center_to_polyline_dist": float(center_to_poly),
        "boundary_to_tail_dist": float(boundary_to_tail),
        "vertical_bias_px": float(vertical_bias),
        "top_clearance_px": float(top_clearance),
        "candidate_type": "last_loop_opening_band",
    }
    return ellipse_info, metrics


def evaluate_last_loop_opening_band_candidates(opening_band_local_mask, roi_box, tail_tip, polyline_points):
    if opening_band_local_mask is None or opening_band_local_mask.size == 0:
        return {
            "all_contours_full": [],
            "kept_contours_full": [],
            "direct_candidates": [],
            "best_candidate": None,
        }

    rx1, ry1, _, _ = roi_box
    contours, _ = cv2.findContours(opening_band_local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    evaluation = {
        "all_contours_full": [],
        "kept_contours_full": [],
        "direct_candidates": [],
        "best_candidate": None,
    }

    for contour in contours:
        cnt_full = contour.copy().astype(np.int32)
        cnt_full[:, 0, 0] += int(rx1)
        cnt_full[:, 0, 1] += int(ry1)
        evaluation["all_contours_full"].append(cnt_full)

        component_mask = np.zeros_like(opening_band_local_mask)
        cv2.drawContours(component_mask, [contour], -1, 255, thickness=-1)

        ellipse_info, metrics = score_last_loop_opening_band_candidate(
            component_mask=component_mask,
            contour=contour,
            roi_box=roi_box,
            tail_tip=tail_tip,
            polyline_points=polyline_points,
        )
        if ellipse_info is None or metrics is None:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        candidate_record = {
            "ellipse_info": {
                "center": tuple(ellipse_info["center"]),
                "axes": tuple(ellipse_info["axes"]),
                "rotation_deg": float(ellipse_info["rotation_deg"]),
                "method": ellipse_info.get("method"),
            },
            "metrics": dict(metrics),
            "contour_bbox": [int(x + rx1), int(y + ry1), int(x + rx1 + w), int(y + ry1 + h)],
            "contour_point_count": int(len(contour)),
            "contour_span_x": int(w),
            "contour_span_y": int(h),
            "contour_full": cnt_full,
        }

        evaluation["kept_contours_full"].append(cnt_full)
        evaluation["direct_candidates"].append(candidate_record)

        candidate = {
            "ellipse_info": ellipse_info,
            "metrics": metrics,
        }
        if evaluation["best_candidate"] is None or metrics["score"] < evaluation["best_candidate"]["metrics"]["score"]:
            evaluation["best_candidate"] = candidate

    return evaluation


def ellipse_local_coordinates(point, ellipse_info):
    px, py = point
    cx, cy = ellipse_info["center"]
    a, b = ellipse_info["axes"]
    rot_deg = ellipse_info["rotation_deg"]

    if a <= 1e-6 or b <= 1e-6:
        return None

    theta = math.radians(rot_deg)
    dx = px - cx
    dy = py - cy

    xr = dx * math.cos(theta) + dy * math.sin(theta)
    yr = -dx * math.sin(theta) + dy * math.cos(theta)
    return xr, yr


def ellipse_normalized_radius(point, ellipse_info):
    local = ellipse_local_coordinates(point, ellipse_info)
    if local is None:
        return None

    xr, yr = local
    a, b = ellipse_info["axes"]
    return float(math.sqrt((xr * xr) / (a * a) + (yr * yr) / (b * b)))


def ellipse_boundary_distance(point, ellipse_info):
    rho = ellipse_normalized_radius(point, ellipse_info)
    if rho is None:
        return None
    a, b = ellipse_info["axes"]
    return float(abs(rho - 1.0) * max(a, b))


def ellipse_parameter_angle_deg(point, ellipse_info):
    local = ellipse_local_coordinates(point, ellipse_info)
    if local is None:
        return None
    xr, yr = local
    a, b = ellipse_info["axes"]
    t = math.atan2(yr / b, xr / a)
    deg = math.degrees(t)
    if deg < 0:
        deg += 360.0
    return float(deg)


def ellipse_horizontal_extent(ellipse_info):
    """Return the ellipse half-width after rotation in image coordinates."""
    a, b = ellipse_info["axes"]
    theta = math.radians(ellipse_info["rotation_deg"])
    return float(math.sqrt((a * math.cos(theta)) ** 2 + (b * math.sin(theta)) ** 2))


def shape_prior_horizontal_extent(shape_prior=None):
    prior = ELLIPSE_SHAPE_PRIOR if shape_prior is None else shape_prior
    ellipse_info = {
        "center": (0.0, 0.0),
        "axes": (float(prior["a"]), float(prior["b"])),
        "rotation_deg": float(prior["rotation_deg"]),
    }
    return ellipse_horizontal_extent(ellipse_info)


def contour_angle_coverage_deg(contour_points, ellipse_info):
    if contour_points is None or len(contour_points) < 2:
        return 0.0

    angles = []
    for x, y in contour_points:
        angle = ellipse_parameter_angle_deg((float(x), float(y)), ellipse_info)
        if angle is not None:
            angles.append(angle)

    if len(angles) < 2:
        return 0.0

    angles = sorted(angles)
    gaps = []
    for i in range(len(angles) - 1):
        gaps.append(angles[i + 1] - angles[i])
    gaps.append((angles[0] + 360.0) - angles[-1])

    return float(360.0 - max(gaps))


def ellipse_point_at_angle(ellipse_info, angle_deg):
    cx, cy = ellipse_info["center"]
    a, b = ellipse_info["axes"]
    rot_deg = ellipse_info["rotation_deg"]

    t = math.radians(angle_deg)
    theta = math.radians(rot_deg)
    x_local = a * math.cos(t)
    y_local = b * math.sin(t)

    x = cx + x_local * math.cos(theta) - y_local * math.sin(theta)
    y = cy + x_local * math.sin(theta) + y_local * math.cos(theta)
    return float(x), float(y)


def ellipse_polyline_crossing_anchors(ellipse_points, polyline_points, center_y):
    candidates = []
    for point in ellipse_points:
        dist = point_to_polyline_distance(point[0], point[1], polyline_points)
        if dist is not None:
            candidates.append((dist, point))

    if not candidates:
        return None, None, None, None

    upper = [(dist, point) for dist, point in candidates if point[1] <= center_y]
    lower = [(dist, point) for dist, point in candidates if point[1] > center_y]

    if not upper or not lower:
        return None, None, None, None

    top_dist, top = min(upper, key=lambda item: item[0])
    bottom_dist, bottom = min(lower, key=lambda item: item[0])
    return top, bottom, top_dist, bottom_dist


def ellipse_anchor_points(ellipse_info, polyline_points=None, step_deg=2):
    points = [ellipse_point_at_angle(ellipse_info, angle) for angle in range(0, 360, step_deg)]

    top = None
    bottom = None
    top_poly_dist = None
    bottom_poly_dist = None

    if polyline_points is not None:
        _, center_y = ellipse_info["center"]
        top, bottom, top_poly_dist, bottom_poly_dist = ellipse_polyline_crossing_anchors(
            points,
            polyline_points,
            center_y,
        )

    if top is None or bottom is None:
        top = min(points, key=lambda p: p[1])
        bottom = max(points, key=lambda p: p[1])
        if polyline_points is not None:
            top_poly_dist = point_to_polyline_distance(top[0], top[1], polyline_points)
            bottom_poly_dist = point_to_polyline_distance(bottom[0], bottom[1], polyline_points)

    left = min(points, key=lambda p: p[0])
    right = max(points, key=lambda p: p[0])

    return {
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "top_polyline_dist": top_poly_dist,
        "bottom_polyline_dist": bottom_poly_dist,
    }


def compute_ellipse_polyline_distance(
    ellipse_info,
    segment_xy,
    image_shape,
    polyline_points,
    tail_hint=None,
    samples=180,
    max_search=1200,
):
    """
    Find the ellipse boundary point nearest the supplied polyline and measure
    the distance from that ellipse point to the provided segment (tail body)
    along the local polyline direction.

    Returns a dict with keys:
      - ellipse_point: (x,y) on ellipse nearest the polyline
      - polyline_distance_px: distance from that ellipse point to the polyline
      - distance_along_polyline_px: distance from the ellipse point along the
        polyline direction to the first pixel of the supplied segment mask
        (None if no intersection within `max_search`)
      - intersection_point: (x,y) on the segment where the line hits (or None)
      - euclidean_distance_px: straight-line distance to the nearest segment pixel
    """
    if ellipse_info is None or segment_xy is None or image_shape is None:
        return None

    # 1) Find ellipse point closest to the polyline by sampling angles
    # Prefer the upper half (top) of the ellipse relative to its center.
    best_pt = None
    best_poly_dist = float("inf")
    cx, cy = ellipse_info["center"]
    # first pass: only consider top half (y <= center_y)
    for angle in np.linspace(0.0, 360.0, samples, endpoint=False):
        pt = ellipse_point_at_angle(ellipse_info, angle)
        if pt is None:
            continue
        # prefer top hemisphere
        if pt[1] > cy:
            continue
        d_poly = point_to_polyline_distance(pt[0], pt[1], polyline_points)
        if d_poly is None:
            continue
        if d_poly < best_poly_dist:
            best_poly_dist = d_poly
            best_pt = pt

    # fallback: if no top-half candidate, use full circumference
    if best_pt is None:
        for angle in np.linspace(0.0, 360.0, samples, endpoint=False):
            pt = ellipse_point_at_angle(ellipse_info, angle)
            if pt is None:
                continue
            d_poly = point_to_polyline_distance(pt[0], pt[1], polyline_points)
            if d_poly is None:
                continue
            if d_poly < best_poly_dist:
                best_poly_dist = d_poly
                best_pt = pt

    if best_pt is None:
        return None

    ellipse_x, ellipse_y = float(best_pt[0]), float(best_pt[1])

    # 2) Build a mask for the provided segment polygon
    seg_mask = polygon_to_mask(segment_xy, image_shape)

    # 3) Compute nearest Euclidean distance to the segment (for fallback / info)
    ys, xs = np.where(seg_mask > 0)
    euclid_dist = None
    euclidean_segment_point = None
    if len(xs) > 0:
        dists = np.hypot(xs - ellipse_x, ys - ellipse_y)
        nearest_index = int(np.argmin(dists))
        euclid_dist = float(dists[nearest_index])
        euclidean_segment_point = (float(xs[nearest_index]), float(ys[nearest_index]))

    # 4) Estimate local polyline tangent (direction vector)
    pts = get_sorted_polyline(polyline_points)
    # find segment that spans ellipse_y
    dir_vec = None
    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        if (y1 <= ellipse_y <= y2) or (y2 <= ellipse_y <= y1):
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            norm = math.hypot(dx, dy)
            if norm > 1e-6:
                dir_vec = (dx / norm, dy / norm)
            break
    if dir_vec is None:
        # fallback to overall polyline direction
        x1, y1 = pts[0]
        x2, y2 = pts[-1]
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        norm = math.hypot(dx, dy)
        if norm > 1e-6:
            dir_vec = (dx / norm, dy / norm)
        else:
            dir_vec = (1.0, 0.0)

    ux, uy = dir_vec

    # 5) Orient direction toward the segment (so search moves from ellipse -> tail)
    cent = segment_centroid(segment_xy)
    if cent is not None:
        vx = cent[0] - ellipse_x
        vy = cent[1] - ellipse_y
        if (vx * ux + vy * uy) < 0:
            ux, uy = -ux, -uy

    # 6) March from ellipse point along (ux,uy) until we hit the segment mask
    h, w = image_shape[:2]
    intersection = None
    distance_along = None
    for step in range(0, int(max_search) + 1):
        sx = int(round(ellipse_x + ux * step))
        sy = int(round(ellipse_y + uy * step))
        if sx < 0 or sx >= w or sy < 0 or sy >= h:
            break
        if seg_mask[sy, sx] > 0:
            intersection = (float(sx), float(sy))
            distance_along = float(step)
            break

    return {
        "ellipse_point": (float(ellipse_x), float(ellipse_y)),
        "polyline_distance_px": float(best_poly_dist),
        "distance_along_polyline_px": distance_along,
        "distance_value_px": euclid_dist,
        "distance_method": "euclidean",
        "euclidean_segment_point": euclidean_segment_point,
        "intersection_point": intersection,
        "euclidean_distance_px": euclid_dist,
    }


def mask_distance_transform(mask):
    if mask is None or mask.size == 0:
        return None

    mask_u8 = mask_to_vis(mask)
    if mask_u8 is None:
        return None

    binary = np.where(mask_u8 > 0, 255, 0).astype(np.uint8)
    inverse = np.where(binary > 0, 0, 255).astype(np.uint8)
    return cv2.distanceTransform(inverse, cv2.DIST_L2, 3)


def distance_to_mask(point, dist_transform):
    if dist_transform is None:
        return None

    x, y = point
    h, w = dist_transform.shape[:2]
    xi = int(round(x))
    yi = int(round(y))

    if xi < 0 or yi < 0 or xi >= w or yi >= h:
        return None

    return float(dist_transform[yi, xi])


def angular_difference_deg(a_deg, b_deg):
    diff = abs(a_deg - b_deg) % 360.0
    return min(diff, 360.0 - diff)


def rasterize_ellipse_mask(image_shape, ellipse_info, thickness=ELLIPSE_OVERLAP_THICKNESS):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    cx, cy = ellipse_info["center"]
    a, b = ellipse_info["axes"]
    rot = ellipse_info["rotation_deg"]

    center_i = (int(round(cx)), int(round(cy)))
    axes_i = (max(1, int(round(a))), max(1, int(round(b))))

    cv2.ellipse(mask, center_i, axes_i, rot, 0, 360, 255, thickness)
    return mask


def compute_ellipse_overlap_ratio(frame_shape, ellipse_info, coil_mask):
    ellipse_mask = rasterize_ellipse_mask(frame_shape, ellipse_info)

    ellipse_pixels = np.count_nonzero(ellipse_mask)
    if ellipse_pixels == 0:
        return 0.0

    overlap = np.count_nonzero((ellipse_mask > 0) & (coil_mask > 0))
    return float(overlap / ellipse_pixels)


def center_inside_roi_with_margin(center, roi_box, margin=ELLIPSE_CENTER_ROI_MARGIN):
    cx, cy = center
    x1, y1, x2, y2 = roi_box
    return (
        x1 - margin <= cx <= x2 + margin and
        y1 - margin <= cy <= y2 + margin
    )


def contour_points_full(contour, roi_offset):
    rx1, ry1 = roi_offset
    pts = contour.reshape(-1, 2).astype(np.float32)
    pts[:, 0] += rx1
    pts[:, 1] += ry1
    return pts


def contour_support_ratio_on_ellipse(contour_points, ellipse_info, tolerance=0.16):
    if contour_points is None or len(contour_points) == 0:
        return 0.0

    rhos = [ellipse_normalized_radius((float(x), float(y)), ellipse_info) for x, y in contour_points]
    rhos = [r for r in rhos if r is not None]
    if not rhos:
        return 0.0

    return float(np.mean([abs(r - 1.0) <= tolerance for r in rhos]))


def contour_tail_side_support_ratio(contour_points, ellipse_info, tail_point, angle_window_deg=40.0, tolerance=0.18):
    if contour_points is None or len(contour_points) == 0:
        return 0.0

    tail_angle = ellipse_parameter_angle_deg(tail_point, ellipse_info)
    if tail_angle is None:
        return 0.0

    support_hits = 0
    considered = 0
    for x, y in contour_points:
        pt = (float(x), float(y))
        angle = ellipse_parameter_angle_deg(pt, ellipse_info)
        rho = ellipse_normalized_radius(pt, ellipse_info)
        if angle is None or rho is None:
            continue
        if angular_difference_deg(angle, tail_angle) <= angle_window_deg:
            considered += 1
            if abs(rho - 1.0) <= tolerance:
                support_hits += 1

    if considered == 0:
        return 0.0
    return float(support_hits / considered)


def score_direct_ellipse_candidate(
    ellipse_info,
    contour,
    contour_points,
    tail_point,
    polyline_points,
    roi_box,
    coil_mask,
    frame_shape,
    dist_transform=None,
):
    cx, cy = ellipse_info["center"]
    a, b = ellipse_info["axes"]
    tail_cx, tail_cy = tail_point

    if not (ELLIPSE_MIN_MAJOR <= a <= ELLIPSE_MAX_MAJOR):
        return None
    if not (ELLIPSE_MIN_MINOR <= b <= ELLIPSE_MAX_MINOR):
        return None
    if a / max(b, 1e-6) > MAX_ELLIPSE_ASPECT_RATIO:
        return None

    if not center_inside_roi_with_margin((cx, cy), roi_box, margin=ELLIPSE_CENTER_ROI_MARGIN):
        return None

    center_to_tail = float(np.hypot(cx - tail_cx, cy - tail_cy))
    if center_to_tail < MIN_CENTER_TO_TAIL_DIST or center_to_tail > MAX_CENTER_TO_TAIL_DIST:
        return None

    center_dy = cy - tail_cy
    if center_dy < -CENTER_ALLOWED_ABOVE_TAIL or center_dy > CENTER_ALLOWED_BELOW_TAIL:
        return None

    center_to_poly = point_to_polyline_distance(cx, cy, polyline_points)
    if center_to_poly is None or center_to_poly > MAX_CENTER_TO_POLYLINE_DIST:
        return None

    tail_to_boundary = ellipse_boundary_distance((tail_cx, tail_cy), ellipse_info)
    if tail_to_boundary is None or tail_to_boundary > MAX_TAIL_TO_ELLIPSE_BOUNDARY_DIST:
        return None

    overlap_ratio = compute_ellipse_overlap_ratio(frame_shape, ellipse_info, coil_mask)
    if overlap_ratio < MIN_ELLIPSE_MASK_OVERLAP_RATIO:
        return None

    support_len = contour_support_score(contour)
    support_ratio = contour_support_ratio_on_ellipse(contour_points, ellipse_info)
    tail_side_support_ratio = contour_tail_side_support_ratio(contour_points, ellipse_info, tail_point)
    angle_coverage_deg = contour_angle_coverage_deg(contour_points, ellipse_info)
    if angle_coverage_deg < DIRECT_MIN_ANGLE_COVERAGE_DEG:
        return None

    horizontal_extent = ellipse_horizontal_extent(ellipse_info)
    horizontal_prior = shape_prior_horizontal_extent()
    horizontal_extent_ratio = horizontal_extent / max(horizontal_prior, 1e-6)
    if horizontal_extent_ratio > MAX_DIRECT_HORIZONTAL_EXTENT_SCALE:
        return None

    if support_ratio < MIN_CONTOUR_SUPPORT_RATIO:
        return None
    if tail_side_support_ratio < MIN_TAIL_SIDE_SUPPORT_RATIO:
        return None

    anchors = ellipse_anchor_points(ellipse_info, polyline_points=polyline_points)
    top_mask_dist = distance_to_mask(anchors["top"], dist_transform)
    left_mask_dist = distance_to_mask(anchors["left"], dist_transform)
    right_mask_dist = distance_to_mask(anchors["right"], dist_transform)
    if top_mask_dist is None or left_mask_dist is None or right_mask_dist is None:
        return None
    if top_mask_dist > DIRECT_TOP_VISIBLE_MAX_MASK_DIST:
        return None
    if left_mask_dist > DIRECT_VISIBLE_MAX_MASK_DIST:
        return None
    if right_mask_dist > DIRECT_VISIBLE_MAX_MASK_DIST:
        return None

    horizontal_prior_penalty = max(0.0, horizontal_extent - horizontal_prior)
    axis_prior_weight = DIRECT_AXIS_PRIOR_SCORE_WEIGHT
    partial_arc_penalty = 0.0
    if angle_coverage_deg < PARTIAL_ARC_COVERAGE_DEG:
        axis_prior_weight = PARTIAL_ARC_AXIS_PRIOR_SCORE_WEIGHT
        partial_arc_penalty = PARTIAL_ARC_DIRECT_SCORE_PENALTY

    score = (
        2.5 * tail_to_boundary +
        1.8 * center_to_poly +
        0.4 * abs(center_dy) +
        0.15 * center_to_tail -
        220.0 * overlap_ratio -
        140.0 * support_ratio -
        110.0 * tail_side_support_ratio -
        1.2 * top_mask_dist +
        1.6 * left_mask_dist +
        1.6 * right_mask_dist -
        0.02 * support_len +
        axis_prior_weight * horizontal_prior_penalty +
        partial_arc_penalty
    )

    accepted = bool(score <= MAX_DIRECT_FIT_SCORE)
    return {
        "score": float(score),
        "accepted": accepted,
        "center_to_polyline_dist": float(center_to_poly),
        "center_to_tail_dist": float(center_to_tail),
        "tail_to_ellipse_boundary_dist": float(tail_to_boundary),
        "contour_support": float(support_len),
        "contour_support_ratio": float(support_ratio),
        "tail_side_support_ratio": float(tail_side_support_ratio),
        "top_visible_mask_dist": float(top_mask_dist),
        "left_visible_mask_dist": float(left_mask_dist),
        "right_visible_mask_dist": float(right_mask_dist),
        "angle_coverage_deg": float(angle_coverage_deg),
        "horizontal_extent_px": float(horizontal_extent),
        "horizontal_extent_prior_px": float(horizontal_prior),
        "horizontal_extent_ratio": float(horizontal_extent_ratio),
        "horizontal_prior_penalty": float(horizontal_prior_penalty),
        "partial_arc_penalty": float(partial_arc_penalty),
        "ellipse_overlap_ratio": float(overlap_ratio),
    }


def fit_final_loop_ellipse_from_segment(
    frame,
    segment_xy,
    tail_tip,
    tail_base=None,
    bbox=None,
    remove_segment_from_edges=True,
    accumulation_frames=None,
):
    base_candidate_mask, debug_masks = build_loop_candidate_mask(
        frame,
        segment_xy,
        tail_tip,
        tail_base=tail_base,
        bbox=bbox,
        remove_segment_fill=remove_segment_from_edges,
    )

    local_mask, roi_box = crop_guided_loop_roi_from_segment(base_candidate_mask, tail_tip, BODY_POLYLINE_POINTS)
    boundary_candidate_mask = debug_masks.get("boundary_candidate_mask")
    boundary_candidate_local_mask = None if boundary_candidate_mask is None else crop_guided_loop_roi_from_segment(
        boundary_candidate_mask,
        tail_tip,
        BODY_POLYLINE_POINTS,
    )[0]
    opening_band_mask = debug_masks.get("loop_opening_band_mask")
    opening_band_local_mask = None if opening_band_mask is None else crop_guided_loop_roi_from_segment(
        opening_band_mask,
        tail_tip,
        BODY_POLYLINE_POINTS,
    )[0]
    rx1, ry1, rx2, ry2 = roi_box

    accumulation_debug = None
    candidate_mask_for_fit = base_candidate_mask
    local_mask_for_fit = local_mask
    mask_source = "base_candidate_mask"

    def evaluate_candidate_mask(candidate_mask, local_candidate_mask, opening_band_local_mask_eval=None, boundary_candidate_local_mask_eval=None):
        merged_local_mask = merge_local_candidate_mask(local_candidate_mask)
        contours, _ = cv2.findContours(merged_local_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        dist_transform = mask_distance_transform(candidate_mask)

        evaluation = {
            "merged_local_mask": merged_local_mask,
            "contours": contours,
            "all_contours_full": [],
            "kept_contours_full": [],
            "direct_candidates": [],
            "best_candidate": None,
        }

        boundary_candidate_evaluation = evaluate_boundary_candidate_mask(
            boundary_candidate_local_mask_eval,
            roi_box=roi_box,
        )
        if boundary_candidate_evaluation.get("all_contours_full"):
            evaluation["all_contours_full"].extend(boundary_candidate_evaluation["all_contours_full"])
        if boundary_candidate_evaluation.get("kept_contours_full"):
            evaluation["kept_contours_full"].extend(boundary_candidate_evaluation["kept_contours_full"])
        if boundary_candidate_evaluation.get("direct_candidates"):
            evaluation["direct_candidates"].extend(boundary_candidate_evaluation["direct_candidates"])
        if boundary_candidate_evaluation.get("best_candidate") is not None:
            evaluation["best_candidate"] = boundary_candidate_evaluation["best_candidate"]
            return evaluation

        opening_evaluation = evaluate_last_loop_opening_band_candidates(
            opening_band_local_mask_eval,
            roi_box=roi_box,
            tail_tip=tail_tip,
            polyline_points=BODY_POLYLINE_POINTS,
        )
        if opening_evaluation.get("all_contours_full"):
            evaluation["all_contours_full"].extend(opening_evaluation["all_contours_full"])
        if opening_evaluation.get("kept_contours_full"):
            evaluation["kept_contours_full"].extend(opening_evaluation["kept_contours_full"])
        if opening_evaluation.get("direct_candidates"):
            evaluation["direct_candidates"].extend(opening_evaluation["direct_candidates"])
        if opening_evaluation.get("best_candidate") is not None:
            evaluation["best_candidate"] = opening_evaluation["best_candidate"]
            return evaluation

        arc_evaluation = evaluate_last_loop_arc_candidates(
            merged_local_mask,
            roi_box=roi_box,
            tail_tip=tail_tip,
            polyline_points=BODY_POLYLINE_POINTS,
        )
        if arc_evaluation.get("all_contours_full"):
            evaluation["all_contours_full"].extend(arc_evaluation["all_contours_full"])
        if arc_evaluation.get("kept_contours_full"):
            evaluation["kept_contours_full"].extend(arc_evaluation["kept_contours_full"])
        if arc_evaluation.get("direct_candidates"):
            evaluation["direct_candidates"].extend(arc_evaluation["direct_candidates"])
        if arc_evaluation.get("best_candidate") is not None:
            evaluation["best_candidate"] = arc_evaluation["best_candidate"]
            return evaluation

        hole_evaluation = evaluate_last_loop_hole_candidates(
            merged_local_mask,
            roi_box=roi_box,
            tail_tip=tail_tip,
            polyline_points=BODY_POLYLINE_POINTS,
        )
        if hole_evaluation.get("all_contours_full"):
            evaluation["all_contours_full"].extend(hole_evaluation["all_contours_full"])
        if hole_evaluation.get("kept_contours_full"):
            evaluation["kept_contours_full"].extend(hole_evaluation["kept_contours_full"])
        if hole_evaluation.get("direct_candidates"):
            evaluation["direct_candidates"].extend(hole_evaluation["direct_candidates"])
        if hole_evaluation.get("best_candidate") is not None:
            evaluation["best_candidate"] = hole_evaluation["best_candidate"]
            return evaluation

        tail_point = (float(tail_tip[0]), float(tail_tip[1]))

        for cnt in contours:
            cnt_full = cnt.copy().astype(np.int32)
            cnt_full[:, 0, 0] += int(rx1)
            cnt_full[:, 0, 1] += int(ry1)
            evaluation["all_contours_full"].append(cnt_full)

            if len(cnt) < MIN_CONTOUR_POINTS_FOR_ELLIPSE:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w < MIN_CONTOUR_SPAN_X or h < MIN_CONTOUR_SPAN_Y:
                continue

            ellipse_info = fit_ellipse_from_contour(cnt, (rx1, ry1))
            if ellipse_info is None:
                continue

            a, b = ellipse_info["axes"]
            if a > 1.35 * w or b > 1.8 * h:
                continue

            contour_points = contour_points_full(cnt, (rx1, ry1))
            metrics = score_direct_ellipse_candidate(
                ellipse_info=ellipse_info,
                contour=cnt,
                contour_points=contour_points,
                tail_point=tail_point,
                polyline_points=BODY_POLYLINE_POINTS,
                roi_box=roi_box,
                coil_mask=candidate_mask,
                frame_shape=frame.shape,
                dist_transform=dist_transform,
            )
            if metrics is None:
                continue

            candidate_record = {
                "ellipse_info": {
                    "center": tuple(ellipse_info["center"]),
                    "axes": tuple(ellipse_info["axes"]),
                    "rotation_deg": float(ellipse_info["rotation_deg"]),
                    "method": ellipse_info.get("method"),
                },
                "metrics": dict(metrics),
                "contour_bbox": [int(x + rx1), int(y + ry1), int(x + rx1 + w), int(y + ry1 + h)],
                "contour_point_count": int(len(cnt)),
                "contour_span_x": int(w),
                "contour_span_y": int(h),
                "contour_full": cnt_full,
            }

            evaluation["kept_contours_full"].append(cnt_full)
            evaluation["direct_candidates"].append(candidate_record)

            candidate = {
                "ellipse_info": ellipse_info,
                "metrics": metrics,
            }
            if evaluation["best_candidate"] is None or metrics["score"] < evaluation["best_candidate"]["metrics"]["score"]:
                evaluation["best_candidate"] = candidate

        return evaluation

    if accumulation_frames is not None and len(accumulation_frames) > 0:
        accumulated_mask, accumulation_debug = accumulate_loop_candidate_masks(
            accumulation_frames,
            segment_xy=segment_xy,
            tail_tip=tail_tip,
            tail_base=tail_base,
            bbox=bbox,
            roi_box=roi_box,
            remove_segment_from_edges=remove_segment_from_edges,
            min_votes=LOOP_ACCUMULATION_MIN_VOTES,
        )

        if accumulated_mask is not None:
            candidate_mask_for_fit = accumulated_mask
            local_mask_for_fit = accumulated_mask[ry1:ry2, rx1:rx2]
            mask_source = "accumulated_candidate_mask"

    evaluation = evaluate_candidate_mask(
        candidate_mask_for_fit,
        local_mask_for_fit,
        opening_band_local_mask,
        boundary_candidate_local_mask,
    )

    if (
        mask_source == "accumulated_candidate_mask"
        and evaluation["best_candidate"] is None
    ):
        candidate_mask_for_fit = base_candidate_mask
        local_mask_for_fit = local_mask
        mask_source = "base_candidate_mask_fallback"
        evaluation = evaluate_candidate_mask(
            candidate_mask_for_fit,
            local_mask_for_fit,
            opening_band_local_mask,
            boundary_candidate_local_mask,
        )

        if accumulation_debug is None:
            accumulation_debug = {}
        accumulation_debug["fallback_applied"] = True
        accumulation_debug["fallback_reason"] = "no_viable_direct_candidates_from_accumulated_mask"
    elif accumulation_debug is not None:
        accumulation_debug["fallback_applied"] = False

    debug_masks["candidate_mask"] = candidate_mask_for_fit
    debug_masks["base_candidate_mask"] = base_candidate_mask
    debug_masks["local_mask"] = local_mask_for_fit
    debug_masks["base_local_mask"] = local_mask
    debug_masks["boundary_candidate_local_mask"] = boundary_candidate_local_mask
    debug_masks["opening_band_local_mask"] = opening_band_local_mask
    debug_masks["merged_local_mask"] = evaluation["merged_local_mask"]
    debug_masks["all_contours_full"] = evaluation["all_contours_full"]
    debug_masks["kept_contours_full"] = evaluation["kept_contours_full"]
    debug_masks["direct_candidates"] = evaluation["direct_candidates"]
    debug_masks["accumulation_debug"] = accumulation_debug
    debug_masks["candidate_mask_source"] = mask_source

    if not evaluation["contours"]:
        return None, candidate_mask_for_fit, roi_box, None, debug_masks

    best_candidate = evaluation["best_candidate"]

    if best_candidate is None:
        return None, candidate_mask_for_fit, roi_box, None, debug_masks

    ellipse_info = best_candidate["ellipse_info"]
    ellipse_info["roi_box"] = roi_box
    return ellipse_info, candidate_mask_for_fit, roi_box, best_candidate["metrics"], debug_masks


def build_direct_anchor_ellipse(direct_ellipse, top_anchor, top_angle_deg, axis_scale=1.0, horizontal_scale=1.0):
    direct_a, direct_b = direct_ellipse["axes"]
    rot_deg = direct_ellipse["rotation_deg"]
    a = float(direct_a) * float(axis_scale) * float(horizontal_scale)
    b = float(direct_b) * float(axis_scale)

    t = math.radians(top_angle_deg)
    theta = math.radians(rot_deg)
    x_local = a * math.cos(t)
    y_local = b * math.sin(t)

    offset_x = x_local * math.cos(theta) - y_local * math.sin(theta)
    offset_y = x_local * math.sin(theta) + y_local * math.cos(theta)

    return {
        "center": (float(top_anchor[0] - offset_x), float(top_anchor[1] - offset_y)),
        "axes": (float(a), float(b)),
        "rotation_deg": float(rot_deg),
        "method": "anchor_guided_direct_ellipse",
        "anchor_method": "direct_upper_arc_fixed",
    }


def score_direct_anchor_ellipse_candidate(
    ellipse_info,
    direct_ellipse,
    top_anchor,
    tail_point,
    candidate_mask,
    frame_shape,
    dist_transform,
):
    anchors = ellipse_anchor_points(ellipse_info)

    top_drift = float(np.hypot(
        anchors["top"][0] - top_anchor[0],
        anchors["top"][1] - top_anchor[1],
    ))
    if top_drift > ANCHOR_MAX_TOP_DRIFT_PX:
        return None

    top_mask_dist = distance_to_mask(anchors["top"], dist_transform)
    left_mask_dist = distance_to_mask(anchors["left"], dist_transform)
    right_mask_dist = distance_to_mask(anchors["right"], dist_transform)
    if top_mask_dist is None or left_mask_dist is None or right_mask_dist is None:
        return None
    if top_mask_dist > ANCHOR_VISIBLE_MAX_MASK_DIST:
        return None
    if left_mask_dist > ANCHOR_VISIBLE_MAX_MASK_DIST:
        return None
    if right_mask_dist > ANCHOR_VISIBLE_MAX_MASK_DIST:
        return None

    tail_boundary = ellipse_boundary_distance(tail_point, ellipse_info)
    if tail_boundary is None or tail_boundary > MAX_TAIL_TO_ELLIPSE_BOUNDARY_DIST:
        return None

    overlap_ratio = compute_ellipse_overlap_ratio(frame_shape, ellipse_info, candidate_mask)
    if overlap_ratio < ANCHOR_MIN_MASK_OVERLAP_RATIO:
        return None

    direct_a, direct_b = direct_ellipse["axes"]
    a, b = ellipse_info["axes"]
    axis_change = abs(a - direct_a) + abs(b - direct_b)

    score = (
        2.3 * tail_boundary +
        3.0 * top_drift +
        2.6 * top_mask_dist +
        2.2 * left_mask_dist +
        2.2 * right_mask_dist +
        0.25 * axis_change -
        260.0 * overlap_ratio
    )

    return {
        "score": float(score),
        "accepted": True,
        "tail_to_ellipse_boundary_dist": float(tail_boundary),
        "top_anchor_drift": float(top_drift),
        "top_visible_mask_dist": float(top_mask_dist),
        "left_visible_mask_dist": float(left_mask_dist),
        "right_visible_mask_dist": float(right_mask_dist),
        "ellipse_overlap_ratio": float(overlap_ratio),
        "axis_change_px": float(axis_change),
        "anchor_points": {
            name: [round(float(point[0]), 2), round(float(point[1]), 2)]
            for name, point in anchors.items()
            if isinstance(point, tuple)
        },
        "source_direct_ellipse": {
            "center": [round(float(direct_ellipse["center"][0]), 2), round(float(direct_ellipse["center"][1]), 2)],
            "axes": [round(float(direct_a), 2), round(float(direct_b), 2)],
            "rotation_deg": round(float(direct_ellipse["rotation_deg"]), 2),
        },
    }


def fit_anchor_guided_ellipse(direct_ellipse, tail_tip, candidate_mask, frame_shape):
    if direct_ellipse is None:
        return None, None

    dist_transform = mask_distance_transform(candidate_mask)
    if dist_transform is None:
        return None, None

    direct_anchors = ellipse_anchor_points(direct_ellipse)
    top_anchor = direct_anchors["top"]
    top_angle_deg = ellipse_parameter_angle_deg(top_anchor, direct_ellipse)
    if top_angle_deg is None:
        return None, None

    best_ellipse = None
    best_metrics = None
    best_axis_scale = None
    best_horizontal_scale = None

    for axis_scale in ANCHOR_AXIS_SCALE_FACTORS:
        for horizontal_scale in ANCHOR_HORIZONTAL_SCALE_FACTORS:
            ellipse_info = build_direct_anchor_ellipse(
                direct_ellipse,
                top_anchor,
                top_angle_deg,
                axis_scale=axis_scale,
                horizontal_scale=horizontal_scale,
            )
            metrics = score_direct_anchor_ellipse_candidate(
                ellipse_info,
                direct_ellipse=direct_ellipse,
                top_anchor=top_anchor,
                tail_point=(float(tail_tip[0]), float(tail_tip[1])),
                candidate_mask=candidate_mask,
                frame_shape=frame_shape,
                dist_transform=dist_transform,
            )
            if metrics is not None and (best_metrics is None or metrics["score"] < best_metrics["score"]):
                best_ellipse = ellipse_info
                best_metrics = metrics
                best_axis_scale = axis_scale
                best_horizontal_scale = horizontal_scale

    if best_ellipse is None:
        return None, None

    best_metrics["axis_scale"] = float(best_axis_scale)
    best_metrics["horizontal_scale"] = float(best_horizontal_scale)
    best_metrics["top_anchor_angle_deg"] = float(top_angle_deg)
    return best_ellipse, best_metrics


def select_final_loop_model(frame, segment_xy, tail_tip, tail_base=None, bbox=None, accumulation_frames=None):
    thermal_ellipse, thermal_mask, thermal_roi_box, thermal_metrics, thermal_debug = fit_thermal_last_loop_ellipse(
        frame,
        segment_xy=segment_xy,
        tail_tip=tail_tip,
        tail_base=tail_base,
        bbox=bbox,
        polyline_points=BODY_POLYLINE_POINTS,
    )

    thermal_accepted = thermal_ellipse is not None and thermal_metrics is not None and thermal_metrics.get("accepted", False)
    if thermal_accepted:
        thermal_ellipse["roi_box"] = thermal_roi_box
        thermal_ellipse["method"] = THERMAL_LOOP_METHOD
        return thermal_ellipse, thermal_metrics, {
            "roi_box": thermal_roi_box,
            "candidate_mask": thermal_mask,
            "debug_masks": {},
            "thermal": {
                "ellipse": thermal_ellipse,
                "metrics": thermal_metrics,
                "candidate_mask": thermal_mask,
                "roi_box": thermal_roi_box,
                "debug_masks": thermal_debug,
            },
            "direct": {
                "ellipse": None,
                "metrics": None,
                "candidate_mask": None,
                "roi_box": None,
                "candidate_count": 0,
                "candidates": [],
                "skipped_reason": "thermal_polyline_selected",
            },
            "anchor_guided": {
                "ellipse": None,
                "metrics": None,
            },
            "chosen_method": THERMAL_LOOP_METHOD,
            "selection_reason": "thermal_polyline_selected",
        }

    if callable(accumulation_frames):
        accumulation_frames = accumulation_frames()

    direct_ellipse, direct_candidate_mask, direct_roi_box, direct_metrics, debug_masks = fit_final_loop_ellipse_from_segment(
        frame,
        segment_xy=segment_xy,
        tail_tip=tail_tip,
        tail_base=tail_base,
        bbox=bbox,
        remove_segment_from_edges=True,
        accumulation_frames=accumulation_frames,
    )

    anchor_ellipse = None
    anchor_metrics = None
    if (
        direct_candidate_mask is not None
        and direct_ellipse is not None
        and direct_ellipse.get("method") == "direct_observed_ellipse"
    ):
        anchor_ellipse, anchor_metrics = fit_anchor_guided_ellipse(
            direct_ellipse,
            tail_tip,
            direct_candidate_mask,
            frame.shape,
        )

    chosen_ellipse = None
    chosen_metrics = None
    chosen_method = None
    chosen_roi_box = direct_roi_box
    chosen_candidate_mask = direct_candidate_mask
    selection_reason = None

    direct_accepted = direct_ellipse is not None and direct_metrics is not None and direct_metrics.get("accepted", False)
    if (
        direct_ellipse is not None
        and direct_ellipse.get("method") in ("last_loop_direct_hole_ellipse", "last_loop_opening_band_ellipse", "last_loop_direct_arc_ellipse")
        and direct_accepted
    ):
        chosen_ellipse = direct_ellipse
        chosen_metrics = direct_metrics
        chosen_method = direct_ellipse.get("method")
        selection_reason = f"{chosen_method}_selected"
    elif anchor_ellipse is not None and anchor_metrics is not None:
        chosen_ellipse = anchor_ellipse
        chosen_metrics = anchor_metrics
        chosen_method = "anchor_guided_direct_ellipse"
        selection_reason = "anchor_guided_from_direct"
    elif direct_accepted:
        chosen_ellipse = direct_ellipse
        chosen_metrics = direct_metrics
        chosen_method = direct_ellipse.get("method", "direct_observed_ellipse")
        selection_reason = "direct_accepted"
    else:
        selection_reason = "no_model_selected"

    if chosen_ellipse is not None:
        chosen_ellipse["roi_box"] = chosen_roi_box
        chosen_ellipse["method"] = chosen_method

    diagnostics = {
        "roi_box": chosen_roi_box,
        "candidate_mask": chosen_candidate_mask,
        "debug_masks": debug_masks,
        "thermal": {
            "ellipse": thermal_ellipse,
            "metrics": thermal_metrics,
            "candidate_mask": thermal_mask,
            "roi_box": thermal_roi_box,
            "debug_masks": thermal_debug,
        },
        "direct": {
            "ellipse": direct_ellipse,
            "metrics": direct_metrics,
            "candidate_mask": direct_candidate_mask,
            "roi_box": direct_roi_box,
            "candidate_count": len(debug_masks.get("direct_candidates", [])) if debug_masks is not None else 0,
            "candidates": debug_masks.get("direct_candidates", []) if debug_masks is not None else [],
        },
        "anchor_guided": {
            "ellipse": anchor_ellipse,
            "metrics": anchor_metrics,
        },
        "chosen_method": chosen_method,
        "selection_reason": selection_reason,
    }
    return chosen_ellipse, chosen_metrics, diagnostics


def point_to_ellipse_angle_deg(point, ellipse_info):
    return ellipse_parameter_angle_deg(point, ellipse_info)


def draw_loop_ellipse(frame, ellipse_info, color=(255, 255, 0), thickness=2, draw_center=True):
    cx, cy = ellipse_info["center"]
    a, b = ellipse_info["axes"]
    rot = ellipse_info["rotation_deg"]

    center_i = (int(round(cx)), int(round(cy)))
    axes_i = (int(round(a)), int(round(b)))

    cv2.ellipse(frame, center_i, axes_i, rot, 0, 360, color, thickness)
    if draw_center:
        cv2.circle(frame, center_i, 6, color, -1)


def draw_loop_search_roi(frame, roi_box, color=(120, 120, 120), thickness=1):
    x1, y1, x2, y2 = roi_box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def mask_to_vis(mask):
    if mask is None:
        return None
    vis = np.asarray(mask)
    if vis.ndim == 2:
        if vis.dtype != np.uint8:
            vis = np.clip(vis, 0, 255).astype(np.uint8)
        if np.max(vis) <= 1:
            vis = (vis * 255).astype(np.uint8)
        return vis
    return vis


def sanitize_for_json(value):
    if isinstance(value, dict):
        return {str(k): sanitize_for_json(v) for k, v in value.items() if k != "contour_full"}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, np.ndarray):
        summary = {
            "array_shape": list(value.shape),
            "array_dtype": str(value.dtype),
        }
        if value.size > 0 and np.issubdtype(value.dtype, np.number):
            summary["nonzero_count"] = int(np.count_nonzero(value))
            summary["min"] = float(np.min(value))
            summary["max"] = float(np.max(value))
        return summary
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def save_debug_image(path, image):
    if image is None:
        return
    cv2.imwrite(str(path), image)


def overlay_common_geometry(image, segment_xy, tail_tip=None, tail_base=None, roi_box=None):
    out = image.copy()
    draw_segment_outline(out, segment_xy, color=(0, 255, 255), thickness=2)
    if tail_tip is not None:
        cv2.circle(out, (int(round(tail_tip[0])), int(round(tail_tip[1]))), 6, (0, 255, 255), -1)
    if tail_base is not None:
        cv2.circle(out, (int(round(tail_base[0])), int(round(tail_base[1]))), 5, (255, 180, 0), -1)
    draw_polyline(out, BODY_POLYLINE_POINTS, color=(255, 0, 255), thickness=2)
    if roi_box is not None:
        draw_loop_search_roi(out, roi_box, color=(160, 160, 160), thickness=1)
    return out


def save_loop_fit_diagnostics(debug_dir, frame, segment_xy, tail_tip, tail_base, annotated_final, loop_diagnostics, final_label_info=None):
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    roi_box = None if loop_diagnostics is None else loop_diagnostics.get("roi_box")
    debug_masks = {} if loop_diagnostics is None else (loop_diagnostics.get("debug_masks") or {})
    direct_info = {} if loop_diagnostics is None else (loop_diagnostics.get("direct") or {})
    anchor_info = {} if loop_diagnostics is None else (loop_diagnostics.get("anchor_guided") or {})
    thermal_info = {} if loop_diagnostics is None else (loop_diagnostics.get("thermal") or {})
    thermal_debug = thermal_info.get("debug_masks") or {}

    base_overlay = overlay_common_geometry(
        frame,
        segment_xy,
        tail_tip=tail_tip,
        tail_base=tail_base,
        roi_box=roi_box,
    )

    save_debug_image(debug_dir / "01_frame.jpg", frame)
    save_debug_image(debug_dir / "02_segment_outline.jpg", base_overlay)
    save_debug_image(debug_dir / "03_edge_mask.png", mask_to_vis(debug_masks.get("edge_mask")))
    save_debug_image(debug_dir / "03a_coil_color_mask.png", mask_to_vis(debug_masks.get("coil_color_mask")))
    save_debug_image(debug_dir / "03aa_coil_envelope_mask.png", mask_to_vis(debug_masks.get("coil_envelope_mask")))
    save_debug_image(debug_dir / "03b_coil_support_mask.png", mask_to_vis(debug_masks.get("coil_support_mask")))
    save_debug_image(debug_dir / "03c_coil_boundary_mask.png", mask_to_vis(debug_masks.get("coil_boundary_mask")))
    save_debug_image(debug_dir / "03ca_boundary_candidate_mask.png", mask_to_vis(debug_masks.get("boundary_candidate_mask")))
    save_debug_image(debug_dir / "03d_loop_opening_mask.png", mask_to_vis(debug_masks.get("loop_opening_mask")))
    save_debug_image(debug_dir / "03e_loop_opening_band_mask.png", mask_to_vis(debug_masks.get("loop_opening_band_mask")))
    save_debug_image(debug_dir / "04_segment_boundary_mask.png", mask_to_vis(debug_masks.get("segment_boundary")))
    save_debug_image(debug_dir / "05_tail_exclusion_mask.png", mask_to_vis(debug_masks.get("tail_exclusion_mask")))
    save_debug_image(debug_dir / "06_candidate_mask.png", mask_to_vis(loop_diagnostics.get("candidate_mask") if loop_diagnostics else None))
    save_debug_image(debug_dir / "06a_base_candidate_mask.png", mask_to_vis(debug_masks.get("base_candidate_mask")))
    save_debug_image(debug_dir / "06b_local_mask.png", mask_to_vis(debug_masks.get("local_mask")))
    save_debug_image(debug_dir / "06c_merged_local_mask.png", mask_to_vis(debug_masks.get("merged_local_mask")))
    save_debug_image(debug_dir / "06f_thermal_heat_score.png", mask_to_vis(thermal_debug.get("heat_mask")))
    save_debug_image(debug_dir / "06g_thermal_geometry_roi.png", mask_to_vis(thermal_debug.get("geometry_roi_mask")))
    save_debug_image(debug_dir / "06h_thermal_material_mask.png", mask_to_vis(thermal_debug.get("material_mask")))
    save_debug_image(debug_dir / "06i_thermal_exclusion_mask.png", mask_to_vis(thermal_debug.get("exclusion_mask")))
    save_debug_image(debug_dir / "06j_thermal_selected_ring.png", mask_to_vis(thermal_debug.get("selected_loop_ring_mask")))
    save_debug_image(debug_dir / "06k_thermal_observed_loop.png", mask_to_vis(thermal_info.get("candidate_mask")))
    save_debug_image(debug_dir / "06m_thermal_centerline.png", mask_to_vis(thermal_debug.get("selected_loop_centerline_mask")))

    thermal_overlay = base_overlay.copy()
    thermal_mask = thermal_info.get("candidate_mask")
    if thermal_mask is not None:
        thermal_overlay[thermal_mask > 0] = (255, 0, 255)
    thermal_ellipse = thermal_info.get("ellipse")
    if thermal_ellipse is not None:
        draw_loop_ellipse(thermal_overlay, thermal_ellipse, color=(255, 0, 255), thickness=3)
    save_debug_image(debug_dir / "06l_thermal_overlay.jpg", thermal_overlay)

    accum_debug = debug_masks.get("accumulation_debug") or {}
    save_debug_image(debug_dir / "06d_accumulated_local_mask.png", mask_to_vis(accum_debug.get("accumulated_local_mask")))

    vote_mask = accum_debug.get("vote_count_mask")
    if vote_mask is not None:
        vote_vis = vote_mask.astype(np.float32)
        if np.max(vote_vis) > 0:
            vote_vis = (255.0 * vote_vis / np.max(vote_vis)).astype(np.uint8)
        else:
            vote_vis = vote_vis.astype(np.uint8)
        save_debug_image(debug_dir / "06e_vote_count_mask.png", vote_vis)

    contours_overlay = base_overlay.copy()
    for cnt in debug_masks.get("all_contours_full", []):
        cv2.polylines(contours_overlay, [cnt.astype(np.int32)], False, (0, 0, 255), 1)
    for cnt in debug_masks.get("kept_contours_full", []):
        cv2.polylines(contours_overlay, [cnt.astype(np.int32)], False, (0, 255, 0), 2)
    save_debug_image(debug_dir / "07_contours_kept.jpg", contours_overlay)

    candidates_overlay = base_overlay.copy()
    candidates = list(direct_info.get("candidates", []))
    candidates.sort(key=lambda c: float((c.get("metrics") or {}).get("score", 1e18)))

    for idx, cand in enumerate(candidates[:DEBUG_SAVE_TOP_K_DIRECT], start=1):
        ellipse_info = cand.get("ellipse_info")
        if ellipse_info is None:
            continue

        draw_loop_ellipse(candidates_overlay, ellipse_info, color=(255, 0, 0), thickness=2)

        cx, cy = ellipse_info["center"]
        metrics = cand.get("metrics") or {}
        score = metrics.get("score", None)
        span_x = cand.get("contour_span_x")
        span_y = cand.get("contour_span_y")

        if score is None:
            label = f"D{idx}"
        else:
            label = f"D{idx}:{float(score):.1f}"

        if span_x is not None and span_y is not None:
            label += f" [{span_x}x{span_y}]"

        cv2.putText(
            candidates_overlay,
            label,
            (int(round(cx)) + 8, int(round(cy)) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2,
        )

        bbox = cand.get("contour_bbox")
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(candidates_overlay, (x1, y1), (x2, y2), (255, 0, 0), 1)

    save_debug_image(debug_dir / "08_direct_candidates.jpg", candidates_overlay)

    anchor_overlay = base_overlay.copy()
    anchor_ellipse = anchor_info.get("ellipse")
    if anchor_ellipse is not None:
        draw_loop_ellipse(anchor_overlay, anchor_ellipse, color=(0, 220, 255), thickness=2)
        m = anchor_info.get("metrics") or {}
        cx, cy = anchor_ellipse["center"]
        score = m.get("score")
        label = "A" if score is None else f"A:{float(score):.1f}"

        cv2.putText(
            anchor_overlay,
            label,
            (int(round(cx)) + 8, int(round(cy)) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 255),
            2,
        )

        for name, point in (m.get("anchor_points") or {}).items():
            px, py = point
            cv2.circle(anchor_overlay, (int(round(px)), int(round(py))), 5, (0, 220, 255), -1)
            cv2.putText(
                anchor_overlay,
                name[0].upper(),
                (int(round(px)) + 6, int(round(py)) + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 220, 255),
                1,
            )

    save_debug_image(debug_dir / "09_anchor_guided_candidate.jpg", anchor_overlay)

    save_debug_image(debug_dir / "10_final_choice.jpg", annotated_final)

    diagnostics_payload = {
        "loop_fit_diagnostics": sanitize_for_json(loop_diagnostics),
        "final_label_info": sanitize_for_json(final_label_info),
    }
    with open(debug_dir / "diagnostics.json", "w") as f:
        json.dump(diagnostics_payload, f, indent=2)
