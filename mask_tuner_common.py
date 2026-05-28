import cv2
import numpy as np

from coil_cv import (
    build_last_loop_mask as build_production_last_loop_mask,
    build_loop_candidate_mask,
    ellipse_params_to_mask,
    find_last_loop_mask,
)

ELLIPSE_COLOR = (255, 255, 0)
ELLIPSE_THICKNESS = 2
AUTO_TUNE_KEYS = [
    ("COIL_MASK_CLOSE_KERNEL", 2, 1, 61),
    ("COIL_MASK_OPEN_KERNEL", 2, 1, 61),
    ("COIL_SUPPORT_DILATE_KERNEL", 2, 1, 61),
    ("COIL_ENVELOPE_CLOSE_KERNEL", 2, 1, 121),
    ("COIL_ENVELOPE_DILATE_KERNEL", 2, 1, 61),
    ("COIL_RELAXED_PAD_X", 10, 0, 1200),
    ("COIL_RELAXED_PAD_Y", 10, 0, 800),
]


def ellipse_from_points(start_pt, end_pt):
    x0, y0 = start_pt
    x1, y1 = end_pt
    cx = float(x0 + x1) / 2.0
    cy = float(y0 + y1) / 2.0
    rx = max(2, abs(x1 - x0) // 2)
    ry = max(2, abs(y1 - y0) // 2)
    return (int(round(cx)), int(round(cy))), (int(rx), int(ry)), 0.0


def ellipse_to_mask(image_shape, ellipse_params):
    return ellipse_params_to_mask(image_shape, ellipse_params)


def build_last_loop_mask(case_data, settings=None, ellipse_params=None):
    return build_production_last_loop_mask(case_data, settings=settings, ellipse_params=ellipse_params)


def rescale_ellipse_params(ellipse_params, src_shape, dst_shape):
    if ellipse_params is None:
        return None
    src_h, src_w = src_shape
    dst_w, dst_h = dst_shape
    center, axes, angle = ellipse_params
    cx, cy = center
    rx, ry = axes

    scale = min(dst_w / max(1, src_w), dst_h / max(1, src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    x0 = (dst_w - new_w) / 2.0
    y0 = (dst_h - new_h) / 2.0

    if cx < x0 or cx > x0 + new_w or cy < y0 or cy > y0 + new_h:
        return None

    inv_scale = 1.0 / scale
    return (
        (int(round((cx - x0) * inv_scale)), int(round((cy - y0) * inv_scale))),
        (int(round(rx * inv_scale)), int(round(ry * inv_scale))),
        angle,
    )


def project_ellipse_params(ellipse_params, src_shape, dst_shape):
    if ellipse_params is None:
        return None

    src_h, src_w = src_shape[:2]
    dst_w, dst_h = dst_shape
    center, axes, angle = ellipse_params
    cx, cy = center
    rx, ry = axes

    scale = min(dst_w / max(1, src_w), dst_h / max(1, src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    x0 = (dst_w - new_w) / 2.0
    y0 = (dst_h - new_h) / 2.0

    return (
        (int(round(x0 + (cx * scale))), int(round(y0 + (cy * scale)))),
        (max(1, int(round(rx * scale))), max(1, int(round(ry * scale)))),
        angle,
    )


def ellipse_match_score(candidate_mask, ellipse_mask):
    if candidate_mask is None or ellipse_mask is None:
        return 0.0
    candidate_bool = candidate_mask > 0
    ellipse_bool = ellipse_mask > 0
    if not np.any(candidate_bool) or not np.any(ellipse_bool):
        return 0.0

    dilation = np.ones((15, 15), np.uint8)
    dilated_candidate = cv2.dilate(candidate_bool.astype(np.uint8) * 255, dilation, iterations=1) > 0
    dilated_ellipse = cv2.dilate(ellipse_bool.astype(np.uint8) * 255, dilation, iterations=1) > 0

    union = np.count_nonzero(dilated_candidate | dilated_ellipse)
    if union == 0:
        return 0.0
    intersection = np.count_nonzero(dilated_candidate & dilated_ellipse)

    boundary_candidate = cv2.morphologyEx(candidate_bool.astype(np.uint8) * 255, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0
    boundary_ellipse = cv2.morphologyEx(ellipse_bool.astype(np.uint8) * 255, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0
    union_boundary = np.count_nonzero(boundary_candidate | boundary_ellipse)
    boundary_iou = 0.0
    if union_boundary > 0:
        boundary_iou = float(np.count_nonzero(boundary_candidate & boundary_ellipse)) / float(union_boundary)

    mask_iou = float(intersection) / float(union)
    return 0.6 * mask_iou + 0.4 * boundary_iou


def build_preview_mask(case_data, settings):
    candidate_mask, _debug = build_production_last_loop_mask(case_data, settings=settings)
    if candidate_mask is not None:
        return candidate_mask

    candidate_mask, _ = build_loop_candidate_mask(
        case_data["frame"],
        segment_xy=case_data["segment_xy"],
        tail_tip=case_data["tail_tip"],
        tail_base=case_data["tail_base"],
        bbox=case_data["bbox"],
        settings=settings,
    )
    return candidate_mask


def search_mask_parameters(case_data, base_settings, ellipse_params, max_iterations=12):
    if ellipse_params is None:
        return base_settings, 0.0

    frame = case_data["frame"]
    ellipse_mask = ellipse_to_mask(frame.shape, ellipse_params)
    current = dict(base_settings)
    best_score = ellipse_match_score(build_preview_mask(case_data, current), ellipse_mask)
    improved = True
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for key, step, min_value, max_value in AUTO_TUNE_KEYS:
            for delta in (-step, step):
                candidate = dict(current)
                candidate[key] = int(round(candidate[key] + delta))
                candidate[key] = max(min_value, min(max_value, candidate[key]))
                if candidate[key] == current[key]:
                    continue
                score = ellipse_match_score(build_preview_mask(case_data, candidate), ellipse_mask)
                if score > best_score + 1e-4:
                    best_score = score
                    current = candidate
                    improved = True
        if improved:
            continue
    return current, best_score
