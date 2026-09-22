import math

import cv2
import numpy as np


BOUNDARY_THICKNESS_PX = 7


def _clean_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_ellipse(raw):
    if raw is None:
        return None
    return {
        "cx": _clean_float(raw.get("cx")),
        "cy": _clean_float(raw.get("cy")),
        "rx": max(0.1, _clean_float(raw.get("rx"), 1.0)),
        "ry": max(0.1, _clean_float(raw.get("ry"), 1.0)),
        "rotation_deg": _clean_float(raw.get("rotation_deg")),
    }


def angle_error_deg(a_deg, b_deg):
    raw = abs((_clean_float(a_deg) - _clean_float(b_deg)) % 180.0)
    return min(raw, 180.0 - raw)


def ellipse_mask(image_shape, ellipse, filled=True, thickness=BOUNDARY_THICKNESS_PX):
    h, w = int(image_shape[0]), int(image_shape[1])
    mask = np.zeros((h, w), dtype=np.uint8)
    if h <= 0 or w <= 0 or ellipse is None:
        return mask

    center = (int(round(ellipse["cx"])), int(round(ellipse["cy"])))
    axes = (max(1, int(round(ellipse["rx"]))), max(1, int(round(ellipse["ry"]))))
    draw_thickness = -1 if filled else max(1, int(round(thickness)))
    cv2.ellipse(mask, center, axes, float(ellipse["rotation_deg"]), 0, 360, 255, draw_thickness)
    return mask


def mask_iou(mask_a, mask_b):
    a = mask_a > 0
    b = mask_b > 0
    union = np.count_nonzero(a | b)
    if union == 0:
        return 0.0
    return float(np.count_nonzero(a & b)) / float(union)


def ellipse_error_components(truth_ellipse, prediction_ellipse):
    truth = normalize_ellipse(truth_ellipse)
    pred = normalize_ellipse(prediction_ellipse)
    if truth is None or pred is None:
        raise ValueError("Both truth and prediction ellipses are required.")

    center_error_px = float(math.hypot(pred["cx"] - truth["cx"], pred["cy"] - truth["cy"]))
    mean_truth_radius = max(1.0, (truth["rx"] + truth["ry"]) / 2.0)
    center_score = math.exp(-center_error_px / mean_truth_radius)

    rx_error_pct = abs(pred["rx"] - truth["rx"]) / max(1.0, truth["rx"])
    ry_error_pct = abs(pred["ry"] - truth["ry"]) / max(1.0, truth["ry"])
    axis_score = math.exp(-(abs(math.log(pred["rx"] / truth["rx"])) + abs(math.log(pred["ry"] / truth["ry"]))))

    truth_aspect = max(truth["rx"], truth["ry"]) / max(0.1, min(truth["rx"], truth["ry"]))
    pred_aspect = max(pred["rx"], pred["ry"]) / max(0.1, min(pred["rx"], pred["ry"]))
    aspect_ratio_score = math.exp(-abs(math.log(pred_aspect / truth_aspect)))

    angle_error = angle_error_deg(pred["rotation_deg"], truth["rotation_deg"])
    circularity = min(truth["rx"], truth["ry"]) / max(truth["rx"], truth["ry"])
    angle_weight = 0.0 if circularity > 0.9 else 0.05
    angle_score = math.exp(-angle_error / 30.0)

    return {
        "truth": truth,
        "pred": pred,
        "center_error_px": center_error_px,
        "center_score": center_score,
        "rx_error_pct": rx_error_pct,
        "ry_error_pct": ry_error_pct,
        "axis_score": axis_score,
        "truth_aspect_ratio": truth_aspect,
        "prediction_aspect_ratio": pred_aspect,
        "aspect_ratio_score": aspect_ratio_score,
        "angle_error_deg": angle_error,
        "angle_score": angle_score,
        "angle_weight": angle_weight,
    }


def _rounded_metrics(weighted, components, boundary_iou, filled_iou, extra=None):
    metrics = {
        "score_0_1": round(float(weighted), 6),
        "score_0_100": round(float(weighted * 100.0), 2),
        "center_error_px": round(float(components["center_error_px"]), 3),
        "center_score": round(float(components["center_score"]), 6),
        "rx_error_pct": round(float(components["rx_error_pct"]), 6),
        "ry_error_pct": round(float(components["ry_error_pct"]), 6),
        "axis_score": round(float(components["axis_score"]), 6),
        "truth_aspect_ratio": round(float(components["truth_aspect_ratio"]), 6),
        "prediction_aspect_ratio": round(float(components["prediction_aspect_ratio"]), 6),
        "aspect_ratio_score": round(float(components["aspect_ratio_score"]), 6),
        "angle_error_deg": round(float(components["angle_error_deg"]), 3),
        "angle_score": round(float(components["angle_score"]), 6),
        "angle_weight": round(float(components["angle_weight"]), 3),
        "boundary_iou": round(float(boundary_iou), 6),
        "filled_iou": round(float(filled_iou), 6),
        "boundary_thickness_px": int(BOUNDARY_THICKNESS_PX),
    }
    if extra:
        metrics.update(extra)
    return metrics


def score_ellipses(truth_ellipse, prediction_ellipse, image_width, image_height):
    components = ellipse_error_components(truth_ellipse, prediction_ellipse)
    truth = components["truth"]
    pred = components["pred"]

    image_shape = (int(image_height), int(image_width))
    if image_shape[0] <= 0 or image_shape[1] <= 0:
        raise ValueError("A positive image width and height are required.")

    truth_filled = ellipse_mask(image_shape, truth, filled=True)
    pred_filled = ellipse_mask(image_shape, pred, filled=True)
    filled_iou = mask_iou(truth_filled, pred_filled)

    truth_boundary = ellipse_mask(image_shape, truth, filled=False)
    pred_boundary = ellipse_mask(image_shape, pred, filled=False)
    boundary_iou = mask_iou(truth_boundary, pred_boundary)

    return _rounded_metrics(filled_iou, components, boundary_iou, filled_iou)


def shape_canvas_shape(truth, pred):
    max_radius = max(truth["rx"], truth["ry"], pred["rx"], pred["ry"], 1.0)
    size = int(math.ceil(max_radius * 2.5 + BOUNDARY_THICKNESS_PX * 4))
    size = max(64, min(size, 4096))
    return (size, size)


def score_ellipse_shape(truth_ellipse, prediction_ellipse):
    components = ellipse_error_components(truth_ellipse, prediction_ellipse)
    truth = dict(components["truth"])
    pred = dict(components["pred"])

    image_shape = shape_canvas_shape(truth, pred)
    center = {"cx": image_shape[1] / 2.0, "cy": image_shape[0] / 2.0}
    truth.update(center)
    pred.update(center)

    truth_filled = ellipse_mask(image_shape, truth, filled=True)
    pred_filled = ellipse_mask(image_shape, pred, filled=True)
    filled_iou = mask_iou(truth_filled, pred_filled)

    truth_boundary = ellipse_mask(image_shape, truth, filled=False)
    pred_boundary = ellipse_mask(image_shape, pred, filled=False)
    boundary_iou = mask_iou(truth_boundary, pred_boundary)

    return _rounded_metrics(
        filled_iou,
        components,
        boundary_iou,
        filled_iou,
        {
            "shape_only": True,
            "center_score": None,
            "center_error_px": None,
            "shape_canvas_width": image_shape[1],
            "shape_canvas_height": image_shape[0],
        },
    )


def labels_from_payload(label_payload):
    if label_payload is None:
        return []
    if isinstance(label_payload, list):
        candidates = label_payload
    elif isinstance(label_payload, dict) and isinstance(label_payload.get("labels"), list):
        candidates = label_payload.get("labels")
    elif isinstance(label_payload, dict):
        candidates = [label_payload]
    else:
        candidates = []
    return [label for label in candidates if isinstance(label, dict) and label.get("ellipse") is not None]


def label_frame_index(label):
    frame = label.get("frame") or {}
    index = frame.get("index")
    if index is None:
        return None
    return int(index)


def closest_label_for_prediction(labels, prediction_frame_index):
    if prediction_frame_index is None:
        return None
    pred_index = int(prediction_frame_index)
    return min(labels, key=lambda label: abs(label_frame_index(label) - pred_index))


def score_capture_labels(label_payload, prediction):
    labels = labels_from_payload(label_payload)
    if not labels:
        return {"ok": False, "error": "No saved truth label found for this capture."}
    if prediction is None or prediction.get("ellipse") is None:
        return {"ok": False, "error": "No CV ellipse prediction found for this capture."}

    prediction_frame_index = prediction.get("frame_index")
    if prediction_frame_index is None:
        return {"ok": False, "error": "Prediction must include a frame index."}

    indexed_labels = [label for label in labels if label_frame_index(label) is not None]
    if not indexed_labels:
        return {"ok": False, "error": "Saved truth labels must include frame indices."}

    prediction_frame_index = int(prediction_frame_index)
    exact_labels = [label for label in indexed_labels if label_frame_index(label) == prediction_frame_index]
    label = exact_labels[0] if exact_labels else closest_label_for_prediction(indexed_labels, prediction_frame_index)
    truth_frame = label.get("frame") or {}
    truth_frame_index = label_frame_index(label)

    if truth_frame_index == prediction_frame_index:
        image_width = truth_frame.get("image_width")
        image_height = truth_frame.get("image_height")
        metrics = score_ellipses(label.get("ellipse"), prediction.get("ellipse"), image_width, image_height)
        score_status = "scored_same_frame"
    else:
        metrics = score_ellipse_shape(label.get("ellipse"), prediction.get("ellipse"))
        score_status = "scored_shape_only_closest_frame"

    return {
        "ok": True,
        "score_status": score_status,
        "frame_index": truth_frame_index,
        "truth_frame_index": truth_frame_index,
        "prediction_frame_index": prediction_frame_index,
        "frame_delta": abs(truth_frame_index - prediction_frame_index),
        "label_count": len(indexed_labels),
        "prediction_method": prediction.get("method"),
        "prediction_json_path": prediction.get("json_path"),
        "prediction_ellipse": prediction.get("ellipse"),
        "truth_ellipse": label.get("ellipse"),
        "metrics": metrics,
    }
