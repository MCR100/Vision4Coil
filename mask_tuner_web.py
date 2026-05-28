import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request

from coil_cv import BODY_POLYLINE_POINTS, draw_polyline, get_mask_tuning_defaults
from mask_tuner_common import (
  ELLIPSE_COLOR,
  ELLIPSE_THICKNESS,
  build_last_loop_mask,
  ellipse_to_mask,
  project_ellipse_params,
  rescale_ellipse_params,
  search_mask_parameters,
  find_last_loop_mask,
)


app = Flask(__name__)
SETTINGS_SAVE_PATH = Path("mask_tuner_settings.json")
PANEL_W = 560
PANEL_H = 315

TRACKBAR_SPECS = [
    {"name": "COIL_RED_MIN", "source_key": "COIL_RED_MIN", "min": 0, "max": 255, "step": 1},
    {"name": "COIL_EXCESS_RED_MIN", "source_key": "COIL_EXCESS_RED_MIN", "min": 0, "max": 255, "step": 1},
    {"name": "COIL_LAB_A_MIN", "source_key": "COIL_LAB_A_MIN", "min": 0, "max": 255, "step": 1},
    {"name": "COIL_SAT_MIN", "source_key": "COIL_SAT_MIN", "min": 0, "max": 255, "step": 1},
    {"name": "COIL_VALUE_MIN", "source_key": "COIL_VALUE_MIN", "min": 0, "max": 255, "step": 1},
    {"name": "COIL_WHITEHOT_RED_MIN", "source_key": "COIL_WHITEHOT_RED_MIN", "min": 0, "max": 255, "step": 1},
    {"name": "COIL_ADAPTIVE_RED_FLOOR", "source_key": "COIL_ADAPTIVE_RED_FLOOR", "min": 0, "max": 255, "step": 1},
    {"name": "COIL_ADAPTIVE_EXCESS_FLOOR", "source_key": "COIL_ADAPTIVE_EXCESS_FLOOR", "min": 0, "max": 255, "step": 1},
    {"name": "COIL_ADAPTIVE_A_FLOOR", "source_key": "COIL_ADAPTIVE_A_FLOOR", "min": 0, "max": 255, "step": 1},
    {"name": "COIL_RELAXED_RED_FLOOR", "source_key": "COIL_RELAXED_RED_FLOOR", "min": 0, "max": 255, "step": 1},
    {"name": "COIL_RELAXED_EXCESS_FLOOR", "source_key": "COIL_RELAXED_EXCESS_FLOOR", "min": -80, "max": 255, "step": 1},
    {"name": "COIL_RELAXED_A_FLOOR", "source_key": "COIL_RELAXED_A_FLOOR", "min": 0, "max": 255, "step": 1},
    {"name": "COIL_RELAXED_VALUE_FLOOR", "source_key": "COIL_RELAXED_VALUE_FLOOR", "min": 0, "max": 255, "step": 1},
    {"name": "COIL_RELAXED_SAT_FLOOR", "source_key": "COIL_RELAXED_SAT_FLOOR", "min": 0, "max": 255, "step": 1},
    {"name": "COIL_RELAXED_SEGMENT_COVERAGE", "source_key": "COIL_RELAXED_SEGMENT_COVERAGE", "min": 0.05, "max": 4.0, "step": 0.01},
    {"name": "COIL_RELAXED_SEGMENT_INTERSECTION", "source_key": "COIL_RELAXED_SEGMENT_INTERSECTION", "min": 0.0, "max": 1.0, "step": 0.01},
    {"name": "COIL_RELAXED_PAD_X", "source_key": "COIL_RELAXED_PAD_X", "min": 0, "max": 1200, "step": 1},
    {"name": "COIL_RELAXED_PAD_Y", "source_key": "COIL_RELAXED_PAD_Y", "min": 0, "max": 800, "step": 1},
    {"name": "COIL_MASK_CLOSE_KERNEL", "source_key": "COIL_MASK_CLOSE_KERNEL", "min": 1, "max": 61, "step": 2},
    {"name": "COIL_MASK_OPEN_KERNEL", "source_key": "COIL_MASK_OPEN_KERNEL", "min": 1, "max": 61, "step": 2},
    {"name": "COIL_SUPPORT_DILATE_KERNEL", "source_key": "COIL_SUPPORT_DILATE_KERNEL", "min": 1, "max": 61, "step": 2},
    {"name": "COIL_ENVELOPE_CLOSE_KERNEL", "source_key": "COIL_ENVELOPE_CLOSE_KERNEL", "min": 1, "max": 121, "step": 2},
    {"name": "COIL_ENVELOPE_DILATE_KERNEL", "source_key": "COIL_ENVELOPE_DILATE_KERNEL", "min": 1, "max": 61, "step": 2},
    {"name": "COIL_ENVELOPE_MIN_AREA", "source_key": "COIL_ENVELOPE_MIN_AREA", "min": 1, "max": 200000, "step": 500},
]


CASE_DATA = None


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


def recover_segment_from_debug(case_dir):
    boundary_path = case_dir / "debug" / "04_segment_boundary_mask.png"
    if not boundary_path.exists():
        return None

    boundary = cv2.imread(str(boundary_path), cv2.IMREAD_GRAYSCALE)
    if boundary is None or np.count_nonzero(boundary) == 0:
        return None

    _ret, thresh = cv2.threshold(boundary, 127, 255, cv2.THRESH_BINARY)
    contours, _hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 20.0:
        return None

    epsilon = max(2.0, 0.0025 * cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if approx is None or len(approx) < 3:
        return None

    return approx.reshape(-1, 2).astype(np.float32)


def load_case(case_path):
    case_dir = normalize_case_path(case_path)
    json_path, image_path = pick_case_assets(case_dir)

    data = json.loads(json_path.read_text())
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    segment_xy = data.get("segment_xy")
    segment_source = "json_segment_xy"
    if segment_xy:
        segment_xy = np.asarray(segment_xy, dtype=np.float32)
    else:
        segment_xy = recover_segment_from_debug(case_dir)
        if segment_xy is not None:
            segment_source = "debug_boundary_reconstruction"
        else:
            bbox = data.get("bbox")
            if bbox is None or len(bbox) != 4:
                raise ValueError(f"No segment geometry available in {json_path}")
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            segment_xy = np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            segment_source = "bbox_fallback"

    tail_tip = tuple(float(v) for v in data["tail_tip"])
    tail_base = None if data.get("tail_base") is None else tuple(float(v) for v in data["tail_base"])
    bbox = None if data.get("bbox") is None else [int(round(v)) for v in data["bbox"]]
    polyline = data.get("body_polyline_points") or BODY_POLYLINE_POINTS
    polyline = [(int(round(x)), int(round(y))) for x, y in polyline]

    return {
        "case_dir": str(case_dir),
        "json_path": str(json_path),
        "image_path": str(image_path),
        "frame": frame,
        "segment_xy": segment_xy,
        "tail_tip": tail_tip,
        "tail_base": tail_base,
        "bbox": bbox,
        "polyline": polyline,
        "segment_source": segment_source,
    }


def fit_panel_image(image, width=PANEL_W, height=PANEL_H):
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
    out = image.copy()
    cv2.putText(out, text, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return out


def mask_to_bgr(mask):
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def encode_image(image, ext=".jpg"):
    ok, buf = cv2.imencode(ext, image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("Failed to encode image.")
    mime = "image/jpeg" if ext == ".jpg" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(buf.tobytes()).decode('ascii')}"


def sanitize_settings(raw_settings):
    defaults = get_mask_tuning_defaults()
    settings = {}
    for spec in TRACKBAR_SPECS:
        key = spec["source_key"]
        value = raw_settings.get(key, defaults[key])
        if spec["step"] >= 1:
            value = int(round(float(value)))
        else:
            value = float(value)
        if value < spec["min"]:
            value = spec["min"]
        if value > spec["max"]:
            value = spec["max"]
        settings[key] = value
    return settings


def render_preview(case_data, settings, ellipse_params=None):
    frame = case_data["frame"]
    segment_xy = case_data["segment_xy"]
    tail_tip = case_data["tail_tip"]
    tail_base = case_data["tail_base"]

    frame_ellipse = None
    if ellipse_params is not None:
        frame_ellipse = rescale_ellipse_params(ellipse_params, frame.shape[:2], (PANEL_W, PANEL_H))

    candidate_mask, debug = build_last_loop_mask(case_data, settings=settings, ellipse_params=frame_ellipse)
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

    raw_frame = fit_panel_image(overlay)
    if ellipse_params is not None:
        center, axes, angle = ellipse_params
        cv2.ellipse(raw_frame, center, axes, angle, 0, 360, ELLIPSE_COLOR, ELLIPSE_THICKNESS)

    candidate_overlay = overlay.copy()
    candidate_overlay[candidate_mask > 0] = (255, 0, 255)

    panels = {
        "frame": label_panel(raw_frame, "Frame + Geometry"),
        "heat": label_panel(fit_panel_image(mask_to_bgr(debug["heat_mask"])), "Heat Score"),
        "material": label_panel(fit_panel_image(mask_to_bgr(debug["material_mask"])), "Hot Material"),
        "roi": label_panel(fit_panel_image(mask_to_bgr(debug["geometry_roi_mask"])), "Geometry ROI"),
        "exclusion": label_panel(fit_panel_image(mask_to_bgr(debug["exclusion_mask"])), "Tail Exclusion"),
        "selected_ring": label_panel(fit_panel_image(mask_to_bgr(debug["selected_loop_ring_mask"])), "Selected Ring"),
        "selected_fill": label_panel(fit_panel_image(mask_to_bgr(debug["selected_loop_fill_mask"])), "Selected Fill"),
        "candidate": label_panel(fit_panel_image(candidate_overlay), "Observed Loop"),
    }

    images = {name: encode_image(img, ext=".jpg") for name, img in panels.items()}
    metrics = debug.get("best_metrics") or {}
    stats = {
        "heat_px": int(np.count_nonzero(debug["heat_mask"])),
        "material_px": int(np.count_nonzero(debug["material_mask"])),
        "roi_px": int(np.count_nonzero(debug["geometry_roi_mask"])),
        "excluded_px": int(np.count_nonzero(debug["exclusion_mask"])),
        "ring_px": int(np.count_nonzero(debug["selected_loop_ring_mask"])),
        "observed_px": int(np.count_nonzero(candidate_mask)),
        "candidates_scored": int(debug.get("candidates_scored", 0)),
    }
    if metrics:
        stats.update({
            "score": round(float(metrics.get("score", 0.0)), 4),
            "ring_coverage": round(float(metrics.get("ring_coverage", 0.0)), 4),
            "polyline_dist": round(float(metrics.get("center_to_polyline_dist", metrics.get("polyline_distance")) or 0.0), 2),
            "topness": round(float(metrics.get("topness", 0.0)), 4),
        })
    if frame_ellipse is not None:
        ellipse_mask = ellipse_to_mask(frame.shape, frame_ellipse)
        stats["guide_overlap_px"] = int(np.count_nonzero((candidate_mask > 0) & (ellipse_mask > 0)))
    return images, stats


@app.get("/")
def index():
    return render_template_string(
        """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Mask Tuner</title>
  <style>
    :root {
      --bg:#111318;
      --panel:#1a1f27;
      --panel2:#202632;
      --text:#eef2f7;
      --muted:#9aa6b2;
      --line:#2e3642;
      --accent:#ff6b3d;
    }
    body {
      margin:0;
      background:linear-gradient(180deg, #141821 0%, #0f1218 100%);
      color:var(--text);
      font-family: ui-sans-serif, system-ui, sans-serif;
    }
    .layout {
      display:grid;
      grid-template-columns: 420px 1fr;
      min-height:100vh;
    }
    .sidebar {
      padding:18px;
      background:rgba(18, 22, 29, 0.96);
      border-right:1px solid var(--line);
      overflow:auto;
    }
    .main {
      padding:18px;
      overflow:auto;
    }
    .card {
      background:var(--panel);
      border:1px solid var(--line);
      border-radius:16px;
      padding:14px;
      box-shadow:0 20px 40px rgba(0,0,0,0.18);
    }
    .header {
      margin-bottom:16px;
    }
    .title {
      font-size:28px;
      font-weight:700;
      margin:0 0 6px 0;
    }
    .muted {
      color:var(--muted);
      font-size:14px;
      line-height:1.4;
    }
    .grid {
      display:grid;
      grid-template-columns: repeat(2, minmax(280px, 1fr));
      gap:16px;
    }
    .panel img {
      width:100%;
      display:block;
      border-radius:12px;
      background:#000;
    }
    .frame-canvas-wrapper {
      position: relative;
    }
    .frame-canvas-wrapper canvas {
      position: absolute;
      left: 0;
      top: 0;
      width: 100%;
      height: 100%;
      cursor: crosshair;
      pointer-events: auto;
      border-radius: 12px;
    }
    .panel-title {
      font-weight:700;
      margin-bottom:10px;
    }
    .group {
      margin:16px 0;
      padding-top:12px;
      border-top:1px solid var(--line);
    }
    .group:first-of-type {
      border-top:none;
      padding-top:0;
    }
    .group h3 {
      margin:0 0 10px 0;
      font-size:14px;
      text-transform:uppercase;
      letter-spacing:0.08em;
      color:var(--muted);
    }
    .control {
      margin:10px 0 14px 0;
    }
    .control-head {
      display:flex;
      justify-content:space-between;
      gap:8px;
      margin-bottom:6px;
      font-size:13px;
    }
    .control input[type=range] {
      width:100%;
    }
    .row {
      display:flex;
      gap:10px;
      flex-wrap:wrap;
      margin-top:16px;
    }
    button {
      border:none;
      border-radius:12px;
      padding:10px 14px;
      background:var(--panel2);
      color:var(--text);
      cursor:pointer;
      font-weight:600;
    }
    button.primary {
      background:var(--accent);
      color:#fff;
    }
    pre {
      margin:0;
      padding:12px;
      background:#0d1015;
      border-radius:12px;
      border:1px solid var(--line);
      font-size:12px;
      overflow:auto;
    }
    .stats {
      margin-top:16px;
      display:grid;
      grid-template-columns: repeat(2, minmax(120px, 1fr));
      gap:10px;
    }
    .stat {
      padding:12px;
      background:var(--panel2);
      border-radius:12px;
      border:1px solid var(--line);
    }
    .stat .k {
      color:var(--muted);
      font-size:12px;
    }
    .stat .v {
      font-size:18px;
      font-weight:700;
      margin-top:4px;
    }
  </style>
</head>
<body>
  <div class="layout">
    <div class="sidebar">
      <div class="header">
        <h1 class="title">Mask Tuner</h1>
        <div class="muted" id="caseInfo"></div>
      </div>

      <div class="card">
        <div class="muted">Tune the fresh last-loop masking path in the browser. White pixels in the mask views are the pixels kept by that stage. Black pixels are rejected by that stage.</div>

        <div class="group">
          <h3>Panel Meanings</h3>
          <div class="muted">
            <div><b>Frame + Geometry</b>: source frame with the tail segment, conveyor polyline, selected ellipse, and optional guide ellipse.</div>
            <div><b>Heat Score</b>: red/brightness heat map used to find hottest material.</div>
            <div><b>Hot Material</b>: thresholded coil pixels inside the conveyor-aligned search region.</div>
            <div><b>Geometry ROI</b>: search band centered on the hardcoded conveyor polyline.</div>
            <div><b>Tail Exclusion</b>: material intentionally removed from the candidate search.</div>
            <div><b>Selected Ring</b>: best-scoring last-loop ellipse band.</div>
            <div><b>Selected Fill</b>: filled ellipse for judging center/shape overlap.</div>
            <div><b>Observed Loop</b>: hot pixels on the selected ring after tail removal.</div>
          </div>
        </div>

        <div class="row">
          <button class="primary" onclick="saveSettings()">Save Settings</button>
          <button onclick="resetSettings()">Reset Defaults</button>
          <button onclick="autoTune()">Auto Tune</button>
          <button onclick="guidedMask()">Guided Mask</button>
        </div>
        <div class="muted">Draw an ellipse on the Frame panel with left drag, then click Auto Tune. Right-click the frame to clear.</div>
        <div class="group">
          <div id="statusMessage" class="muted"></div>
        </div>

        <div id="controls"></div>

        <div class="group">
          <h3>Current Settings</h3>
          <pre id="settingsJson"></pre>
        </div>
      </div>
    </div>

    <div class="main">
      <div class="grid" id="imageGrid"></div>
      <div class="stats" id="stats"></div>
    </div>
  </div>

<script>
const trackbarSpecs = {{ specs|tojson }};
const defaultSettings = {{ defaults|tojson }};
const caseMeta = {{ case_meta|tojson }};
const panelTitles = {
  frame: 'Frame + Geometry',
  heat: 'Heat Score',
  material: 'Hot Material',
  roi: 'Geometry ROI',
  exclusion: 'Tail Exclusion',
  selected_ring: 'Selected Ring',
  selected_fill: 'Selected Fill',
  candidate: 'Observed Loop',
};
let state = {...defaultSettings};
let ellipse = null;
let dragStart = null;
let dragActive = false;
let debounce = null;

function buildControls() {
  const groups = [
    ['COIL_RED_MIN','COIL_EXCESS_RED_MIN','COIL_LAB_A_MIN','COIL_SAT_MIN','COIL_VALUE_MIN','COIL_WHITEHOT_RED_MIN'],
    ['COIL_ADAPTIVE_RED_FLOOR','COIL_ADAPTIVE_EXCESS_FLOOR','COIL_ADAPTIVE_A_FLOOR'],
    ['COIL_RELAXED_RED_FLOOR','COIL_RELAXED_EXCESS_FLOOR','COIL_RELAXED_A_FLOOR','COIL_RELAXED_VALUE_FLOOR','COIL_RELAXED_SAT_FLOOR','COIL_RELAXED_SEGMENT_COVERAGE','COIL_RELAXED_SEGMENT_INTERSECTION','COIL_RELAXED_PAD_X','COIL_RELAXED_PAD_Y'],
    ['COIL_MASK_CLOSE_KERNEL','COIL_MASK_OPEN_KERNEL','COIL_SUPPORT_DILATE_KERNEL','COIL_ENVELOPE_CLOSE_KERNEL','COIL_ENVELOPE_DILATE_KERNEL','COIL_ENVELOPE_MIN_AREA'],
  ];
  const groupLabels = ['Base Color', 'Adaptive Floors', 'Relaxed Fallback', 'Morphology'];
  const controls = document.getElementById('controls');
  controls.innerHTML = '';

  groups.forEach((keys, index) => {
    const group = document.createElement('div');
    group.className = 'group';
    group.innerHTML = `<h3>${groupLabels[index]}</h3>`;

    keys.forEach((key) => {
      const spec = trackbarSpecs.find(s => s.source_key === key);
      const control = document.createElement('div');
      control.className = 'control';
      control.innerHTML = `
        <div class="control-head">
          <span>${spec.name}</span>
          <span id="value-${key}">${state[key]}</span>
        </div>
        <input type="range" min="${spec.min}" max="${spec.max}" step="${spec.step}" value="${state[key]}" id="slider-${key}" />
      `;
      group.appendChild(control);
    });
    controls.appendChild(group);
  });

  trackbarSpecs.forEach((spec) => {
    document.getElementById(`slider-${spec.source_key}`).addEventListener('input', (event) => {
      const raw = event.target.value;
      state[spec.source_key] = spec.step >= 1 ? parseInt(raw, 10) : parseFloat(raw);
      document.getElementById(`value-${spec.source_key}`).textContent = state[spec.source_key];
      document.getElementById('settingsJson').textContent = JSON.stringify(state, null, 2);
      queueRefresh();
    });
  });
}

function renderCaseMeta() {
  const sourceLabel = {
    json_segment_xy: 'saved tail segment polygon from JSON',
    debug_boundary_reconstruction: 'reconstructed tail segment polygon from debug boundary mask',
    bbox_fallback: 'bbox fallback because no segment polygon was available'
  }[caseMeta.segment_source] || caseMeta.segment_source;
  document.getElementById('caseInfo').innerHTML =
    `Case: ${caseMeta.case_dir}<br>Frame: ${caseMeta.image_path}<br>JSON: ${caseMeta.json_path}<br>Segment source: ${sourceLabel}`;
}

function renderImages(images) {
  const grid = document.getElementById('imageGrid');
  grid.innerHTML = '';
  Object.entries(images).forEach(([key, src]) => {
    const panel = document.createElement('div');
    panel.className = 'card panel';
    if (key === 'frame') {
      panel.innerHTML = `
        <div class="panel-title">${panelTitles[key] || key}</div>
        <div class="frame-canvas-wrapper">
          <img id="frame-image" src="${src}" alt="${key}" />
          <canvas id="frame-canvas"></canvas>
        </div>
      `;
    } else {
      panel.innerHTML = `<div class="panel-title">${panelTitles[key] || key}</div><img src="${src}" alt="${key}" />`;
    }
    grid.appendChild(panel);
  });
  initFrameCanvas();
}

function renderStats(stats) {
  const box = document.getElementById('stats');
  box.innerHTML = '';
  Object.entries(stats).forEach(([key, value]) => {
    const item = document.createElement('div');
    item.className = 'stat';
    item.innerHTML = `<div class="k">${key}</div><div class="v">${value}</div>`;
    box.appendChild(item);
  });
}

async function refreshPreview() {
  const response = await fetch('/api/preview', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({settings: state, ellipse}),
  });
  const payload = await response.json();
  renderImages(payload.images);
  renderStats(payload.stats);
}

function queueRefresh() {
  if (debounce) clearTimeout(debounce);
  debounce = setTimeout(refreshPreview, 120);
}

async function saveSettings() {
  await fetch('/api/save', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({settings: state}),
  });
}

function resetSettings() {
  state = {...defaultSettings};
  ellipse = null;
  dragStart = null;
  dragActive = false;
  buildControls();
  document.getElementById('settingsJson').textContent = JSON.stringify(state, null, 2);
  queueRefresh();
}

function clearEllipse() {
  ellipse = null;
  dragStart = null;
  dragActive = false;
  updateEllipseCanvas();
  queueRefresh();
}

function setEllipseParams(start, end) {
  ellipse = {
    cx: Math.round((start.x + end.x) / 2),
    cy: Math.round((start.y + end.y) / 2),
    rx: Math.max(2, Math.round(Math.abs(end.x - start.x) / 2)),
    ry: Math.max(2, Math.round(Math.abs(end.y - start.y) / 2)),
  };
}

function drawEllipseOnCanvas(canvas) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!ellipse) {
    return;
  }
  ctx.strokeStyle = '#ffb347';
  ctx.lineWidth = Math.max(2, canvas.width / 300);
  ctx.setLineDash([8, 6]);
  ctx.beginPath();
  ctx.ellipse(ellipse.cx, ellipse.cy, ellipse.rx, ellipse.ry, 0, 0, 2 * Math.PI);
  ctx.stroke();
}

function updateEllipseCanvas() {
  const canvas = document.getElementById('frame-canvas');
  if (!canvas) {
    return;
  }
  const img = document.getElementById('frame-image');
  if (!img || img.naturalWidth === 0) {
    return;
  }
  const rect = img.getBoundingClientRect();
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;
  drawEllipseOnCanvas(canvas);
}

function pointOnCanvas(event, canvas) {
  const rect = canvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) * (canvas.width / rect.width);
  const y = (event.clientY - rect.top) * (canvas.height / rect.height);
  return {
    x: Math.max(0, Math.min(canvas.width, x)),
    y: Math.max(0, Math.min(canvas.height, y)),
  };
}

function initFrameCanvas() {
  const img = document.getElementById('frame-image');
  const canvas = document.getElementById('frame-canvas');
  if (!img || !canvas) {
    return;
  }
  if (img.naturalWidth === 0) {
    img.onload = () => initFrameCanvas();
    return;
  }
  const rect = img.getBoundingClientRect();
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;
  canvas.addEventListener('mousedown', (event) => {
    if (event.button !== 0) {
      return;
    }
    dragActive = true;
    dragStart = pointOnCanvas(event, canvas);
    setEllipseParams(dragStart, dragStart);
    updateEllipseCanvas();
  });
  canvas.addEventListener('mousemove', (event) => {
    if (!dragActive || !dragStart) {
      return;
    }
    const current = pointOnCanvas(event, canvas);
    setEllipseParams(dragStart, current);
    updateEllipseCanvas();
  });
  canvas.addEventListener('mouseup', (event) => {
    if (!dragActive || !dragStart) {
      return;
    }
    dragActive = false;
    const current = pointOnCanvas(event, canvas);
    setEllipseParams(dragStart, current);
    updateEllipseCanvas();
    queueRefresh();
  });
  canvas.addEventListener('mouseleave', () => {
    if (!dragActive) {
      return;
    }
    dragActive = false;
    updateEllipseCanvas();
    queueRefresh();
  });
  canvas.addEventListener('contextmenu', (event) => {
    event.preventDefault();
    clearEllipse();
  });
  updateEllipseCanvas();
}

async function autoTune() {
  if (!ellipse) {
    alert('Draw an ellipse first before running auto-tune.');
    return;
  }
  const response = await fetch('/api/auto_tune', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({settings: state, ellipse}),
  });
  const payload = await response.json();
  if (!payload.ok) {
    alert(payload.error || 'Auto-tune failed');
    return;
  }
  state = payload.settings;
  document.getElementById('settingsJson').textContent = JSON.stringify(state, null, 2);
  document.getElementById('statusMessage').textContent = payload.status || (payload.score !== undefined ? `Auto-tune completed (score=${payload.score.toFixed(4)})` : 'Auto-tune completed');
  buildControls();
  renderImages(payload.images);
  renderStats(payload.stats);
}

async function guidedMask() {
  if (!ellipse) {
    alert('Draw an ellipse first before running guided mask.');
    return;
  }
  document.getElementById('statusMessage').textContent = 'Running guided mask...';
  const response = await fetch('/api/guided_mask', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({settings: state, ellipse}),
  });
  const payload = await response.json();
  if (!payload.ok) {
    alert(payload.error || 'Guided mask failed');
    return;
  }
  document.getElementById('statusMessage').textContent = payload.status || 'Guided mask completed';
  // show results
  renderImages(payload.images);
  renderStats(payload.stats);
}

renderCaseMeta();
buildControls();
document.getElementById('settingsJson').textContent = JSON.stringify(state, null, 2);
refreshPreview();
</script>
</body>
</html>
        """,
        specs=TRACKBAR_SPECS,
        defaults=get_mask_tuning_defaults(),
        case_meta={
            "case_dir": CASE_DATA["case_dir"],
            "image_path": CASE_DATA["image_path"],
            "json_path": CASE_DATA["json_path"],
            "segment_source": CASE_DATA["segment_source"],
        },
    )


@app.post("/api/preview")
def api_preview():
    payload = request.get_json(force=True) or {}
    settings = sanitize_settings(payload.get("settings") or {})
    ellipse = payload.get("ellipse")
    ellipse_params = None
    if isinstance(ellipse, dict):
        ellipse_params = (
            (int(ellipse.get("cx", 0)), int(ellipse.get("cy", 0))),
            (int(ellipse.get("rx", 0)), int(ellipse.get("ry", 0))),
            0.0,
        )
    images, stats = render_preview(CASE_DATA, settings, ellipse_params=ellipse_params)
    return jsonify({"images": images, "stats": stats, "settings": settings})


@app.post("/api/auto_tune")
def api_auto_tune():
    payload = request.get_json(force=True) or {}
    settings = sanitize_settings(payload.get("settings") or {})
    ellipse = payload.get("ellipse")
    if not isinstance(ellipse, dict):
        return jsonify({"ok": False, "error": "Ellipse data required for auto-tune."}), 400
    ellipse_params = (
        (int(ellipse.get("cx", 0)), int(ellipse.get("cy", 0))),
        (int(ellipse.get("rx", 0)), int(ellipse.get("ry", 0))),
        0.0,
    )
    rescaled_ellipse = rescale_ellipse_params(ellipse_params, CASE_DATA["frame"].shape[:2], (PANEL_W, PANEL_H))
    status = None
    if rescaled_ellipse is None:
        status = "Ellipse could not be rescaled from panel to frame space; running search with the raw ellipse coordinates."
        rescaled_ellipse = ellipse_params
    tuned_settings, score = search_mask_parameters(CASE_DATA, settings, rescaled_ellipse)
    if tuned_settings == settings:
        status = status or f"Auto-tune completed; no improvement found (score={score:.4f})."
    else:
        status = status or f"Auto-tune completed; score={score:.4f}."
    images, stats = render_preview(CASE_DATA, tuned_settings, ellipse_params=ellipse_params)
    return jsonify({"ok": True, "status": status, "settings": tuned_settings, "score": score, "images": images, "stats": stats})


@app.post('/api/guided_mask')
def api_guided_mask():
    payload = request.get_json(force=True) or {}
    settings = sanitize_settings(payload.get('settings') or {})
    ellipse = payload.get('ellipse')
    if not isinstance(ellipse, dict):
        return jsonify({'ok': False, 'error': 'Ellipse data required for guided mask.'}), 400

    ellipse_params = (
        (int(ellipse.get('cx', 0)), int(ellipse.get('cy', 0))),
        (int(ellipse.get('rx', 0)), int(ellipse.get('ry', 0))),
        0.0,
    )
    rescaled = rescale_ellipse_params(ellipse_params, CASE_DATA['frame'].shape[:2], (PANEL_W, PANEL_H))
    if rescaled is None:
        rescaled = ellipse_params

    guided_mask, debug = find_last_loop_mask(CASE_DATA, settings=settings, ellipse_params=rescaled, return_debug=True)
    if guided_mask is None:
        return jsonify({'ok': False, 'error': 'No guided mask found.'}), 200

    overlay = CASE_DATA['frame'].copy()
    selected_ellipse = debug.get('best_ellipse')
    if selected_ellipse is not None:
        center, axes, angle = selected_ellipse
        cv2.ellipse(overlay, center, axes, angle, 0, 360, (255, 0, 255), 4)
    overlay[guided_mask > 0] = (255, 0, 255)

    panel_guide = project_ellipse_params(rescaled, CASE_DATA['frame'].shape[:2], (PANEL_W, PANEL_H))
    frame_panel = fit_panel_image(overlay)
    if panel_guide is not None:
        center, axes, angle = panel_guide
        cv2.ellipse(frame_panel, center, axes, angle, 0, 360, ELLIPSE_COLOR, ELLIPSE_THICKNESS)

    panels = {
        'frame': label_panel(frame_panel, 'Frame + Guided Mask'),
        'guided_mask': label_panel(fit_panel_image(mask_to_bgr(guided_mask)), 'Observed Loop'),
        'selected_ring': label_panel(fit_panel_image(mask_to_bgr(debug['selected_loop_ring_mask'])), 'Selected Ring'),
        'selected_fill': label_panel(fit_panel_image(mask_to_bgr(debug['selected_loop_fill_mask'])), 'Selected Fill'),
    }
    images = {name: encode_image(img, ext='.jpg') for name, img in panels.items()}
    metrics = debug.get('best_metrics') or {}
    stats = {
        'guided_px': int(np.count_nonzero(guided_mask)),
        'candidates_scored': int(debug.get('candidates_scored', 0)),
    }
    if metrics:
        stats.update({
            'score': round(float(metrics.get('score', 0.0)), 4),
            'ring_coverage': round(float(metrics.get('ring_coverage', 0.0)), 4),
            'polyline_dist': round(float(metrics.get('polyline_distance') or 0.0), 2),
        })
    status = f'Guided mask found ({stats["guided_px"]} px)'
    return jsonify({'ok': True, 'status': status, 'images': images, 'stats': stats})


@app.post("/api/save")
def api_save():
    payload = request.get_json(force=True) or {}
    settings = sanitize_settings(payload.get("settings") or {})
    SETTINGS_SAVE_PATH.write_text(json.dumps(settings, indent=2))
    return jsonify({"ok": True, "saved_to": str(SETTINGS_SAVE_PATH), "settings": settings})


def parse_args():
    parser = argparse.ArgumentParser(description="Browser-based mask tuner for saved output cases.")
    parser.add_argument(
        "case",
        nargs="?",
        default=None,
        help="Case folder, detection JSON, or detection JPG. Defaults to the latest output case.",
    )
    parser.add_argument("--port", type=int, default=8001, help="Port to serve on. Default: 8001")
    return parser.parse_args()


def main():
    global CASE_DATA
    args = parse_args()
    case_path = args.case or str(find_latest_case())
    CASE_DATA = load_case(case_path)

    print(f"Loaded case: {CASE_DATA['case_dir']}")
    print(f"Frame source: {CASE_DATA['image_path']}")
    print(f"Detection JSON: {CASE_DATA['json_path']}")
    print(f"Segment source: {CASE_DATA['segment_source']}")
    print(f"Open http://127.0.0.1:{args.port} in your browser.")

    app.run(host="0.0.0.0", port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
