import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
from flask import Flask, abort, jsonify, render_template_string, request, send_file

from coil_cv import select_final_loop_model
from ellipse_scoring import score_capture_labels


app = Flask(__name__)
CAPTURE_ROOT = Path("output").resolve()
LABEL_REL_PATH = Path("labels") / "true_ellipse.json"
LABEL_FRAME_GLOB = "true_ellipse_frame_*.json"
SCORE_LOG_PATH = Path(__file__).resolve().parent / "score_log.csv"
SCORE_LOG_COLUMNS = [
    "logged_utc",
    "capture_id",
    "method_name",
    "score_status",
    "truth_frame_index",
    "prediction_frame_index",
    "frame_delta",
    "label_count",
    "prediction_json_path",
    "score_0_1",
    "score_0_100",
    "center_error_px",
    "center_score",
    "rx_error_pct",
    "ry_error_pct",
    "axis_score",
    "truth_aspect_ratio",
    "prediction_aspect_ratio",
    "aspect_ratio_score",
    "angle_error_deg",
    "angle_score",
    "angle_weight",
    "boundary_iou",
    "filled_iou",
    "boundary_thickness_px",
    "shape_only",
    "shape_canvas_width",
    "shape_canvas_height",
]


HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Capture Labeler</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101114;
      --panel: #17191d;
      --panel-2: #202329;
      --line: #343943;
      --text: #f2f4f8;
      --muted: #aeb6c4;
      --accent: #f5b642;
      --good: #3ccf91;
      --warn: #f06d5f;
      --blue: #67b7dc;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    button, input, textarea {
      font: inherit;
    }
    button {
      border: 1px solid var(--line);
      background: var(--panel-2);
      color: var(--text);
      padding: 8px 11px;
      border-radius: 6px;
      cursor: pointer;
    }
    button:hover { border-color: var(--accent); }
    button.primary { background: #2d5f78; border-color: #4b91b2; }
    button.warn { background: #63322f; border-color: #94504b; }
    input, textarea {
      width: 100%;
      border: 1px solid var(--line);
      background: #0c0d10;
      color: var(--text);
      border-radius: 6px;
      padding: 7px 9px;
    }
    .app {
      display: grid;
      grid-template-columns: 300px minmax(0, 1fr) 320px;
      min-height: 100vh;
    }
    .sidebar, .tools {
      background: var(--panel);
      border-right: 1px solid var(--line);
      min-height: 100vh;
      overflow: auto;
    }
    .tools { border-right: 0; border-left: 1px solid var(--line); }
    .head {
      padding: 14px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      gap: 8px;
      justify-content: space-between;
    }
    h1, h2, h3 { margin: 0; font-size: 15px; letter-spacing: 0; }
    .muted { color: var(--muted); font-size: 12px; }
    .capture-list { padding: 8px; display: grid; gap: 7px; }
    .capture-item {
      width: 100%;
      text-align: left;
      display: grid;
      gap: 4px;
      background: #121418;
    }
    .capture-item.active { border-color: var(--accent); }
    .capture-top { display: flex; justify-content: space-between; gap: 8px; align-items: center; }
    .badge { font-size: 11px; color: #0b0d10; border-radius: 999px; padding: 2px 7px; background: var(--muted); white-space: nowrap; }
    .badge.done { background: var(--good); }
    .main {
      min-width: 0;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      min-height: 100vh;
    }
    .toolbar {
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .stage-wrap {
      min-height: 0;
      overflow: auto;
      padding: 14px;
      display: grid;
      align-items: start;
      justify-items: center;
      background: #090a0c;
    }
    .stage {
      position: relative;
      max-width: 100%;
      line-height: 0;
      box-shadow: 0 0 0 1px #000, 0 18px 60px rgba(0,0,0,0.35);
      background: #000;
    }
    .stage img {
      display: block;
      max-width: min(100%, 1400px);
      max-height: calc(100vh - 230px);
      width: auto;
      height: auto;
    }
    .stage canvas {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      touch-action: none;
      cursor: crosshair;
    }
    .film {
      border-top: 1px solid var(--line);
      background: var(--panel);
      padding: 10px 12px;
      display: grid;
      gap: 9px;
    }
    .range-row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      align-items: center;
    }
    .thumbs {
      display: flex;
      gap: 8px;
      overflow-x: auto;
      padding-bottom: 3px;
    }
    .thumb {
      flex: 0 0 88px;
      height: 56px;
      border: 2px solid transparent;
      background: #050608;
      border-radius: 6px;
      object-fit: cover;
      cursor: pointer;
    }
    .thumb.active { border-color: var(--accent); }
    .tool-section {
      border-bottom: 1px solid var(--line);
      padding: 14px;
      display: grid;
      gap: 10px;
    }
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; }
    label { display: grid; gap: 4px; font-size: 12px; color: var(--muted); }
    .row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    .status { color: var(--muted); font-size: 12px; min-height: 18px; white-space: pre-wrap; }
    .empty {
      display: grid;
      place-items: center;
      min-height: 100vh;
      color: var(--muted);
      text-align: center;
      padding: 24px;
    }
    @media (max-width: 1100px) {
      .app { grid-template-columns: 240px minmax(0, 1fr); }
      .tools { grid-column: 1 / span 2; min-height: auto; border-left: 0; border-top: 1px solid var(--line); }
    }
    @media (max-width: 760px) {
      .app { display: block; }
      .sidebar, .tools { min-height: auto; }
      .stage img { max-height: 58vh; }
    }
  </style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <div class="head">
      <h1>Captures</h1>
      <button id="refreshBtn">Refresh</button>
    </div>
    <div id="captureList" class="capture-list"></div>
  </aside>

  <main class="main">
    <div class="toolbar">
      <h2 id="captureTitle">No capture selected</h2>
      <span id="frameMeta" class="muted"></span>
      <div style="flex:1"></div>
      <button id="drawBtn">New Ellipse</button>
      <button id="perspectiveBtn">Perspective Mode</button>
      <button id="predictionBtn">Use Prediction</button>
      <button id="scoreBtn">Score</button>
      <button id="clearBtn" class="warn">Clear</button>
      <button id="saveBtn" class="primary">Save Label</button>
    </div>
    <div class="stage-wrap">
      <div id="empty" class="empty">No captures found under output/ with manifest.json and frames/.</div>
      <div id="stage" class="stage" style="display:none">
        <img id="frameImage" alt="capture frame" />
        <canvas id="overlay"></canvas>
      </div>
    </div>
    <div class="film" id="film" style="display:none">
      <div class="range-row">
        <input id="frameRange" type="range" min="0" max="0" step="1" value="0" />
        <span id="frameCount" class="muted"></span>
      </div>
      <div id="thumbs" class="thumbs"></div>
    </div>
  </main>

  <aside class="tools">
    <div class="head"><h2>Truth Ellipse</h2></div>
    <div class="tool-section">
      <div class="grid-2">
        <label>Center X<input id="cx" type="number" step="0.1" /></label>
        <label>Center Y<input id="cy" type="number" step="0.1" /></label>
        <label>Radius X<input id="rx" type="number" step="0.1" min="0.1" /></label>
        <label>Radius Y<input id="ry" type="number" step="0.1" min="0.1" /></label>
      </div>
      <label>Rotation<input id="rot" type="number" step="0.1" /></label>
      <div class="grid-3">
        <button id="nudgeLeft">Left</button>
        <button id="nudgeUp">Up</button>
        <button id="nudgeRight">Right</button>
      </div>
      <div class="grid-3">
        <button id="shrink">Shrink</button>
        <button id="nudgeDown">Down</button>
        <button id="grow">Grow</button>
      </div>
    </div>
    <div class="tool-section">
      <label>Notes<textarea id="notes" rows="5"></textarea></label>
      <div id="labelPath" class="muted"></div>
      <div id="status" class="status"></div>
    </div>
  </aside>
</div>

<script>
const $ = (id) => document.getElementById(id);
const state = {
  captures: [],
  capture: null,
  frameIndex: 0,
  ellipse: null,
  perspective: null,
  editMode: 'ellipse',
  scorePrediction: null,
  mode: 'select',
  action: null,
  pointerId: null,
  dragStart: null,
  actionStart: null,
  perspectiveCornerIndex: null,
  label: null,
};

const fields = ['cx', 'cy', 'rx', 'ry', 'rot'];

function setStatus(text) {
  $('status').textContent = text || '';
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || response.statusText);
  }
  return payload;
}

function fmt(value, digits = 1) {
  return Number.isFinite(value) ? value.toFixed(digits) : '';
}

function frameUrl(captureId, frame) {
  const name = frame.path.split('/').pop();
  return `/frames/${encodeURIComponent(captureId)}/${encodeURIComponent(name)}`;
}

async function loadCaptures() {
  state.captures = await fetchJson('/api/captures');
  renderCaptureList();
  if (!state.capture && state.captures.length) {
    await selectCapture(state.captures[0].id);
  }
}

function renderCaptureList() {
  const list = $('captureList');
  list.innerHTML = '';
  state.captures.forEach((capture) => {
    const item = document.createElement('button');
    item.className = 'capture-item' + (state.capture && state.capture.id === capture.id ? ' active' : '');
    item.innerHTML = `
      <div class="capture-top"><strong>${capture.id}</strong><span class="badge ${capture.labeled ? 'done' : ''}">${capture.labeled ? 'labeled' : 'open'}</span></div>
      <div class="muted">${capture.frame_count} frames - ${fmt(capture.segment_duration_s || 0, 2)}s</div>
    `;
    item.addEventListener('click', () => selectCapture(capture.id));
    list.appendChild(item);
  });
}

async function selectCapture(id) {
  state.capture = await fetchJson(`/api/captures/${encodeURIComponent(id)}`);
  state.label = state.capture.label || null;
  $('captureTitle').textContent = state.capture.id;
  $('labelPath').textContent = `${state.capture.id}/labels/true_ellipse.json`;
  $('notes').value = state.label && state.label.notes ? state.label.notes : '';
  state.scorePrediction = null;
  state.perspective = state.label && state.label.perspective ? normalizePerspective(state.label.perspective) : null;
  state.editMode = state.label && state.label.editor_mode === 'perspective' && state.perspective ? 'perspective' : 'ellipse';

  if (state.label && state.label.ellipse) {
    state.ellipse = normalizeEllipse(state.label.ellipse);
    state.frameIndex = Math.min(Math.max(0, state.label.frame.index || 0), state.capture.frames.length - 1);
  } else {
    state.ellipse = null;
    state.frameIndex = suggestedFrameIndex(state.capture);
  }

  $('empty').style.display = 'none';
  $('stage').style.display = '';
  $('film').style.display = '';
  $('frameRange').max = Math.max(0, state.capture.frames.length - 1);
  renderCaptureList();
  renderThumbs();
  setFrame(state.frameIndex);
  syncFields();
  updateModeButtons();
  setStatus(state.label ? 'Loaded saved label.' : 'Ready.');
}

function suggestedFrameIndex(capture) {
  if (capture.prediction && Number.isInteger(capture.prediction.frame_index)) {
    return Math.min(Math.max(0, capture.prediction.frame_index), capture.frames.length - 1);
  }
  return Math.floor((capture.frames.length - 1) / 2);
}

function renderThumbs() {
  const thumbs = $('thumbs');
  thumbs.innerHTML = '';
  state.capture.frames.forEach((frame, index) => {
    const img = document.createElement('img');
    img.className = 'thumb' + (index === state.frameIndex ? ' active' : '');
    img.loading = 'lazy';
    img.src = frameUrl(state.capture.id, frame);
    img.alt = `frame ${index}`;
    img.addEventListener('click', () => setFrame(index));
    thumbs.appendChild(img);
  });
}

function setFrame(index) {
  if (!state.capture) return;
  state.frameIndex = Math.min(Math.max(0, Number(index)), state.capture.frames.length - 1);
  const frame = state.capture.frames[state.frameIndex];
  $('frameImage').src = frameUrl(state.capture.id, frame);
  $('frameRange').value = state.frameIndex;
  $('frameCount').textContent = `${state.frameIndex + 1} / ${state.capture.frames.length}`;
  $('frameMeta').textContent = `Frame ${state.frameIndex} - t=${fmt(frame.t_s || 0, 2)}s - FFT=${fmt(frame.fft_intensity || 0, 2)}`;
  document.querySelectorAll('.thumb').forEach((thumb, i) => thumb.classList.toggle('active', i === state.frameIndex));
  drawOverlay();
}

function normalizeEllipse(raw) {
  return {
    cx: Number(raw.cx || 0),
    cy: Number(raw.cy || 0),
    rx: Math.max(0.1, Number(raw.rx || raw.radius_x || 1)),
    ry: Math.max(0.1, Number(raw.ry || raw.radius_y || 1)),
    rotation_deg: Number(raw.rotation_deg || 0),
  };
}

function cloneEllipse(e) {
  return e ? {cx: e.cx, cy: e.cy, rx: e.rx, ry: e.ry, rotation_deg: e.rotation_deg} : null;
}

function normalizePoint(raw) {
  return {x: Number(raw.x || 0), y: Number(raw.y || 0)};
}

function normalizePerspective(raw) {
  if (!raw || !Array.isArray(raw.corners) || raw.corners.length !== 4 || !raw.base_ellipse) {
    return null;
  }
  return {
    mode: 'homography_quad',
    base_ellipse: normalizeEllipse(raw.base_ellipse),
    corners: raw.corners.map(normalizePoint),
  };
}

function updateModeButtons() {
  const p = $('perspectiveBtn');
  if (p) p.classList.toggle('primary', state.editMode === 'perspective');
}

function ellipseSourceCorners(e) {
  return [
    localToWorld(e, -e.rx, -e.ry),
    localToWorld(e, e.rx, -e.ry),
    localToWorld(e, e.rx, e.ry),
    localToWorld(e, -e.rx, e.ry),
  ];
}

function initPerspectiveFromEllipse() {
  if (!state.ellipse) return null;
  const base = cloneEllipse(state.ellipse);
  const corners = ellipseSourceCorners(base);
  state.perspective = {mode: 'homography_quad', base_ellipse: base, corners};
  return state.perspective;
}

function invalidatePerspectiveFromEllipseEdit() {
  if (state.editMode === 'ellipse') state.perspective = null;
}

function togglePerspectiveMode() {
  if (state.editMode === 'perspective') {
    state.editMode = 'ellipse';
  } else {
    if (!state.perspective) {
      if (!state.ellipse) {
        setStatus('Create or use a prediction ellipse before entering perspective mode.');
        return;
      }
      initPerspectiveFromEllipse();
    }
    state.editMode = 'perspective';
  }
  state.mode = 'select';
  updateModeButtons();
  drawOverlay();
}

function solveLinearSystem(matrix, rhs) {
  const n = rhs.length;
  const a = matrix.map((row, i) => row.concat([rhs[i]]));
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
    }
    if (Math.abs(a[pivot][col]) < 1e-9) return null;
    [a[col], a[pivot]] = [a[pivot], a[col]];
    const div = a[col][col];
    for (let k = col; k <= n; k++) a[col][k] /= div;
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = a[row][col];
      for (let k = col; k <= n; k++) a[row][k] -= factor * a[col][k];
    }
  }
  return a.map(row => row[n]);
}

function homographyFromPoints(src, dst) {
  const matrix = [];
  const rhs = [];
  for (let i = 0; i < 4; i++) {
    const x = src[i].x;
    const y = src[i].y;
    const u = dst[i].x;
    const v = dst[i].y;
    matrix.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
    rhs.push(u);
    matrix.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
    rhs.push(v);
  }
  const h = solveLinearSystem(matrix, rhs);
  if (!h) return null;
  return [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1];
}

function transformHomography(h, p) {
  const den = h[6] * p.x + h[7] * p.y + h[8];
  if (Math.abs(den) < 1e-9) return {x: p.x, y: p.y};
  return {
    x: (h[0] * p.x + h[1] * p.y + h[2]) / den,
    y: (h[3] * p.x + h[4] * p.y + h[5]) / den,
  };
}

function perspectiveEllipsePoints(perspective, samples = 160) {
  if (!perspective || !perspective.base_ellipse || !perspective.corners) return [];
  const src = ellipseSourceCorners(perspective.base_ellipse);
  const h = homographyFromPoints(src, perspective.corners);
  if (!h) return [];
  const e = perspective.base_ellipse;
  const pts = [];
  for (let i = 0; i <= samples; i++) {
    const t = (i / samples) * Math.PI * 2;
    const p = localToWorld(e, Math.cos(t) * e.rx, Math.sin(t) * e.ry);
    pts.push(transformHomography(h, p));
  }
  return pts;
}

function fitEllipseFromPoints(points) {
  if (!points || points.length < 8) return null;
  const cx = points.reduce((sum, p) => sum + p.x, 0) / points.length;
  const cy = points.reduce((sum, p) => sum + p.y, 0) / points.length;
  let sxx = 0, syy = 0, sxy = 0;
  points.forEach((p) => {
    const dx = p.x - cx;
    const dy = p.y - cy;
    sxx += dx * dx;
    syy += dy * dy;
    sxy += dx * dy;
  });
  sxx /= points.length;
  syy /= points.length;
  sxy /= points.length;
  const trace = sxx + syy;
  const diff = sxx - syy;
  const root = Math.sqrt(diff * diff + 4 * sxy * sxy);
  const l1 = Math.max(1, (trace + root) / 2);
  const l2 = Math.max(1, (trace - root) / 2);
  const angle = deg(0.5 * Math.atan2(2 * sxy, diff));
  return {
    cx,
    cy,
    rx: Math.sqrt(2 * l1),
    ry: Math.sqrt(2 * l2),
    rotation_deg: angle,
  };
}

function currentSaveEllipse() {
  if (state.editMode === 'perspective' && state.perspective) {
    const fitted = fitEllipseFromPoints(perspectiveEllipsePoints(state.perspective));
    if (fitted) return fitted;
  }
  return state.ellipse;
}

function serializePerspective() {
  if (state.editMode !== 'perspective' || !state.perspective) return null;
  return {
    mode: 'homography_quad',
    base_ellipse: cloneEllipse(state.perspective.base_ellipse),
    corners: state.perspective.corners.map(p => ({x: p.x, y: p.y})),
  };
}


function usePrediction() {
  if (!state.capture || !state.capture.prediction || !state.capture.prediction.ellipse) {
    setStatus('No model prediction found for this capture.');
    return;
  }
  state.ellipse = normalizeEllipse(state.capture.prediction.ellipse);
  state.perspective = null;
  state.editMode = 'ellipse';
  updateModeButtons();
  state.scorePrediction = null;
  if (Number.isInteger(state.capture.prediction.frame_index)) {
    setFrame(state.capture.prediction.frame_index);
  }
  syncFields();
  drawOverlay();
  setStatus('Prediction copied into the editable truth ellipse.');
}

function clearEllipse() {
  state.ellipse = null;
  state.perspective = null;
  state.editMode = 'ellipse';
  updateModeButtons();
  state.scorePrediction = null;
  syncFields();
  drawOverlay();
}

function syncFields() {
  const e = state.ellipse;
  $('cx').value = e ? fmt(e.cx) : '';
  $('cy').value = e ? fmt(e.cy) : '';
  $('rx').value = e ? fmt(e.rx) : '';
  $('ry').value = e ? fmt(e.ry) : '';
  $('rot').value = e ? fmt(e.rotation_deg) : '';
}

function readFields() {
  if (!state.ellipse) return;
  invalidatePerspectiveFromEllipseEdit();
  state.ellipse.cx = Number($('cx').value || 0);
  state.ellipse.cy = Number($('cy').value || 0);
  state.ellipse.rx = Math.max(0.1, Number($('rx').value || 0.1));
  state.ellipse.ry = Math.max(0.1, Number($('ry').value || 0.1));
  state.ellipse.rotation_deg = Number($('rot').value || 0);
  drawOverlay();
}

function imageScale() {
  const img = $('frameImage');
  const rect = img.getBoundingClientRect();
  return rect.width > 0 ? img.naturalWidth / rect.width : 1;
}

function pointerPoint(event) {
  const img = $('frameImage');
  const rect = img.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * (img.naturalWidth / rect.width),
    y: (event.clientY - rect.top) * (img.naturalHeight / rect.height),
  };
}

function resizeCanvas() {
  const img = $('frameImage');
  const canvas = $('overlay');
  if (!img.naturalWidth) return;
  const rect = img.getBoundingClientRect();
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;
  drawOverlay();
}

function rad(deg) { return deg * Math.PI / 180; }
function deg(radValue) { return radValue * 180 / Math.PI; }

function localToWorld(e, x, y) {
  const a = rad(e.rotation_deg);
  return {
    x: e.cx + x * Math.cos(a) - y * Math.sin(a),
    y: e.cy + x * Math.sin(a) + y * Math.cos(a),
  };
}

function worldToLocal(e, x, y) {
  const a = rad(-e.rotation_deg);
  const dx = x - e.cx;
  const dy = y - e.cy;
  return {
    x: dx * Math.cos(a) - dy * Math.sin(a),
    y: dx * Math.sin(a) + dy * Math.cos(a),
  };
}

function handles(e) {
  return [
    {name: 'rx+', kind: 'resize-x', ...localToWorld(e, e.rx, 0)},
    {name: 'rx-', kind: 'resize-x', ...localToWorld(e, -e.rx, 0)},
    {name: 'ry+', kind: 'resize-y', ...localToWorld(e, 0, e.ry)},
    {name: 'ry-', kind: 'resize-y', ...localToWorld(e, 0, -e.ry)},
    {name: 'rotate', kind: 'rotate', ...localToWorld(e, 0, -e.ry - 48)},
  ];
}

function drawEllipseStroke(ctx, ellipse, color, dash, lineWidth, alpha = 1.0) {
  if (!ellipse) return;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.translate(ellipse.cx, ellipse.cy);
  ctx.rotate(rad(ellipse.rotation_deg));
  ctx.lineWidth = lineWidth;
  ctx.strokeStyle = color;
  ctx.setLineDash(dash || []);
  ctx.beginPath();
  ctx.ellipse(0, 0, ellipse.rx, ellipse.ry, 0, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

function drawPathStroke(ctx, points, color, dash, lineWidth, alpha = 1.0) {
  if (!points || points.length < 2) return;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.lineWidth = lineWidth;
  ctx.strokeStyle = color;
  ctx.setLineDash(dash || []);
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y);
  ctx.stroke();
  ctx.restore();
}

function perspectiveCornerHandles() {
  if (!state.perspective) return [];
  return state.perspective.corners.map((p, index) => ({index, x: p.x, y: p.y}));
}

function drawPerspectiveOverlay(ctx) {
  const p = state.perspective;
  if (!p) return;
  const canvas = $('overlay');
  const lineWidth = Math.max(2, canvas.width / 800);
  const handleRadius = 7 * imageScale();
  drawPathStroke(ctx, perspectiveEllipsePoints(p), '#f5b642', [], Math.max(3, canvas.width / 650));
  drawPathStroke(ctx, [...p.corners, p.corners[0]], '#ff7ad9', [10, 7], lineWidth, 0.95);
  perspectiveCornerHandles().forEach((h) => {
    ctx.beginPath();
    ctx.fillStyle = '#ff7ad9';
    ctx.strokeStyle = '#0c0d10';
    ctx.lineWidth = 2;
    ctx.arc(h.x, h.y, handleRadius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  });
}

function drawOverlay() {
  const canvas = $('overlay');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (state.scorePrediction && state.scorePrediction.frame_index === state.frameIndex) {
    drawEllipseStroke(ctx, state.scorePrediction.ellipse, '#48ff8a', [], Math.max(3, canvas.width / 650), 0.95);
  }

  if (state.editMode === 'perspective') {
    if (!state.perspective && state.ellipse) initPerspectiveFromEllipse();
    drawPerspectiveOverlay(ctx);
    return;
  }

  if (!state.ellipse) return;
  const e = state.ellipse;
  drawEllipseStroke(ctx, e, '#f5b642', [12, 8], Math.max(2, canvas.width / 800));

  ctx.save();
  ctx.translate(e.cx, e.cy);
  ctx.rotate(rad(e.rotation_deg));
  ctx.strokeStyle = '#67b7dc';
  ctx.lineWidth = Math.max(2, canvas.width / 800);
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(e.rx, 0);
  ctx.moveTo(0, 0);
  ctx.lineTo(0, -e.ry);
  ctx.stroke();
  ctx.restore();

  handles(e).forEach((h) => {
    ctx.beginPath();
    ctx.fillStyle = h.kind === 'rotate' ? '#67b7dc' : '#f2f4f8';
    ctx.strokeStyle = '#0c0d10';
    ctx.lineWidth = 2;
    ctx.arc(h.x, h.y, 6 * imageScale(), 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  });
}

function hitTest(point) {
  if (state.editMode === 'perspective') {
    if (!state.perspective) return null;
    const threshold = 13 * imageScale();
    for (const h of perspectiveCornerHandles()) {
      if (Math.hypot(point.x - h.x, point.y - h.y) <= threshold) {
        return {type: 'perspective-corner', index: h.index};
      }
    }
    return null;
  }
  if (!state.ellipse) return null;
  const e = state.ellipse;
  const threshold = 12 * imageScale();
  for (const h of handles(e)) {
    if (Math.hypot(point.x - h.x, point.y - h.y) <= threshold) {
      return {type: h.kind, handle: h.name};
    }
  }
  const local = worldToLocal(e, point.x, point.y);
  const value = (local.x * local.x) / (e.rx * e.rx) + (local.y * local.y) / (e.ry * e.ry);
  if (value <= 1.0) return {type: 'move'};
  return null;
}

function canvasDown(event) {
  if (!state.capture || event.button !== 0) return;
  const canvas = $('overlay');
  canvas.setPointerCapture(event.pointerId);
  const point = pointerPoint(event);
  state.pointerId = event.pointerId;
  state.dragStart = point;

  if (state.editMode === 'perspective') {
    if (!state.perspective) initPerspectiveFromEllipse();
    const hit = hitTest(point);
    if (!hit) {
      state.action = null;
      canvas.releasePointerCapture(event.pointerId);
      state.pointerId = null;
      drawOverlay();
      return;
    }
    state.action = hit.type;
    state.perspectiveCornerIndex = hit.index;
    state.actionStart = JSON.parse(JSON.stringify(state.perspective));
    drawOverlay();
    return;
  }

  if (state.mode === 'draw' || !state.ellipse) {
    invalidatePerspectiveFromEllipseEdit();
    state.action = 'draw';
    state.ellipse = {cx: point.x, cy: point.y, rx: 1, ry: 1, rotation_deg: 0};
  } else {
    const hit = hitTest(point);
    state.action = hit ? hit.type : 'draw';
    if (state.action === 'draw') {
      invalidatePerspectiveFromEllipseEdit();
      state.ellipse = {cx: point.x, cy: point.y, rx: 1, ry: 1, rotation_deg: 0};
    }
  }

  state.actionStart = JSON.parse(JSON.stringify(state.ellipse));
  drawOverlay();
}

function canvasMove(event) {
  if (state.pointerId !== event.pointerId || !state.action) return;
  const point = pointerPoint(event);
  const start = state.actionStart;
  const drag = state.dragStart;
  if (state.action === 'perspective-corner') {
    if (!state.perspective || state.perspectiveCornerIndex === null) return;
    state.perspective.corners[state.perspectiveCornerIndex] = {x: point.x, y: point.y};
    const fitted = currentSaveEllipse();
    if (fitted) state.ellipse = fitted;
    syncFields();
    drawOverlay();
    return;
  }
  if (!state.ellipse) return;
  invalidatePerspectiveFromEllipseEdit();
  if (state.action === 'draw') {
    state.ellipse.cx = (drag.x + point.x) / 2;
    state.ellipse.cy = (drag.y + point.y) / 2;
    state.ellipse.rx = Math.max(1, Math.abs(point.x - drag.x) / 2);
    state.ellipse.ry = Math.max(1, Math.abs(point.y - drag.y) / 2);
    state.ellipse.rotation_deg = 0;
  } else if (state.action === 'move') {
    state.ellipse.cx = start.cx + (point.x - drag.x);
    state.ellipse.cy = start.cy + (point.y - drag.y);
  } else if (state.action === 'resize-x') {
    const local = worldToLocal(start, point.x, point.y);
    state.ellipse.rx = Math.max(1, Math.abs(local.x));
  } else if (state.action === 'resize-y') {
    const local = worldToLocal(start, point.x, point.y);
    state.ellipse.ry = Math.max(1, Math.abs(local.y));
  } else if (state.action === 'rotate') {
    state.ellipse.rotation_deg = deg(Math.atan2(point.y - start.cy, point.x - start.cx)) + 90;
  }
  syncFields();
  drawOverlay();
}

function canvasUp(event) {
  if (state.pointerId !== event.pointerId) return;
  state.pointerId = null;
  state.action = null;
  state.perspectiveCornerIndex = null;
  state.mode = 'select';
  $('drawBtn').classList.remove('primary');
  syncFields();
  drawOverlay();
}

function nudge(dx, dy) {
  if (!state.ellipse) return;
  invalidatePerspectiveFromEllipseEdit();
  state.ellipse.cx += dx;
  state.ellipse.cy += dy;
  syncFields();
  drawOverlay();
}

function scaleEllipse(amount) {
  if (!state.ellipse) return;
  invalidatePerspectiveFromEllipseEdit();
  state.ellipse.rx = Math.max(1, state.ellipse.rx + amount);
  state.ellipse.ry = Math.max(1, state.ellipse.ry + amount);
  syncFields();
  drawOverlay();
}

async function saveLabel() {
  const ellipse = currentSaveEllipse();
  if (!state.capture || !ellipse) {
    setStatus('Create an ellipse before saving.');
    return;
  }
  const img = $('frameImage');
  const frame = state.capture.frames[state.frameIndex];
  const payload = {
    frame_index: state.frameIndex,
    frame_path: frame.path,
    frame_t_s: frame.t_s,
    frame_fft_intensity: frame.fft_intensity,
    image_width: img.naturalWidth,
    image_height: img.naturalHeight,
    ellipse,
    editor_mode: state.editMode,
    perspective: serializePerspective(),
    notes: $('notes').value || '',
  };
  const saved = await fetchJson(`/api/captures/${encodeURIComponent(state.capture.id)}/label`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  });
  state.label = saved.label;
  setStatus(`Saved ${saved.path}`);
  await loadCaptures();
}

async function scoreCapture() {
  if (!state.capture) {
    setStatus('Select a capture before scoring.');
    return;
  }
  const payload = await fetchJson(`/api/captures/${encodeURIComponent(state.capture.id)}/score`, {method: 'POST'});
  if (payload.prediction && payload.prediction.ellipse && Number.isInteger(payload.prediction.frame_index)) {
    state.scorePrediction = {
      frame_index: payload.prediction.frame_index,
      ellipse: normalizeEllipse(payload.prediction.ellipse),
    };
    setFrame(payload.prediction.frame_index);
  }

  if (!payload.ok) {
    setStatus(payload.error || 'Unable to score this capture.');
    drawOverlay();
    return;
  }

  const m = payload.metrics;
  const isShapeOnly = payload.score_status === 'scored_shape_only_closest_frame' || m.shape_only;
  const lines = isShapeOnly ? [
    `Partial shape score: ${m.score_0_100.toFixed(2)} / 100`,
    `Truth frame: ${payload.truth_frame_index}`,
    `Prediction frame: ${payload.prediction_frame_index}`,
    `Frame delta: ${payload.frame_delta}`,
    `Boundary IoU (center-aligned): ${m.boundary_iou.toFixed(4)}`,
    `Filled IoU (center-aligned): ${m.filled_iou.toFixed(4)}`,
    `Axis score: ${m.axis_score.toFixed(4)}`,
    `Aspect score: ${m.aspect_ratio_score.toFixed(4)}`,
    `Angle error: ${m.angle_error_deg.toFixed(2)} deg`,
    `Method: ${payload.prediction_method || 'unknown'}`,
  ] : [
    `Score: ${m.score_0_100.toFixed(2)} / 100`,
    `Frame: ${payload.frame_index}`,
    `Boundary IoU: ${m.boundary_iou.toFixed(4)}`,
    `Filled IoU: ${m.filled_iou.toFixed(4)}`,
    `Center error: ${m.center_error_px.toFixed(2)} px`,
    `Axis score: ${m.axis_score.toFixed(4)}`,
    `Aspect score: ${m.aspect_ratio_score.toFixed(4)}`,
    `Angle error: ${m.angle_error_deg.toFixed(2)} deg`,
    `Method: ${payload.prediction_method || 'unknown'}`,
  ];
  if (payload.score_log) {
    if (payload.score_log.duplicate) {
      lines.push('Score log: duplicate skipped');
    } else if (payload.score_log.saved) {
      lines.push(`Score log: saved to ${payload.score_log.path}`);
    }
  }
  setStatus(lines.join('\n'));
  drawOverlay();
}


$('refreshBtn').addEventListener('click', loadCaptures);
$('frameRange').addEventListener('input', (event) => setFrame(Number(event.target.value)));
$('frameImage').addEventListener('load', resizeCanvas);
window.addEventListener('resize', resizeCanvas);
$('overlay').addEventListener('pointerdown', canvasDown);
$('overlay').addEventListener('pointermove', canvasMove);
$('overlay').addEventListener('pointerup', canvasUp);
$('overlay').addEventListener('pointercancel', canvasUp);
$('drawBtn').addEventListener('click', () => {
  state.editMode = 'ellipse';
  state.perspective = null;
  updateModeButtons();
  state.mode = 'draw';
  $('drawBtn').classList.add('primary');
});
$('perspectiveBtn').addEventListener('click', togglePerspectiveMode);
$('predictionBtn').addEventListener('click', usePrediction);
$('scoreBtn').addEventListener('click', scoreCapture);
$('clearBtn').addEventListener('click', clearEllipse);
$('saveBtn').addEventListener('click', saveLabel);
fields.forEach((id) => $(id).addEventListener('input', readFields));
$('nudgeLeft').addEventListener('click', () => nudge(-1, 0));
$('nudgeRight').addEventListener('click', () => nudge(1, 0));
$('nudgeUp').addEventListener('click', () => nudge(0, -1));
$('nudgeDown').addEventListener('click', () => nudge(0, 1));
$('grow').addEventListener('click', () => scaleEllipse(1));
$('shrink').addEventListener('click', () => scaleEllipse(-1));
document.addEventListener('keydown', (event) => {
  if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') return;
  const step = event.shiftKey ? 10 : 1;
  if (event.key === 'ArrowLeft') nudge(-step, 0);
  if (event.key === 'ArrowRight') nudge(step, 0);
  if (event.key === 'ArrowUp') nudge(0, -step);
  if (event.key === 'ArrowDown') nudge(0, step);
});

loadCaptures().catch((error) => setStatus(error.message));
</script>
</body>
</html>
"""


def capture_path(capture_id):
    path = (CAPTURE_ROOT / capture_id).resolve()
    if CAPTURE_ROOT not in path.parents and path != CAPTURE_ROOT:
        abort(404)
    if not path.is_dir():
        abort(404)
    return path


def load_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def frame_label_path(capture_dir, frame_index):
    return capture_dir / "labels" / f"true_ellipse_frame_{int(frame_index):06d}.json"


def label_frame_index(label):
    if not isinstance(label, dict):
        return None
    frame = label.get("frame") or {}
    index = frame.get("index")
    if index is None:
        return None
    try:
        return int(index)
    except (TypeError, ValueError):
        return None


def labels_from_payload(payload):
    if payload is None:
        return []
    if isinstance(payload, list):
        candidates = payload
    elif isinstance(payload, dict) and isinstance(payload.get("labels"), list):
        candidates = payload.get("labels")
    elif isinstance(payload, dict):
        candidates = [payload]
    else:
        candidates = []
    return [label for label in candidates if isinstance(label, dict) and label.get("ellipse") is not None]


def load_truth_labels(capture_dir):
    labels_by_frame = {}
    for path in sorted((capture_dir / "labels").glob(LABEL_FRAME_GLOB)):
        for label in labels_from_payload(load_json(path)):
            index = label_frame_index(label)
            if index is not None:
                labels_by_frame[index] = label
    for label in labels_from_payload(load_json(capture_dir / LABEL_REL_PATH)):
        index = label_frame_index(label)
        if index is not None:
            labels_by_frame[index] = label
    return [labels_by_frame[index] for index in sorted(labels_by_frame)]


def load_manifest(capture_dir):
    manifest = load_json(capture_dir / "manifest.json")
    if manifest is None:
        return None
    frames_dir = capture_dir / "frames"
    if not frames_dir.is_dir():
        return None
    frames = manifest.get("frames") or []
    if not frames:
        return None
    return manifest


def detection_info(capture_dir):
    json_files = sorted(capture_dir.glob("tail_detected_*.json"))
    if not json_files:
        return None
    data = load_json(json_files[-1]) or {}
    fit = data.get("final_loop_fit") or {}
    diagnostics = data.get("loop_fit_diagnostics") or {}
    method_name = data.get("method") or fit.get("method") or diagnostics.get("method")
    if not method_name:
        method_name = datetime.now(timezone.utc).date().isoformat()
    center = fit.get("ellipse_center")
    axes = fit.get("ellipse_axes_semi")
    if center is None or axes is None:
        ellipse = None
    else:
        ellipse = {
            "cx": float(center[0]),
            "cy": float(center[1]),
            "rx": float(axes[0]),
            "ry": float(axes[1]),
            "rotation_deg": float(fit.get("ellipse_rotation_deg") or 0.0),
        }
    return {
        "json_path": str(json_files[-1]),
        "frame_index": diagnostics.get("best_frame_index"),
        "ellipse": ellipse,
        "method": method_name,
    }


def recompute_detection_info(capture_dir):
    """Run the current CV model from a capture's saved frame and tail geometry."""
    json_files = sorted(capture_dir.glob("tail_detected_*.json"))
    if not json_files:
        return None

    prediction_path = json_files[-1]
    data = load_json(prediction_path) or {}
    diagnostics = data.get("loop_fit_diagnostics") or {}
    manifest = load_manifest(capture_dir)
    frames = [] if manifest is None else (manifest.get("frames") or [])
    frame_index = diagnostics.get("best_frame_index")
    if frame_index is None or not frames:
        return None

    frame_index = int(frame_index)
    frame_record = next((frame for frame in frames if int(frame.get("index", -1)) == frame_index), None)
    if frame_record is None:
        return None

    frame_path = (capture_dir / frame_record.get("path", "")).resolve()
    if capture_dir.resolve() not in frame_path.parents or not frame_path.is_file():
        return None
    frame = cv2.imread(str(frame_path))
    if frame is None:
        return None

    segment_xy = data.get("segment_xy")
    tail_tip = data.get("tail_tip")
    if not segment_xy or tail_tip is None:
        return None

    ellipse_info, metrics, _ = select_final_loop_model(
        frame,
        segment_xy=segment_xy,
        tail_tip=tail_tip,
        tail_base=data.get("tail_base"),
        bbox=data.get("bbox"),
    )
    if ellipse_info is None:
        return {
            "json_path": f"{prediction_path} (recomputed)",
            "frame_index": frame_index,
            "ellipse": None,
            "method": None,
            "metrics": metrics,
            "source": "current_cv_model",
        }

    center = ellipse_info.get("center")
    axes = ellipse_info.get("axes")
    return {
        "json_path": f"{prediction_path} (recomputed)",
        "frame_index": frame_index,
        "ellipse": {
            "cx": float(center[0]),
            "cy": float(center[1]),
            "rx": float(axes[0]),
            "ry": float(axes[1]),
            "rotation_deg": float(ellipse_info.get("rotation_deg") or 0.0),
        },
        "method": ellipse_info.get("method"),
        "metrics": metrics,
        "source": "current_cv_model",
    }


def csv_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def score_log_row(capture_id, result):
    metrics = result.get("metrics") or {}
    row = {column: "" for column in SCORE_LOG_COLUMNS}
    row.update({
        "logged_utc": datetime.now(timezone.utc).isoformat(),
        "capture_id": capture_id,
        "method_name": result.get("prediction_method") or datetime.now(timezone.utc).date().isoformat(),
        "score_status": result.get("score_status"),
        "truth_frame_index": result.get("truth_frame_index"),
        "prediction_frame_index": result.get("prediction_frame_index"),
        "frame_delta": result.get("frame_delta"),
        "label_count": result.get("label_count"),
        "prediction_json_path": result.get("prediction_json_path"),
    })
    for key, value in metrics.items():
        if key in row:
            row[key] = value
    return {key: csv_value(row.get(key)) for key in SCORE_LOG_COLUMNS}


def score_log_duplicate(existing_row, new_row):
    return all(
        csv_value(existing_row.get(column)) == csv_value(new_row.get(column))
        for column in SCORE_LOG_COLUMNS
        if column != "logged_utc"
    )


def append_score_log(capture_id, result):
    if not result.get("ok") or not result.get("metrics"):
        return {"saved": False, "duplicate": False, "path": str(SCORE_LOG_PATH)}

    row = score_log_row(capture_id, result)
    existing_rows = []
    if SCORE_LOG_PATH.exists():
        with SCORE_LOG_PATH.open("r", newline="") as handle:
            existing_rows = list(csv.DictReader(handle))
    if any(score_log_duplicate(existing, row) for existing in existing_rows):
        return {"saved": False, "duplicate": True, "path": str(SCORE_LOG_PATH)}

    write_header = not SCORE_LOG_PATH.exists() or SCORE_LOG_PATH.stat().st_size == 0
    with SCORE_LOG_PATH.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORE_LOG_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return {"saved": True, "duplicate": False, "path": str(SCORE_LOG_PATH)}


def capture_summary(capture_dir, manifest):
    label_path = capture_dir / LABEL_REL_PATH
    return {
        "id": capture_dir.name,
        "frame_count": int(manifest.get("frame_count") or len(manifest.get("frames") or [])),
        "segment_start_s": manifest.get("segment_start_s"),
        "segment_end_s": manifest.get("segment_end_s"),
        "segment_duration_s": manifest.get("segment_duration_s"),
        "labeled": label_path.exists(),
    }


def all_captures():
    if not CAPTURE_ROOT.exists():
        return []
    captures = []
    for capture_dir in sorted([p for p in CAPTURE_ROOT.iterdir() if p.is_dir()], key=lambda p: p.name):
        manifest = load_manifest(capture_dir)
        if manifest is not None:
            captures.append((capture_dir, manifest))
    return captures


def clean_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clean_ellipse(raw):
    raw = raw or {}
    return {
        "cx": round(clean_float(raw.get("cx")), 4),
        "cy": round(clean_float(raw.get("cy")), 4),
        "rx": round(max(0.1, clean_float(raw.get("rx"), 1.0)), 4),
        "ry": round(max(0.1, clean_float(raw.get("ry"), 1.0)), 4),
        "rotation_deg": round(clean_float(raw.get("rotation_deg")), 4),
    }


def label_ellipse(raw):
    ellipse = clean_ellipse(raw)
    ellipse.update({
        "coordinate_space": "image_pixels",
        "units": "pixels",
        "offscreen_allowed": True,
    })
    return ellipse


def clean_point(raw):
    raw = raw or {}
    return {
        "x": round(clean_float(raw.get("x")), 4),
        "y": round(clean_float(raw.get("y")), 4),
    }


def clean_perspective(raw):
    if not isinstance(raw, dict):
        return None
    corners = raw.get("corners")
    base = raw.get("base_ellipse")
    if not isinstance(corners, list) or len(corners) != 4 or not isinstance(base, dict):
        return None
    return {
        "mode": "homography_quad",
        "base_ellipse": clean_ellipse(base),
        "corners": [clean_point(point) for point in corners],
    }


@app.get("/")
def index():
    return render_template_string(HTML)


@app.get("/api/captures")
def api_captures():
    return jsonify([capture_summary(capture_dir, manifest) for capture_dir, manifest in all_captures()])


@app.get("/api/captures/<capture_id>")
def api_capture(capture_id):
    capture_dir = capture_path(capture_id)
    manifest = load_manifest(capture_dir)
    if manifest is None:
        abort(404)
    payload = dict(manifest)
    payload["id"] = capture_dir.name
    payload["label"] = load_json(capture_dir / LABEL_REL_PATH)
    payload["prediction"] = detection_info(capture_dir)
    return jsonify(payload)


@app.get("/frames/<capture_id>/<filename>")
def frame_file(capture_id, filename):
    capture_dir = capture_path(capture_id)
    path = (capture_dir / "frames" / filename).resolve()
    frames_dir = (capture_dir / "frames").resolve()
    if frames_dir not in path.parents or not path.is_file():
        abort(404)
    return send_file(path)


@app.get("/api/captures/<capture_id>/label")
def api_get_label(capture_id):
    capture_dir = capture_path(capture_id)
    label = load_json(capture_dir / LABEL_REL_PATH)
    if label is None:
        return jsonify({"label": None})
    return jsonify({"label": label})


@app.post("/api/captures/<capture_id>/label")
def api_save_label(capture_id):
    capture_dir = capture_path(capture_id)
    manifest = load_manifest(capture_dir)
    if manifest is None:
        abort(404)

    payload = request.get_json(force=True) or {}
    ellipse_raw = payload.get("ellipse") or {}
    frame_index = int(payload.get("frame_index") or 0)
    frames = manifest.get("frames") or []
    if frame_index < 0 or frame_index >= len(frames):
        return jsonify({"error": "frame_index is outside this capture."}), 400

    now = datetime.now(timezone.utc).isoformat()
    existing = load_json(capture_dir / LABEL_REL_PATH) or {}
    ellipse = label_ellipse(ellipse_raw)
    editor_mode = payload.get("editor_mode") if payload.get("editor_mode") in {"ellipse", "perspective"} else "ellipse"
    perspective = clean_perspective(payload.get("perspective"))
    if editor_mode == "perspective" and perspective is None:
        return jsonify({"error": "Perspective labels require four corners and a base ellipse."}), 400
    frame_record = frames[frame_index]
    label = {
        "schema_version": 1,
        "label_type": "true_loop_ellipse",
        "capture_id": capture_dir.name,
        "created_utc": existing.get("created_utc") or now,
        "updated_utc": now,
        "frame": {
            "index": frame_index,
            "path": payload.get("frame_path") or frame_record.get("path"),
            "t_s": payload.get("frame_t_s", frame_record.get("t_s")),
            "fft_intensity": payload.get("frame_fft_intensity", frame_record.get("fft_intensity")),
            "image_width": int(payload.get("image_width") or 0),
            "image_height": int(payload.get("image_height") or 0),
        },
        "ellipse": ellipse,
        "editor_mode": editor_mode,
        "notes": payload.get("notes") or "",
    }
    if perspective is not None:
        label["perspective"] = perspective

    label_path = capture_dir / LABEL_REL_PATH
    per_frame_label_path = frame_label_path(capture_dir, frame_index)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_json = json.dumps(label, indent=2)
    label_path.write_text(label_json)
    per_frame_label_path.write_text(label_json)
    return jsonify({"ok": True, "path": str(label_path), "frame_path": str(per_frame_label_path), "label": label})


@app.post("/api/captures/<capture_id>/score")
def api_score_capture(capture_id):
    capture_dir = capture_path(capture_id)
    manifest = load_manifest(capture_dir)
    if manifest is None:
        abort(404)

    labels = load_truth_labels(capture_dir)
    prediction = recompute_detection_info(capture_dir)
    result = score_capture_labels(labels, prediction)
    result["prediction"] = prediction
    result["score_log"] = append_score_log(capture_dir.name, result)
    return jsonify(result)


def parse_args():
    parser = argparse.ArgumentParser(description="Browser-based capture ellipse labeler.")
    parser.add_argument("--root", default="output", help="Folder containing capture output folders.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8050)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    CAPTURE_ROOT = Path(args.root).resolve()
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
