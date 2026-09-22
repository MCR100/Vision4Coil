"""Browser tool for creating and using a planar pixel-to-world homography.

The homography saved by this tool accepts pixel coordinates measured in the
*undistorted* image.  Run the server with::

    python planar_homography_web.py --calibration calibration.npz

Then open http://localhost:5002, upload a frame, click the four reference
corners in TL, TR, BR, BL order, and save the mapping.
"""

from __future__ import annotations

import argparse
import io
import json
import threading
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024

CALIBRATION_PATH = Path("calibration.npz")
OUTPUT_PATH = Path("planar_homography.npz")
ALPHA = 0.0
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Planar Homography Calibration</title>
  <style>
    :root { color-scheme: dark; --bg:#101217; --panel:#191c23; --line:#39404d;
      --text:#eef1f6; --muted:#aab2c0; --accent:#58b8e6; --good:#55d692; --bad:#ff796f; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font:14px system-ui,sans-serif; }
    header, aside { background:var(--panel); }
    header { padding:14px 18px; border-bottom:1px solid var(--line); }
    h1 { margin:0 0 5px; font-size:19px; }
    p { margin:5px 0; color:var(--muted); }
    main { display:grid; grid-template-columns:310px minmax(0,1fr); min-height:calc(100vh - 76px); }
    aside { padding:16px; border-right:1px solid var(--line); display:grid; align-content:start; gap:14px; }
    label { display:grid; gap:5px; color:var(--muted); }
    input, button { font:inherit; }
    input { width:100%; background:#0d0f13; color:var(--text); border:1px solid var(--line);
      border-radius:6px; padding:8px; }
    input[type=file] { padding:6px; }
    .dims { display:grid; grid-template-columns:1fr 1fr; gap:9px; }
    button { padding:9px 11px; color:var(--text); background:#252a34; border:1px solid var(--line);
      border-radius:6px; cursor:pointer; }
    button:hover { border-color:var(--accent); }
    button.primary { background:#216282; border-color:#4698be; }
    button:disabled { opacity:.45; cursor:not-allowed; }
    .buttons { display:flex; gap:8px; flex-wrap:wrap; }
    .status { min-height:42px; white-space:pre-wrap; color:var(--muted); }
    .status.error { color:var(--bad); }
    .status.good { color:var(--good); }
    .order { line-height:1.55; color:var(--muted); }
    .stage-wrap { min-width:0; overflow:auto; display:grid; place-items:start center; padding:16px; background:#090a0d; }
    .stage { position:relative; line-height:0; max-width:100%; box-shadow:0 12px 45px #0009; }
    #image { display:block; max-width:100%; max-height:calc(100vh - 115px); width:auto; height:auto; }
    #overlay { position:absolute; inset:0; width:100%; height:100%; cursor:crosshair; touch-action:none; }
    .placeholder { margin:auto; color:var(--muted); text-align:center; padding:30px; }
    code { color:#c9e9f8; }
    @media(max-width:760px) { main{display:block} aside{border-right:0;border-bottom:1px solid var(--line)} }
  </style>
</head>
<body>
<header>
  <h1>Planar Homography Calibration</h1>
  <p>Lens-correct a captured frame, select a measured planar rectangle, and save an undistorted-pixel → real-space mapping.</p>
</header>
<main>
  <aside>
    <label>Captured frame (JPG or PNG)
      <input id="file" type="file" accept="image/jpeg,image/png">
    </label>
    <div class="dims">
      <label>Width (mm)<input id="width" type="number" value="326" min="0.001" step="any"></label>
      <label>Height (mm)<input id="height" type="number" value="86" min="0.001" step="any"></label>
    </div>
    <div class="order">
      Click the matching corners in this order:<br>
      <b>1.</b> top-left &nbsp; <b>2.</b> top-right<br>
      <b>3.</b> bottom-right &nbsp; <b>4.</b> bottom-left
    </div>
    <div class="buttons">
      <button id="undo" disabled>Undo point</button>
      <button id="clear" disabled>Clear</button>
      <button id="save" class="primary" disabled>Save mapping</button>
    </div>
    <div id="status" class="status">Upload a frame to begin.</div>
    <p>The defaults describe the 350×110 mm outer rectangle inset by 12 mm on each side. Change them if you select different corners.</p>
    <p>Output: <code>{{ output_path }}</code></p>
  </aside>
  <section class="stage-wrap">
    <div id="placeholder" class="placeholder">The undistorted image will appear here.</div>
    <div id="stage" class="stage" hidden>
      <img id="image" alt="Undistorted uploaded frame">
      <canvas id="overlay"></canvas>
    </div>
  </section>
</main>
<script>
const file = document.querySelector('#file');
const image = document.querySelector('#image');
const canvas = document.querySelector('#overlay');
const ctx = canvas.getContext('2d');
const statusEl = document.querySelector('#status');
const undo = document.querySelector('#undo');
const clear = document.querySelector('#clear');
const save = document.querySelector('#save');
let jobId = null;
let points = [];

function status(text, kind='') { statusEl.textContent=text; statusEl.className='status '+kind; }
function syncCanvas() {
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  draw();
}
function draw() {
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if (!points.length) return;
  ctx.lineWidth = Math.max(2, canvas.width/800);
  ctx.strokeStyle = '#58b8e6';
  ctx.fillStyle = '#58b8e6';
  ctx.beginPath(); ctx.moveTo(points[0][0],points[0][1]);
  for (let i=1;i<points.length;i++) ctx.lineTo(points[i][0],points[i][1]);
  if (points.length===4) ctx.closePath();
  ctx.stroke();
  points.forEach((p,i) => {
    const r=Math.max(5,canvas.width/300);
    ctx.beginPath(); ctx.arc(p[0],p[1],r,0,Math.PI*2); ctx.fill();
    ctx.fillStyle='#081018'; ctx.font=`bold ${Math.max(12,canvas.width/100)}px system-ui`;
    ctx.textAlign='center'; ctx.textBaseline='middle'; ctx.fillText(String(i+1),p[0],p[1]);
    ctx.fillStyle='#58b8e6';
  });
}
function updateButtons() {
  undo.disabled=!points.length; clear.disabled=!points.length;
  save.disabled=!(jobId && points.length===4);
  if (jobId) status(`${points.length}/4 points selected.`);
}
file.addEventListener('change', async () => {
  if (!file.files.length) return;
  const body=new FormData(); body.append('frame',file.files[0]);
  status('Uploading and undistorting…');
  try {
    const response=await fetch('/upload',{method:'POST',body});
    const data=await response.json();
    if (!response.ok) throw new Error(data.error || 'Upload failed');
    jobId=data.job_id; points=[];
    image.onload=() => { syncCanvas(); document.querySelector('#placeholder').hidden=true;
      document.querySelector('#stage').hidden=false; updateButtons(); };
    image.src=`/image/${jobId}?v=${Date.now()}`;
  } catch (e) { status(e.message,'error'); }
});
canvas.addEventListener('pointerdown', e => {
  if (!jobId || points.length>=4) return;
  const r=canvas.getBoundingClientRect();
  points.push([(e.clientX-r.left)*canvas.width/r.width, (e.clientY-r.top)*canvas.height/r.height]);
  draw(); updateButtons();
});
undo.onclick=() => { points.pop(); draw(); updateButtons(); };
clear.onclick=() => { points=[]; draw(); updateButtons(); };
window.addEventListener('resize', draw);
save.onclick=async () => {
  const width=Number(document.querySelector('#width').value);
  const height=Number(document.querySelector('#height').value);
  if (!(width>0 && height>0)) { status('Width and height must be positive.','error'); return; }
  save.disabled=true; status('Saving mapping…');
  try {
    const response=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({job_id:jobId,points,width_mm:width,height_mm:height})});
    const data=await response.json();
    if (!response.ok) throw new Error(data.error || 'Save failed');
    status(`Saved ${data.output}\nMean corner reprojection error: ${data.reprojection_error_px.toFixed(4)} px`,'good');
  } catch(e) { status(e.message,'error'); save.disabled=false; }
};
</script>
</body>
</html>
"""


def load_camera_calibration(path: str | Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Load K/dist/img_size, with common alternative key names supported."""
    with np.load(path) as data:
        k_key = next((key for key in ("K", "camera_matrix", "mtx") if key in data), None)
        d_key = next((key for key in ("dist", "dist_coeffs", "distortion_coefficients") if key in data), None)
        if k_key is None or d_key is None:
            raise ValueError("Calibration must contain K (or camera_matrix/mtx) and dist (or dist_coeffs).")
        matrix = np.asarray(data[k_key], dtype=np.float64)
        distortion = np.asarray(data[d_key], dtype=np.float64)
        if "img_size" in data:
            size = tuple(int(v) for v in np.asarray(data["img_size"]).ravel())
        elif "image_size" in data:
            size = tuple(int(v) for v in np.asarray(data["image_size"]).ravel())
        else:
            size = (int(round(matrix[0, 2] * 2)), int(round(matrix[1, 2] * 2)))
    if matrix.shape != (3, 3) or len(size) != 2 or min(size) <= 0:
        raise ValueError("Invalid camera matrix or calibration image size.")
    return matrix, distortion, size


def scaled_camera_matrix(matrix: np.ndarray, from_size: tuple[int, int], to_size: tuple[int, int]) -> np.ndarray:
    """Scale camera intrinsics from calibration resolution to frame resolution."""
    sx, sy = to_size[0] / from_size[0], to_size[1] / from_size[1]
    result = matrix.copy()
    result[0, :] *= sx
    result[1, :] *= sy
    result[2, :] = (0.0, 0.0, 1.0)
    return result


def load_homography(path: str | Path = "planar_homography.npz") -> np.ndarray:
    """Load the undistorted-pixel to real-space homography."""
    with np.load(path) as data:
        return np.asarray(data["image_to_world"], dtype=np.float64)


def pixel_to_real(points_xy, homography_or_path="planar_homography.npz") -> np.ndarray:
    """Convert one or more undistorted pixel coordinates to real-space millimetres."""
    homography = (
        load_homography(homography_or_path)
        if isinstance(homography_or_path, (str, Path))
        else np.asarray(homography_or_path, dtype=np.float64)
    )
    points = np.asarray(points_xy, dtype=np.float64)
    original_shape = points.shape
    if original_shape == (2,):
        points = points.reshape(1, 2)
    elif points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy must have shape (2,) or (N, 2).")
    converted = cv2.perspectiveTransform(points.reshape(-1, 1, 2), homography).reshape(-1, 2)
    return converted[0] if original_shape == (2,) else converted


@app.get("/")
def index():
    return render_template_string(HTML, output_path=str(OUTPUT_PATH))


@app.post("/upload")
def upload():
    uploaded = request.files.get("frame")
    if uploaded is None or not uploaded.filename:
        return jsonify(error="Choose a JPG or PNG frame."), 400
    raw = uploaded.read()
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify(error="The uploaded file is not a readable image."), 400

    try:
        camera_matrix, distortion, calibration_size = load_camera_calibration(CALIBRATION_PATH)
    except (OSError, ValueError) as exc:
        return jsonify(error=f"Could not load calibration: {exc}"), 500

    frame_size = (frame.shape[1], frame.shape[0])
    scaled_k = scaled_camera_matrix(camera_matrix, calibration_size, frame_size)
    new_k, roi = cv2.getOptimalNewCameraMatrix(scaled_k, distortion, frame_size, ALPHA, frame_size)
    undistorted = cv2.undistort(frame, scaled_k, distortion, None, new_k)
    ok, encoded = cv2.imencode(".jpg", undistorted, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not ok:
        return jsonify(error="OpenCV could not encode the undistorted image."), 500

    job_id = uuid.uuid4().hex
    job = {
        "jpeg": encoded.tobytes(), "frame_size": frame_size,
        "calibration_size": calibration_size, "camera_matrix": camera_matrix,
        "scaled_camera_matrix": scaled_k, "new_camera_matrix": new_k,
        "distortion": distortion, "roi": tuple(int(v) for v in roi),
        "source_name": Path(uploaded.filename).name,
    }
    with _jobs_lock:
        _jobs.clear()  # This is intentionally a single-user local calibration utility.
        _jobs[job_id] = job
    return jsonify(job_id=job_id, width=frame_size[0], height=frame_size[1])


@app.get("/image/<job_id>")
def image(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify(error="Uploaded image is no longer available."), 404
    return Response(job["jpeg"], mimetype="image/jpeg")


@app.post("/save")
def save():
    payload = request.get_json(silent=True) or {}
    with _jobs_lock:
        job = _jobs.get(str(payload.get("job_id", "")))
    if job is None:
        return jsonify(error="Upload a frame again before saving."), 400
    try:
        points = np.asarray(payload["points"], dtype=np.float64)
        width_mm = float(payload["width_mm"])
        height_mm = float(payload["height_mm"])
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Four points and valid rectangle dimensions are required."), 400
    if points.shape != (4, 2) or not np.isfinite(points).all():
        return jsonify(error="Exactly four finite image points are required."), 400
    if not np.isfinite((width_mm, height_mm)).all() or width_mm <= 0 or height_mm <= 0:
        return jsonify(error="Rectangle dimensions must be positive."), 400
    contour = points.astype(np.float32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour) or abs(cv2.contourArea(contour)) < 25:
        return jsonify(error="Points must form a non-crossing convex quadrilateral in TL, TR, BR, BL order."), 400

    world_points = np.array(
        [[0.0, 0.0], [width_mm, 0.0], [width_mm, height_mm], [0.0, height_mm]],
        dtype=np.float64,
    )
    homography = cv2.getPerspectiveTransform(points.astype(np.float32), world_points.astype(np.float32))
    world_to_image = np.linalg.inv(homography)
    reconstructed = cv2.perspectiveTransform(
        world_points.reshape(-1, 1, 2), world_to_image
    ).reshape(-1, 2)
    error_px = float(np.mean(np.linalg.norm(reconstructed - points, axis=1)))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        image_to_world=homography,
        world_to_image=world_to_image,
        image_points=points,
        world_points=world_points,
        rectangle_size_mm=np.array([width_mm, height_mm]),
        frame_size=np.array(job["frame_size"]),
        calibration_size=np.array(job["calibration_size"]),
        camera_matrix=job["camera_matrix"],
        scaled_camera_matrix=job["scaled_camera_matrix"],
        new_camera_matrix=job["new_camera_matrix"],
        dist_coeffs=job["distortion"],
        undistort_alpha=np.array(ALPHA),
        valid_roi=np.array(job["roi"]),
        source_name=np.array(job["source_name"]),
    )
    temp_path = OUTPUT_PATH.with_suffix(OUTPUT_PATH.suffix + ".tmp")
    temp_path.write_bytes(buffer.getvalue())
    temp_path.replace(OUTPUT_PATH)

    metadata = {
        "coordinate_contract": "image_to_world expects pixels from an image undistorted with the saved scaled_camera_matrix, dist_coeffs, and new_camera_matrix",
        "units": "mm", "source_name": job["source_name"],
        "frame_size": list(job["frame_size"]), "calibration_size": list(job["calibration_size"]),
        "rectangle_size_mm": [width_mm, height_mm], "image_points": points.tolist(),
        "world_points": world_points.tolist(), "image_to_world": homography.tolist(),
        "mean_corner_reprojection_error_px": error_px,
    }
    json_path = OUTPUT_PATH.with_suffix(".json")
    json_temp = json_path.with_suffix(json_path.suffix + ".tmp")
    json_temp.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    json_temp.replace(json_path)
    return jsonify(output=str(OUTPUT_PATH), metadata=str(json_path), reprojection_error_px=error_px)


def main():
    global CALIBRATION_PATH, OUTPUT_PATH, ALPHA
    parser = argparse.ArgumentParser(description="Browser-based planar homography calibration tool.")
    parser.add_argument("--calibration", type=Path, default=CALIBRATION_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--alpha", type=float, default=0.0, help="Undistortion free-scaling parameter (0 crops invalid edges; 1 retains all pixels).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5002)
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be between 0 and 1")
    CALIBRATION_PATH = args.calibration.resolve()
    OUTPUT_PATH = args.output.resolve()
    ALPHA = args.alpha
    load_camera_calibration(CALIBRATION_PATH)  # Fail early with a useful traceback.
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
