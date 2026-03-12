import threading
import time
from collections import deque

import cv2
from flask import Flask, Response, jsonify, render_template_string, request

import FFT_RTSP


class WebSink(FFT_RTSP.BaseSink):
    def __init__(self, max_points=2000):
        self.lock = threading.Lock()
        self.latest_roi_jpg = None
        self.latest_frame_jpg = None
        self.latest_t = 0.0
        self.latest_intensity = 0.0
        self.series_t = deque(maxlen=max_points)
        self.series_y = deque(maxlen=max_points)
        self._stop = False
        self.running = False

    def _encode(self, img, quality=80):
        ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        return jpg.tobytes() if ok else None

    def on_roi(self, roi_view, t_s: float, intensity: float):
        with self.lock:
            self.latest_roi_jpg = self._encode(roi_view)

    def on_frame(self, frame, t_s: float, intensity: float):
        with self.lock:
            self.latest_frame_jpg = self._encode(frame)
            self.latest_t = float(t_s)
            self.latest_intensity = float(intensity)
            self.series_t.append(float(t_s))
            self.series_y.append(float(intensity))

    def on_series(self, t_list, y_list):
        # optional: if using the original arrays rather than deque appends
        pass

    def should_stop(self) -> bool:
        with self.lock:
            return self._stop

    def stop(self):
        with self.lock:
            self._stop = True

    def snapshot_state(self):
        with self.lock:
            return {
                "running": self.running,
                "time": self.latest_t,
                "intensity": self.latest_intensity,
                "series_t": list(self.series_t),
                "series_y": list(self.series_y),
            }

    def snapshot_jpg(self, which: str):
        with self.lock:
            return self.latest_roi_jpg if which == "roi" else self.latest_frame_jpg


app = Flask(__name__)
sink = WebSink()
worker_thread = None

# default ROI points
roi_points = [(677, 1288), (1325, 1418), (1425, 1171), (893, 1051)]


def run_worker(source: str):
    sink.running = True
    try:
        FFT_RTSP.process_rtsp_stream(source, roi_points, sink=sink)
    finally:
        sink.running = False


@app.get("/")
def index():
    return render_template_string("""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>CV Web UI</title>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
  <style>
    body { margin:0; font-family: system-ui; background:#0f0f10; color:#eaeaea; }
    .wrap { display:grid; grid-template-columns: 1fr 1fr; gap:12px; padding:12px; }
    .card { background:#16161a; border-radius:14px; padding:12px; }
    img { width:100%; border-radius:10px; background:#000; }
    .row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
    input, button { padding:8px 10px; border-radius:10px; border:1px solid #333; background:#0f0f10; color:#eaeaea; }
    button { cursor:pointer; }
    #plot { height: 360px; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="row">
        <div><b>ROI View</b></div>
        <div style="margin-left:auto" id="status">—</div>
      </div>
      <img src="/mjpeg/roi" />
    </div>

    <div class="card">
      <div class="row"><b>Full Frame</b></div>
      <img src="/mjpeg/frame" />
    </div>

    <div class="card" style="grid-column:1 / span 2;">
      <div class="row" style="justify-content:space-between;">
        <b>Intensity</b>
        <div class="row">
          <input id="src" style="min-width:340px" value="temp/21_08_2025_P1-00.00.00.000-01.05.57.296.mov" />
          <button onclick="start()">Start</button>
          <button onclick="stop()">Stop</button>
        </div>
      </div>
      <div id="plot"></div>
    </div>
  </div>

<script>
let initialized = false;

async function refresh() {
  const r = await fetch('/api/state');
  const s = await r.json();

  document.getElementById('status').textContent =
    (s.running ? 'RUNNING' : 'STOPPED') + ' | t=' + s.time.toFixed(2) + 's | intensity=' + s.intensity.toFixed(2);

  const x = s.series_t;
  const y = s.series_y;

  if (!initialized) {
    Plotly.newPlot('plot', [{x, y, mode:'lines', name:'Intensity'}],
      {margin:{t:10,l:50,r:20,b:40}, paper_bgcolor:'#16161a', plot_bgcolor:'#16161a',
       xaxis:{title:'Time (s)'}, yaxis:{title:'Intensity'}});
    initialized = true;
  } else {
    Plotly.react('plot', [{x, y, mode:'lines', name:'Intensity'}],
      {margin:{t:10,l:50,r:20,b:40}, paper_bgcolor:'#16161a', plot_bgcolor:'#16161a',
       xaxis:{title:'Time (s)'}, yaxis:{title:'Intensity'}});
  }
}

async function start() {
  await fetch('/api/start', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({source: document.getElementById('src').value})
  });
}

async function stop() {
  await fetch('/api/stop', {method:'POST'});
}

setInterval(refresh, 500);
refresh();
</script>
</body>
</html>
""")


@app.get("/api/state")
def api_state():
    return jsonify(sink.snapshot_state())


@app.post("/api/start")
def api_start():
    global worker_thread
    data = request.get_json(force=True) or {}
    source = data.get("source") or FFT_RTSP.RTSP_URL

    if sink.running:
        return jsonify({"ok": True, "already_running": True})

    sink._stop = False
    worker_thread = threading.Thread(target=run_worker, args=(source,), daemon=True)
    worker_thread.start()
    return jsonify({"ok": True, "source": source})


@app.post("/api/stop")
def api_stop():
    sink.stop()
    return jsonify({"ok": True})


def mjpeg_stream(which: str):
    boundary = b"frame"
    while True:
        jpg = sink.snapshot_jpg(which)
        if jpg is None:
            time.sleep(0.05)
            continue
        yield b"--" + boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        time.sleep(0.03)


@app.get("/mjpeg/roi")
def mjpeg_roi():
    return Response(mjpeg_stream("roi"), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/mjpeg/frame")
def mjpeg_frame():
    return Response(mjpeg_stream("frame"), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)