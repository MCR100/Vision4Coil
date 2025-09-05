import cv2
import numpy as np
import os
from datetime import timedelta, datetime
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
import json
import torch
import os, time
from collections import deque
import threading
HEADLESS = not os.environ.get("DISPLAY")
if HEADLESS:
    import matplotlib
    matplotlib.use("Agg")

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|max_delay;500000|stimeout;5000000" #Undo this
# ----- CONFIG -----
THRESHOLD = 4264.8 # 4264.8 for DB16 ; 3200 for R5.5 ; 3900 for R8.5
TARGET_SIZE = (480, 270)
DEVICE = 0 if torch.cuda.is_available() else 'cpu'
#USERNAME = "admin"
#PASSWORD = "passkey"
#CAMERA_IP = "cam_ip" #192.168.1.100
RTSP_URL = f"rtsp://172.21.164.154:8554/mystream"

model = YOLO("best_latest.pt")

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
    os.makedirs(folder_name, exist_ok=True)
    return folder_name


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

def detect_tail_and_save(frames, roi_points, save_path, conf_thresh=0.6, negatives_to_save=3):
    best_conf = 0
    best_frame = None
    best_box = None
    best_cls = None

    frames_batch = list(reversed(frames[-10:]))

    results = model.predict(
    frames_batch,
    device=DEVICE,      # e.g., 0 for GPU0, or 'cpu'
    imgsz=640,
    conf=conf_thresh,
    verbose=False
    )
    #Saving diagnostic frames
    frame_max_confs = []

    for frame_i, res in zip(frames_batch, results):
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            frame_max_confs.append(0.0)
            continue

        # record max conf for this frame (for reporting)
        try:
            frame_max_confs.append(float(boxes.conf.max().item()))
        except Exception:
            frame_max_confs.append(float(boxes.conf.max()))

        for box in boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            if conf > best_conf:
                # xyxy is a 1x4 tensor; convert to ints
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                best_conf = conf
                best_cls  = cls_id
                best_box  = [x1, y1, x2, y2]
                best_frame = frame_i.copy()

                cv2.rectangle(best_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(best_frame, f"{conf:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
        Path(save_path).mkdir(parents=True, exist_ok=True)
        if best_conf > 0 and best_frame is not None:
            #Path(save_path).mkdir(parents=True, exist_ok=True)
            img_path = os.path.join(save_path, f"tail_detected_{best_conf:.2f}.jpg")
            cv2.imwrite(img_path, best_frame)
            print(f"Saved tail image: {img_path}")

            # Save JSON label data
            import json
            label_info = {
                "class_id": best_cls,
                "confidence": round(best_conf, 4),
                "bbox": best_box
            }
            json_path = os.path.join(save_path, f"tail_detected_{best_conf:.2f}.json")
            with open(json_path, "w") as f:
                json.dump(label_info, f, indent=2)
            print(f" Saved label info: {json_path}")

        else:
            if not frames_batch:
                print("No frames available to save for negative case.")
            else:
                f_out = frames_batch[0].copy()

                out_path = os.path.join(save_path, "no_tail.jpg")
                cv2.imwrite(out_path, f_out)

                meta = {
                    "reason": "no_detection_above_threshold",
                    "conf_threshold": conf_thresh}
                     
                with open(os.path.join(save_path, "no_tail_meta.json"), "w") as f:
                                json.dump(meta, f, indent=2)

                print(f"Saved negative frame: {out_path}")
                    

def process_rtsp_stream(rtsp_url, roi_points):
    print("Real-time stream started.")

    # cap = cv2.VideoCapture(rtsp_url)
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Error: Cannot open RTSP stream.")
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Reader thread puts the latest frame into a 1-slot deque
    frame_buf = deque(maxlen=25)
    stop_read = False

    def reader():
        while not stop_read:
            ret, f = cap.read()
            if not ret: break
            frame_buf.append((time.perf_counter(), f))

    t = threading.Thread(target=reader, daemon=True); t.start()

    
    fps = 30  # assumed 30 - not required #livestream
    frame_idx = 0
    in_segment = False
    segment_start = None
    segment_frames = []
    segment_time = []
    segment_intensities = []

    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], color='black')
    ax.set_title("Live Frequency Intensity")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Intensity")
    ax.set_xlim(0, 30)
    graph_time = []
    graph_intensity = []
    segment_starts = []

    UI_UPDATE_EVERY = 20  # draw every 20 frames (tune as you like)
    last_seen_ts = None


    while cap.isOpened():
        if not frame_buf:
            if not t.is_alive():  # reader thread died => stream ended
                print("Stream ended or not receiving frames.")
                break
            time.sleep(0.002)
            continue

        arrival_ts, frame = frame_buf[-1]

        if arrival_ts == last_seen_ts:
            time.sleep(0.001)
            continue

        last_seen_ts = arrival_ts

        current_time = frame_idx / fps
        intensity, roi_view = compute_fft_spectrum(frame, roi_points)
        frame_idx += 1

        if not HEADLESS:
            cv2.imshow("ROI View", roi_view)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Update graph
        graph_time.append(current_time)
        graph_intensity.append(intensity)

        # Start segment
        if not in_segment and intensity > THRESHOLD:
            in_segment = True
            segment_start = round(current_time, 2)
            segment_frames = []
            segment_time = []
            segment_intensities = []
            print(f"Segment START at {segment_start:.2f}s")

        # End segment
        elif in_segment and intensity < THRESHOLD:
            segment_end = round(current_time, 2)
            in_segment = False
            segment_duration = segment_end - segment_start
            print(f"Segment END at {segment_end:.2f}s")

            if segment_duration >= 10:
                segment_starts.append(segment_start)
                if len(segment_starts) >= 2:
                    cutoff = segment_starts[-1]
                    graph_time = [t for t in graph_time if t >= cutoff]
                    graph_intensity = graph_intensity[-len(graph_time):]

                start_dt = datetime.now() - timedelta(seconds=segment_duration)
                end_dt = datetime.now()
                folder = create_timestamped_folder(start_dt, end_dt)
                base = os.path.join(folder, os.path.basename(folder))
                save_results_txt(segment_time, segment_intensities, base)
                save_results_html(segment_time, segment_intensities, base)
                detect_tail_and_save(segment_frames, roi_points, folder)
            else:
                print(f"Segment duration {segment_duration:.2f}s too short. Skipped.")

            segment_frames.clear()
            segment_time.clear()
            segment_intensities.clear()

        if in_segment:
            segment_frames.append(frame)
            segment_time.append(current_time)
            segment_intensities.append(intensity)

        if (frame_idx % UI_UPDATE_EVERY == 0) and graph_time and (not HEADLESS):
            line.set_xdata(graph_time)
            line.set_ydata(graph_intensity)
            ax.set_xlim(min(graph_time), max(graph_time) + 5)
            ax.set_ylim(min(graph_intensity) - 50, max(graph_intensity) + 50)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

        frame_idx += 1


    stop_read = True
    t.join(timeout=1.0)
    cap.release()
    if not HEADLESS:
        cv2.destroyAllWindows()

    # Matplotlib end
    if HEADLESS:
        plt.close(fig)   # no GUI
    else:
        plt.show()
    print("RTSP stream processing complete.")


# --- Run ---
roi_points = [(677, 1288), (1325, 1418), (1425, 1171), (893, 1051)]
video_path = "/home/drizvi/Data/drizvi/Infrabuild/20250806-100300 to 20250806-220539.mov"
process_rtsp_stream(RTSP_URL, roi_points)
#process_rtsp_stream(video_path, roi_points)
