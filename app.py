from flask import Flask, Response, render_template, jsonify
import cv2
from ultralytics import YOLO
import numpy as np
import time
from collections import defaultdict, deque

app = Flask(__name__)

VIDEO_PATH = r"C:\Users\abcte\OneDrive\ドキュメント\Majorproject\AccidentDetection\WhatsApp Video 2026-02-21 at 09.43.43.mp4"
camera = cv2.VideoCapture(VIDEO_PATH)

VIDEO_FPS = camera.get(cv2.CAP_PROP_FPS)
if VIDEO_FPS is None or VIDEO_FPS == 0:
    VIDEO_FPS = 25  
FRAME_DELAY = 1.0 / VIDEO_FPS

model = YOLO("yolo11s.pt")

VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle"]

DECEL_THRESHOLD = 5.0
DIST_COLLAPSE = 45
CONFIRM_FRAMES = 4
ACCIDENT_HOLD_TIME = 5

positions = defaultdict(lambda: deque(maxlen=5))
speeds = defaultdict(lambda: deque(maxlen=5))
accident_votes = defaultdict(int)

accident_active = False
accident_start_time = 0

current_accident = False
current_conf = 0.05
current_detections = 0

def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def distance(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

def generate_frames():
    global current_accident, current_conf, current_detections
    global accident_active, accident_start_time

    last_frame_time = time.time()

    while True:
        ret, frame = camera.read()
        if not ret:
            camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        results = model.track(
            frame,
            persist=True,
            conf=0.35,
            iou=0.5,
            verbose=False
        )

        vehicles = {}

        for r in results:
            for box in r.boxes:
                if box.id is None:
                    continue

                cls = int(box.cls[0])
                label = model.names[cls]
                if label not in VEHICLE_CLASSES:
                    continue

                tid = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                c = center((x1, y1, x2, y2))

                vehicles[tid] = c
                positions[tid].append(c)

                if len(positions[tid]) >= 2:
                    p1, p2 = positions[tid][-2], positions[tid][-1]
                    speeds[tid].append(distance(p1, p2))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ID:{tid}",
                            (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)

        for tid, sp in speeds.items():
            if len(sp) >= 3:
                if sp[-2] - sp[-1] > DECEL_THRESHOLD:
                    accident_votes[tid] += 1

        ids = list(vehicles.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if distance(vehicles[ids[i]], vehicles[ids[j]]) < DIST_COLLAPSE:
                    accident_votes[ids[i]] += 1
                    accident_votes[ids[j]] += 1

        now = time.time()

        if not accident_active:
            for v in accident_votes.values():
                if v >= CONFIRM_FRAMES:
                    accident_active = True
                    accident_start_time = now
                    accident_votes.clear()
                    positions.clear()
                    speeds.clear()
                    break

        if accident_active:
            current_accident = True
            current_conf = min(0.95, current_conf + 0.15)

            if now - accident_start_time > ACCIDENT_HOLD_TIME:
                accident_active = False
                current_accident = False
                current_conf = 0.05
        else:
            current_accident = False
            current_conf = max(0.05, current_conf - 0.05)

        current_detections += len(vehicles)

        if current_accident:
            cv2.putText(frame, "🚨 ACCIDENT CONFIRMED",
                        (40, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255), 3)

        elapsed = time.time() - last_frame_time
        if elapsed < FRAME_DELAY:
            time.sleep(FRAME_DELAY - elapsed)
        last_frame_time = time.time()

        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status")
def status():
    return jsonify({
        "accident": current_accident,
        "accident_conf": round(current_conf, 2),
        "detections": current_detections
    })


if __name__ == "__main__":
    print(f"🚀 Server running at http://127.0.0.1:5000  (FPS ≈ {VIDEO_FPS:.1f})")
    app.run(threaded=True, debug=False, use_reloader=False)