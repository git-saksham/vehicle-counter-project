import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import pandas as pd
import time
from ultralytics import YOLO

# -------- CONFIG --------
VIDEO_PATH = "video.mp4"
MODEL_PATH = "yolov8n.pt"
CSV_OUT = "lane_vehicle_counts.csv"
LINE_POS_RATIO = 0.72
MIN_GREEN_TIME = 5
MAX_GREEN_TIME = 30 # New config for maximum green time
# ------------------------

print("📥 Loading YOLOv8 model...")
model = YOLO(MODEL_PATH)
print("✅ Model loaded successfully")

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ Error: Could not open video file at {VIDEO_PATH}")
    raise SystemExit

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
count_line_y = int(frame_height * LINE_POS_RATIO)
divider_x = frame_width // 2

# Counters and Queues
left_count = 0
right_count = 0
uncounted_vehicles = {"left": 0, "right": 0} 
counted_left_ids = set()
counted_right_ids = set()

# Traffic signal states
signal_status = {"left": "RED", "right": "RED"}
current_green = "left"
last_switch_time = time.time()
green_duration = MIN_GREEN_TIME

log = []
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1

    # Run YOLOv8 tracking
    results = model.track(frame, persist=True, show=False)

    # Draw divider + counting line
    cv2.line(frame, (0, count_line_y), (frame_width, count_line_y), (0, 255, 255), 3)
    cv2.line(frame, (divider_x, 0), (divider_x, frame_height), (255, 0, 0), 2)

    uncounted_vehicles = {"left": 0, "right": 0}

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy()
        xyxy = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        for i, track_id in enumerate(ids):
            x1, y1, x2, y2 = xyxy[i]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            
            if classes[i] in [2, 3, 5, 7]:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {int(track_id)}", (cx, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Count vehicles in the queue
                if cy > count_line_y * 0.5:
                    if cx < divider_x:
                        uncounted_vehicles["left"] += 1
                    else:
                        uncounted_vehicles["right"] += 1

                # Count vehicles that have crossed
                if cx < divider_x:
                    if track_id not in counted_left_ids and cy > count_line_y:
                        left_count += 1
                        counted_left_ids.add(track_id)
                        print(f"Object {track_id} crossed LEFT lane")
                        log.append({
                            "frame": frame_id, "time_s": round(frame_id/fps, 2), "id": int(track_id),
                            "lane": "left", "left_total": left_count, "right_total": right_count
                        })
                else:
                    if track_id not in counted_right_ids and cy > count_line_y:
                        right_count += 1
                        counted_right_ids.add(track_id)
                        print(f"Object {track_id} crossed RIGHT lane")
                        log.append({
                            "frame": frame_id, "time_s": round(frame_id/fps, 2), "id": int(track_id),
                            "lane": "right", "left_total": left_count, "right_total": right_count
                        })

    # 🚦 Dynamic Signal Switching and Countdown Timer
    now = time.time()
    elapsed_time = now - last_switch_time

    # Decide to switch based on timer and traffic volume
    if elapsed_time >= MIN_GREEN_TIME:
        if current_green == "left" and (elapsed_time >= MAX_GREEN_TIME or uncounted_vehicles["right"] > uncounted_vehicles["left"]):
            current_green = "right"
            last_switch_time = now
        elif current_green == "right" and (elapsed_time >= MAX_GREEN_TIME or uncounted_vehicles["left"] > uncounted_vehicles["right"]):
            current_green = "left"
            last_switch_time = now

    # Update signal status
    if current_green == "left":
        signal_status = {"left": "GREEN", "right": "RED"}
    else:
        signal_status = {"left": "RED", "right": "GREEN"}

    # Show counts
    cv2.putText(frame, f"Left Lane: {left_count}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Right Lane: {right_count}", (frame_width//2 + 30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Show traffic lights
    if signal_status["left"] == "GREEN":
        cv2.circle(frame, (100, 150), 30, (0, 255, 0), -1)
        cv2.circle(frame, (100, 220), 30, (0, 0, 255), 2)
    else:
        cv2.circle(frame, (100, 150), 30, (0, 255, 0), 2)
        cv2.circle(frame, (100, 220), 30, (0, 0, 255), -1)
    cv2.putText(frame, "Left Signal", (60, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if signal_status["right"] == "GREEN":
        cv2.circle(frame, (frame_width - 150, 150), 30, (0, 255, 0), -1)
        cv2.circle(frame, (frame_width - 150, 220), 30, (0, 0, 255), 2)
    else:
        cv2.circle(frame, (frame_width - 150, 150), 30, (0, 255, 0), 2)
        cv2.circle(frame, (frame_width - 150, 220), 30, (0, 0, 255), -1)
    cv2.putText(frame, "Right Signal", (frame_width - 200, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Countdown timer
    time_remaining = max(0, int(MAX_GREEN_TIME - elapsed_time))
    timer_text = f"Switch in: {time_remaining}s"
    cv2.putText(frame, timer_text, (frame_width // 2 - 100, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Vehicle Counter with Signals", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if log:
    df = pd.DataFrame(log)
    df.to_csv(CSV_OUT, index=False)
    print("✅ Saved counts to:", CSV_OUT)

print(f"Finished. Left lane: {left_count}, Right lane: {right_count}")