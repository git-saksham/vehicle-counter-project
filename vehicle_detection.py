import cv2
import pandas as pd
from ultralytics import YOLO

# -------- CONFIG --------
VIDEO_PATH = "video.mp4"              # change to your video file
MODEL_PATH = "yolov8n.pt"            # YOLOv8 model (small & fast)
CSV_OUT = "lane_vehicle_counts.csv"  # output file
LINE_POS_RATIO = 0.72                # horizontal counting line position
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
divider_x = frame_width // 2  # middle divider (adjust manually if needed)

# Counters
left_count = 0
right_count = 0
counted_left_ids = set()
counted_right_ids = set()

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

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy()
        xyxy = results[0].boxes.xyxy.cpu().numpy()

        for i, track_id in enumerate(ids):
            x1, y1, x2, y2 = xyxy[i]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {int(track_id)}", (cx, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # LEFT lane
            if cx < divider_x:
                if track_id not in counted_left_ids and cy > count_line_y:
                    left_count += 1
                    counted_left_ids.add(track_id)
                    print(f"Object {track_id} crossed LEFT lane")
                    log.append({
                        "frame": frame_id,
                        "time_s": round(frame_id/fps, 2),
                        "id": int(track_id),
                        "lane": "left",
                        "left_total": left_count,
                        "right_total": right_count
                    })

            # RIGHT lane
            else:
                if track_id not in counted_right_ids and cy > count_line_y:
                    right_count += 1
                    counted_right_ids.add(track_id)
                    print(f"Object {track_id} crossed RIGHT lane")
                    log.append({
                        "frame": frame_id,
                        "time_s": round(frame_id/fps, 2),
                        "id": int(track_id),
                        "lane": "right",
                        "left_total": left_count,
                        "right_total": right_count
                    })

    # Show counts on screen
    cv2.putText(frame, f"Left Lane: {left_count}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Right Lane: {right_count}", (frame_width//2 + 30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Vehicle Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save CSV
if log:
    df = pd.DataFrame(log)
    df.to_csv(CSV_OUT, index=False)
    print("✅ Saved counts to:", CSV_OUT)

print(f"Finished. Left lane: {left_count}, Right lane: {right_count}")
