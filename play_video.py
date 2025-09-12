# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 09:22:32 2025

@author: harik
"""
import cv2
import numpy as np
import pandas as pd
import math
import time

# --------- CONFIG ----------
VIDEO_PATH = "video.mp4"
USE_BG_SUBTRACTOR = "MOG2"   # "MOG2" or "KNN"
ROI_TOP_RATIO = 0.40         # ignore top 40% of frame (trees/sky)
LINE_POS_RATIO = 0.72        # place counting line at 72% of frame height (tweak if needed)
MIN_AREA_RATIO = 1/500.0     # min contour area as fraction of frame area (dynamic)
MAX_MATCH_DISTANCE = 60      # tracker matching distance (px) - increase if resolution large
SAVE_CSV = True
CSV_OUT = "vehicle_counts_improved.csv"
DIRECTION = "down"           # 'down' means count when object moves downward across line
# ---------------------------

class EuclideanDistTracker:
    def __init__(self, maxDisappeared=40, maxDistance=50):
        self.nextObjectID = 0
        self.objects = dict()        # objectID -> (cx,cy)
        self.previous = dict()       # objectID -> previous (cx,cy)
        self.disappeared = dict()    # objectID -> frames disappeared
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.previous[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.previous[objectID]
        del self.disappeared[objectID]

    def update(self, inputCentroids):
        # inputCentroids: list of (cx,cy)
        if len(inputCentroids) == 0:
            # mark disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.array(inputCentroids)

        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                self.register(tuple(inputCentroids[i]))
            return self.objects

        objectIDs = list(self.objects.keys())
        objectCentroids = np.array(list(self.objects.values()))

        # distance matrix between existing objects and new centroids
        D = np.linalg.norm(objectCentroids[:, None, :] - inputCentroids[None, :, :], axis=2)

        # greedy matching (smallest distances first)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        usedRows, usedCols = set(), set()

        for (row, col) in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue
            if D[row, col] > self.maxDistance:
                continue

            objectID = objectIDs[row]
            self.previous[objectID] = self.objects[objectID]
            self.objects[objectID] = tuple(inputCentroids[col])
            self.disappeared[objectID] = 0

            usedRows.add(row)
            usedCols.add(col)

        # handle unmatched existing objects
        unusedRows = set(range(0, D.shape[0])) - usedRows
        for row in unusedRows:
            objectID = objectIDs[row]
            self.disappeared[objectID] += 1
            if self.disappeared[objectID] > self.maxDisappeared:
                self.deregister(objectID)

        # handle unmatched new centroids
        unusedCols = set(range(0, D.shape[1])) - usedCols
        for col in unusedCols:
            self.register(tuple(inputCentroids[col]))

        return self.objects

# ---------- main ----------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: cannot open video:", VIDEO_PATH)
    raise SystemExit

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_area = frame_width * frame_height

# dynamic thresholds
min_area = max(500, int(frame_area * MIN_AREA_RATIO))  # at least 500 px
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# ROI and counting line
roi_top = int(frame_height * ROI_TOP_RATIO)
count_line_y = int(frame_height * LINE_POS_RATIO)

# background subtractor selection
if USE_BG_SUBTRACTOR == "KNN":
    bg_sub = cv2.createBackgroundSubtractorKNN(detectShadows=True)
else:
    bg_sub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

tracker = EuclideanDistTracker(maxDisappeared=40, maxDistance=MAX_MATCH_DISTANCE)
counted_ids = set()
total_count = 0
log = []
frame_id = 0
start_time = time.time()

print(f"Video: {VIDEO_PATH} | {frame_width}x{frame_height} | FPS: {fps:.1f}")
print(f"ROI top: {roi_top}px  | Counting line y: {count_line_y}px  | min_area: {min_area}px")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1

    # Foreground mask
    fg = bg_sub.apply(frame)
    # Clean mask
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    fg = cv2.dilate(fg, kernel, iterations=2)
    # optional: remove shadows (pixel value 127 are shadows for MOG2)
    _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

    # Find contours on the mask
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Collect centroids for tracker only for contours that pass filters
    centroids = []
    boxes = []  # store for drawing
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        # filter by ROI (ignore objects above ROI top)
        cy = int(y + h/2)
        if cy < roi_top:
            continue
        # ignore objects touching frame borders (likely noise)
        if x <= 5 or y <= 5 or x + w >= frame_width - 5:
            continue
        # aspect ratio check (vehicles are wider than very tall, but allow flexibility)
        aspect_ratio = w / float(h) if h > 0 else 0
        if aspect_ratio < 0.3:  # too tall / thin -> likely noise or pole
            continue
        # optionally check solidity (area / (w*h)) to exclude thin contours
        rect_area = w * h
        solidity = area / float(rect_area) if rect_area > 0 else 0
        if solidity < 0.2:
            continue

        cx = int(x + w/2)
        cy = int(y + h/2)
        centroids.append((cx, cy))
        boxes.append((x, y, w, h))

    # Update tracker with centroids
    objects = tracker.update(centroids)

    # Draw ROI and counting line
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, roi_top), (frame_width, frame_height), (0, 0, 0), -1)
    alpha = 0.15
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.line(frame, (0, count_line_y), (frame_width, count_line_y), (0, 165, 255), 3)
    cv2.putText(frame, "ROI shown (shaded). Adjust ROI_TOP_RATIO if needed.", (10, roi_top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Draw bounding boxes (match centroids->boxes by proximity)
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # For each tracked object, check crossing
    for objectID, centroid in objects.items():
        cx, cy = centroid
        prev_cx, prev_cy = tracker.previous.get(objectID, (cx, cy))
        # draw id and centroid
        cv2.putText(frame, f"ID {objectID}", (cx - 10, cy - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        # check crossing depending on direction
        if objectID not in counted_ids:
            if DIRECTION == "down":
                if prev_cy < count_line_y <= cy:
                    total_count += 1
                    counted_ids.add(objectID)
                    timestamp = frame_id / fps
                    log.append({"frame": frame_id, "time_s": round(timestamp, 2),
                                "id": objectID, "total": total_count})
            elif DIRECTION == "up":
                if prev_cy > count_line_y >= cy:
                    total_count += 1
                    counted_ids.add(objectID)
                    timestamp = frame_id / fps
                    log.append({"frame": frame_id, "time_s": round(timestamp, 2),
                                "id": objectID, "total": total_count})
            else:  # both directions
                if (prev_cy < count_line_y <= cy) or (prev_cy > count_line_y >= cy):
                    total_count += 1
                    counted_ids.add(objectID)
                    timestamp = frame_id / fps
                    log.append({"frame": frame_id, "time_s": round(timestamp, 2),
                                "id": objectID, "total": total_count})

    cv2.putText(frame, f"Count: {total_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)

    # show windows
    cv2.imshow("Video", frame)
    cv2.imshow("Foreground Mask", fg)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cleanup
cap.release()
cv2.destroyAllWindows()

# save CSV if requested
if SAVE_CSV and log:
    df = pd.DataFrame(log)
    df.to_csv(CSV_OUT, index=False)
    print("Saved counts to:", CSV_OUT)

print("Finished. Total vehicles counted:", total_count)
