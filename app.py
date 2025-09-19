# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 08:49:23 2025

@author: harik
"""

import os
from flask import Flask, render_template_string, request, Response
from werkzeug.utils import secure_filename
import cv2
import time
from ultralytics import YOLO

# -------- CONFIG --------
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
MODEL_PATH = "yolov8n.pt"
LINE_POS_RATIO = 0.72
MIN_GREEN_TIME = 5 # Minimum time a light stays green (in seconds)
MAX_GREEN_TIME = 30 # Maximum time a light stays green (in seconds)
# ------------------------

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load YOLO model once
print("📥 Loading YOLOv8 model...")
model = YOLO(MODEL_PATH)
print("✅ Model loaded successfully")

# Global variables for state management
# These will be reset for each new video upload
left_count = 0
right_count = 0
uncounted_vehicles = {"left": 0, "right": 0}
counted_left_ids = set()
counted_right_ids = set()
signal_status = {"left": "RED", "right": "RED"}
current_green = "left"
last_switch_time = time.time()
video_path = None

def allowed_file(filename):
    """Check if the uploaded file has a valid extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video_stream():
    """
    Generator function to process video frames and yield them as JPEG images.
    """
    global left_count, right_count, uncounted_vehicles, counted_left_ids, counted_right_ids, \
            signal_status, current_green, last_switch_time, video_path

    if not video_path:
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file at {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count_line_y = int(frame_height * LINE_POS_RATIO)
    divider_x = frame_width // 2

    # Reset counts for the new video
    left_count = 0
    right_count = 0
    counted_left_ids.clear()
    counted_right_ids.clear()
    current_green = "left"
    last_switch_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            # End of video stream, loop the video
            cap.release()
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if not ret:
                break
        
        results = model.track(frame, persist=True, show=False)
        
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
                
                if classes[i] in [2, 3, 5, 7]: # Car, Motorcycle, Bus, Truck
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID {int(track_id)}", (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    if cy > count_line_y * 0.5:
                        if cx < divider_x:
                            uncounted_vehicles["left"] += 1
                        else:
                            uncounted_vehicles["right"] += 1
                    
                    if cx < divider_x:
                        if track_id not in counted_left_ids and cy > count_line_y:
                            left_count += 1
                            counted_left_ids.add(track_id)
                    else:
                        if track_id not in counted_right_ids and cy > count_line_y:
                            right_count += 1
                            counted_right_ids.add(track_id)
        
        now = time.time()
        elapsed_time = now - last_switch_time

        if elapsed_time >= MIN_GREEN_TIME:
            if current_green == "left" and (elapsed_time >= MAX_GREEN_TIME or uncounted_vehicles["right"] > uncounted_vehicles["left"]):
                current_green = "right"
                last_switch_time = now
            elif current_green == "right" and (elapsed_time >= MAX_GREEN_TIME or uncounted_vehicles["left"] > uncounted_vehicles["right"]):
                current_green = "left"
                last_switch_time = now

        if current_green == "left":
            signal_status = {"left": "GREEN", "right": "RED"}
        else:
            signal_status = {"left": "RED", "right": "GREEN"}
            
        # Draw counts on frame
        cv2.putText(frame, f"Left Lane: {left_count}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Right Lane: {right_count}", (frame_width//2 + 30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global video_path
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)
            return """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Analysis in Progress</title>
                <style>
                    body {
                        font-family: sans-serif;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        background-color: #333;
                        color: white;
                        padding: 20px;
                        box-sizing: border-box;
                    }
                    .main-container {
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        width: 95%;
                        max-width: 650px;
                    }
                    #video-container {
                        border: 5px solid #00ff99;
                        box-shadow: 0 0 20px rgba(0, 255, 153, 0.5);
                        margin-bottom: 20px;
                        width: 100%;
                        height: auto;
                    }
                    #video-feed {
                        display: block;
                        width: 100%;
                        height: auto;
                    }
                    .controls {
                        display: flex;
                        justify-content: space-around;
                        width: 100%;
                        flex-wrap: wrap;
                    }
                    .signal-box {
                        background-color: #444;
                        border-radius: 10px;
                        padding: 20px;
                        text-align: center;
                        flex: 1;
                        min-width: 250px;
                        margin: 10px;
                    }
                    .signal-light {
                        width: 50px;
                        height: 50px;
                        border-radius: 50%;
                        margin: 10px auto;
                        border: 2px solid #fff;
                    }
                    .red-light { background-color: #ff6666; }
                    .green-light { background-color: #66ff66; }
                    .count {
                        font-size: 2em;
                        font-weight: bold;
                        color: #fff;
                    }
                </style>
            </head>
            <body>
                <div class="main-container">
                    <div id="video-container">
                        <img id="video-feed" src="/video">
                    </div>
                    <div class="controls">
                        <div class="signal-box">
                            <h2>Left Lane</h2>
                            <div id="left-signal" class="signal-light"></div>
                            <p>Vehicle Count: <span id="left-count" class="count">0</span></p>
                        </div>
                        <div class="signal-box">
                            <h2>Right Lane</h2>
                            <div id="right-signal" class="signal-light"></div>
                            <p>Vehicle Count: <span id="right-count" class="count">0</span></p>
                        </div>
                    </div>
                </div>
                <script>
                    const leftCountElem = document.getElementById('left-count');
                    const rightCountElem = document.getElementById('right-count');
                    const leftSignalElem = document.getElementById('left-signal');
                    const rightSignalElem = document.getElementById('right-signal');
                    
                    function fetchData() {
                        fetch('/data')
                            .then(response => response.json())
                            .then(data => {
                                leftCountElem.textContent = data.left_count;
                                rightCountElem.textContent = data.right_count;
                                if (data.signal_status.left === 'GREEN') {
                                    leftSignalElem.classList.add('green-light');
                                    leftSignalElem.classList.remove('red-light');
                                    rightSignalElem.classList.add('red-light');
                                    rightSignalElem.classList.remove('green-light');
                                } else {
                                    leftSignalElem.classList.add('red-light');
                                    leftSignalElem.classList.remove('green-light');
                                    rightSignalElem.classList.add('green-light');
                                    rightSignalElem.classList.remove('red-light');
                                }
                            })
                            .catch(error => console.error('Error fetching data:', error));
                    }
                    
                    setInterval(fetchData, 1000); // Fetch data every second
                </script>
            </body>
            </html>
            """
    
    # HTML for the initial upload form
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Upload Video for Analysis</title>
        <style>
            body {
                font-family: sans-serif;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                min-height: 100vh;
                background-color: #333;
                color: white;
            }
            .upload-card {
                background-color: #444;
                padding: 40px;
                border-radius: 10px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.5);
                text-align: center;
            }
            .upload-card h1 {
                margin-bottom: 20px;
            }
            .upload-form {
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            .file-input {
                margin-bottom: 20px;
                padding: 10px;
                border-radius: 5px;
                background-color: #555;
                color: white;
                border: 1px solid #777;
            }
            .submit-btn {
                padding: 10px 20px;
                font-size: 1em;
                background-color: #00ff99;
                color: #333;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                transition: background-color 0.3s;
            }
            .submit-btn:hover {
                background-color: #00cc88;
            }
        </style>
    </head>
    <body>
        <div class="upload-card">
            <h1>Upload Traffic Video</h1>
            <form method="POST" enctype="multipart/form-data" class="upload-form">
                <input type="file" name="file" accept="video/*" class="file-input">
                <button type="submit" class="submit-btn">Start Analysis</button>
            </form>
        </div>
    </body>
    </html>
    """

@app.route('/video')
def video_feed():
    return Response(process_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data')
def get_data():
    global left_count, right_count, signal_status
    return {
        "left_count": left_count,
        "right_count": right_count,
        "signal_status": signal_status
    }

if __name__ == '__main__':
    app.run(debug=True)
