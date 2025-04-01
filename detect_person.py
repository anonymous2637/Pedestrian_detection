import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import sqlite3
import pandas as pd

# RTSP Camera URL
ip_camera_url = "rtsp://admin:admin123@192.168.1.213/cam/realmonitor?channel=1&subtype=0"

# Initialize Video Capture with FFMPEG (Better Performance)
cap = cv2.VideoCapture(ip_camera_url, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Load YOLOv5mu Model (More Optimized)
model = YOLO("yolov5mu.pt")

# Confidence Threshold for Detection
confidence_threshold = 0.

# SQLite Database Setup
db_conn = sqlite3.connect("pedestrian_data.db")
cursor = db_conn.cursor()

# Create Table if Not Exists
cursor.execute("""
    CREATE TABLE IF NOT EXISTS person_detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        time TEXT,
        person_count INTEGER
    )
""")
db_conn.commit()

# Excel Data Storage
excel_data = []

print("🚀 Pedestrian detection started... Press 'q' to exit.")

while True:
    for _ in range(5): 
        cap.grab()  # Skip old frames to reduce lag

    ret, frame = cap.read()
    if not ret:
        print("Reconnecting...")
        cap.release()
        cap = cv2.VideoCapture(ip_camera_url, cv2.CAP_FFMPEG)
        continue

    # Resize for Performance
    frame = cv2.resize(frame, (640, 480))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Denoising (Reduces Noise in Low Light)
    frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

    # Run YOLOv5mu for Person Detection
    results = model(frame, conf=confidence_threshold)

    person_count = 0
    person_detected = False  # To track if person is detected

    for result in results:
        people = [box for box in result.boxes if int(box.cls[0]) == 0 and box.conf[0] > confidence_threshold]
        person_count = len(people)

        for box in people:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw Box
            cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if person_count > 0:
            person_detected = True  # Set to True when a person is detected

    # Show "Person Detected" with LED simulation when person is detected
    if person_detected:
        # Draw the LED (glowing effect)
        cv2.circle(frame, (600, 30), 15, (0, 255, 0), -1)  # LED circle
        cv2.putText(frame, "Person Detected", (620, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)  # Text

    # Get Date & Time (12-Hour Format)
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time_12hr = now.strftime("%I:%M:%S %p")  # 12-Hour Format

    # Save Data to SQLite Database
    cursor.execute("INSERT INTO person_detections (date, time, person_count) VALUES (?, ?, ?)", (date, time_12hr, person_count))
    db_conn.commit()

    # Append Data for Excel File
    excel_data.append([date, time_12hr, person_count])

    # Display the Frame
    cv2.imshow("Person Detection", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # Press 'q' to Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save Data to Excel File
df = pd.DataFrame(excel_data, columns=["Date", "Time", "Person Count"])
df.to_excel("pedestrian_data.xlsx", index=False)

# Cleanup
cap.release()
cv2.destroyAllWindows()
db_conn.close()
print("✅ Data saved successfully to pedestrian_data.db and pedestrian_data.xlsx")
