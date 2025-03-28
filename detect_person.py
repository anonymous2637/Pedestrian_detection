import cv2
import time
import sqlite3
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
from openpyxl import load_workbook

# RTSP Camera URL
ip_camera_url = "rtsp://admin:admin123@192.168.1.213/cam/realmonitor?channel=1&subtype=0"

# Load YOLO model
model = YOLO("yolov5mu.pt")
confidence_threshold = 0.3  # Only detect persons with confidence > 0.5

# Connect to SQLite database
conn = sqlite3.connect("detections.db")
cursor = conn.cursor()

# Create table if it doesn't exist
cursor.execute("""
    CREATE TABLE IF NOT EXISTS person_detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        time TEXT,
        person_count INTEGER
    )
""")
conn.commit()

# Function to insert detection into database and Excel
def save_detection(person_count):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")  # YYYY-MM-DD
    time_12hr = now.strftime("%I:%M:%S %p")  # 12-hour format

    # Save to SQLite
    cursor.execute("INSERT INTO person_detections (date, time, person_count) VALUES (?, ?, ?)",
                   (date, time_12hr, person_count))
    conn.commit()

    # Save to Excel
    df = pd.DataFrame([[date, time_12hr, person_count]], columns=["Date", "Time", "Person Count"])
    
    excel_file = "detections.xlsx"
    
    try:
        with open(excel_file, "rb"):
            existing_file = True
    except FileNotFoundError:
        existing_file = False

    if existing_file:
        with pd.ExcelWriter(excel_file, mode="a", if_sheet_exists="overlay", engine="openpyxl") as writer:
            workbook = load_workbook(excel_file)
            sheet = workbook.active
            df.to_excel(writer, index=False, header=False, sheet_name="Detections", startrow=sheet.max_row)
    else:
        df.to_excel(excel_file, index=False, sheet_name="Detections")

# Function to connect to the IP camera and get full resolution
def connect_camera(url):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Get original resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, width, height

cap, width, height = connect_camera(ip_camera_url)
cv2.namedWindow("Person Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Person Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

last_detection_time = 0  # To avoid duplicate logging

while True:
    for _ in range(5):
        cap.grab()  # Drop outdated frames

    ret, frame = cap.read()
    
    if not ret:
        print("Lost connection. Reconnecting...")
        cap.release()
        time.sleep(1)
        cap, width, height = connect_camera(ip_camera_url)
        continue

    frame = cv2.resize(frame, (width, height))  # Ensure full resolution
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run YOLO model
    results = model(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), conf=confidence_threshold)

    for result in results:
        people = [box for box in result.boxes if int(box.cls[0]) == 0 and box.conf[0] > confidence_threshold]

        # Count the number of detected persons
        person_count = len(people)

        # Draw bounding boxes and labels
        for box in people:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save to database & Excel (avoid duplicate logging within 2 second)
        current_time = time.time()
        if person_count > 0 and current_time - last_detection_time > 2:
            save_detection(person_count)
            last_detection_time = current_time

    cv2.imshow("Person Detection", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()  # Close database connection