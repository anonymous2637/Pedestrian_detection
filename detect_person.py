import cv2
import time
from ultralytics import YOLO
from db import save_to_db
from excel import save_to_excel
from roi import is_inside_roi, draw_roi  # Your ROI module

# Camera URL and YOLO model
ip_camera_url = "rtsp://admin:admin123@192.168.1.213/cam/realmonitor?channel=1&subtype=0"
model = YOLO("yolov5mu.pt")
confidence_threshold = 0.3

# Connect to IP camera
def connect_camera(url):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

# Main video processing function
def process_frames():
    cap = connect_camera(ip_camera_url)
    last_detection_time = 0

    cv2.namedWindow("IP Camera Feed", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("IP Camera Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        for _ in range(5):
            cap.grab()

        success, frame = cap.read()
        if not success:
            print("Lost connection. Reconnecting...")
            cap.release()
            time.sleep(1)
            cap = connect_camera(ip_camera_url)
            continue

        draw_roi(frame)

        results = model(frame, conf=confidence_threshold)
        for result in results:
            people = [box for box in result.boxes if int(box.cls[0]) == 0 and box.conf[0] > confidence_threshold]
            person_count = 0

            for box in people:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if is_inside_roi(cx, cy):
                    person_count += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if person_count > 0 and time.time() - last_detection_time > 2:
                save_to_db(person_count)
                save_to_excel(person_count)
                last_detection_time = time.time()

        cv2.imshow("IP Camera Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_frames()


