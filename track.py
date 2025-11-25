import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

model = YOLO("yolov8n.pt")  # using the nano version for faster inference

tracker = DeepSort(
    max_age=60,               # How many frames to keep a track alive without detection
    n_init=3,                 # Minimum frames before confirming a track
    max_iou_distance=0.6,     # IOU threshold for associating new boxes with existing tracks
    max_cosine_distance=0.3,  # Appearance feature similarity threshold
    nn_budget=100             # Max number of feature vectors to store per track
)

cap = cv2.VideoCapture("videos/sydney_walking.mp4")
if not cap.isOpened():
    print("Could not open video file")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    frame_count += 1
    print(f"\nProcessing frame {frame_count}")

    results = model(frame, verbose=False)[0]

    detections = []

    # Parse YOLOv8 results and keep only 'person' class (class 0)
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls != 0:
            continue  # skip non-person classes

        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        # Convert to (x, y, w, h) format for DeepSORT
        bbox = [x1, y1, x2 - x1, y2 - y1]
        detections.append((bbox, conf, "person"))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue  # ignore unconfirmed or lost tracks

        track_id = track.track_id
        ltrb = track.to_ltrb()  # returns left, top, right, bottom
        x1, y1, x2, y2 = map(int, ltrb)

        # Draw bounding box and ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame, f"ID: {track_id}", (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

    # Show the frame with tracking
    cv2.imshow("YOLOv8 + DeepSORT Realtime", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
