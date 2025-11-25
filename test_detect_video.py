from ultralytics import YOLO
import cv2

# Load the model
model = YOLO("yolov8n.pt")  # Use 'yolov8s.pt' or your own trained model if needed

# Load the video
video_path = "classroom.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Prepare output writer
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run inference
    results = model(frame, stream=True)

    # Draw results on frame
    for r in results:
        annotated_frame = r.plot()

    # Display and save
    cv2.imshow("YOLOv8 Video Detection", annotated_frame)
    out.write(annotated_frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
out.release()
cv2.destroyAllWindows()
