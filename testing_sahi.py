import cv2
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel
from ultralytics import YOLO
import numpy as np

# Initialize detection model
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path='files/models/trackbox_yolov8n.pt',
    confidence_threshold=0.3,
    device="cpu",  # or 'cpu'
)

# Open the video file
video_path = "files/videos/input/trackbox_short.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('files/videos/output/test_video.mp4', fourcc, fps, (frame_width, frame_height))

# Process each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    result = get_sliced_prediction(
        frame,
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )

    # Draw bounding boxes without labels
    for detection in result.object_prediction_list:
        bbox = detection.bbox
        x1, y1, x2, y2 = map(int, [bbox.minx, bbox.miny, bbox.maxx, bbox.maxy])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Write the frame to the output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()