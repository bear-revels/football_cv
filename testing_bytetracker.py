import cv2
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel
import numpy as np
import supervision as sv

# Initialize detection model
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path='files/models/trackbox_yolov8n.pt',
    confidence_threshold=0.3,
    device="cpu"  # or 'cuda' if you have a GPU
)

# Initialize ByteTrack tracker
tracker = sv.ByteTrack()

# Open the video file
video_path = "files/videos/input/trackbox_short.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('files/videos/output/test_video.mp4', fourcc, fps, (frame_width, frame_height))

def process_frame(frame):
    result = get_sliced_prediction(
        frame,
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )

    detections = sv.Detections.from_sahi(result)
    detections = tracker.update_with_detections(detections)

    labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
    
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    annotated_frame = bounding_box_annotator.annotate(
        scene=frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels)

    return annotated_frame

# Process each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = process_frame(frame)
    out.write(processed_frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()