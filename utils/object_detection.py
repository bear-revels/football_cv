"""
Module for object detection in video frames.
"""

from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel
import torch
import os

class ObjectDetection:
    def __init__(self, model_path='files/models/trackbox_yolov8n.pt'):
        self.model_path = model_path
        
        # Initialize the SAHI detection model with YOLOv8
        self.model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=model_path,
            confidence_threshold=0.3,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Download YOLOv8 model if not already available
        if not os.path.exists(model_path):
            from sahi.utils.yolov8 import download_yolov8s_model
            download_yolov8s_model(model_path)

    def detect(self, frame):
        """
        Detect objects in the given frame using SAHI and YOLOv8.

        Args:
            frame (numpy.ndarray): The input video frame.

        Returns:
            list: A list of detected objects.
        """
        # Use SAHI to slice the image and get predictions
        result = get_sliced_prediction(
            image=frame,
            detection_model=self.model,
            slice_height=256,
            slice_width=256,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )
        
        detections = []
        for pred in result.object_prediction_list:
            bbox = pred.bbox
            detections.append({
                'bbox': [bbox.minx, bbox.miny, bbox.maxx, bbox.maxy],  # Bounding box coordinates
                'confidence': pred.score.value,  # Confidence score
                'class': pred.category.name  # Class label
            })
        return detections