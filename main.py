"""
Main script for processing soccer game videos.

This script ingests a video file, processes it to correct distortion,
extracts homography, identifies objects, identifies players, recognizes jersey numbers,
tracks objects, and saves the processed video.
"""

from utils.video_utils import ProcessVideo
from utils.distortion_correction import DistortionCorrection
from utils.homography_extraction import HomographyExtraction
from utils.object_detection import ObjectDetection
from utils.player_identification import PlayerIdentification
from utils.object_tracking import ObjectTracking
import numpy as np

def main(input_video_path, output_video_path):
    # Initialize utility classes
    video_processor = ProcessVideo(input_video_path, output_video_path)
    distortion_corrector = DistortionCorrection()
    homography_extractor = HomographyExtraction()
    object_detector = ObjectDetection()
    player_identifier = PlayerIdentification()
    object_tracker = ObjectTracking()

    # Load video
    video, output_writer = video_processor.load_video()

    # Process video frame by frame
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # Correct distortion
        frame = distortion_corrector.correct(frame)

        # Extract homography
        homography_matrix = homography_extractor.extract(frame)

        # Apply homography
        frame = homography_extractor.apply_homography(frame, homography_matrix)

        # Detect objects
        objects = object_detector.detect(frame)

        # Convert objects to numpy array for the tracker, including a dummy class ID
        objects_np = np.array(
            [[*obj['bbox'], obj['confidence'], 0] for obj in objects],  # Add 0 as the class ID
            dtype=np.float32  # Ensure the correct numeric data type
        )

        # Identify players and recognize jersey numbers
        player_info = player_identifier.identify(objects, frame)

        # Track objects
        tracked_objects = object_tracker.track(objects_np, frame)

        # Annotate frame with detections, teams, and jerseys
        frame = annotate_frame(frame, tracked_objects, player_info)

        # Save frame to output video
        video_processor.save_video(frame, output_writer)

    video.release()
    output_writer.release()

def annotate_frame(frame, tracked_objects, player_info):
    # Add annotations to the frame (implement as needed)
    return frame

if __name__ == "__main__":
    input_video_path = 'files/videos/input/trackbox_short.mp4'
    output_video_path = 'files/videos/output/trackbox_short_processed.mp4'
    main(input_video_path, output_video_path)