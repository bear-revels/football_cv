"""
Module for video processing utilities including loading and saving videos.
"""

import cv2

class ProcessVideo:
    def __init__(self, input_video_path, output_video_path):
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path

    def load_video(self):
        """
        Load the input video.

        Returns:
            cv2.VideoCapture: The video capture object.
            cv2.VideoWriter: The video writer object for saving the output.
        """
        video = cv2.VideoCapture(self.input_video_path)

        # Get video properties
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_writer = cv2.VideoWriter(self.output_video_path, fourcc, fps, (frame_width, frame_height))

        return video, output_writer

    def save_video(self, frame, output_writer):
        """
        Save a frame to the output video.

        Args:
            frame (numpy.ndarray): The frame to be saved.
            output_writer (cv2.VideoWriter): The video writer object.
        """
        output_writer.write(frame)