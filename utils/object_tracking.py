"""
Module for tracking objects in video frames.
"""

from bytetracker import BYTETracker
import numpy as np

class ObjectTracking:
    def __init__(self):
        self.tracker = BYTETracker()

    def track(self, objects, frame):
        """
        Track objects across video frames using ByteTrack.

        Args:
            objects (numpy.ndarray): The detected objects as numpy array.
            frame (numpy.ndarray): The current video frame.

        Returns:
            list: A list of tracked objects.
        """
        # Convert objects to the format required by ByteTrack
        if len(objects) > 0:
            dets = np.array(objects, dtype=np.float32)  # Ensure correct dtype
        else:
            dets = np.empty((0, 6), dtype=np.float32)

        return self.tracker.update(dets, frame)
