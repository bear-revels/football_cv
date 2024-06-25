"""
Module for identifying players based on jersey colors and recognizing jersey numbers using OCR.
"""

import cv2
import easyocr

class PlayerIdentification:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])

    def identify(self, objects, frame):
        """
        Identify teams and recognize jersey numbers for the detected objects.

        Args:
            objects (list): The detected objects.
            frame (numpy.ndarray): The input video frame.

        Returns:
            list: A list of player information including team and jersey number.
        """
        player_info = []
        for obj in objects:
            team = self.identify_team(obj, frame)
            jersey_number = self.recognize_jersey(obj, frame)
            player_info.append({'object': obj, 'team': team, 'jersey_number': jersey_number})
        return player_info

    def identify_team(self, obj, frame):
        """
        Identify the team for the detected object based on jersey color.

        Args:
            obj (dict): The detected object.
            frame (numpy.ndarray): The input video frame.

        Returns:
            str: The identified team.
        """
        x, y, w, h = map(int, obj['bbox'])  # Convert coordinates to integers
        roi = frame[y:y+h, x:x+w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        team1_color_range = ((0, 100, 100), (10, 255, 255))  # Example color range for team 1
        team2_color_range = ((110, 100, 100), (130, 255, 255))  # Example color range for team 2

        mask_team1 = cv2.inRange(hsv_roi, team1_color_range[0], team1_color_range[1])
        mask_team2 = cv2.inRange(hsv_roi, team2_color_range[0], team2_color_range[1])

        if cv2.countNonZero(mask_team1) > cv2.countNonZero(mask_team2):
            return 'team1'
        else:
            return 'team2'

    def recognize_jersey(self, obj, frame):
        """
        Recognize the jersey number for the detected object using OCR.

        Args:
            obj (dict): The detected object.
            frame (numpy.ndarray): The input video frame.

        Returns:
            str: The recognized jersey number.
        """
        x, y, w, h = map(int, obj['bbox'])  # Convert coordinates to integers
        roi = frame[y:y+h, x:x+w]
        result = self.reader.readtext(roi)
        if result:
            return result[0][-2]  # Return the recognized text
        return None