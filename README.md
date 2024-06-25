# Soccer Game Video Processing

## Overview
This project processes soccer game videos to correct distortion, extract homography, identify objects, identify teams, recognize jersey numbers, track objects, and save the processed video. 

## Project Structure
```
project/
│
│── files/
│ └── models/
│ └── videos/
│ └── workbooks/
│── utils/
│ └── distortion_correction.py
│ └── homography_extraction.py
│ └── tiling.py
│ └── object_detection.py
│ └── team_identification.py
│ └── jersey_recognition.py
│ └── object_tracking.py
└── main.py
```

## Instructions
1. Place your input video in the `files/videos/input/` folder.
2. Run the `main.py` script.
3. The processed video will be saved in the `files/videos/output/` folder.

## Code Base
| Step | Process                   | Library/Approach                                                                                                    |
|------|---------------------------|---------------------------------------------------------------------------------------------------------------------|
| 1    | Distortion Correction     | Discorpy (correct radial distortion without a calibration target)                                                   |
| 2    | Homography Extraction     | OpenCV (cv.findHomography, cv.warpPerspective)                                                                      |
| 3    | Tiling                    | SAHI by Ultralytics (Slicing Aided Hyper Inference for partitioning large images and optimizing detection)          |
| 4    | Object Identification     | Ultralytics YOLOv8n (state-of-the-art object detection model, easy to use and highly accurate)                      |
| 5    | Team Identification       | OpenCV (cv2.inRange for color segmentation to identify teams)                                                       |
| 6    | Jersey Number Recognition | EasyOCR (deep learning-based OCR tool that is highly accurate and easy to use)                                      |
| 7    | Object Tracking           | ByteTrack by Supervision (one-shot detection-based approach)                                                        |

## Requirements
- Python 3.8+
- OpenCV
- NumPy
- EasyOCR
- Ultralytics YOLOv8
- BYTETracker
