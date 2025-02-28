# Full Body Detection with OpenCV Python

## Overview

The **Full Body Detection OpenCV Python** project is designed to detect and count humans in real-time using a webcam or custom video/image input. The project leverages the OpenCV library, which is widely used in computer vision applications such as pedestrian detection, criminal identification, and healthcare monitoring. The system utilizes **Histogram of Oriented Gradients (HOG)**, a popular technique for object detection, along with **Support Vector Machine (SVM)** to perform human body detection.

## Features

- **Real-time Human Detection**: Detects humans in real-time via webcam or video input.
- **Object Detection**: Identifies and locates the human body within the frame.
- **Pedestrian Detection**: Can be used for pedestrian detection in various scenarios like surveillance.
- **Count Detection**: Counts the number of humans detected in the video or image.
- **Input Options**: Accepts webcam, image file, or video file as input.
- **Output**: Displays frames with detected humans highlighted by bounding boxes and can save the output to a file.

## Prerequisites

- Python 3.x installed.
- Basic knowledge of Python programming and OpenCV.

### Required Libraries

To install the required libraries, run the following commands:

```bash
pip install opencv-python
pip install imutils
pip install numpy
```

### Libraries Used

- **OpenCV**: A library for machine learning and computer vision tasks.
- **Imutils**: A library for image processing.
- **Numpy**: For scientific computing and handling images as numpy arrays.
- **Argparse**: For handling command-line arguments.

## Steps to Build the Project

1. **Import Libraries**: Import the necessary libraries for the project.
2. **Create Detection Model**: Use OpenCV's HOGDescriptor with SVM to create a model for human detection.
3. **Frame Detection**: Detect humans in each frame of the video or image.
4. **Human Detection Methods**: 
    - **From Webcam**: Capture video from the webcam.
    - **From Video File**: Read frames from a video file.
    - **From Image**: Detect humans in a given image.
5. **Process and Display Results**: Detect humans and display them with bounding boxes in real-time.
6. **Command-line Interface**: Use argparse to handle input from the command line for videos, images, and camera input.

## Code Walkthrough

### Step 1: Import Libraries
```python
import cv2
import imutils
import numpy as np
import argparse
```

### Step 2: Create Human Detection Model
```python
hog = cv2.HOGDescriptor_getDefaultPeopleDetector()
```

### Step 3: Detect Method
```python
def detect(frame):
    (humans, _) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
    for (x, y, w, h) in humans:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame
```

### Step 4: Human Detection by Camera
```python
def detect_by_camera():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect(frame)
        cv2.imshow('Human Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
```

### Step 5: Human Detection by Path (Video or Image)
```python
def detect_by_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect(frame)
        cv2.imshow('Human Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
```

### Step 6: Command-line Argument Parsing
```python
def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time Human Detection")
    parser.add_argument('-v', '--video', type=str, help="Path to video file")
    parser.add_argument('-i', '--image', type=str, help="Path to image file")
    parser.add_argument('-c', '--camera', type=bool, default=False, help="Use webcam for real-time detection")
    parser.add_argument('-o', '--output', type=str, help="Output file name to save results")
    return parser.parse_args()
```

### Step 7: Main Function
```python
def main():
    args = parse_args()

    if args.camera:
        detect_by_camera()
    elif args.video:
        detect_by_video(args.video)
    elif args.image:
        image = cv2.imread(args.image)
        image = detect(image)
        cv2.imshow('Human Detection', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
```

## Running the Project

### To run the project, use the following commands:

1. **For webcam detection**:
   ```bash
   python main.py -c True
   ```

2. **For video file input**:
   ```bash
   python main.py -v 'Path_to_video'
   ```

3. **For image file input**:
   ```bash
   python main.py -i 'Path_to_image'
   ```

4. **To save the output**:
   ```bash
   python main.py -c True -o 'output_file_name'
   ```

## Conclusion

This **Full Body Detection** project in Python using OpenCV allows for real-time human detection in webcam streams, images, and videos. It utilizes efficient algorithms like HOG and SVM for accurate detection and is a great starting point for computer vision applications in various fields like security, healthcare, and more.
