# ai-virtual-mouse

## Overview
This project implements a virtual mouse system using hand tracking and gesture recognition with the help of OpenCV, MediaPipe, and AutoPy libraries. The system allows you to control your mouse cursor and perform click actions using hand gestures detected from a webcam feed.

## Features
- **Hand Tracking:** Detect and track hand landmarks using the MediaPipe library.
- **Virtual Mouse Control:** Move the mouse cursor with your hand movements.
- **Click Action:** Perform a left-click action by pinching the index and middle fingers together.
- **Smooth Movement:** Smoothen mouse movements for a more user-friendly experience.
- **Dynamic Control Area:** Control the movement area of the virtual mouse with a predefined frame.

## Project Setup
### Prerequisites
- Python 3.7+
- OpenCV: For capturing webcam feed and displaying images.
- MediaPipe: For detecting and tracking hand landmarks.
- AutoPy: For controlling mouse movements and clicks programmatically.
- NumPy: For handling array operations and calculations.
- Anaconda or Miniconda installed on your machine

### Installation
1. Clone the repository:
```bash
git clone https://github.com/Kanon14/ai-virtual-mouse.git
cd ai-virtual-mouse
``` 

2. Create and activate a Conda environment:
```bash
conda create -n ai_mouse python=3.8 -y
conda activate ai_mouse
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## How to Run
1. Execute the project:
```bash
python ai_mouse.py
```
2. Hand Gestures for Control:
- **Move Mouse Cursor:** Raise only the index finger and move it around to control the cursor.
- **Left Click:** Raise both the index and middle fingers and bring them together (pinch) to perform a left-click.
3. Adjusting Parameters:
- You can modify parameters like `frameR` (frame reduction) and `smoothening` in the script to fine-tune the control sensitivity and smoothness.

## Limitations
- Requires good lighting conditions for accurate hand tracking.
- May not work well with multiple hands in the frame.
- Requires the hand to be in the defined frame area for proper control.

## Future Improvements
- Add support for right-click and scrolling gestures.
- Implement dynamic gesture recognition for more controls.
- Enhance the tracking accuracy using more advanced machine learning models.