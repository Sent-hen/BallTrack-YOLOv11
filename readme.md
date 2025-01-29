# Ball Detection Game

A real-time interactive game that uses computer vision to detect a sports ball and track its position relative to a target box on the screen. Built with YOLO object detection and OpenCV.

## Description

This application uses your computer's webcam to detect a sports ball in real-time. Players must move the ball into a semi-transparent green box on the screen. Once the ball has been successfully held in the box five times, the box will randomly move to a new location and the challenge continues.

## Features

- Real-time ball detection using YOLOv11
- Score tracking system
- Moving target box with transparency
- Performance counter for completed box challenges
- High-resolution webcam support (1600x900)
- 30 FPS capture rate

## Requirements

- Python 3.x
- OpenCV (cv2)
- Ultralytics YOLO
- NumPy
- A webcam
- A sports ball (basketball, soccer ball, tennis ball, etc.)

## Installation

1. Install the required Python packages:
```bash
pip install ultralytics opencv-python numpy
```

2. Download the YOLO model file:
- Place `yolo11n.pt` in your project directory

## Usage

1. Run the script:
```bash
python ball_detection_game.py
```

2. Hold up a sports ball in view of your webcam
3. Move the ball into the green box on screen
4. Keep the ball in the box to score points
5. After 5 points, the box will move to a new random location
6. Press 'q' to quit the application

## Game Rules

- The green box serves as your target area
- Successfully holding the ball in the box increases your score
- The box turns darker green when the ball is correctly positioned
- After reaching 5 points, the box moves to a random location
- The "Boxes Completed" counter tracks how many times you've successfully moved the box

## Controls

- `q` - Quit the application
- Physical ball movement controls the game

## Technical Details

- Camera Resolution: 1600x900
- Frame Rate: 30 FPS
- Detection Model: YOLOv11
- Object Class: Sports Ball (Class ID: 32)

## Troubleshooting

If you encounter issues:

1. Ensure your webcam is properly connected and accessible
2. Check that the YOLO model file is in the correct directory
3. Verify all dependencies are installed correctly
4. Ensure adequate lighting for better ball detection

## Notes

- The application is optimized for standard webcam usage
- Performance may vary based on your hardware capabilities
- Good lighting conditions will improve ball detection accuracy

## Contributing

Feel free to fork this repository and submit pull requests for any improvements you develop.
