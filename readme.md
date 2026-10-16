# BallTrack (YOLOv11)

A small webcam game. YOLOv11 finds a ball in the camera feed, and you have to move it into a green box drawn on the screen. Put it in the box five times and the box jumps somewhere else.

## Setup

You need Python 3, a webcam and some kind of ball (basketball, soccer ball, tennis ball, anything YOLO calls a "sports ball").

```bash
pip install ultralytics opencv-python numpy
```

The model weights (`yolo11n.pt`) are already in the repo.

## Running it

```bash
python src/app.py
```

Hold the ball up to the camera and move it into the green box. The box turns darker when the ball is inside. Each time the ball goes into the box you get a point. At 5 points the box moves to a random spot, the score resets, and the "Boxes Completed" counter goes up by one. Press `q` to quit.

## Details

- The camera is set to 1600×900 at 30 FPS.
- Only COCO class 32 (sports ball) counts. The class list is in a comment at the top of `app.py`.
- Detection gets noticeably worse in dim light, so play somewhere bright.
- If nothing shows up, check that the webcam isn't in use by another app.
