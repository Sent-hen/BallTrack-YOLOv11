from ultralytics import YOLO
import cv2

model = YOLO('yolo11n.pt')
cap = cv2.VideoCapture(0)

class_names = model.names

for class_id, class_name in class_names.items():
    print(f"Class ID: {class_id}, Class Name: {class_name}")


if not cap.isOpened():
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow('YOLOv11 Live Object Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
