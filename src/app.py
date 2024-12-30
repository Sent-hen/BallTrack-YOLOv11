from ultralytics import YOLO
import cv2
import numpy as np 
import random


model = YOLO('yolo11n.pt')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 900) 
cap.set(cv2.CAP_PROP_FPS, 30)
class_names = model.names

#for class_id, class_name in class_names.items():
#    print(f"Class ID: {class_id}, Class Name: {class_name}")

"""
Class ID: 0, Class Name: person
Class ID: 1, Class Name: bicycle
Class ID: 2, Class Name: car
Class ID: 3, Class Name: motorcycle
Class ID: 4, Class Name: airplane
Class ID: 5, Class Name: bus
Class ID: 6, Class Name: train
Class ID: 7, Class Name: truck
Class ID: 8, Class Name: boat
Class ID: 9, Class Name: traffic light
Class ID: 10, Class Name: fire hydrant
Class ID: 11, Class Name: stop sign
Class ID: 12, Class Name: parking meter
Class ID: 13, Class Name: bench
Class ID: 14, Class Name: bird
Class ID: 15, Class Name: cat
Class ID: 16, Class Name: dog
Class ID: 17, Class Name: horse
Class ID: 18, Class Name: sheep
Class ID: 19, Class Name: cow
Class ID: 20, Class Name: elephant
Class ID: 21, Class Name: bear
Class ID: 22, Class Name: zebra
Class ID: 23, Class Name: giraffe
Class ID: 24, Class Name: backpack
Class ID: 25, Class Name: umbrella
Class ID: 26, Class Name: handbag
Class ID: 27, Class Name: tie
Class ID: 28, Class Name: suitcase
Class ID: 29, Class Name: frisbee
Class ID: 30, Class Name: skis
Class ID: 31, Class Name: snowboard
Class ID: 32, Class Name: sports ball
Class ID: 33, Class Name: kite
Class ID: 34, Class Name: baseball bat
Class ID: 35, Class Name: baseball glove
Class ID: 36, Class Name: skateboard
Class ID: 37, Class Name: surfboard
Class ID: 38, Class Name: tennis racket
Class ID: 39, Class Name: bottle
Class ID: 40, Class Name: wine glass
Class ID: 41, Class Name: cup
Class ID: 42, Class Name: fork
Class ID: 43, Class Name: knife
Class ID: 44, Class Name: spoon
Class ID: 45, Class Name: bowl
Class ID: 46, Class Name: banana
Class ID: 47, Class Name: apple
Class ID: 48, Class Name: sandwich
Class ID: 49, Class Name: orange
Class ID: 50, Class Name: broccoli
Class ID: 51, Class Name: carrot
Class ID: 52, Class Name: hot dog
Class ID: 53, Class Name: pizza
Class ID: 54, Class Name: donut
Class ID: 55, Class Name: cake
Class ID: 56, Class Name: chair
Class ID: 57, Class Name: couch
Class ID: 58, Class Name: potted plant
Class ID: 59, Class Name: bed
Class ID: 60, Class Name: dining table
Class ID: 61, Class Name: toilet
Class ID: 62, Class Name: tv
Class ID: 63, Class Name: laptop
Class ID: 64, Class Name: mouse
Class ID: 65, Class Name: remote
Class ID: 66, Class Name: keyboard
Class ID: 67, Class Name: cell phone
Class ID: 68, Class Name: microwave
Class ID: 69, Class Name: oven
Class ID: 70, Class Name: toaster
Class ID: 71, Class Name: sink
Class ID: 72, Class Name: refrigerator
Class ID: 73, Class Name: book
Class ID: 74, Class Name: clock
Class ID: 75, Class Name: vase
Class ID: 76, Class Name: scissors
Class ID: 77, Class Name: teddy bear
Class ID: 78, Class Name: hair drier
Class ID: 79, Class Name: toothbrush
"""

# Define the semi-transparent box properties
box_color = (0, 255, 0)  # Initial color (green)
box_position = (600, 300, 1000, 600)  # x1, y1, x2, y2 (box coordinates)
score = 0  # Point tracker
box_scored = False  
boxes_completed = 0

def check_ball_in_box(center_x, center_y):
    global box_color, score, box_scored
    x1, y1, x2, y2 = box_position
    
    if x1 <= center_x <= x2 and y1 <= center_y <= y2:
        if not box_scored:  
            box_color = (0, 100, 0)  
            score += 1
            box_scored = True  
    else:
        box_color = (0, 255, 0)  
        box_scored = False  

def move_box_randomly(frame_width, frame_height):
    global box_position, score, boxes_completed
    
    # Only move the box if score reaches 5
    if score >= 5:
        # Get current box width and height
        box_width = box_position[2] - box_position[0]
        box_height = box_position[3] - box_position[1]
        
        # Generate new random position for the box, keeping its size constant
        new_x1 = random.randint(0, frame_width - box_width)
        new_y1 = random.randint(0, frame_height - box_height)
        
        # Calculate new x2 and y2 based on the same box size
        new_x2 = new_x1 + box_width
        new_y2 = new_y1 + box_height
        
        # Update the box's position
        box_position = (new_x1, new_y1, new_x2, new_y2)
        
        # Reset score and increment the boxes completed counter
        score = 0
        boxes_completed += 1

if not cap.isOpened():
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = frame.copy()

    overlay = annotated_frame.copy()
    x1, y1, x2, y2 = box_position
    cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, -1)
    cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            if int(box.cls) == 32: 
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                frame_height = 900
                conf = float(box.conf)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                radius = min(x2 - x1, y2 - y1) // 2

                
                check_ball_in_box(center_x, center_y)
                move_box_randomly(1600, 900)

                cv2.circle(annotated_frame, (center_x, center_y), radius, (255, 0, 0), 2)

    cv2.putText(annotated_frame, f"Score: {score}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Boxes Completed: {boxes_completed}",
                (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)


    cv2.imshow('YOLOv11 Live Object Detection', annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
