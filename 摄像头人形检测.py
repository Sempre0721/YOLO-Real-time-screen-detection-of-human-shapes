import cv2
from ultralytics import YOLO
model = YOLO('yolov8n-pose.pt')
cap = cv2.VideoCapture(0) 
while cap.isOpened():
    success, frame_init = cap.read()
    frame = cv2.flip(frame_init, 1)
    if success:
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow('YOLOv8 Inference',annotated_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
