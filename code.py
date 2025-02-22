import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Load Pre-trained YOLO Model
model = YOLO("yolov8_damage.pt")  # Ensure this model exists

# Load Image
image_path = "car_damage.jpg"  # Use the downloaded image
image = cv2.imread(image_path)

# Run Damage Detection
results = model(image)

# Display Results
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()  # Extract bounding boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw box

cv2.imshow("Damage Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
