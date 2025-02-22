
## How to Run the Code on Your Image

### 1. Download the Generated Image
Click on the AI-generated image provided earlier and save it as `car_damage.jpg`.

### 2. Install Dependencies
Ensure you have the necessary libraries installed:
```bash
pip install ultralytics opencv-python numpy torch
```

### 3. Run the Code with Adjustments
Update and execute the following Python script to process your image:

```python
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
```

### 4. Expected Output
- The model will detect damaged areas and highlight them with green bounding boxes.
- The processed image will be displayed in an OpenCV window titled "Damage Detection".

### 5. Troubleshooting
If the detection is not working as expected, try the following:
- Ensure `yolov8_damage.pt` exists in the same directory.
- Lower the confidence threshold in the model:
  ```python
  results = model(image, conf=0.4)
  ```
- Check that `car_damage.jpg` is properly loaded.

    


