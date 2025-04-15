from ultralytics import YOLO
import cv2

# Test on an image
model = YOLO("yolo11s_AOD3.pt")

# Replace file in quotes with a path to your test image  
results = model("test_image.jpg", show=True)

cv2.waitKey(0)
cv2.destroyAllWindows()