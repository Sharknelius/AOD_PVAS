from ultralytics import YOLO
import cv2

model = YOLO("best.pt")

# Test on an image
results = model("test_image2.jpg", conf=0.1, show=True)

cv2.waitKey(0)
cv2.destroyAllWindows()