from ultralytics import YOLO
import cv2

model = YOLO("AOD_weapons.pt")

# Test on an image
results = model("test_image2.jpg", show=True)

cv2.waitKey(0)
cv2.destroyAllWindows()