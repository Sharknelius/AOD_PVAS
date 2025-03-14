import os
import cv2
from tracker import ObjectCounter  # Importing ObjectCounter from tracker.py

# Define the mouse callback function
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  # Check for mouse movement
        point = [x, y]
        print(f"Mouse moved to: {point}")

ret = False # Set to true if recording live

if ret:
    cap = cv2.VideoCapture(0) # Live video feed
else:
    cap = cv2.VideoCapture('test3.mp4') # Load mp4

# Get the default frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define region points for counting
region_points = [(100,352), (900, 352)]

# Initialize the object counter
counter = ObjectCounter(
    region=region_points,
    model=os.path.join(os.getcwd(), "yolo11s.pt"), # Update to test other models (n, s, or m)
    classes=[0,1,2,3,5,7],
    show_in=True,
    show_out=True,
    line_width=2,
)

# Create a named window and set the mouse callback
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

count = 0

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break
        # If video ends, reset to the beginning
#        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#        continue
    count += 1
    if count % 2 != 0:  # Skip odd frames
        continue

    frame = cv2.resize(frame, (1020, 500))

    # Process the frame with the object counter
    frame1 = counter.count(frame)
   
    # Show the frame
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()