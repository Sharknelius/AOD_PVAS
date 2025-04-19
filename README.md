# Accelerating Object and Weapons Detection System

## Setup (Windows)
Before running the code, have Python (any version between 3.8 to before 3.13) and pip installed.
### Dependencies
Most requirements can be installed using pip (--user might be necessary to run as admin):
```
pip install ultralytics==8.3.76 --user
pip install opencv-python>=4.11.0.86 -- user
```
Once repo is downloaded, cd into "YOLO11-Object-Tracking/ultralytics" and run "pip install -r requirements.txt" to install all necessary Python libraries.
## Run Code
To run the code, cd into "YOLO11-Object-Tracking/ultralytics/detect" and run the command "python runTracker.py." This will run the current model defined in runTracker.py. The model will detect any vehicles or weapons in the live video feed passed into the tracker.py file. The video feed and model being used can be modified in runTracker.py.

The list of commands to run once in the root directory:
```
cd YOLO11-Object-Tracking/ultralytics/detect
python runTracker.py
```
This will create a live video feed, which will draw bounding boxes around detected objects and estimate the speed of moving vehicles. Here is an example of the video feed:
![alt text](https://github.com/Sharknelius/AOD_PVAS/blob/main/Documentation/GitHubImages/previewAODImage.png?raw=true)
## Code changes for IP camera
Within YOLO11-Object-Tracking/ultralytics/detect/runTracker.py file the 'cap' variable at the beginning of the document can be changed to support an IP camera. Below the initialization of the variable there is a commented line that shows an example initialization of the 'cap' variable with an IP camera rather than a basic webcam. Change the IP in this line of code to match the IP of the camera which will be used.

For further explanation of the subsystem go to [AOD_DESC.md](Documentation/AOD_DESC.md)
