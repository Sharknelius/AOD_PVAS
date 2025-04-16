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
To run the code, cd into "YOLO11-Object-Tracking/ultralytics/detect" and run the command "python runTracker.py." This will run the current model defined in runTracker.py. The model will detect any vehicles or weapons it finds in the live video feed passed into the tracker.py file. The video feed and model being used can be modified in runTracker.py.

The list of commands to run once in the root directory:
```
cd YOLO11-Object-Tracking/ultralytics/detect
python runTracker.py
```
This will output a video with bounding box annotations.
