# AOD Subsystem Explanation
## Branches
**Main:** This includes the necessary code to run the system on the **Unigen Cupcake server** and uses security cameras connected through IP addresses.<br/>
**Develop:** This includes all the code from **main** as well as the code to train (**trainModel.py**) and test(**testModel.py**) **YOLO** models. The main processing script, **tracker.py** contains code using **Opencv-Python** that allows videos to be output after processing live video feed and non-live video files, and for those files to be given numbered names when saved to the output file using **Glob**. <br/>
Other branches were used for development.

## Dependency Explanation
**Ultralytics:**
**YOLO11:**
**Opencv-Python:**
**Glob:**

## Subsystem's Current State
