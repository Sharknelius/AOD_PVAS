# AOD Subsystem Explanation
## Branches
**Main:** This includes the necessary code to run the system on the **Unigen Cupcake server** and uses security cameras connected through IP addresses.<br/>
**Develop:** This includes all the code from **main** as well as the code to train (**trainModel.py**) and test(**testModel.py**) **YOLO** models. The main processing script, **tracker.py** contains code using **Opencv-Python** that allows videos to be output after processing live video feed and non-live video files, and for those files to be given numbered names when saved to the output file using **Glob**. <br/>
Other branches were used for development.

## Dependency Explanation
**Ultralytics:** Used for tracking objects between frames and creating bounding boxes of differing colors for different objects in the main branch. Used for training the YOLO model in the develop branch. Here is [Ultralytics' GitHub Repository](https://github.com/ultralytics) and [ultralytics' website](https://www.ultralytics.com). <br/>
**YOLO11:** You Only Look Once (YOLO) provides real-time object detection. Here is [Yolo's GitHub Repository](https://github.com/ultralytics/ultralytics) and the [documentation for YOLO](https://docs.ultralytics.com/).<br/>
**OpenCV-Python:** A library of Python bindings designed to solve computer vision problems. Here is [OpenCV's GitHub Repository](https://github.com/opencv/opencv) and [OpenCV's website](https://opencv.org).<br/>
**Glob:** Used for output avi files in the Develop branch. This allows for multiple output videos to be saved after running the model so they can be reviewed later. Glob is actually a built-in Python library with [in-depth documentation](https://docs.python.org/3/library/glob.html), and Glob2 is an extention with more versatility. Here's more [information on Glob2](https://pypi.org/project/glob2/0.4.1/).<br/>

## Subsystem's Current State
The subsystem can detect objects traveling up to about 60 mph. This includes objects commonly found on roads that could cause damage to a patrol vehicle in a collision such as cars, trucks, motorcycles, and buses. Weapons, this includes guns and sharp-objects like knives, can also be detected. The YOLO model is more suited in detecting handguns over larger guns like rifles. If any weapon or vehicle that is determined to be heading in the direction of the cameras (south) at an accelerating speed (currently a 40% increase in speed, though this can be more fine-tuned), then a placeholder printout alert can be seen generated on the console. Based on performance metrics, the system struggles at recalling objects at further distances and may exhibit less adequate results if using a lower quality camera. </br>
</br>
**Integrating on Unigen Cupcake Server:** The system is able to be cloned and ran on the Unigen Cupcake, but showcases lower processing speeds and output due to the the Cupcakes computing limitations. The system is configured to process every even numbered frame of the video feed (this means the model accesses these frames and makes the detections), but the frequency may need to be lower to accommodate the Cupake (possibly every 5 or 10 frames). Currently, a CV2 window is generated and shown while the system is running which allowed our team to see the YOLO model working at detecting and annotating the objects every other frame; this is not necessary for the system to run, and likely uses more resources. This output should be removed for the final implementation. </br>
</br>
**Output:**</br>
