<H1 align="center">
YOLOv8 Object Detection with DeepSORT Tracking(ID + Trails) </H1>

## Object Detection and Tracking (ID + Trails)  using YOLOv8 on Custom Data

## YOLOv8 Segmentation with DeepSORT Object Tracking

[`Github Repo Link`](https://github.com/MuhammadMoinFaisal/YOLOv8_Segmentation_DeepSORT_Object_Tracking.git)

## Steps to run Code


- Goto the cloned folder.
```
cd YOLOv8-DeepSORT-Object-Tracking/ultralytics/yolo/v8/detect
```
- Install the dependecies
```
pip install -e '.[dev]'
```
- Downloading a Sample Video from the Google Drive
```
gdown "https://drive.google.com/uc?id=1rjBn8Fl1E_9d0EMVtL24S9aNQOJAveR5&confirm=t"
```

- Run the code with mentioned command below.
```
python predict.py model=yolov8l.pt source="test3.mp4" show=True
```

### RESULTS

#### Vehicles Detection, Tracking and Counting 
![](./figure/figure1.png)

