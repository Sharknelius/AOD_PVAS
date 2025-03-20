from ultralytics import YOLO

model = YOLO("yolo11s.pt")

model.train(data="../../../cfg/datasets/weapons-data.yaml", epochs=1, time=2.0, dropout = 0.25, plots=True)
