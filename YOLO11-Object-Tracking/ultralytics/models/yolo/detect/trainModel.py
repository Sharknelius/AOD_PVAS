from ultralytics import YOLO
import torch

if __name__ ==  '__main__':
    # Pretrained model
    model = YOLO("yolo11s_AOD3.pt")

    # Change file path to your data.yaml that redirects to your dataset
    model.train(data="../../../cfg/datasets/coco-and-weapons-data.yaml", epochs=50, imgsz=640, dropout = 0.25, plots=True, batch=16, device=0)

    model.val()
