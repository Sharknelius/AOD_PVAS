from ultralytics import YOLO
import torch

if __name__ ==  '__main__':
    #model = YOLO("yolo11.yaml")

    #model.train(data="../../../cfg/datasets/coco-and-weapons-data.yaml", epochs=25, dropout = 0.25, plots=True, batch=16, device=0)

    # Pretrained model
    model = YOLO("AOD_weapons.pt")

    model.train(data="../../../cfg/datasets/coco-and-weapons-data.yaml", epochs=40, dropout = 0.25, plots=True, batch=16, device=0)

    model.val()
