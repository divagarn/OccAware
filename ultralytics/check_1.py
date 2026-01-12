from ultralytics import YOLO
import torch

model = YOLO("ultralytics/cfg/models/v8/yolov8n.yaml")
model.model.eval()

dummy = torch.randn(1, 3, 640, 640)

with torch.no_grad():
    preds = model.model(dummy)

print("✅ Forward pass OK")

