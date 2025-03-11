import torch
from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')  

scripted_model = model.model  
scripted_model = torch.jit.script(scripted_model)  

scripted_model.save("yolov8n-seg.torchscript.pt")
