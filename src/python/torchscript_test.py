import torch

loaded_model = torch.jit.load("yolov8n-seg.torchscript.pt")
loaded_model.eval()

sample_input = torch.randn(1, 3, 640, 640)  
output = loaded_model(sample_input)

print(output)  
