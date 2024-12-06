from ultralytics import YOLO

# Load your YOLOv8 model
model = YOLO('best.pt')  # Load your trained weights

# Export to ONNX
model.export(format='onnx', imgsz=416)  # Adjust img size as necessary

