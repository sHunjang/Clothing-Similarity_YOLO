from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n-cls.pt")

# # Dataset Path
# Data_set_10 = 'Dataset_10'

# # Dataset Path
# Data_set_50 = 'Dataset_50'

# Dataset Path
Data_set_100 = 'Dataset_100'

# # Save Model Path-Dataset_10
# Train_Result_10 = 'Train_Result-Dataset_10'

# # Save Model Path-Dataset_50
# Train_Result_50 = 'Train_Result-Dataset_50'

# Save Model Path-Dataset_100
Train_Result_100 = 'Train_Result-Dataset_100'

# Model Train
result = model.train(
    data=Data_set_100,
    epochs=100,
    batch=8,
    optimizer='AdamW',
    project=Train_Result_100,
    device='mps',
    imgsz='224',
)