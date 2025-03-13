from ultralytics import YOLO

# Load Pretrained Model
Pretrained_Model = YOLO('Train_Result-Dataset_100/train7/weights/best.pt')

# Test Clothing Image Path
Predict_Images_Path = 'test_set'

# Save Predict Result Path-Dataset_10
Predict_Result_Path_10 = 'Predict_Result-Dataset_10'

# Save Predict Result Path-Dataset_50
Predict_Result_Path_50 = 'Predict_Result-Dataset_50'

# Save Predict Result Path-Dataset_100
Predict_Result_Path_100 = 'Predict_Result-Dataset_100'

# Predict Model
result_combination = Pretrained_Model.predict(
    source=Predict_Images_Path,
    save=True,
    save_txt=True,
    project=Predict_Result_Path_100,
    device='mps',
)