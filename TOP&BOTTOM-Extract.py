from ultralytics import YOLO

# Model Load
model = YOLO('TOP&BOTTOM_Detection.pt')

# Image path
image = 'Dataset_10/train/Sporty/Data-8.png'

# Save Results
extractClothing = 'TOP&BOTTOM'

model.predict(
    source=image,
    conf=0.8,
    project=extractClothing,
    agnostic_nms=True,
    save=True,
    device='mps',
    save_crop=True,
    save_conf=True,
)