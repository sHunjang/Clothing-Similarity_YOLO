import cv2
import torch
from pathlib import Path
from ultralytics import YOLO  # YOLO 모델 로드

# YOLOv8 모델 불러오기
model = YOLO('TOP&BOTTOM_Detection.pt')  # 사용자 모델 경로

# 이미지 불러오기
image_path = 'Dataset/train/Wannabe/001_Data-21.png'  # 처리할 이미지 경로
img = cv2.imread(image_path)

# 모델을 통해 이미지에서 객체 탐지
results = model(img)

# 탐지된 객체 정보 얻기
boxes = results[0].boxes  # results[0]에서 bounding box 정보 추출
names = results[0].names  # 객체 클래스 이름 추출
confidences = results[0].conf  # 신뢰도 정보 추출

# 출력 디렉토리 생성
output_dir = Path("extracted_objects")
output_dir.mkdir(parents=True, exist_ok=True)

# 신뢰도가 50% 이상인 탐지된 상의와 하의 객체 추출 및 저장
for i, (box, confidence) in enumerate(zip(boxes, confidences)):
    if confidence > 0.5:  # 신뢰도 50% 이상인 경우만 처리
        class_id = int(box.cls)  # 객체 클래스 ID
        class_name = names[class_id]  # 클래스 이름

        if class_name == 'top':  # 'top' 클래스인 경우
            x1, y1, x2, y2 = map(int, box.xywh[0])  # bounding box 좌표 추출
            cropped_img = img[y1:y2, x1:x2]  # 상의 객체 이미지 자르기
            cv2.imwrite(str(output_dir / f"top_{i}.jpg"), cropped_img)
        elif class_name == 'bottom':  # 'bottom' 클래스인 경우
            x1, y1, x2, y2 = map(int, box.xywh[0])  # bounding box 좌표 추출
            cropped_img = img[y1:y2, x1:x2]  # 하의 객체 이미지 자르기
            cv2.imwrite(str(output_dir / f"bottom_{i}.jpg"), cropped_img)

print(f"상/하의 객체 이미지가 {output_dir}에 저장되었습니다.")