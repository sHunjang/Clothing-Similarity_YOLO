import os
import random
import shutil


# 원본 데이터셋 경로
original_dataset_path = "Dataset"

# 출력 데이터셋 경로
output_base_path = "Dataset_output"

# 생성할 데이터셋 크기 (50, 100)
data_sizes = [50, 100]

# 데이터 비율
split_ratios = {"train": 0.8, "test": 0.1, "val": 0.1}

# 클래스 이름
classes = ["Amekaji", "Business", "Sporty", "Wannabe", "Casual"]


def copy_with_repetition(files, num_needed, source_dir, dest_dir):
    """
    부족한 이미지가 있을 경우 중복 복사로 채움.
    """
    os.makedirs(dest_dir, exist_ok=True)
    num_files = len(files)

    # 필요한 수만큼 파일 복사
    for i in range(num_needed):
        src_file = os.path.join(source_dir, files[i % num_files])
        dest_file = os.path.join(dest_dir, f"{i+1:03d}_{os.path.basename(src_file)}")
        shutil.copy(src_file, dest_file)


def create_dataset_with_ratios(original_path, output_path, sizes, ratios, classes):
    for size in sizes:
        size_folder = os.path.join(output_path, f"dataset_{size}")
        os.makedirs(size_folder, exist_ok=True)

        for split_type in ["train", "test", "val"]:
            for class_name in classes:
                # 원본 데이터 경로
                source_class_path = os.path.join(original_path, split_type, class_name)
                if not os.path.exists(source_class_path):
                    print(f"Class path not found: {source_class_path}")
                    continue

                # 이미지 파일 목록 가져오기
                image_files = [f for f in os.listdir(source_class_path) if f.endswith(".png")]

                # 필요한 데이터 크기 계산
                num_images_needed = int(size * ratios[split_type])

                if len(image_files) == 0:
                    print(f"Error: No images found in {source_class_path}. Skipping this class.")
                    continue

                # 출력 경로 설정
                dest_class_path = os.path.join(size_folder, split_type, class_name)
                os.makedirs(dest_class_path, exist_ok=True)

                if len(image_files) < num_images_needed:
                    print(
                        f"Warning: Not enough images in {split_type}/{class_name}. "
                        f"Found {len(image_files)}, required {num_images_needed}. Using repeated copies."
                    )
                    copy_with_repetition(image_files, num_images_needed, source_class_path, dest_class_path)
                else:
                    # 필요한 개수만큼 샘플링
                    sampled_files = random.sample(image_files, num_images_needed)
                    for file in sampled_files:
                        src = os.path.join(source_class_path, file)
                        dst = os.path.join(dest_class_path, file)
                        shutil.copy(src, dst)

                print(f"{split_type}/{class_name}: Copied {num_images_needed} images for dataset_{size}")


# 실행
create_dataset_with_ratios(original_dataset_path, output_base_path, data_sizes, split_ratios, classes)