import os
from PIL import Image

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path)
            images.append((filename, img))
    return images

def resize_image(image, target_width):
    width_percent = target_width / float(image.size[0])
    target_height = int(float(image.size[1]) * float(width_percent))
    return image.resize((target_width, target_height), Image.BICUBIC)

def create_outfit_combinations(tops, bottoms, output_folder):
    for top_name, top in tops:
        for bottom_name, bottom in bottoms:
            # 상의와 하의의 최대 너비와 높이를 얻기
            max_height = max(top.height, bottom.height)
            total_height = top.height + bottom.height
            
            
            # 하의를 상의의 너비에 맞추어 크기 조정
            if bottom.height < top.height:
                bottom = resize_image(bottom, top.height)
            
            # 새로운 빈 이미지를 생성 (가장 큰 너비와 합쳐진 높이)
            new_img = Image.new('RGB', (max_height, total_height), (0, 0, 0))  # 검은색 배경
            
            # 상의와 하의를 새 이미지에 붙여넣기 (중앙에 배치)
            top_x = (max_height - top.width) // 2
            new_img.paste(top, (top_x, 0))
            
            bottom_x = (max_height - bottom.width) // 2
            new_img.paste(bottom, (bottom_x, top.height))
            
            # 새로운 이미지 이름 생성 및 저장
            new_img_name = f"{os.path.splitext(top_name)[0]}_{os.path.splitext(bottom_name)[0]}.png"
            new_img.save(os.path.join(output_folder, new_img_name))

# 상의와 하의 이미지 폴더 경로
tops_folder = 'User_Clothing_List/Tops'
bottoms_folder = 'User_Clothing_List/Bottoms'
output_folder = 'Combination'

# 출력 폴더가 없으면 생성
os.makedirs(output_folder, exist_ok=True)

# 이미지 로드
tops = load_images_from_folder(tops_folder)
bottoms = load_images_from_folder(bottoms_folder)

# 조합 생성 및 저장
create_outfit_combinations(tops, bottoms, output_folder)