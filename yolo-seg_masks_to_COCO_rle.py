import cv2
from ultralytics import YOLO
import numpy as np
import os
import cv2
from pycocotools import mask as mask_util
import json
from PIL import Image

model = YOLO('s70.pt') 
images_path = "./images"
output_annotations_path = './annotations/coco_annotations.json'
categories = {}
print(model.names)
# Ваши категории объектов
for index, name in model.names.items():
    categories[index+1] = name

# Список аннотаций
coco_annotations = {
    "images": [],
    "annotations": [],
    "categories": []
}

# Добавляем категории в формате COCO
for category_id, category_name in categories.items():
    coco_annotations["categories"].append({
        "id": category_id,
        "name": category_name,
        "supercategory": "object"
    })

annotation_id = 1

# Функция для создания RLE-сегментации
def create_rle(mask, image_size):
    # Меняем размер маски до разрешения изображения
    resized_mask = cv2.resize(mask, (image_size[1],image_size[0]), interpolation=cv2.INTER_LANCZOS4) 
    resized_mask = resized_mask.astype(np.uint8)
    # Преобразуем маску в RLE
    rle = mask_util.encode(np.asfortranarray(resized_mask))
    rle['counts'] = rle['counts'].decode('utf-8')  # Для совместимости с JSON
    return rle

# Проходим по всем изображениям в папке
for image_id, image_name in enumerate(os.listdir(images_path), start=1):
    image_path = os.path.join(images_path, image_name)
    img = Image.open(image_path)
    img_width, img_height = img.size

    # Добавляем информацию о изображении в COCO формат
    coco_annotations["images"].append({
        "id": image_id,
        "file_name": image_name,
        "width": img_width,
        "height": img_height
    })

    # Прогоняем изображение через модель
    results = model.predict(image_path, save=False, show=False, retina_masks=True)[0]
    # Для каждого объекта на изображении
    for index in range(len(results.boxes)):
        class_id = int(results.boxes.cls[index]) + 1  # Класс объекта
        mask = results.masks.data[index].cpu().numpy() # Получаем маску объекта (в формате 384x640)
        
        rle = create_rle(mask, results.masks.orig_shape)

        # Добавляем аннотацию в COCO формат
        coco_annotations["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": class_id,
            "segmentation": rle,
            "area": int(np.sum(mask)),  
            "bbox": results.boxes.xyxy[index].tolist(), 
            "iscrowd": 0  # Не является скоплением объектов
        })

        annotation_id += 1

# Сохраняем аннотации в файл
os.makedirs(os.path.dirname(output_annotations_path), exist_ok=True)
with open(output_annotations_path, 'w') as f:
    json.dump(coco_annotations, f, indent=4)

print(f"COCO-аннотации сохранены в {output_annotations_path}")
