import json
import os
import cv2

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def write_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def generate_segmentation(bbox):
    x, y, w, h = bbox
    return [x, y, x + w, y, x + w, y + h, x, y + h]

def add_segmentation_to_coco(coco_data, image_folder):
    for annotation in coco_data["annotations"]:
        if "bbox" in annotation:
            bbox = annotation["bbox"]
            segmentation = generate_segmentation(bbox)
            annotation["segmentation"] = [segmentation]
    return coco_data

# 示例用法
coco_file_path = '/media/Storage2/wlw/Federated/easyFL/flgo/benchmark/RAW_DATA/Dataset_Fetus_Object_Detection/annotations/4c/test.json'
image_folder = '/media/Storage2/wlw/Federated/easyFL/flgo/benchmark/RAW_DATA/Dataset_Fetus_Object_Detection/4C'

# 读取 COCO 文件
coco_data = read_json(coco_file_path)

# 为注释添加 segmentation 属性
coco_data = add_segmentation_to_coco(coco_data, image_folder)

# 保存更新后的 COCO 文件
output_file_path = '/media/Storage2/wlw/Federated/easyFL/flgo/benchmark/RAW_DATA/Dataset_Fetus_Object_Detection/annotations/4c/test.json'
write_json(coco_data, output_file_path)
