import json
import copy

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def merge_coco_files(file_paths, output_file):
    merged_data = {
        "categories": [],
        "images": [],
        "annotations": []
    }
    image_id_offset = 0
    annotation_id_offset = 0

    # Read and merge each file
    for from_id, file_path in enumerate(file_paths, start=1):
        data = read_json(file_path)

        # Copy categories from the first file
        if not merged_data["categories"]:
            merged_data["categories"] = data["categories"]

        # Merge images and annotations with updated IDs
        for image in data["images"]:
            new_image = copy.deepcopy(image)
            new_image["id"] += image_id_offset
            new_image["from"] = from_id
            merged_data["images"].append(new_image)

        for annotation in data["annotations"]:
            new_annotation = copy.deepcopy(annotation)
            new_annotation["id"] += annotation_id_offset
            new_annotation["image_id"] += image_id_offset
            new_annotation["from"] = from_id
            merged_data["annotations"].append(new_annotation)

        image_id_offset += len(data["images"])
        annotation_id_offset += len(data["annotations"])

    # Write the merged data to the output file
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=4)

# Example usage
file_paths = ['/media/Storage2/wlw/Federated/easyFL/flgo/benchmark/RAW_DATA/Dataset_Fetus_Object_Detection/annotations/4c/c1/train/annotation.json', '/media/Storage2/wlw/Federated/easyFL/flgo/benchmark/RAW_DATA/Dataset_Fetus_Object_Detection/annotations/4c/c2/train/annotation.json', '/media/Storage2/wlw/Federated/easyFL/flgo/benchmark/RAW_DATA/Dataset_Fetus_Object_Detection/annotations/4c/c3/train.json']
output_file = '/media/Storage2/wlw/Federated/easyFL/flgo/benchmark/RAW_DATA/Dataset_Fetus_Object_Detection/annotations/4c/train.json'
merge_coco_files(file_paths, output_file)
