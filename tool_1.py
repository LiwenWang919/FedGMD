import os
import shutil

def merge_folders(source_folders, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for folder in source_folders:
        for root, dirs, files in os.walk(folder):
            for file in files:
                source_file = os.path.join(root, file)
                target_file = os.path.join(target_folder, file)

                # 如果文件名重复，可以选择重命名或覆盖
                if os.path.exists(target_file):
                    base, ext = os.path.splitext(file)
                    counter = 1
                    while os.path.exists(target_file):
                        new_file_name = f"{base}_{counter}{ext}"
                        target_file = os.path.join(target_folder, new_file_name)
                        counter += 1

                shutil.copy2(source_file, target_file)
                print(f"Copied {source_file} to {target_file}")

# 示例用法
source_folders = ['/media/Storage2/wlw/Federated/easyFL/flgo/benchmark/RAW_DATA/Dataset_Fetus_Object_Detection/Hospital_1/four_chamber_heart', '/media/Storage2/wlw/Federated/easyFL/flgo/benchmark/RAW_DATA/Dataset_Fetus_Object_Detection/Hospital_2/four_chamber_heart', '/media/Storage2/wlw/Federated/easyFL/flgo/benchmark/RAW_DATA/Dataset_Fetus_Object_Detection/Hospital_3/four_chamber_heart']
target_folder = '/media/Storage2/wlw/Federated/easyFL/flgo/benchmark/RAW_DATA/Dataset_Fetus_Object_Detection/4C'
merge_folders(source_folders, target_folder)
