import os
import shutil


def clear_folder(folder_path):
    # Check if the path exists and is a directory
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a directory or does not exist.")
        return

    # Traverse all files and subdirectories in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            # if it is a file, then delete
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 删除文件
            # If it is a folder, recursively delete it."
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 递归删除文件夹
        except Exception as e:
            print(f"Error while deleting {file_path}: {e}")


# use case
folder_to_clear = "E:\pyCausalVision\car_project/runs\cnn_experiment"
clear_folder(folder_to_clear)