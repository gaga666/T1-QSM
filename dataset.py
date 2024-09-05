import os

# 定义数据集路径
dataset_dir = "../PD_cluster/QSM"

# 初始化文件路径和标签列表
nii_files = []
labels = []

# 遍历每个类别的文件夹
for label, class_name in enumerate(os.listdir(dataset_dir)):
    class_dir = os.path.join(dataset_dir, class_name)
    if os.path.isdir(class_dir):
        # 遍历每个类别文件夹中的NIfTI文件
        for nii_file in os.listdir(class_dir):
            if nii_file.endswith(".nii") or nii_file.endswith(".nii.gz"):
                nii_files.append(os.path.join(class_dir, nii_file))
                labels.append(label)

# 检查生成的文件路径和标签
print("Number of files:", len(nii_files))
print("Example file path:", nii_files)
print("Corresponding label:", labels)
