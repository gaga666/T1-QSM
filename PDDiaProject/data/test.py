import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm

# =====================
root_dir = ""
target_shape = (256, 256, 128)
# =====================

for group in ["HC", "PD"]:

    input_dir = os.path.join(root_dir, group)

    nii_files = [
        f for f in os.listdir(input_dir)
        if f.endswith(".nii") or f.endswith(".nii.gz")
    ]

    for fname in tqdm(nii_files, desc=f"Processing {group}"):

        file_path = os.path.join(input_dir, fname)

        img = nib.load(file_path)
        data = img.get_fdata()

        original_shape = data.shape

        if len(original_shape) != 3:
            print(f"跳过非3D文件: {fname}, shape={original_shape}")
            continue

        # 已经是目标尺寸直接跳过
        if original_shape == target_shape:
            print(f"跳过 {fname}，已经是 {target_shape}")
            continue

        zoom_factors = [
            target_shape[i] / original_shape[i]
            for i in range(3)
        ]

        resized_data = zoom(
            data,
            zoom_factors,
            order=1
        ).astype(np.float32)

        new_img = nib.Nifti1Image(
            resized_data,
            affine=img.affine,
            header=img.header
        )

        new_img.header.set_data_shape(target_shape)

        # 直接覆盖原文件
        nib.save(new_img, file_path)

        print(
            f"{fname}: "
            f"{original_shape} -> {resized_data.shape}"
        )

print("全部完成")