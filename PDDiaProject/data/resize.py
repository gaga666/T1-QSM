import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm


input_root = "./qsm"
output_root = "./qsm_re"
target_shape = (256, 256, 128)
# =====================

os.makedirs(output_root, exist_ok=True)

for group in ["HC", "PD"]:
    input_dir = os.path.join(input_root, group)
    output_dir = os.path.join(output_root, group)
    os.makedirs(output_dir, exist_ok=True)

    nii_files = [
        f for f in os.listdir(input_dir)
        if f.endswith(".nii") or f.endswith(".nii.gz")
    ]

    for fname in tqdm(nii_files, desc=f"Processing {group}"):
        in_path = os.path.join(input_dir, fname)

        img = nib.load(in_path)
        data = img.get_fdata()

        original_shape = data.shape

        if len(original_shape) != 3:
            print(f"跳过非3D文件: {fname}, shape={original_shape}")
            continue

        zoom_factors = [
            target_shape[i] / original_shape[i]
            for i in range(3)
        ]

        resized_data = zoom(data, zoom_factors, order=1)

        resized_data = resized_data.astype(np.float32)

        new_img = nib.Nifti1Image(
            resized_data,
            affine=img.affine,
            header=img.header
        )

        new_img.header.set_data_shape(target_shape)

        out_name = fname.replace(".nii.gz", ".nii")
        out_path = os.path.join(output_dir, out_name)

        nib.save(new_img, out_path)

        print(f"{fname}: {original_shape} -> {resized_data.shape}")

print("全部完成")