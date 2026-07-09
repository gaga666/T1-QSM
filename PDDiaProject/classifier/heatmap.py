import os
import numpy as np
import nibabel as nib
import torch
import scipy.io as sio

from classifier_model.classifierNet import ClassifierNet


def load_nii_as_slices(nii_path):
    img = nib.load(nii_path)
    data = img.get_fdata().astype(np.float32)

    if data.shape == (256, 256, 128):
        data = np.transpose(data, (2, 0, 1))

    elif data.shape == (128, 256, 256):
        pass

    else:
        raise ValueError(
            f"数据尺寸不符合要求: {data.shape}"
        )

    data = (data - data.mean()) / (data.std() + 1e-8)

    return data


def save_patient_heatmap(
    model_path,
    nii_path,
    save_dir,
    patient_name="patient"
):
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = ClassifierNet().to(device)

    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    model.eval()

    data = load_nii_as_slices(nii_path)

    # data: [Z,256,256]
    Z, H, W = data.shape

    probs = []

    with torch.no_grad():
        for z in range(Z):
            slice_img = data[z]

            x = np.expand_dims(slice_img, axis=0)
            x = np.expand_dims(x, axis=0)

            x = torch.tensor(
                x,
                dtype=torch.float32
            ).to(device)

            logit = model(x)

            prob = torch.sigmoid(logit).item()

            probs.append(prob)

    probs = np.array(probs, dtype=np.float32)

    # 生成heatmap mask
    # 每张slice整张图赋值为该slice的PD概率
    heatmap_mask = np.zeros(
        (Z, H, W),
        dtype=np.float32
    )

    for z in range(Z):
        heatmap_mask[z, :, :] = probs[z]

    npy_path = os.path.join(
        save_dir,
        f"{patient_name}_heatmap_mask.npy"
    )

    mat_path = os.path.join(
        save_dir,
        f"{patient_name}_heatmap_mask.mat"
    )

    prob_path = os.path.join(
        save_dir,
        f"{patient_name}_slice_probs.npy"
    )

    np.save(npy_path, heatmap_mask)
    np.save(prob_path, probs)

    sio.savemat(
        mat_path,
        {
            "heatmap_mask": heatmap_mask,
            "slice_probs": probs
        }
    )

    print("保存完成:")
    print(npy_path)
    print(mat_path)
    print(prob_path)


if __name__ == "__main__":

    model_path = "./result/classifier_cv/best_fold_1.pth"

    nii_path = "./data/qsm_test/PD/PD_005.nii"

    save_dir = "./result/heatmap"

    patient_name = "PD_005"

    save_patient_heatmap(
        model_path=model_path,
        nii_path=nii_path,
        save_dir=save_dir,
        patient_name=patient_name
    )