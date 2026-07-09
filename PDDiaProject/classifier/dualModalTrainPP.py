import os
import csv
import numpy as np
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve
)

from feature_ex.dualModal import DualModalClassifierNet
from earlyStopping import EarlyStopping


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.float()

        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none"
        )

        probs = torch.sigmoid(logits)

        pt = torch.where(
            targets == 1,
            probs,
            1 - probs
        )

        alpha_t = torch.where(
            targets == 1,
            self.alpha,
            1 - self.alpha
        )

        loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def load_nii_to_zhw(path, name):
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)

    if data.shape == (256, 256, 128):
        data = np.transpose(data, (2, 0, 1))
    elif data.shape == (128, 256, 256):
        pass
    else:
        raise ValueError(
            f"{name} 的数据尺寸不符合要求: {data.shape}，"
            f"期望为 (256,256,128) 或 (128,256,256)"
        )

    data = (data - data.mean()) / (data.std() + 1e-8)

    return data


def collect_dual_samples(qsm_root, t1_root):
    samples = []

    for cls_name, label in [("HC", 0), ("PD", 1)]:
        qsm_dir = os.path.join(qsm_root, cls_name)
        t1_dir = os.path.join(t1_root, cls_name)

        qsm_files = [
            f for f in os.listdir(qsm_dir)
            if f.endswith(".nii") or f.endswith(".nii.gz")
        ]

        for fname in qsm_files:
            qsm_path = os.path.join(qsm_dir, fname)
            t1_path = os.path.join(t1_dir, fname)

            if not os.path.exists(t1_path):
                raise FileNotFoundError(
                    f"T1中找不到与QSM对应的文件: {t1_path}"
                )

            samples.append(
                (
                    qsm_path,
                    t1_path,
                    label,
                    fname
                )
            )

    return samples


class DualNiiSliceDataset(Dataset):
    def __init__(self, samples):
        self.slice_samples = []

        for qsm_path, t1_path, label, name in samples:
            qsm_data = load_nii_to_zhw(qsm_path, name)
            t1_data = load_nii_to_zhw(t1_path, name)

            if qsm_data.shape != t1_data.shape:
                raise ValueError(
                    f"{name} 的QSM和T1尺寸不一致: "
                    f"QSM={qsm_data.shape}, T1={t1_data.shape}"
                )

            for z in range(qsm_data.shape[0]):
                self.slice_samples.append(
                    (
                        qsm_data[z],
                        t1_data[z],
                        label,
                        name,
                        z
                    )
                )

    def __len__(self):
        return len(self.slice_samples)

    def __getitem__(self, index):
        qsm_slice, t1_slice, label, name, z = self.slice_samples[index]

        qsm_slice = np.expand_dims(qsm_slice, axis=0)
        t1_slice = np.expand_dims(t1_slice, axis=0)

        return (
            torch.tensor(qsm_slice, dtype=torch.float32),
            torch.tensor(t1_slice, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
            name,
            z
        )


class DualNiiPatientDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        qsm_path, t1_path, label, name = self.samples[index]

        qsm_data = load_nii_to_zhw(qsm_path, name)
        t1_data = load_nii_to_zhw(t1_path, name)

        if qsm_data.shape != t1_data.shape:
            raise ValueError(
                f"{name} 的QSM和T1尺寸不一致: "
                f"QSM={qsm_data.shape}, T1={t1_data.shape}"
            )

        qsm_data = np.expand_dims(qsm_data, axis=1)
        t1_data = np.expand_dims(t1_data, axis=1)

        return (
            torch.tensor(qsm_data, dtype=torch.float32),
            torch.tensor(t1_data, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
            name
        )


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0

    for qsm, t1, labels, names, z in loader:
        qsm = qsm.to(device)
        t1 = t1.to(device)
        labels = labels.to(device).view(-1, 1)

        optimizer.zero_grad()

        logits = model(qsm, t1)

        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_slice_level(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0

    all_slice_probs = []
    all_slice_labels = []

    patient_names = []
    patient_labels = []
    patient_slice_probs = []

    with torch.no_grad():
        for qsm_volume, t1_volume, label, name in loader:
            qsm_volume = qsm_volume.to(device)
            t1_volume = t1_volume.to(device)
            label = label.to(device)

            B, Z, C, H, W = qsm_volume.shape

            qsm_volume = qsm_volume.reshape(B * Z, C, H, W)
            t1_volume = t1_volume.reshape(B * Z, C, H, W)

            logits = model(qsm_volume, t1_volume)

            slice_labels = label.view(B, 1).repeat(1, Z).reshape(B * Z, 1)

            loss = criterion(logits, slice_labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)

            slice_probs = probs.squeeze(-1).cpu().numpy()
            slice_labels_np = slice_labels.squeeze(-1).cpu().numpy()

            all_slice_probs.extend(slice_probs.tolist())
            all_slice_labels.extend(slice_labels_np.tolist())

            patient_names.append(name[0])
            patient_labels.append(int(label.item()))
            patient_slice_probs.append(slice_probs.tolist())

    avg_loss = total_loss / len(loader)

    slice_pred_labels = [
        1 if p >= 0.5 else 0
        for p in all_slice_probs
    ]

    try:
        auc = roc_auc_score(all_slice_labels, all_slice_probs)
    except:
        auc = 0.0

    acc = accuracy_score(all_slice_labels, slice_pred_labels)

    sen = recall_score(
        all_slice_labels,
        slice_pred_labels,
        zero_division=0
    )

    pre = precision_score(
        all_slice_labels,
        slice_pred_labels,
        zero_division=0
    )

    f1 = f1_score(
        all_slice_labels,
        slice_pred_labels,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        all_slice_labels,
        slice_pred_labels,
        labels=[0, 1]
    ).ravel()

    spe = tn / (tn + fp + 1e-8)

    try:
        fpr, tpr, thresholds = roc_curve(
            all_slice_labels,
            all_slice_probs
        )
    except:
        fpr = np.array([0.0, 1.0])
        tpr = np.array([0.0, 1.0])
        thresholds = np.array([1.0, 0.0])

    return {
        "loss": avg_loss,
        "auc": auc,
        "acc": acc,
        "sen": sen,
        "spe": spe,
        "pre": pre,
        "f1": f1,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "names": patient_names,
        "labels": patient_labels,
        "slice_probs": patient_slice_probs
    }


def save_slice_results_for_postprocess(save_path, names, labels, slice_probs):
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "patient_name",
            "true_label",
            "slice_probs"
        ])

        for n, y, sp in zip(names, labels, slice_probs):
            writer.writerow([
                n,
                int(y),
                sp
            ])


def save_roc_data(save_path, fpr, tpr, thresholds):
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "fpr",
            "tpr",
            "threshold"
        ])

        for a, b, c in zip(fpr, tpr, thresholds):
            writer.writerow([
                float(a),
                float(b),
                float(c)
            ])


def main():

    qsm_train_root = "./data/qsm_train"
    t1_train_root = "./data/t1_train"

    qsm_test_root = "./data/qsm_test"
    t1_test_root = "./data/t1_test"

    save_dir = "./result/dual_modal_slice_cv"

    os.makedirs(save_dir, exist_ok=True)

    batch_size = 8
    num_epochs = 300
    patience = 50
    n_splits = 5

    lr = 1e-4
    weight_decay = 1e-5

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_samples = collect_dual_samples(
        qsm_train_root,
        t1_train_root
    )

    test_samples = collect_dual_samples(
        qsm_test_root,
        t1_test_root
    )

    labels = [
        s[2]
        for s in train_samples
    ]

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=2025
    )

    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(train_samples, labels)
    ):

        print(
            f"\n========== Fold {fold + 1}/{n_splits} =========="
        )

        fold_train_samples = [
            train_samples[i]
            for i in train_idx
        ]

        fold_val_samples = [
            train_samples[i]
            for i in val_idx
        ]

        train_dataset = DualNiiSliceDataset(
            fold_train_samples
        )

        val_dataset = DualNiiPatientDataset(
            fold_val_samples
        )

        test_dataset = DualNiiPatientDataset(
            test_samples
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )

        model = DualModalClassifierNet().to(device)

        criterion = BinaryFocalLoss(
            alpha=0.25,
            gamma=2.0
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=10,
            min_lr=1e-7
        )

        best_model_path = os.path.join(
            save_dir,
            f"best_fold_{fold + 1}.pth"
        )

        early_stopping = EarlyStopping(
            patience=patience,
            save_path=best_model_path
        )

        for epoch in range(num_epochs):

            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device
            )

            val_result = evaluate_slice_level(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device
            )

            val_auc = val_result["auc"]

            scheduler.step(val_auc)

            now_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Fold [{fold + 1}/{n_splits}] "
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f} "
                f"Val Slice Loss: {val_result['loss']:.4f} "
                f"Val Slice AUC: {val_result['auc']:.4f} "
                f"Val Slice ACC: {val_result['acc']:.4f} "
                f"Val Slice SEN: {val_result['sen']:.4f} "
                f"Val Slice SPE: {val_result['spe']:.4f} "
                f"Val Slice PRE: {val_result['pre']:.4f} "
                f"Val Slice F1: {val_result['f1']:.4f} "
                f"LR: {now_lr:.2e}"
            )

            early_stopping(
                auc=val_auc,
                model=model
            )

            if early_stopping.early_stop:
                print(
                    f"Fold {fold + 1} 早停止"
                )
                break

        model.load_state_dict(
            torch.load(
                best_model_path,
                map_location=device
            )
        )

        val_result = evaluate_slice_level(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device
        )

        test_result = evaluate_slice_level(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device
        )

        save_slice_results_for_postprocess(
            os.path.join(
                save_dir,
                f"fold_{fold + 1}_val_slice_results.csv"
            ),
            val_result["names"],
            val_result["labels"],
            val_result["slice_probs"]
        )

        save_slice_results_for_postprocess(
            os.path.join(
                save_dir,
                f"fold_{fold + 1}_test_slice_results.csv"
            ),
            test_result["names"],
            test_result["labels"],
            test_result["slice_probs"]
        )

        save_roc_data(
            os.path.join(
                save_dir,
                f"fold_{fold + 1}_val_slice_roc.csv"
            ),
            val_result["fpr"],
            val_result["tpr"],
            val_result["thresholds"]
        )

        save_roc_data(
            os.path.join(
                save_dir,
                f"fold_{fold + 1}_test_slice_roc.csv"
            ),
            test_result["fpr"],
            test_result["tpr"],
            test_result["thresholds"]
        )

        fold_metrics.append([
            fold + 1,

            val_result["auc"],
            val_result["acc"],
            val_result["sen"],
            val_result["spe"],
            val_result["pre"],
            val_result["f1"],

            test_result["auc"],
            test_result["acc"],
            test_result["sen"],
            test_result["spe"],
            test_result["pre"],
            test_result["f1"]
        ])

        print(
            f"\nFold {fold + 1} Test Slice: "
            f"AUC={test_result['auc']:.4f}, "
            f"ACC={test_result['acc']:.4f}, "
            f"SEN={test_result['sen']:.4f}, "
            f"SPE={test_result['spe']:.4f}, "
            f"PRE={test_result['pre']:.4f}, "
            f"F1={test_result['f1']:.4f}"
        )

    metrics_path = os.path.join(
        save_dir,
        "five_fold_slice_metrics.csv"
    )

    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "fold",

            "val_slice_auc",
            "val_slice_acc",
            "val_slice_sen",
            "val_slice_spe",
            "val_slice_pre",
            "val_slice_f1",

            "test_slice_auc",
            "test_slice_acc",
            "test_slice_sen",
            "test_slice_spe",
            "test_slice_pre",
            "test_slice_f1"
        ])

        writer.writerows(fold_metrics)

    fold_metrics_np = np.array(fold_metrics)

    print("\n========== 双模态 slice 级五折交叉验证完成 ==========")
    print(f"平均 Val Slice AUC : {fold_metrics_np[:, 1].mean():.4f}")
    print(f"平均 Test Slice AUC: {fold_metrics_np[:, 7].mean():.4f}")


if __name__ == "__main__":
    main()