import os
import csv
import numpy as np
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve
)

from relabel.relabelingNet import RelabelingNet
from earlyStopping import EarlyStopping


class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss
    输入 logits，不需要提前 sigmoid
    """

    def __init__(self, alpha=0.5, gamma=2.0, reduction="mean"):
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
            1.0 - probs
        )

        alpha_t = torch.where(
            targets == 1,
            self.alpha,
            1.0 - self.alpha
        )

        focal_loss = alpha_t * (1.0 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class NiiPatientDataset(Dataset):
    """
    一个 nii 文件 = 一个病人

    输出:
        image: [Z,1,256,256]
        label: 0/1
        name: 文件名
    """

    def __init__(self, root_dir):
        self.samples = []

        for cls_name, label in [("HC", 0), ("PD", 1)]:
            cls_dir = os.path.join(root_dir, cls_name)

            for fname in os.listdir(cls_dir):
                if fname.endswith(".nii") or fname.endswith(".nii.gz"):
                    self.samples.append(
                        (
                            os.path.join(cls_dir, fname),
                            label,
                            fname
                        )
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, name = self.samples[index]

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

        data = np.expand_dims(data, axis=1)

        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return data, label, name


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    relabel_round
):
    """
    训练阶段：
        按 slice 级别计算 loss

    logits:
        [B,Z,1]

    slice_labels:
        [B,Z,1]
    """

    model.train()

    total_loss = 0.0

    for images, labels, names in loader:
        images = images.to(device)
        labels = labels.to(device)

        B, Z, C, H, W = images.shape

        optimizer.zero_grad()

        if relabel_round == 0:
            logits, pred_labels = model.forward_first_round(images)
        else:
            logits, pred_labels = model.forward_later_round(images)

        # 病人标签复制到每一张 slice
        slice_labels = labels.view(B, 1, 1).repeat(1, Z, 1)

        loss = criterion(logits, slice_labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate_slice_level(
    model,
    loader,
    criterion,
    device,
    relabel_round
):
    """
    验证阶段：
        完全按 slice 级别计算 loss 和指标
        不再计算 patient_prob
        不再 mean 聚合
    """

    model.eval()

    total_loss = 0.0

    all_slice_probs = []
    all_slice_labels = []

    with torch.no_grad():
        for images, labels, names in loader:
            images = images.to(device)
            labels = labels.to(device)

            B, Z, C, H, W = images.shape

            if relabel_round == 0:
                logits, pred_labels = model.forward_first_round(images)
            else:
                logits, pred_labels = model.forward_later_round(images)

            slice_labels = labels.view(B, 1, 1).repeat(1, Z, 1)

            loss = criterion(logits, slice_labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)

            # [B,Z,1] -> [B*Z]
            probs_np = probs.reshape(-1).cpu().numpy()
            labels_np = slice_labels.reshape(-1).cpu().numpy()

            all_slice_probs.extend(probs_np.tolist())
            all_slice_labels.extend(labels_np.tolist())

    val_loss = total_loss / len(loader)

    slice_preds = [
        1 if p >= 0.5 else 0
        for p in all_slice_probs
    ]

    try:
        val_auc = roc_auc_score(
            all_slice_labels,
            all_slice_probs
        )
    except:
        val_auc = 0.0

    val_acc = accuracy_score(
        all_slice_labels,
        slice_preds
    )

    val_sen = recall_score(
        all_slice_labels,
        slice_preds,
        zero_division=0
    )

    val_pre = precision_score(
        all_slice_labels,
        slice_preds,
        zero_division=0
    )

    val_f1 = f1_score(
        all_slice_labels,
        slice_preds,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        all_slice_labels,
        slice_preds,
        labels=[0, 1]
    ).ravel()

    val_spe = tn / (tn + fp + 1e-8)

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
        "loss": val_loss,
        "auc": val_auc,
        "acc": val_acc,
        "sen": val_sen,
        "spe": val_spe,
        "pre": val_pre,
        "f1": val_f1,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds
    }


def save_round_results(
    model,
    loader,
    device,
    relabel_round,
    save_dir,
    split_name
):
    """
    保存每一轮的 slice 级预测结果

    保存内容:
        patient_name
        true_label
        slice_probs
        slice_pred_labels

    不再保存:
        patient_prob
        patient_pred_label
    """

    model.eval()

    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(
        save_dir,
        f"round_{relabel_round + 1}_{split_name}_slice_results.csv"
    )

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "patient_name",
            "true_label",
            "slice_probs",
            "slice_pred_labels"
        ])

        with torch.no_grad():
            for images, labels, names in loader:
                images = images.to(device)
                labels = labels.to(device)

                if relabel_round == 0:
                    logits, pred_labels = model.forward_first_round(images)
                else:
                    logits, pred_labels = model.forward_later_round(images)

                probs = torch.sigmoid(logits)

                # [B,Z,1] -> [B,Z]
                slice_probs = probs.squeeze(-1).cpu().numpy()
                slice_pred_labels = pred_labels.cpu().numpy()

                labels_np = labels.cpu().numpy()

                for i in range(len(names)):
                    writer.writerow([
                        names[i],
                        int(labels_np[i]),
                        slice_probs[i].tolist(),
                        slice_pred_labels[i].tolist()
                    ])

    print(f"第 {relabel_round + 1} 轮 {split_name} slice 结果已保存: {save_path}")


def save_roc_data(save_path, fpr, tpr, thresholds):
    """
    保存 slice 级 ROC 曲线数据
    """

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


def set_trainable_for_round(model, relabel_round):
    """
    控制每一轮训练哪些参数

    第1轮:
        训练 feature_ex_1 + z_attention + classifier
        不训练 feature_ex_2

    第2~5轮:
        冻结 feature_ex_1
        训练 feature_ex_2 + z_attention + classifier
    """

    if relabel_round == 0:
        for p in model.feature_ex_1.parameters():
            p.requires_grad = True

        for p in model.feature_ex_2.parameters():
            p.requires_grad = False

        for p in model.z_attention.parameters():
            p.requires_grad = True

        for p in model.classifier.parameters():
            p.requires_grad = True

    else:
        for p in model.feature_ex_1.parameters():
            p.requires_grad = False

        for p in model.feature_ex_2.parameters():
            p.requires_grad = True

        for p in model.z_attention.parameters():
            p.requires_grad = True

        for p in model.classifier.parameters():
            p.requires_grad = True


def main():

    data_root = "./data/qsm_train"
    save_dir = "./data/result"

    batch_size = 1
    num_epochs = 300
    num_relabel_rounds = 5

    round_lrs = [
        1e-4,
        5e-5,
        3e-5,
        1e-5,
        5e-6
    ]

    patience = 50

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = NiiPatientDataset(data_root)

    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(2025)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    model = RelabelingNet(
        threshold=0.5
    ).to(device)

    criterion = BinaryFocalLoss(
        alpha=0.5,
        gamma=2.0
    )

    os.makedirs(save_dir, exist_ok=True)

    round_metrics = []

    for relabel_round in range(num_relabel_rounds):

        print(
            f"\n========== 第 {relabel_round + 1} 轮重标签训练开始 =========="
        )

        set_trainable_for_round(
            model,
            relabel_round
        )

        current_lr = round_lrs[relabel_round]

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=current_lr,
            weight_decay=1e-5
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=10,
            min_lr=1e-7
        )

        print(f"当前初始学习率: {current_lr:.2e}")

        best_model_path = os.path.join(
            save_dir,
            f"best_round_{relabel_round + 1}.pth"
        )

        early_stopping = EarlyStopping(
            patience=patience,
            save_path=best_model_path
        )

        best_auc = 0.0

        for epoch in range(num_epochs):

            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                relabel_round=relabel_round
            )

            val_result = validate_slice_level(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                relabel_round=relabel_round
            )

            val_auc = val_result["auc"]
            best_auc = max(best_auc, val_auc)

            scheduler.step(val_auc)

            now_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Round [{relabel_round + 1}/{num_relabel_rounds}] "
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
                    f"第 {relabel_round + 1} 轮早停止触发"
                )
                break

        if os.path.exists(best_model_path):
            model.load_state_dict(
                torch.load(
                    best_model_path,
                    map_location=device
                )
            )

        final_val_result = validate_slice_level(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            relabel_round=relabel_round
        )

        round_metrics.append([
            relabel_round + 1,
            final_val_result["auc"],
            final_val_result["acc"],
            final_val_result["sen"],
            final_val_result["spe"],
            final_val_result["pre"],
            final_val_result["f1"]
        ])

        save_round_results(
            model=model,
            loader=train_loader,
            device=device,
            relabel_round=relabel_round,
            save_dir=save_dir,
            split_name="train"
        )

        save_round_results(
            model=model,
            loader=val_loader,
            device=device,
            relabel_round=relabel_round,
            save_dir=save_dir,
            split_name="val"
        )

        save_roc_data(
            os.path.join(
                save_dir,
                f"round_{relabel_round + 1}_val_slice_roc.csv"
            ),
            final_val_result["fpr"],
            final_val_result["tpr"],
            final_val_result["thresholds"]
        )

        if relabel_round == 0:
            for p in model.feature_ex_1.parameters():
                p.requires_grad = False

            print("第一轮结束，feature_ex_1 已冻结")

    metrics_path = os.path.join(
        save_dir,
        "relabel_round_slice_metrics.csv"
    )

    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "round",
            "val_slice_auc",
            "val_slice_acc",
            "val_slice_sen",
            "val_slice_spe",
            "val_slice_pre",
            "val_slice_f1"
        ])

        writer.writerows(round_metrics)

    print("\n全部5轮重标签训练完成")


if __name__ == "__main__":
    main()