import os
import csv
import numpy as np
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.metrics import roc_auc_score

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
    一个nii文件 = 一个病人
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

        # 统一保证输出为 [128,256,256]
        # 也就是 [Z,H,W]
        if data.shape == (256, 256, 128):
            data = np.transpose(data, (2, 0, 1))

        elif data.shape == (128, 256, 256):
            pass

        else:
            raise ValueError(
                f"{name} 的数据尺寸不符合要求: {data.shape}，"
                f"期望为 (256,256,128) 或 (128,256,256)"
            )

        # 标准化
        data = (data - data.mean()) / (data.std() + 1e-8)

        # [Z,H,W] -> [Z,1,H,W]
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

        # 病人标签复制到每个slice
        slice_labels = labels.view(B, 1, 1).repeat(1, Z, 1)

        loss = criterion(logits, slice_labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate_one_round(
    model,
    loader,
    criterion,
    device,
    relabel_round
):
    model.eval()

    total_loss = 0.0

    all_probs = []
    all_labels = []

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

            # slice概率平均成病人级概率
            patient_probs = probs.mean(dim=1).squeeze(-1)

            all_probs.extend(patient_probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = total_loss / len(loader)

    try:
        val_auc = roc_auc_score(all_labels, all_probs)
    except:
        val_auc = 0.0

    return val_loss, val_auc


def save_round_results(
    model,
    loader,
    device,
    relabel_round,
    save_dir,
    split_name
):
    """
    保存每一轮的预测概率和离散标签

    保存内容:
        patient_name
        true_label
        patient_prob
        patient_pred_label
        slice_probs
        slice_pred_labels
    """

    model.eval()

    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(
        save_dir,
        f"round_{relabel_round + 1}_{split_name}_results.csv"
    )

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "patient_name",
            "true_label",
            "patient_prob",
            "patient_pred_label",
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

                # 病人级概率
                patient_probs = probs.mean(dim=1).squeeze(-1).cpu().numpy()
                patient_pred_labels = (patient_probs >= 0.5).astype(int)

                labels_np = labels.cpu().numpy()

                for i in range(len(names)):
                    writer.writerow([
                        names[i],
                        int(labels_np[i]),
                        float(patient_probs[i]),
                        int(patient_pred_labels[i]),
                        slice_probs[i].tolist(),
                        slice_pred_labels[i].tolist()
                    ])

    print(f"第 {relabel_round + 1} 轮 {split_name} 结果已保存: {save_path}")


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

    # 每一轮重标签的初始学习率
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

        # 每一轮内部根据验证集AUC自适应降低学习率
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=10,
            min_lr=1e-7
        )

        print(f"当前初始学习率: {current_lr:.2e}")

        early_stopping = EarlyStopping(
            patience=patience,
            save_path=os.path.join(
                save_dir,
                f"best_round_{relabel_round + 1}.pth"
            )
        )

        for epoch in range(num_epochs):

            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                relabel_round=relabel_round
            )

            val_loss, val_auc = validate_one_round(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                relabel_round=relabel_round
            )

            # 根据AUC调整当前轮内部学习率
            scheduler.step(val_auc)

            now_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Round [{relabel_round + 1}/{num_relabel_rounds}] "
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f} "
                f"Val Loss: {val_loss:.4f} "
                f"Val AUC: {val_auc:.4f} "
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

        # 加载当前轮最佳模型
        best_path = os.path.join(
            save_dir,
            f"best_round_{relabel_round + 1}.pth"
        )

        if os.path.exists(best_path):
            model.load_state_dict(
                torch.load(best_path, map_location=device)
            )

        # 保存当前轮训练集和验证集的结果与标签
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

        # 第一轮结束后冻结第一个feature ex
        if relabel_round == 0:
            for p in model.feature_ex_1.parameters():
                p.requires_grad = False

            print("第一轮结束，feature_ex_1 已冻结")

    print("\n全部5轮重标签训练完成")


if __name__ == "__main__":
    main()