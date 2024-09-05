import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom
import dataset as lab
import cov as cov
# 自定义数据集类，用于加载NIfTI文件
class NiiDataset(Dataset):
    def __init__(self, nii_files, labels, covariates=None, transform=None):
        self.nii_files = nii_files
        self.labels = labels
        self.covariates = covariates
        self.transform = transform

    def __len__(self):
        return len(self.nii_files)

    def __getitem__(self, idx):
        nii_path = self.nii_files[idx]
        label = self.labels[idx]
        covariate = self.covariates[idx] if self.covariates is not None else None

        # 读取nii文件
        img = nib.load(nii_path).get_fdata()
        img_resized = zoom(img, (64 / img.shape[0], 64 / img.shape[1], 64 / img.shape[2]), order=1)
        img_normalized = (img_resized - np.mean(img_resized)) / np.std(img_resized)
        img_tensor = torch.tensor(img_normalized, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label, covariate

# 定义ResNet18模型，修改为3D卷积网络，并加入协变量处理
class ResNet18_3D(nn.Module):
    def __init__(self, covariate_dim):
        super(ResNet18_3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)

        self.layer1 = self._make_layer(nn.Conv3d, 64, 64, 2)
        self.layer2 = self._make_layer(nn.Conv3d, 64, 128, 2, stride=2)
        self.layer3 = self._make_layer(nn.Conv3d, 128, 256, 2, stride=2)
        self.layer4 = self._make_layer(nn.Conv3d, 256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=0.5)

        # 处理协变量的全连接层
        self.covariate_fc = nn.Linear(3, 64)
        self.fc = nn.Linear(512 + 64, 2)  # 结合3D特征和协变量特征

    def _make_layer(self, conv, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(conv(in_channels, out_channels, kernel_size=(3, 3, 3), stride=(stride, stride, stride), padding=1, bias=False))
        layers.append(nn.BatchNorm3d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(1, blocks):
            layers.append(conv(out_channels, out_channels, kernel_size=(3, 3, 3), padding=1, bias=False))
            layers.append(nn.BatchNorm3d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x, covariates):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)

        covariates = self.covariate_fc(covariates)
        x = torch.cat((x, covariates), dim=1)  # 合并3D特征和协变量特征
        x = self.fc(x)

        return x

# 准备数据集
nii_files = lab.nii_files
labels = lab.labels
covariates = cov.covariates_array

# 数据集分割
train_files, val_files, train_labels, val_labels, train_covariates, val_covariates = train_test_split(
    nii_files, labels, covariates, test_size=0.2, random_state=42
)

# 数据增强和转换
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),  # 增加旋转增强
])

# 创建数据加载器
train_dataset = NiiDataset(train_files, train_labels, covariates=train_covariates, transform=transform)
val_dataset = NiiDataset(val_files, val_labels, covariates=val_covariates)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# 实例化模型、损失函数和优化器
covariate_dim = len(covariates[0])  # 计算协变量的特征数量
model = ResNet18_3D(covariate_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)  # 合适的学习率
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(train_loader), epochs=50)

# 训练模型
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels, covariates in train_loader:
        inputs, labels, covariates = inputs.to(device), labels.to(device), covariates.to(device)

        optimizer.zero_grad()
        outputs = model(inputs, covariates)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

    # 验证模型
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, labels, covariates in val_loader:
            inputs, labels, covariates = inputs.to(device), labels.to(device), covariates.to(device)
            outputs = model(inputs, covariates)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    accuracy = correct / len(val_dataset)
    print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")

# 保存模型
torch.save(model.state_dict(), "resnet18_3d_with_covariates.pth")
