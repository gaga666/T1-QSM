import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
import torch.nn.functional as F
import dataset_test as lab
import cov_test as cov

# 定义与训练时相同的ResNet18_3D模型结构
class ResNet18_3D_with_covariates(nn.Module):
    def __init__(self, num_covariates):
        super(ResNet18_3D_with_covariates, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)

        self.layer1 = self._make_layer(nn.Conv3d, 64, 64, 2)
        self.layer2 = self._make_layer(nn.Conv3d, 64, 128, 2, stride=2)
        self.layer3 = self._make_layer(nn.Conv3d, 128, 256, 2, stride=2)
        self.layer4 = self._make_layer(nn.Conv3d, 256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_img = nn.Linear(512, 128)  # 图像特征映射到128维
        self.fc_cov = nn.Linear(num_covariates, 32)  # 协变量特征映射到32维
        self.fc_combined = nn.Linear(128 + 32, 2)  # 图像特征和协变量特征结合

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
        img_features = self.fc_img(x)  # 图像特征

        cov_features = self.fc_cov(covariates)  # 协变量特征

        combined_features = torch.cat((img_features, cov_features), dim=1)  # 结合特征

        output = self.fc_combined(combined_features)
        return output



# 准备测试NIfTI文件的预处理函数
def preprocess_data(nii_file, covariate_data):
    # 处理图像
    img = nib.load(nii_file).get_fdata()
    img_resized = zoom(img, (64 / img.shape[0], 64 / img.shape[1], 64 / img.shape[2]), order=1)
    img_normalized = (img_resized - np.mean(img_resized)) / np.std(img_resized)
    img_tensor = torch.tensor(img_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 64, 64, 64)

    # 处理协变量
    covariates_tensor = torch.tensor(covariate_data, dtype=torch.float32).unsqueeze(0)  # (1, num_covariates)

    return img_tensor, covariates_tensor


# 定义模型和加载权重
num_covariates = 3  # 根据实际协变量数量调整
model = ResNet18_3D_with_covariates(num_covariates)
checkpoint = torch.load("resnet18_3d_with_covariates.pth")
model.load_state_dict(checkpoint, strict=False)  # strict=False 可以忽略一些不匹配的权重
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 准备测试数据
nii_files = lab.nii_files
true_labels = lab.labels
covariates_list = cov.covariates_array  # 确保有协变量列表

all_preds = []
all_labels = []

for nii_file, true_label, covariate_data in zip(nii_files, true_labels, covariates_list):
    input_tensor, covariates_tensor = preprocess_data(nii_file, covariate_data)
    input_tensor, covariates_tensor = input_tensor.to(device), covariates_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor, covariates_tensor)
        probs = F.softmax(output, dim=1)  # 获取类别概率
        pred_prob = probs[:, 1].item()  # 获取阳性类别的概率

    all_preds.append(pred_prob)
    all_labels.append(true_label)

# 处理预测结果
threshold = 0.5
all_preds_binary = [1 if prob > threshold else 0 for prob in all_preds]

# 计算评估指标
fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
roc_auc = auc(fpr, tpr)

accuracy = accuracy_score(all_labels, all_preds_binary)
conf_matrix = confusion_matrix(all_labels, all_preds_binary)
tn, fp, fn, tp = conf_matrix.ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# 输出评估指标
print(f'AUC: {roc_auc:.2f}')
print(f'Acc: {accuracy:.2f}')
print(f'Sen: {sensitivity:.2f}')
print(f'Spe: {specificity:.2f}')
