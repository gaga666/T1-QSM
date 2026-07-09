import torch
import torch.nn as nn

from feature_ex.feature import FeatureExtraction
from relabel.selfAttention import ZAxisSelfAttention
from classifier.classifier_model import ClassifierMLP


class RelabelingNet(nn.Module):
    """
    两个Feature_ex的重标签网络

    feature_ex_1:
        第一轮训练，之后冻结

    feature_ex_2:
        后续第2~5轮持续更新
    """

    def __init__(self, threshold=0.5):
        super().__init__()

        self.threshold = threshold

        self.feature_ex_1 = FeatureExtraction()
        self.feature_ex_2 = FeatureExtraction()

        self.z_attention = ZAxisSelfAttention()

        self.classifier = ClassifierMLP()

    def freeze_first_feature_ex(self):
        """
        冻结第一个Feature Extraction
        """

        for param in self.feature_ex_1.parameters():
            param.requires_grad = False

    def forward_first_round(self, x):
        """
        第一轮前向传播

        输入:
            x: [B,Z,1,256,256]

        输出:
            logits: [B,Z,1]
            pred_labels: [B,Z]
        """

        B, Z, C, H, W = x.shape

        x_slice = x.reshape(B * Z, C, H, W)

        feat = self.feature_ex_1(x_slice)

        feat = feat.reshape(B, Z, 256, 256)

        feat = self.z_attention(feat)

        feat = feat.reshape(B * Z, 256, 256)

        logits = self.classifier(feat)

        logits = logits.reshape(B, Z, 1)

        probs = torch.sigmoid(logits)

        pred_labels = (probs >= self.threshold).long().squeeze(-1)

        return logits, pred_labels

    def forward_later_round(self, x):
        """
        第2~5轮前向传播

        输入:
            x: [B,Z,1,256,256]

        输出:
            logits: [B,Z,1]
            pred_labels: [B,Z]
        """

        B, Z, C, H, W = x.shape

        x_slice = x.reshape(B * Z, C, H, W)

        # 第一个Feature_ex已经冻结，只负责提供稳定初始特征
        with torch.no_grad():
            feat = self.feature_ex_1(x_slice)

        # 第二个Feature_ex继续更新
        feat = self.feature_ex_2(feat)

        feat = feat.reshape(B, Z, 256, 256)

        feat = self.z_attention(feat)

        feat = feat.reshape(B * Z, 256, 256)

        logits = self.classifier(feat)

        logits = logits.reshape(B, Z, 1)

        probs = torch.sigmoid(logits)

        pred_labels = (probs >= self.threshold).long().squeeze(-1)

        return logits, pred_labels