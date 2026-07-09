import torch
import torch.nn as nn

from feature_ex.feature import FeatureExtraction
from classifier.classifier_model import ClassifierMLP


class ClassifierNet(nn.Module):
    """
    完整分类网络

    输入:
        x: [B, 1, 256, 256]

    输出:
        out: [B, 1]
    """

    def __init__(self):
        super().__init__()

        # 特征提取模块
        self.feature_extractor = FeatureExtraction()

        # 分类器MLP
        self.classifier = ClassifierMLP()

    def forward(self, x):
        """
        x:
            [B, 1, 256, 256]
        """

        # 提取特征
        feature = self.feature_extractor(x)

        # feature:
        # [B, 256, 256]

        # 分类
        out = self.classifier(feature)

        # out:
        # [B, 1]

        return out
