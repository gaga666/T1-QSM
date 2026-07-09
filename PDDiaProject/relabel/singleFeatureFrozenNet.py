import torch
import torch.nn as nn

from feature_ex.feature import FeatureExtraction
from classifier.classifier_model import ClassifierMLP


class SingleFeatureFrozenNet(nn.Module):
    """
    单Feature Extraction对照实验网络

    Stage 1:
        FeatureExtraction + ClassifierMLP 一起训练

    Stage 2:
        冻结 FeatureExtraction
        只训练 ClassifierMLP
    """

    def __init__(self):
        super().__init__()

        self.feature_extractor = FeatureExtraction()
        self.classifier = ClassifierMLP()

    def freeze_feature_extractor(self):
        """
        冻结Feature Extraction
        """

        for p in self.feature_extractor.parameters():
            p.requires_grad = False

    def unfreeze_feature_extractor(self):
        """
        解冻Feature Extraction
        """

        for p in self.feature_extractor.parameters():
            p.requires_grad = True

    def forward(self, x):
        """
        x:
            [B,1,256,256]

        return:
            [B,1]
        """

        feat = self.feature_extractor(x)

        logits = self.classifier(feat)

        return logits