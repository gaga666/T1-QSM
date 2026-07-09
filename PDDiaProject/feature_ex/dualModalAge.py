import torch
import torch.nn as nn

from feature_ex.feature import FeatureExtraction
from feature_ex.crossAttention import CrossAttentionBlock
from feature_ex.ageEncoding import AgeEncoding
from classifier.classifier_model import ClassifierMLP


class DualModalClassifierNet(nn.Module):
    """
    双模态 + 年龄协变量分类网络

    输入:
        qsm: [B,1,256,256]
        t1 : [B,1,256,256]
        age: [B] 或 [B,1]

    输出:
        out: [B,1]
    """

    def __init__(self):
        super().__init__()

        self.qsm_feature = FeatureExtraction()
        self.t1_feature = FeatureExtraction()

        self.shared_cross_attention = CrossAttentionBlock()

        self.qsm_norm = nn.LayerNorm(256)
        self.t1_norm = nn.LayerNorm(256)

        self.final_cross_attention = CrossAttentionBlock()
        self.final_norm = nn.LayerNorm(256)

        # 年龄Sin/Cos编码：[B] -> [B,256,256]
        self.age_encoder = AgeEncoding(
            embed_dim=256,
            max_age=100.0
        )

        # 年龄融合后的归一化
        self.age_norm = nn.LayerNorm(256)

        self.classifier = ClassifierMLP()

    def forward(self, qsm, t1, age):
        """
        qsm:
            [B,1,256,256]

        t1:
            [B,1,256,256]

        age:
            [B] 或 [B,1]
        """

        # 1. 单模态特征提取
        qsm_feat = self.qsm_feature(qsm)
        t1_feat = self.t1_feature(t1)

        # 2. 共享Cross-Attention特征
        shared_feat = self.shared_cross_attention(
            qsm_feat,
            t1_feat,
            t1_feat
        )

        # 3. shared_feat 同时回流给两个模态
        qsm_shared = self.qsm_norm(
            qsm_feat + shared_feat
        )

        t1_shared = self.t1_norm(
            t1_feat + shared_feat
        )

        # 4. 最终Cross-Attention融合
        fusion_feat = self.final_cross_attention(
            qsm_shared,
            t1_shared,
            t1_shared
        )

        fusion_feat = self.final_norm(
            fusion_feat + qsm_shared
        )

        # 5. Method2：Feature后、分类前加入年龄协变量
        age_feat = self.age_encoder(age)

        # age_feat: [B,256,256]
        # fusion_feat: [B,256,256]
        fusion_feat = self.age_norm(
            fusion_feat + age_feat
        )

        # 6. 分类
        out = self.classifier(fusion_feat)

        return out
