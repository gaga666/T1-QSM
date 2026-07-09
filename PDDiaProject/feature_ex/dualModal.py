import torch
import torch.nn as nn

from feature_ex.feature import FeatureExtraction
from feature_ex.crossAttention import CrossAttentionBlock
from classifier.classifier_model import ClassifierMLP


class DualModalClassifierNet(nn.Module):
    """
    双模态分类网络

    QSM路径:
        QSM -> FeatureExtraction -> qsm_feat

    T1路径:
        T1 -> FeatureExtraction -> t1_feat

    共享Cross-Attention:
        生成 shared_feat
        shared_feat 同时回流给 QSM 和 T1 两条路径

    最终融合:
        再使用 Cross-Attention 对 qsm_shared 和 t1_shared 进行融合

    输入:
        qsm: [B,1,256,256]
        t1 : [B,1,256,256]

    输出:
        out: [B,1]
    """

    def __init__(self):
        super().__init__()

        # QSM单模态特征提取
        self.qsm_feature = FeatureExtraction()

        # T1单模态特征提取
        self.t1_feature = FeatureExtraction()

        # 共享Cross-Attention模块
        self.shared_cross_attention = CrossAttentionBlock()

        # shared_feat 回流后的归一化
        self.qsm_norm = nn.LayerNorm(256)
        self.t1_norm = nn.LayerNorm(256)

        # 最终融合Cross-Attention
        # QSM作为Query，T1作为Key和Value
        self.final_cross_attention = CrossAttentionBlock()

        # 最终融合后的归一化
        self.final_norm = nn.LayerNorm(256)

        # 分类器MLP
        self.classifier = ClassifierMLP()

    def forward(self, qsm, t1):
        """
        qsm:
            [B,1,256,256]

        t1:
            [B,1,256,256]
        """

        # =========================
        # 1. 单模态特征提取
        # =========================
        qsm_feat = self.qsm_feature(qsm)
        t1_feat = self.t1_feature(t1)

        # qsm_feat: [B,256,256]
        # t1_feat : [B,256,256]

        # =========================
        # 2. 生成共享Cross-Attention特征
        # =========================
        shared_feat = self.shared_cross_attention(
            qsm_feat,
            t1_feat,
            t1_feat
        )

        # shared_feat: [B,256,256]

        # =========================
        # 3. shared_feat 同时回流给两个模态
        # =========================
        qsm_shared = self.qsm_norm(
            qsm_feat + shared_feat
        )

        t1_shared = self.t1_norm(
            t1_feat + shared_feat
        )

        # =========================
        # 4. 最终Cross-Attention融合
        # =========================
        fusion_feat = self.final_cross_attention(
            qsm_shared,
            t1_shared,
            t1_shared
        )

        # fusion_feat: [B,256,256]

        fusion_feat = self.final_norm(
            fusion_feat + qsm_shared
        )

        # =========================
        # 5. 分类
        # =========================
        out = self.classifier(fusion_feat)

        # out: [B,1]

        return out
