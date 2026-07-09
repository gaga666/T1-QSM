import torch
import torch.nn as nn

from feature_ex.upper import UpperFeatureExtractor
from feature_ex.middle import TokenToFeatureMap
from feature_ex.lower import LowerFeatureExtractor
from feature_ex.crossAttention import CrossAttentionBlock

class ExtractionBlock(nn.Module):
    """
    单个 Extraction Block

    token_input:
        第1个block: [B,1,256,256]
        后续block: [B,256,256]

    raw_img:
        始终是原始图像 [B,1,256,256]

    输出:
        [B,256,256]
    """

    def __init__(self):
        super().__init__()

        self.upper = UpperFeatureExtractor()
        self.middle = TokenToFeatureMap()
        self.lower = LowerFeatureExtractor()
        self.cross_attention = CrossAttentionBlock()

    def forward(self, token_input, raw_img):

        # 上半部分
        token_feature = self.upper(token_input)

        # 中间层
        middle_feature = self.middle(token_feature)

        # 下半部分始终使用原图
        lower_feature = self.lower(
            raw_img,
            middle_feature
        )

        # Cross-Attention
        output = self.cross_attention(
            token_feature,
            lower_feature,
            lower_feature
        )

        return output


class FeatureExtraction(nn.Module):
    """
    整个 Feature Extraction 模块

    Block × 4
    """

    def __init__(self):
        super().__init__()

        self.block1 = ExtractionBlock()
        self.block2 = ExtractionBlock()
        self.block3 = ExtractionBlock()
        self.block4 = ExtractionBlock()

    def forward(self, x):

        # 保存原始图像
        raw_img = x

        # 第1个block输入原始图像
        x = self.block1(x, raw_img)

        # 后续block输入上一轮输出，但lower分支仍然用raw_img
        x = self.block2(x, raw_img)
        x = self.block3(x, raw_img)
        x = self.block4(x, raw_img)

        return x