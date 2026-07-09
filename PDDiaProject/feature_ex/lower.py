import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DenseFeatureConvBlock(nn.Module):
    """
    DenseNet风格的Feature Convolution模块

    输入:
        [B, 256, 16, 16]

    输出:
        [B, 256, 16, 16]
    """

    def __init__(self, channels=256, growth_channels=128):
        super().__init__()

        # 1×1卷积：通道压缩
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, growth_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(growth_channels),
            nn.ReLU(inplace=True)
        )

        # 3×3卷积：局部特征提取
        self.conv3 = nn.Sequential(
            nn.Conv2d(growth_channels, growth_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(growth_channels),
            nn.ReLU(inplace=True)
        )

        # 拼接后再压缩回256通道
        self.compress = nn.Sequential(
            nn.Conv2d(channels + growth_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv3(out)

        # DenseNet式通道拼接
        out = torch.cat([identity, out], dim=1)

        # 通道压缩回256
        out = self.compress(out)

        return out


class LowerFeatureExtractor(nn.Module):
    """
    图中下半部分Feature Extraction分支

    输入:
        x: [B, 1, 256, 256]
        token_feat: [B, 128, 16, 16]

    输出:
        [B, 256, 256]
    """

    def __init__(self):
        super().__init__()

        # 4个 3×3Conv + BN + ReLU + MaxPooling
        self.cnn_stage = nn.Sequential(
            ConvBNReLU(1, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 256 → 128

            ConvBNReLU(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 128 → 64

            ConvBNReLU(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 64 → 32

            ConvBNReLU(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2)    # 32 → 16
        )

        # 1×1卷积，把CNN分支通道数 256 → 128
        self.conv1x1_reduce = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # 两个连续Feature Convolution模块
        self.feature_conv1 = DenseFeatureConvBlock(
            channels=256,
            growth_channels=128
        )

        self.feature_conv2 = DenseFeatureConvBlock(
            channels=256,
            growth_channels=128
        )

        # 最后的1×1卷积 + BN
        self.final_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, token_feat):
        """
        x:
            原始2D切片 [B, 1, 256, 256]

        token_feat:
            中间层下来的特征 [B, 128, 16, 16]
        """

        # CNN分支特征提取
        x = self.cnn_stage(x)

        # 此时 x: [B, 256, 16, 16]

        # 1×1卷积压缩通道
        x = self.conv1x1_reduce(x)

        # 此时 x: [B, 128, 16, 16]

        # 与中间层下来的token特征做通道拼接
        x = torch.cat([x, token_feat], dim=1)

        # 此时 x: [B, 256, 16, 16]

        # 两个DenseNet风格Feature Convolution
        x = self.feature_conv1(x)
        x = self.feature_conv2(x)

        # 最后1×1卷积 + BN
        x = self.final_conv(x)

        # [B, 256, 16, 16]
        # reshape成 [B, 256, 256]
        B = x.shape[0]
        x = x.reshape(B, 256, 256)

        return x

