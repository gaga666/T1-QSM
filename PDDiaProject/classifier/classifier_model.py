import torch
import torch.nn as nn


class ClassifierMLP(nn.Module):
    """
    分类器 MLP

    输入:
        [B, 256, 256]

    输出:
        [B, 1]
    """

    def __init__(self):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 2),
            nn.ReLU(inplace=True),

            nn.Linear(2, 1)
        )

    def forward(self, x):
        """
        x: [B,256,256]
        """

        # 对256个token做平均池化
        # [B,256,256] → [B,256]
        x = x.mean(dim=1)

        # MLP分类
        x = self.mlp(x)

        return x
