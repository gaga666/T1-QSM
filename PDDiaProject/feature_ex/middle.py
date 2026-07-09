import torch
import torch.nn as nn


class TokenToFeatureMap(nn.Module):
    """
    Token Branch

    Input:
        [B, 256, 256]

    Output:
        [B, 128, 16, 16]
    """

    def __init__(self):
        super().__init__()

        # LayerNorm on token dimension
        self.norm = nn.LayerNorm(256)

        # Channel reduction
        self.conv1x1 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):

        # x : [B,256,256]

        B = x.shape[0]

        # -------------------------
        # LayerNorm
        # -------------------------
        x = self.norm(x)

        # -------------------------
        # Reshape
        # 256 tokens × 256 dim
        # →
        # 16×16×256
        # -------------------------

        x = x.reshape(
            B,
            16,
            16,
            256
        )

        # Conv2d expects:
        # [B,C,H,W]

        x = x.permute(
            0,
            3,
            1,
            2
        )

        # shape:
        # [B,256,16,16]

        # -------------------------
        # 1×1 Conv
        # 256 → 128 channels
        # -------------------------

        x = self.conv1x1(x)

        # output:
        # [B,128,16,16]

        return x
