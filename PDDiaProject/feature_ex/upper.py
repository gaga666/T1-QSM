import torch
import torch.nn as nn


class AttentionFeedForwardBlock(nn.Module):
    def __init__(self, dim=256, num_heads=8):
        super().__init__()

        self.bn = nn.BatchNorm1d(dim)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.ln = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, 512),
            nn.GELU(),
            nn.Linear(512, dim)
        )

    def forward(self, x):

        x_bn = self.bn(x.transpose(1, 2)).transpose(1, 2)

        attn_out, _ = self.self_attn(x_bn, x_bn, x_bn)

        x = x + attn_out

        x_ln = self.ln(x)

        mlp_out = self.mlp(x_ln)

        x = x + mlp_out

        return x


class LinearTokenization(nn.Module):
    def __init__(self, image_size=256, token_dim=256):
        super().__init__()

        self.proj = nn.Linear(image_size, token_dim)

    def forward(self, x):

        x = x.squeeze(1)
        x = self.proj(x)          

        return x


class UpperFeatureExtractor(nn.Module):
    def __init__(self, dim=256, num_heads=8):
        super().__init__()

        self.tokenization = LinearTokenization(
            image_size=256,
            token_dim=dim
        )

        self.block1 = AttentionFeedForwardBlock(
            dim=dim,
            num_heads=num_heads
        )

        self.block2 = AttentionFeedForwardBlock(
            dim=dim,
            num_heads=num_heads
        )

    def forward(self, x):

        x = self.tokenization(x)

        x = self.block1(x)
        x = self.block2(x)

        return x


