import torch
import torch.nn as nn


class ZAxisSelfAttention(nn.Module):
    """
    Z轴向 Self-Attention

    输入:
        [B, Z, H, W]

    输出:
        [B, Z, H, W]
    """

    def __init__(self, attn_dim=64, num_heads=8):
        super().__init__()

        self.attn_dim = attn_dim

        # 把每个空间位置上的标量特征映射到高维特征
        self.input_proj = nn.Linear(1, attn_dim)

        # Z轴向多头自注意力
        self.self_attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # LayerNorm
        self.norm = nn.LayerNorm(attn_dim)

        # 前馈网络
        self.mlp = nn.Sequential(
            nn.Linear(attn_dim, attn_dim * 2),
            nn.GELU(),
            nn.Linear(attn_dim * 2, attn_dim)
        )

        # 投影回单通道
        self.output_proj = nn.Linear(attn_dim, 1)

    def forward(self, x):
        """
        x: [B, Z, H, W]
        """

        B, Z, H, W = x.shape
        x_reshape = x.permute(0, 2, 3, 1)
        x_reshape = x_reshape.reshape(B * H * W, Z, 1)

        feat = self.input_proj(x_reshape)

        # ------------------------------------------------
        # Z轴向 Self-Attention
        # Query = Key = Value = feat
        # ------------------------------------------------

        attn_out, _ = self.self_attn(
            query=feat,
            key=feat,
            value=feat
        )

        # 残差连接
        feat = feat + attn_out

        # 前馈网络 + 残差连接
        feat = feat + self.mlp(self.norm(feat))

        out = self.output_proj(feat)

        # ------------------------------------------------
        # reshape回原始空间结构
        # [B*H*W, Z, 1]
        # →
        # [B, H, W, Z]
        # →
        # [B, Z, H, W]
        # ------------------------------------------------

        out = out.reshape(B, H, W, Z)
        out = out.permute(0, 3, 1, 2)

        return out

