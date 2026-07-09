import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=8):
        super().__init__()

        # 对Q、K、V分别进行归一化
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)

        # 多头交叉注意力
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        # 注意力后的归一化
        self.norm_out = nn.LayerNorm(dim)

        # 前馈网络
        self.mlp = nn.Sequential(
            nn.Linear(dim, 512),
            nn.GELU(),
            nn.Linear(512, dim)
        )

    def forward(self,
                q_feat,
                k_feat,
                v_feat):
        """
        参数
        ----------
        q_feat : [B,256,256]
            Query特征矩阵

        k_feat : [B,256,256]
            Key特征矩阵

        v_feat : [B,256,256]
            Value特征矩阵

        返回
        ----------
        x : [B,256,256]
            Cross-Attention输出特征
        """

        # LayerNorm
        q = self.norm_q(q_feat)
        k = self.norm_k(k_feat)
        v = self.norm_v(v_feat)

        # Cross-Attention
        attn_out, attn_weight = self.cross_attn(
            query=q,
            key=k,
            value=v
        )

        # 第一次残差连接
        x = q_feat + attn_out

        # 前馈网络
        mlp_out = self.mlp(
            self.norm_out(x)
        )

        # 第二次残差连接
        x = x + mlp_out

        return x

