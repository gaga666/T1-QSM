import torch
import torch.nn as nn


class AgeEncoding(nn.Module):
    """
    年龄Sin/Cos编码

    输入:
        age: [B]

    输出:
        age_feature:
        [B,256,256]
    """

    def __init__(
            self,
            embed_dim=256,
            max_age=100.0
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.max_age = max_age

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

    def forward(self, age):

        if age.dim() == 1:
            age = age.unsqueeze(1)

        age = age.float() / self.max_age

        device = age.device

        half_dim = self.embed_dim // 2

        div_term = torch.exp(
            torch.arange(
                half_dim,
                dtype=torch.float32,
                device=device
            )
            *
            (
                -torch.log(
                    torch.tensor(
                        10000.0,
                        device=device
                    )
                )
                / half_dim
            )
        )

        pe = torch.zeros(
            age.shape[0],
            self.embed_dim,
            device=device
        )

        pe[:, 0::2] = torch.sin(age * div_term)
        pe[:, 1::2] = torch.cos(age * div_term)

        pe = self.mlp(pe)

        # [B,256]
        # ↓
        # [B,1,256]
        pe = pe.unsqueeze(1)

        # ↓
        # [B,256,256]
        pe = pe.repeat(
            1,
            256,
            1
        )

        return pe