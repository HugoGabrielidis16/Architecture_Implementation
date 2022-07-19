import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from SelfAttention import SelfAttention


class TransformerBlock(nn.Module):
    """
    A Transformer block consisting of self attention and ff-layer.

    Args:
        d (int): The embedding dimension.
        heads (int): The number of attention heads.
        n_mlp (int): The number of mlp 'blocks'.
    """

    def __init__(self, d: int, heads: int = 8, n_mlp: int = 4):
        super().__init__()

        # The self attention layer.
        self.attention = SelfAttention(d, heads=heads)

        # The two layer norms.
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

        # The feed-forward layer.
        self.ff = nn.Sequential(
            nn.Linear(d, n_mlp * d), nn.ReLU(), nn.Linear(n_mlp * d, d)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: The input embedding of shape [b, l, d].

        Returns:
            Transformer output tensor of shape [b, l, d].
        """
        # Implement the forward pass as shown in the figure above.
        # ----------------
        out = self.attention(x) + x
        out = self.norm1(out)
        out = self.ff(out) + out
        out = self.norm2(out)
        # ----------------
        return out
