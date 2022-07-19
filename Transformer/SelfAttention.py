import torch
import torch.nn as nn
import torch.functional as F
import numpy as np


class SelfAttention(nn.Module):
    def __init__(self, d, heads=8):
        super().__init__()
        self.h = heads

        self.Wq = nn.Linear(d, d * heads, bias=False)
        self.Wk = nn.Linear(d, d * heads, bias=False)
        self.Wv = nn.Linear(d, d * heads, bias=False)

        # This unifies the outputs of the different heads into
        # a single k-dimensional vector.
        self.unifyheads = nn.Linear(heads * d, d)

    def forward(self, x):

        b, l, d = x.size()
        h = self.h

        # Transform the input embeddings x of shape [b, l, d] to queries, keys, values.
        # The output shape is [b, l, d*h] which we transform into [b, l, h, d]. Then,
        # we fold the heads into the batch dimenstion to arrive at [b*h, l, d]
        queries = (
            self.Wq(x).view(b, l, h, d).transpose(1, 2).contiguous().view(b * h, l, d)
        )
        keys = (
            self.Wk(x).view(b, l, h, d).transpose(1, 2).contiguous().view(b * h, l, d)
        )
        values = (
            self.Wv(x).view(b, l, h, d).transpose(1, 2).contiguous().view(b * h, l, d)
        )

        # Compute the product of queries and keys and scale with sqrt(d).
        # The tensor w' has shape (b*h, l, l) containing raw weights.
        # ----------------
        w_prime = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(d)
        # ----------------

        # Compute w by normalizing w' over the last dimension.
        # Shape: [b*h, l, l]
        # ----------------
        w = F.softmax(w_prime, dim=-1)
        # ----------------

        # Apply the self attention to the values.
        # Shape: [b, h, l, d]
        # ----------------
        out = torch.bmm(w, values).view(b, h, l, d)
        # ----------------

        # Swap h, l back.
        # Shape: [b, l, h*d]
        out = out.transpose(1, 2).contiguous().view(b, l, h * d)

        # Unify heads to arrive at shape [b, l, d].
        return self.unifyheads(out)
