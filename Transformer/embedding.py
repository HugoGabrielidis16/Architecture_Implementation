import torch
import torch.nn as nn
from math import sin, cos
from transformers import AutoTokenizer
import numpy as np


class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)


class PositionalEncoding(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        """
        x : [batch_size,seq_length,embed_dim]
        """
        _, s, e = x.shape
        PositionalEncoding = torch.zeros(x.shape)
        for i in range(s):
            for j in range(e):
                if j % 2 == 0:
                    PositionalEncoding[:, i, j] = sin(i / (10000 ** (2 * j / e)))
                else:
                    PositionalEncoding[:, i, j] = cos(i / (10000 ** (2 * j / e)))
        return x + PositionalEncoding


class FinalEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim) -> None:
        super().__init__()
        self.Embedding = Embedding(vocab_size, embedding_dim)
        self.PositionalEncoding = PositionalEncoding()

    def forward(self, x):
        x = self.Embedding(x)
        x = self.PositionalEncoding(x)
        return x
