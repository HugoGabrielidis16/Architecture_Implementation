import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim) -> None:
        super().__init__()
        self.Embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.Embedding(x)


def 