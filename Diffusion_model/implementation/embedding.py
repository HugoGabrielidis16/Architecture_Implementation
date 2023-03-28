import torch 
import torch.nn as nn
from math import cos, sin 



"""
They used the same positional embedding as in the transformer paper.
"""
class PositionalEmbedding(nn.Module):
    def __init__(self):
        super(PositionalEmbedding, self).__init__()
    
    def forward(self, x):
        bs,s,e = x.shape
        encoding = torch.zeros(x.shape)
        for i in range(s):
            for j in range(e):
                if j % 2 == 0:
                    encoding[:, i, j] = sin(i / (10000 ** (2 * j / e)))
                else:
                    encoding[:, i, j] = cos(i / (10000 ** (2 * j / e)))
        return x + encoding