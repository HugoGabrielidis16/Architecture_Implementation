import torch
import torch.nn as nn





"""
Took this code from my Transformer implementation: 
https://github.com/Yuuuugo/Architecture_Implementation/blob/main/Transformer/implementation/Attention.py
"""

class Attention(nn.Module):
    """
    Decided to create a SingleAttentionHead class that implement the scaled dot product operation
    """

    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=False):
        attention_dim = k.shape[-1]
        attention = (
            torch.matmul(q, k.transpose(-1, -2)) / attention_dim**0.5
        )  # Scaled dot product
        if mask:
            filter = torch.tril(attention, diagonal=0)
            attention = attention.masked_fill(
                filter == 0, float("-inf")
            )  # apply a mask in the case of a decoder
        attention = self.softmax(attention)
        attention = torch.matmul(attention, v)
        return attention


class AttentionBlock(nn.Module):  # <-> MultiHeadAttention
    def __init__(self, num_heads, embedding_dim, attention_dim, mask=False):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads

        self.Wq = nn.Linear(embedding_dim, attention_dim)
        self.Wk = nn.Linear(embedding_dim, attention_dim)
        self.Wv = nn.Linear(embedding_dim, attention_dim)

        self.attention = Attention()
        self.Linear = nn.Linear(attention_dim, attention_dim)

        self.mask = mask

    def forward(self, query, key, value):
        batch_size, sequence_length, _ = query.shape
        length = self.attention_dim // self.num_heads

        q = self.Wq(query)  # Create a query matrix
        k = self.Wk(key)  # Create a key matrix
        v = self.Wv(value)  # Create a value matrix

        #
        q = q.view(batch_size, sequence_length, self.num_heads, length).permute(
            0, 2, 1, 3
        )
        k = k.view(batch_size, sequence_length, self.num_heads, length).permute(
            0, 2, 1, 3
        )

        v = v.view(batch_size, sequence_length, self.num_heads, length).permute(
            0, 2, 1, 3
        )

        attention = self.attention(q, k, v, mask=self.mask)
        attention = (
            attention.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, sequence_length, self.attention_dim)
        )
        out = self.Linear(attention)
        return out



class Resblock(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups = 32, ):
        super().__init__()
        

        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()


        if out_channels!= in_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        x1 = self.act1(self.conv1(self.norm1(x)))
        x2 = self.act2(self.conv2(self.norm2(x1)))

        return self.shortcut(x) + x2



class Downblock(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups, ):
