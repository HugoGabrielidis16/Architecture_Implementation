import torch
import torch.nn as nn
from Attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self, n_head, embedding_dim, attention_dim, hidden_dim) -> None:
        super().__init__()

        self.MultiHeadAttention = MultiHeadAttention(
            n_head, embedding_dim, attention_dim
        )
        self.LayerNorm1 = nn.LayerNorm(attention_dim)
        self.MLP = nn.Sequential(
            nn.Linear(attention_dim, hidden_dim),
            nn.Linear(hidden_dim, attention_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.LayerNorm2 = nn.LayerNorm(attention_dim)

    def forward(self, x):
        out = self.MultiHeadAttention(query=x, key=x, value=x)
        out = out + x
        out = self.LayerNorm1(out)
        out2 = self.MLP(out)
        out = out + out2
        out2 = self.LayerNorm2(out2)
        return out2


class Encoder(nn.Module):
    def __init__(
        self, n_head, embedding_dim, attention_dim, hidden_dim, vocab_size, n_layer
    ) -> None:
        super().__init__()
        self.Embedding = nn.Embedding(vocab_size, embedding_dim)
        self.Encoder = nn.ModuleList(
            [
                EncoderLayer(n_head, embedding_dim, attention_dim, hidden_dim)
                for _ in range(n_layer)
            ]
        )

    def forward(self, x):
        x = self.Embedding(x)
        for Layer in self.Encoder:
            x = Layer(x)
        return x


if __name__ == "__main__":
    x = torch.randint(0, 100, (5, 100))
    encoder = Encoder(8, 512, 512, 2048, 100, 6)
    out = encoder(x)
    print(out.shape)
