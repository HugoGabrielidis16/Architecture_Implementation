import torch
import torch.nn as nn
from Attention import MultiHeadAttention
from embedding import FinalEmbedding


class OneEncoderLayer(nn.Module):
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
        out = self.MultiHeadAttention(x)
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
        self.Embedding = FinalEmbedding(vocab_size, embedding_dim)
        self.Encoder = nn.ModuleList(
            [
                OneEncoderLayer(n_head, embedding_dim, attention_dim, hidden_dim)
                for _ in range(n_layer)
            ]
        )

    def forward(self, x):
        x = self.Embedding(x)
        for Layer in self.Encoder:
            x = Layer(x)
        return x


if __name__ == "__main__":
    x = torch.randint(0, 100, (32, 100))
    encoder = Encoder(
        n_head=8,
        embedding_dim=512,
        attention_dim=512,
        hidden_dim=1024,
        vocab_size=100,
        n_layer=6,
    )
    y = encoder(x)
    print(y.shape)
""" class Decoder(nn.Module):
    def __init__(self, n_head, embedding_dim, attention_dim, hidden_dim, vocab_size) -> None:
 """
