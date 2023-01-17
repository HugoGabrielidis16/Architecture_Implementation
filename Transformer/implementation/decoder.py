import torch
import torch.nn as nn
from Attention import MultiHeadAttention


class DecoderLayer(nn.Module):
    def __init__(self, n_head, embedding_dim, attention_dim, hidden_dim) -> None:
        super().__init__()

        self.MaskedMultiHeadAttention = MultiHeadAttention(
            n_head, embedding_dim, attention_dim, mask=True
        )
        self.LayerNorm1 = nn.LayerNorm(attention_dim)
        self.MultiHeadAttention = MultiHeadAttention(
            n_head, attention_dim, attention_dim
        )
        self.LayerNorm2 = nn.LayerNorm(attention_dim)
        self.MLP = nn.Sequential(
            nn.Linear(attention_dim, hidden_dim),
            nn.Linear(hidden_dim, attention_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.LayerNorm3 = nn.LayerNorm(attention_dim)

    def forward(self, x, encoder_output):
        masked_attention = self.MaskedMultiHeadAttention(query=x, key=x, value=x)
        x = x + masked_attention
        x = self.LayerNorm1(x)
        attention = self.MultiHeadAttention(
            query=x, key=encoder_output, value=encoder_output
        )
        x = x + attention
        x = self.LayerNorm2(x)
        MLP_output = self.MLP(x)
        x = x + MLP_output
        x = self.LayerNorm3(x)

        return x


class Decoder(nn.Module):
    def __init__(
        self, n_head, embedding_dim, attention_dim, hidden_dim, vocab_size, n_layer
    ) -> None:
        super().__init__()
        self.Embedding = nn.Embedding(vocab_size, embedding_dim)
        self.Decoder = nn.ModuleList(
            [
                DecoderLayer(
                    n_head=n_head,
                    embedding_dim=embedding_dim,
                    attention_dim=attention_dim,
                    hidden_dim=hidden_dim,
                )
                for _ in range(n_layer)
            ]
        )

    def forward(self, x, encoder_output):
        x = self.Embedding(x)
        for Layer in self.Decoder:
            x = Layer(x, encoder_output)
        return x


def decoder_layer_test():
    x = torch.randn((32, 10, 512))
    encoder_output = torch.randn((32, 10, 512))
    n_head = 8
    embedding_dim = 512
    attention_dim = 512
    hidden_dim = 2048
    model = DecoderLayer(n_head, embedding_dim, attention_dim, hidden_dim)
    output = model(x, encoder_output)
    print(output.shape)


def decoder_test():
    x = torch.randint(0, 10000, (32, 10))
    encoder_output = torch.randn((32, 10, 512))
    n_head = 8
    embedding_dim = 512
    attention_dim = 512
    hidden_dim = 2048
    vocab_size = 10000
    n_layer = 6
    model = Decoder(
        n_head, embedding_dim, attention_dim, hidden_dim, vocab_size, n_layer
    )
    output = model(x, encoder_output)
    print(output.shape)


if __name__ == "__main__":
    decoder_test()
