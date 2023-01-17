import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):
    def __init__(
        self, n_head, embedding_dim, attention_dim, hidden_dim, vocab_size, n_layer
    ) -> None:
        super().__init__()
        self.Encoder = Encoder(
            n_head, embedding_dim, attention_dim, hidden_dim, vocab_size, n_layer
        )
        self.Decoder = Decoder(
            n_head, embedding_dim, attention_dim, hidden_dim, vocab_size, n_layer
        )
        self.Linear = nn.Linear(attention_dim, vocab_size)
        self.Softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        encoder_output = self.Encoder(x)
        decoder_output = self.Decoder(y, encoder_output)
        output = self.Linear(decoder_output)
        return output


if __name__ == "__main__":
    n_head = 8
    embedding_dim = 512
    attention_dim = 512
    hidden_dim = 2048
    vocab_size = 10000
    n_layer = 6
    model = Transformer(
        n_head, embedding_dim, attention_dim, hidden_dim, vocab_size, n_layer
    )
    x = torch.randint(0, 10000, (32, 100))
    y = torch.randint(0, 10000, (32, 100))
    output = model(x, y)
    print(output.shape)
