import torch
import torch.nn as nn
from RNN import Many2OneRNN
from embedding import Embedding


class model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size) -> None:
        super().__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.rnn = Many2OneRNN(embedding_dim, hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        y = self.rnn(x)
        return y


if __name__ == "__main__":
    model = model(100)
