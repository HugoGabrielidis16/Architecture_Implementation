import torch
import torch.nn as nn


class RNNcell(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
    ) -> None:
        super().__init__()

        self.Wih = nn.Linear(input_size, hidden_size)
        self.Whh = nn.Linear(hidden_size, hidden_size)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_, h):
        """
        x : [1, input_size]
        h : [1,hidden_size]
        """
        Wx = self.Wih(x)
        Wh = self.Whh(h)
        h_next = torch.tanh(Wx + Wh)
        return h_next


class myRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layer):
        super().__init__()
        self.n_layer = n_layer
        self.h = nn.Parameter(torch.zeros(n_layer, hidden_size))
        self.RNN = nn.ModuleList(
            [RNNcell(input_size, hidden_size) for _ in range(n_layer)]
        )

    def forward(self, x):
        """
        x : [sequence_length, input_size]
        h : [n_layer, hidden_size]
        """
        for i in range(self.n_layer - 1):
            h_next = self.RNN[i](x[:, i], self.h[i, :])
            self.h[i + 1, :].data = h_next
        return self.h


if __name__ == "__main__":
    x = torch.randn(10, 10)
    rnn = myRNN(input_size=10, hidden_size=32, n_layer=10)
    y = rnn(x)
    print(y.shape)
