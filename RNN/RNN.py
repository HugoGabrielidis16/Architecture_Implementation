import torch
import torch.nn as nn


class RNNcell(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wax = nn.Linear(input_size, hidden_size)
        self.Waa = nn.Linear(hidden_size, hidden_size)

        self.softmax = nn.Softmax()

    def forward(self, x, a_prev=None):
        if a_prev == None:
            a_prev = torch.zeros(1, self.hidden_size, requires_grad=True)
        a_next = torch.tanh(self.Wax(x) + self.Waa(a_prev))

        return a_next


class One2OneRNN(nn.Module):
    """
    x : [batch_size, input_size]
    y : [batch_size, output_size]
    """

    def __init__(self, input_size, hidden_size, output_size) -> None:
        super().__init__()

        self.rnn = RNNcell(input_size, hidden_size)
        self.Wya = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Sigmoid()

    def forward(self, x):
        a_next = self.rnn(x)
        y_pred = self.softmax(self.Wya(a_next))
        return y_pred


class Many2OneRNN(nn.Module):  # n_x & 1
    """
    x : [batch_size, seq_len, input_size]
    y : [batch_size, output_size]
    """

    def __init__(self, input_size, hidden_size, output_size) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = RNNcell(input_size, hidden_size)
        self.Wya = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Sigmoid()

    def forward(self, x, a=None):
        batch_size, sequence_len, _ = x.shape
        if a == None:
            a = torch.zeros(batch_size, sequence_len, self.hidden_size)
        for i in range(sequence_len):
            a[:, i, :] = self.rnn(x[:, i, :], a[:, i - 1, :])
        y_pred = self.softmax(self.Wya(a[:, -1, :]))
        return y_pred


class One2Many(nn.Module):  # 1 &  n_y
    """
    x : [batch_size, input_size]
    y : [batch_size, seq_len,output_size]
    """

    def __init__(self, input_size, hidden_size, output_size, seq_len) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.seq_len = seq_len
        self.rnn = RNNcell(input_size, hidden_size)
        self.rnn2 = RNNcell(output_size, hidden_size)

        self.Wya = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, a=None):
        batch_size, _ = x.shape
        y_pred = torch.zeros(batch_size, self.seq_len, self.output_size)
        if a == None:
            a = torch.zeros(batch_size, self.seq_len, self.hidden_size)
        for i in range(self.seq_len):
            if i == 0:
                a[:, i, :] = self.rnn(x, a[:, i - 1, :])
                y_pred[:, i, :] = self.sigmoid(self.Wya(a[:, i, :]))
            else:
                a[:, i, :] = self.rnn2(y_pred[:, i - 1, :], a[:, i - 1, :])
                y_pred[:, i, :] = self.sigmoid(self.Wya(a[:, i, :]))
        return y_pred


class Many2ManyRNN(nn.Module):  # n_x = n_y
    """
    x : [batch_size, seq_len, input_size]
    y : [batch_size, seq_len,output_size]
    """

    def __init__(self, input_size, hidden_size, output_size) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn = RNNcell(input_size, hidden_size)
        self.Wya = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Sigmoid()

    def forward(self, x, a=None):
        batch_size, sequence_len, _ = x.shape
        if a == None:
            a = torch.zeros(batch_size, sequence_len, self.hidden_size)
        y_pred = torch.zeros(batch_size, sequence_len, self.output_size)
        for i in range(sequence_len):
            a[:, i, :] = self.rnn(x[:, i, :], a[:, i - 1, :])
            y_pred[:, i, :] = self.softmax(self.Wya(a[:, i, :]))
        return y_pred


class Many2ManyRNN(nn.Module):  # n_x != n_y
    """
    x : [batch_size, seq_len_x, input_size]
    y : [batch_size, seq_len_y,output_size]
    """

    def __init__(self, input_size, hidden_size, output_size, output_len) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.output_len = output_len
        self.rnn1 = RNNcell(input_size, hidden_size)
        self.rnn2 = RNNcell(output_size, hidden_size)
        self.Wya = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Sigmoid()

    def forward(self, x, a=None):
        batch_size, sequence_len, _ = x.shape
        if a == None:
            a = torch.zeros(
                batch_size, sequence_len + self.output_len, self.hidden_size
            )
        y_pred = torch.zeros(batch_size, self.output_len, self.output_size)
        for i in range(sequence_len):
            a[:, i, :] = self.rnn1(x[:, i, :], a[:, i - 1, :])

        y_pred[:, 0, :] = self.softmax(self.Wya(a[:, sequence_len - 1, :]))
        for i in range(self.output_len):
            a[:, i + sequence_len, :] = self.rnn2(
                y_pred[:, i, :], a[:, sequence_len + i - 1, :]
            )
            y_pred[:, i, :] = self.softmax(self.Wya(a[:, i + sequence_len, :]))
        return y_pred


if __name__ == "__main__":
    x = torch.randn(32, 5, 64)
    model = Many2ManyRNN(64, 128, 35, 10)
    y = model(x)
    print(y.shape)
