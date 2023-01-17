import torch
import torch.nn as nn


class LSTMcell(nn.Module):
    def __init__(self,input_size,hidden_size):
        super.__init__()

        self.Wx = nn.Linear(input_size,hidden_size)
        self.Wh = nn.Linear(hidden_size, hidden_size)
        self.Wc = 
    def forward(self,x,h,c):


