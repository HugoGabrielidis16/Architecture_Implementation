import torch
import torch.nn as nn


class IdentityBlock(nn.Module):
    def __init__(self, input_channels):
        super(IdentityBlock, self).__init__()

        self.conv1 = nn.Conv2d()
