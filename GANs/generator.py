from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
import torch

"""
The Generator aims to create images that looks like 

"""


class Generator(nn.Module):
    def __init__(self, latent_dims) -> None:
        super(Generator, self).__init__()

        self.linear1 = nn.Linear(latent_dims, 7 * 7 * 64)
        self.conv1 = nn.ConvTranspose2d(64, 10, 2, stride=2)
        self.conv2 = nn.ConvTranspose2d(10, 1, 2, stride=2)

    def forward(self, x):
        x = self.linear1(x)
        x = x.view(-1, 64, 7, 7)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


if __name__ == "__main__":
    model = Generator(62)
    x = torch.randn((10, 62))
    y = model(x)
    print(y.shape)
