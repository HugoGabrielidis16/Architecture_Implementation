import torch
import torch.nn as nn
from torchsummary import summary


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Trim(nn.Module):
    def __init__(self, shape, *args):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x[:, :, : self.shape, : self.shape]


class VAE(nn.Module):
    def __init__(self, shape) -> None:
        super(VAE, self).__init__()

        self.channel, self.width, self.height = shape
        self.encoder = nn.Sequential(
            nn.Conv2d(self.channel, 8, stride=(1, 1), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(8, 16, stride=(2, 2), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(16, 16, stride=(2, 2), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(16, 32, stride=(1, 1), kernel_size=(3, 3), padding=1),
            nn.Flatten(),
        )

        self.mean = nn.Linear(100352, 100)  # Result
        self.var = nn.Linear(100352, 100)

        self.decoder = nn.Sequential(
            nn.Linear(100, 100352),
            Reshape(-1, 32, 56, 56),
            nn.ConvTranspose2d(32, 16, stride=(1, 1), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(16, 16, stride=(2, 2), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(16, 8, stride=(2, 2), kernel_size=(3, 3), padding=0),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(8, 3, stride=(1, 1), kernel_size=(3, 3), padding=0),
            Trim(224, 224),  # 3x225x225 -> 3x224x224
            nn.Sigmoid(),
        )

    def reparameterize(self, mean, var):
        eps = torch.randn(mean.size(0), mean.size(1))
        z = mean + eps * torch.exp(var / 2.0)
        return z

    def forward(self, x):
        encode = self.encoder(x)
        mean = self.mean(encode)
        var = self.var(encode)
        encoded = self.reparameterize(mean, var)
        return self.decoder(encoded), mean, var


if __name__ == "__main__":
    x = torch.randn(6, 3, 224, 224)
    model = VAE((3, 224, 224))
    pred_img, pred_mean, pred_var = model(x)
    # summary(model, (3, 224, 224))
