from pickletools import optimize
from turtle import forward
import torch
import torch.nn as nn
from discriminator import Discriminator
from generator import Generator
import torch.nn.functional as F
import pytorch_lightning as pl


class GAN(pl.LightningModule):
    def __init__(self, latent_dims=100, lr=3e-4):
        super(GAN, self).__init__()
        self.save_hyperparameters()

        self.discriminator = Discriminator()
        self.generator = Generator(latent_dims)

        self.noise = torch.rand(32, self.hpparams.latent_dims)

    def forward(self, x):
        return self.generator(x)

    def adversarial_loss(self, y_pred, y):
        return F.binary_cross_entropy(y_pred, y)

    def training_steps(self,batch, batch_idsm optimizer_idx):

    def validation_steps():


    def configure_optizers(self):
        lr = self.hparams.lr
        optimizer_generator = torch.optim.Adam(self.generator.parameters(), lr)
        optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr)
