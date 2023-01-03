from trainer import Trainer
from data import mnist_dataset
import torch
from loss import KL_loss, MSE_loss
from vae import VAE


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = VAE((1, 28, 28))
    trainer = Trainer(
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        reconstruction_loss=MSE_loss,
        kl_loss=KL_loss,
        criterion=criterion,
        device=device,
        total_epochs=20,
    )
    trainer.train()
