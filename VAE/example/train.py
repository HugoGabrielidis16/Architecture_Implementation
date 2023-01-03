from data import BirdLoader
from trainer import Trainer
from loss import KL_loss, MSE_loss
from vae import VAE
import torch


if __name__ == "__main__":
    print("Generating data loaders...")
    bird_loader = BirdLoader()
    train_loader, test_loader, valid_loader = bird_loader.setup()
    print("Data loaders generated.")
    model = VAE((3, 224, 224))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = torch.optim.Adam(params=model.parameters(), lr=0.001)

    print("Training model...")
    trainer = Trainer(
        model=model,
        trainloader=train_loader,
        testloader=test_loader,
        reconstruction_loss=MSE_loss,
        kl_loss=KL_loss,
        criterion=criterion,
        device=device,
        total_epochs=1,
    )
    trainer.fit()
