from data import BirdLoader
from trainer import Trainer
from loss import KL_loss, MSE_loss
from vae import VAE
import torch
import argparse
from types import SimpleNamespace


default_config = SimpleNamespace(
    batch_size=8,  # Small batch size since we don't have much data
    model="resunet34d",  # resunet34d, unet
    augment=False,  # use data augmentation
    epochs=10,  # for brevity, increase for better results :)
    lr=2e-3,
    mixed_precision=False,  # use automatic mixed precision
    seed=42,
    log_preds=False,
)


def parse_args():
    "Overriding default argments"
    argparser = argparse.ArgumentParser(description="Process hyper-parameters")
    argparser.add_argument(
        "--batch_size", type=int, default=default_config.batch_size, help="batch size"
    )
    argparser.add_argument(
        "--epochs",
        type=int,
        default=default_config.epochs,
        help="number of training epochs",
    )
    argparser.add_argument(
        "--lr", type=float, default=default_config.lr, help="learning rate"
    )
    argparser.add_argument(
        "--model",
        type=str,
        default=default_config.model,
        help="model to train",
    )
    argparser.add_argument(
        "--seed", type=int, default=default_config.seed, help="random seed"
    )
    argparser.add_argument(
        "--log_preds",
        type=bool,
        default=default_config.log_preds,
        help="log model predictions",
    )
    args = argparser.parse_args()
    vars(default_config).update(vars(args))


if __name__ == "__main__":

    parse_args()
    print("Generating data loaders...")
    bird_loader = BirdLoader()
    train_loader, test_loader, valid_loader = bird_loader.setup()
    print("Data loaders generated.")
    model = VAE((3, 224, 224))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = torch.optim.Adam(params=model.parameters(), lr=default_config.lr)

    print("Training model...")
    trainer = Trainer(
        model=model,
        trainloader=train_loader,
        testloader=test_loader,
        reconstruction_loss=MSE_loss,
        kl_loss=KL_loss,
        criterion=criterion,
        device=device,
        total_epochs=default_config.epochs,
    )
    trainer.fit()
