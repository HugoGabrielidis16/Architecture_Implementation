from jax import tree_multimap
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as T


def load_data():
    train_loader = DataLoader(
        torchvision.datasets.MNIST(
            root="./data/",
            train=True,
            download=True,
            tranform=T.compose(
                T.Normalize((0.1307,), (0.3081)),
                T.ToTensor(),
            ),
        ),
        batch_size=32,
        shuffle=True,
    )

    test_loader = DataLoader(
        torchvision.datasets.MNIST(
            root="./data/",
            train=False,
            download=True,
            tranform=T.compose(
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), T.ToTensor()
            ),
        ),
        batch_size=32,
        shuffle=True,
    )
    return train_loader, test_loader
