import torch
import torch.nn as nn
import torchvision
from glob import glob
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from PIL import Image
import matplotlib.pyplot as plt


def get_image_filenames(folder):
    folders = [f for f in listdir(folder)]
    file_path = []
    for folder_name in folders:
        file_path += [
            f"{folder}{folder_name}/{f}"
            for f in listdir(folder + folder_name)
            if isfile(join(folder + folder_name, f))
        ]

    return file_path


class BirdDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file_path,
        train=False,
    ):
        self.file_path = file_path
        if train:
            self.transform = Compose(
                [
                    Resize(height=224, width=224),
                    Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                        max_pixel_value=255,
                    ),
                    ToTensorV2(),
                ]
            )
        else:
            self.transform = Compose([Resize(height=224, width=224), ToTensorV2()])

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, idx):
        image_path = self.file_path[idx]
        image = np.array(Image.open(image_path))
        transformed_image = self.transform(image=image)["image"]
        return (transformed_image).to(torch.float32)


class BirdLoader:
    def __init__(self):
        self.train_path = get_image_filenames("bird_data/train/")
        self.test_path = get_image_filenames("bird_data/test/")
        self.valid_path = get_image_filenames("bird_data/valid/")

    def setup(self):
        self.train_ds = BirdDataset(self.train_path, train=True)
        self.test_ds = BirdDataset(self.test_path)
        self.valid_ds = BirdDataset(self.valid_path)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_ds, batch_size=256, shuffle=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_ds, batch_size=32, shuffle=True
        )
        self.valid_loader = torch.utils.data.DataLoader(
            self.valid_ds, batch_size=32, shuffle=True
        )

        return self.train_loader, self.test_loader, self.valid_loader


if __name__ == "__main__":
    bird_loader = BirdLoader()
    train_loader, test_loader, valid_loader = bird_loader.setup()
    for idx, data in enumerate(train_loader):
        x = data[0].permute((1, 2, 0)).cpu().detach().numpy()
        plt.imshow(x, cmap="turbo")
        plt.show()
        break
