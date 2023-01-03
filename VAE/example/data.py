import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from glob import glob

import os
from os import listdir
from os.path import isfile, join
from PIL import Image


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
    ):
        self.file_path = file_path

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, idx):
        image = self.file_path[idx]
        image = Image.open(image)
        image = self.transform(image)
        return image

    def transform(self, image):
        transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        return transform(image)


class BirdLoader:
    def __init__(self):
        self.train_path = get_image_filenames("bird_data/train/")
        self.test_path = get_image_filenames("bird_data/test/")
        self.valid_path = get_image_filenames("bird_data/valid/")

    def setup(self):
        self.train_ds = BirdDataset(self.train_path)
        self.test_ds = BirdDataset(self.test_path)
        self.valid_ds = BirdDataset(self.valid_path)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_ds, batch_size=1, shuffle=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_ds, batch_size=1, shuffle=True
        )
        self.valid_loader = torch.utils.data.DataLoader(
            self.valid_ds, batch_size=1, shuffle=True
        )

        return self.train_loader, self.test_loader, self.valid_loader


if __name__ == "__main__":
    bird_loader = BirdLoader()
    train_loader, test_loader, valid_loader = bird_loader.setup()
    print(len(train_loader), len(test_loader), len(valid_loader))
