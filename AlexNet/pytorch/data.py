import torch
import torchvision
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import dill as pickle


def load_data():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    batch_size = 64

    trainset = torchvision.datasets.CIFAR10(
        root="../data",
        train=True,
        download=True,
        transform=transform,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="../data",
        train=False,
        download=True,
        transform=transform,
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2,
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
    return trainloader, testloader, classes


if __name__ == "__main__":
    trainloader, testloader, classes = load_data()
    plt.figure(figsize=(5, 10))
    print(len(trainloader))
    for x, y in trainloader:
        print(x.shape, y.shape)
    """ enum = enumerate(trainloader)
    idx, (x, y) = next(enum)
    print(y[0])

    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.imshow(x[i].permute(1, 2, 0))
        plt.axis("off")
        plt.title(classes[y[i]])
    plt.show() """
