from Alexnet_pytorch import AlexNet
from data import load_data
from trainer import Trainer
import torch.nn as nn
import torch.optim as optim
import torch

if __name__ == "__main__":
    trainloader, testloader, classes = load_data()
    model = AlexNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(model, trainloader, testloader, optimizer, criterion, device = device)
    trainer.fit(10)
    torch.save(model.state_dict(), "your_model_path.pt")
