import torch
from data import load_data
from Alexnet_pytorch import AlexNet
import torch.nn as nn
import torch.optim as optim
import numpy as np
from progressbar import progress_bar

if __name__ == "__main__":
    _, testloader, classes = load_data()
    model = AlexNet()
    model.load_state_dict(torch.load("Alexnet.pt", map_location=torch.device("cpu")))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    test_loss = 0
    test_correct = 0
    total = 0
    with torch.no_grad():
        for batch_num, data in enumerate(testloader, 0):
            x, y = data[0].to(device), data[1].to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            test_loss += loss.item()
            prediction = torch.max(y_pred, 1)
            total += x.size(0)
            test_correct += np.sum(prediction[1].cpu().numpy() == y.cpu().numpy())

            progress_bar(
                batch_num,
                len(testloader),
                "Loss: %.4f | Acc: %.3f%% (%d/%d)"
                % (
                    test_loss / (batch_num + 1),
                    100.0 * test_correct / total,
                    test_correct,
                    total,
                ),
            )
