import torch
from tqdm import tqdm
import torch.nn.functional as F
from progressbar import progress_bar
import numpy as np


class Trainer:
    def __init__(
        self, model, trainloader, testloader, optimizer, criterion, device="cpu"
    ):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def fit(self, epochs):
        for epoch in range(epochs):
            print(f"Epoch : {epoch}")
            train_loss = self.train_()
            val_loss = self.val_()

            print(
                f" Epoch : {epoch}/{epochs} - train_loss : {train_loss} - val_loss{val_loss}"
            )

    def train_(self):
        self.model.to(self.device)
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0
        for batch_id, data in enumerate(self.trainloader, 0):

            running_loss = 0
            x, y = data[0].to(self.device), data[1].to(self.device)

            self.optimizer.zero_grad()  # need to initalize the grads to 0 for each batchs

            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            prediction = torch.max(y_pred, 1)
            total += y.size(0)

            # train_correct incremented by one if predicted right
            train_correct += np.sum(prediction[1].cpu().numpy() == y.cpu().numpy())

            progress_bar(
                batch_id,
                len(self.trainloader),
                "Loss: %.4f | Acc: %.3f%% (%d/%d)"
                % (
                    train_loss / (batch_id + 1),
                    100.0 * train_correct / total,
                    train_correct,
                    total,
                ),
            )

        return train_loss, train_correct / total

        return running_loss

    def val_(self):
        test_loss = 0
        test_correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for batch_num,data in enumerate(self.testloader, 0):
                x, y = data[0].to(self.device), data[1].to(self.device)
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)
                test_loss += loss.item()
                prediction = torch.max(y_pred, 1)
                total += x.size(0)
                test_correct += np.sum(prediction[1].cpu().numpy() == y.cpu().numpy())

                progress_bar(
                    batch_num,
                    len(self.testloader),
                    "Loss: %.4f | Acc: %.3f%% (%d/%d)"
                    % (
                        test_loss / (batch_num + 1),
                        100.0 * test_correct / total,
                        test_correct,
                        total,
                    ),
                )

        return test_loss, test_correct / total


def accuracy(y, y_pred):
    y = F.one_hot(y, 10).float()
    y_pred = torch.tensor([[1, 4, 2, 3]])

    _, arg_y = torch.max(y, 1)
    _, arg_ypred = torch.max(y_pred, 1)

    count = (arg_y == arg_ypred).float().sum()
    return count / y.shape[0]


if __name__ == "__main__":
    y = torch.tensor(
        [[1, 3, 0, 0]],
    )
    y_pred = torch.tensor([[1, 4, 20, 3]])
