import torch.nn as nn
import torch.nn.functional as F
import torch


"""
The Discriminator role is to detect if image in input is a fake or not.
So it takes an batchs of (28,28,1) images as inputs and return a float between 0 and 1.
With the value near 0 if it thinks it is a fake and near 1 if it thinks it is real.
"""


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple CNN
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # Flatten the tensor so it can be fed into the FC layers
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return torch.sigmoid(x)


if __name__ == "__main__":
    model = Discriminator()
    x = torch.randn((10, 1, 28, 28))
    y = model(x)
    print(y.shape)
