import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, strides):
        """
        Class for the two ConvBlock in the beginning
        """
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, strides, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
        )

    def forward(self, x):
        return self.block(x)


class LinearBlock(nn.Module):
    def __init__(self, input, output):
        """
        Classic Dense + DropOut in keras
        """
        super(LinearBlock, self).__init__()
        self.block = nn.Sequential(nn.Linear(input, output), nn.ReLU(), nn.Dropout(0.5))

    def forward(self, x):
        return self.block(x)


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.convblock1 = ConvBlock(3, 96, (11, 11), 4)
        self.convblock2 = ConvBlock(96, 256, (5, 5), 1)

        self.conv1 = nn.Conv2d(256, 384, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(384, 384, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(384, 256, (3, 3), padding=1)

        self.maxpool = nn.MaxPool2d((3, 3), 2)
        self.flatten = nn.Flatten()

        self.linear_1 = LinearBlock(9216, 4096)
        self.linear_2 = LinearBlock(4096, 4096)
        self.linear_3 = nn.Linear(4096, 10)
        # We don't need to add a softmax activation function since the CrossEntropy loss from pytorch has it in it

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = self.linear_3(x)
        return x


if __name__ == "__main__":
    model = AlexNet()
    x = torch.randn((1, 3, 224, 224))
    y = model(x)
    print(y.shape)
