def f(x, y):
    return x + y


dict = {"x": 5, "y": 4}

print(f(**dict))


import torch
import torch.nn as nn


dict = {
    "in_channels": 3,
    "out_channels": 32,
    "kernel_size": (1, 1),
    "stride": (1, 1),
    "padding": "same",
}


conv1_1 = nn.Conv2d(**dict)
x = torch.randn((10, 3, 256, 256))
y = conv1_1(x)
print(y.shape)
