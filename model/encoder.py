import torch.nn as nn
from complexLayers import ComplexConv2d
from complexFunctions import complex_relu, complex_max_pool2d

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ComplexConv2d(1, 20, 5, 1)

    def forward(self, xr, xi):
        xr,xi = self.conv1(xr,xi)
        xr,xi = complex_relu(xr,xi)
        xr,xi = complex_max_pool2d(xr,xi, 2, 2)
        return xr, xi
