import torch.nn as nn
from complexLayers import ComplexConv2d, ComplexLinear
from complexModules import *
from resnet import ComplexBlock, make_layers


class ProcessingUnit(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.layers = make_layers(ComplexBlock, 16, 32, blocks, stride=2)
        self.conv1 = ComplexConv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = InvariantBatchNorm()
        self.relu = InvariantReLU()

    def forward(self, xr, xi):
        x = self.layers([xr, xi])
        xr = x[0]
        xi = x[1]
        xr, xi = self.conv1(xr, xi)
        xr, xi = self.bn1(xr, xi)
        xr, xi = self.relu(xr, xi)
        return xr, xi
