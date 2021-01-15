import torch.nn as nn
from complexLayers import ComplexConv2d, ComplexLinear
from complexModules import *
from resnet import ComplexBlock, make_layers, _weights_init

class ProcessingUnit(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = make_layers(ComplexBlock, 16, 32, 3, stride=2)

    def forward(self, xr, xi):
        x = self.layers([xr, xi])
        xr = x[0]
        xi = x[1]
        return xr, xi
