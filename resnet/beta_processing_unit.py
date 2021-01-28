import torch.nn as nn
from complexLayers import ComplexConv2d, ComplexLinear
from complexModules import *
from resnet.resnet_blocks import ComplexBlock, make_layers


class BetaProcessingUnit(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.layers = make_layers(ComplexBlock, 16, 32, blocks, stride=2)

    def forward(self, xr, xi):
        x = self.layers([xr, xi])
        xr = x[0]
        xi = x[1]
        return xr, xi
