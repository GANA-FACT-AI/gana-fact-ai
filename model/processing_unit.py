import torch.nn as nn
from complexLayers import ComplexConv2d
from complexModules import *
from resnet.resnet_blocks import ComplexBlock, make_layers


class ProcessingUnit(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = make_layers(ComplexBlock, 16, 32, 3, stride=2)
        self.conv1 = ComplexConv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = InvariantBatchNorm()
        self.relu = InvariantReLU()
        self.layers = [self.conv1, self.bn1, self.relu]

    def forward(self, xr, xi):
        xr, xi = self.blocks([xr, xi])
        for layer in self.layers:
            xr, xi = layer(xr, xi)
        return xr, xi
