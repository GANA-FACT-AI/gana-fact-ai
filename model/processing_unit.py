import torch.nn as nn
from complexLayers import ComplexConv2d, ComplexLinear
from complexModules import *
from resnet import ComplexBlock, make_layers


class ProcessingUnit(nn.Module):
    def __init__(self):
        super().__init__()
        block = make_layers(ComplexBlock, 16, 32, 3, stride=2)
        conv1 = ComplexConv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        bn1 = InvariantBatchNorm()
        relu = InvariantReLU()
        self.layers = nn.Sequential(block, conv1, bn1, relu)

    def forward(self, xr, xi):
        return self.layers([xr, xi])
