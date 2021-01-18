import torch.nn as nn
from complexLayers import ComplexConv2d, ComplexLinear
from complexModules import *

class ProcessingUnit(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ComplexConv2d(384, 256, kernel_size=3, padding=1)
        self.relu = InvariantReLU(inplace=True)
        self.conv2 = ComplexConv2d(256, 256, kernel_size=3, padding=1)

    def forward(self, xr, xi):
        xr, xi = self.conv1(xr, xi)
        xr, xi = self.relu(xr, xi)
        xr, xi = self.conv2(xr, xi)
        xr, xi = self.relu(xr, xi)
        xr, xi = invmaxpool2d(xr, xi, 3, 2) #TODO: Set appropriate c value

        return xr, xi
