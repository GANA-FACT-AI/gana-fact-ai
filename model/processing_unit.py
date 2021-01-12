import torch.nn as nn
from complexLayers import ComplexConv2d, ComplexLinear
from complexModules import *

class ProcessingUnit(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = InvariantBatchNorm()
        self.relu = InvariantReLU()
        self.conv2 = ComplexConv2d(20, 50, 5, 1)
        self.fc1 = ComplexLinear(4*4*50, 500)
        self.fc2 = ComplexLinear(500, 200)

    def forward(self, xr, xi):
        xr,xi = self.bn(xr,xi)
        xr,xi = self.conv2(xr,xi)
        xr,xi = self.relu(xr,xi)
        xr,xi = invmaxpool2d(xr,xi, 2, 2)
        
        xr = xr.view(-1, 4*4*50)
        xi = xi.view(-1, 4*4*50)
        xr,xi = self.fc1(xr,xi)
        xr,xi = self.relu(xr,xi)
        xr,xi = self.fc2(xr,xi)
        return xr, xi
