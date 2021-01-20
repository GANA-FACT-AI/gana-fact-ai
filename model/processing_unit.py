import torch.nn as nn
from complexLayers import ComplexConv2d, ComplexLinear
from complexModules import *

class ProcessingUnit(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ComplexConv2d(384, 256, kernel_size=3, padding=1, bias=False)
        self.relu = InvariantReLU()
        self.conv2 = ComplexConv2d(256, 256, kernel_size=3, padding=1, bias=False)

    def forward(self, xr, xi):
        xr, xi = self.conv1(xr, xi)
        cr = torch.mean(xr, dim=(0, 2, 3)).view(1, 256, 1, 1)
        ci = torch.mean(xi, dim=(0, 2, 3)).view(1, 256, 1, 1)
        xr, xi = self.relu(xr, xi, cr, ci)
        xr, xi = self.conv2(xr, xi)
        cr = torch.mean(xr, dim=(0, 2, 3)).view(1, 256, 1, 1)
        ci = torch.mean(xi, dim=(0, 2, 3)).view(1, 256, 1, 1)
        xr, xi = self.relu(xr, xi, cr, ci)
        xr, xi = invmaxpool2d(xr, xi, 3, 2)

        return xr, xi
