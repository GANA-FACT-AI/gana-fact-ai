import torch.nn as nn
from complexLayers import ComplexConv2d, ComplexLinear
from complexModules import *
import model.processing_unit


class ProcessingUnit(model.processing_unit.ProcessingUnit):
    def __init__(self):
        super().__init__(blocks=0, init_layers=False)
        self.conv1 = ComplexConv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, bias=False)
        self.tanh = nn.Tanh()
        self.avgpool2d = nn.AvgPool2d(kernel_size=2)
        self.conv2 = ComplexConv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, bias=False)
        self.linear1 = ComplexLinear(in_features=120, out_features=84)
        self.linear2 = ComplexLinear(in_features=84, out_features=10)

    def forward(self, xr, xi):
        xr, xi = self.conv1(xr, xi)
        xr = self.tanh(xr)
        xi = self.tanh(xi)
        xr = self.avgpool2d(xr)
        xi = self.avgpool2d(xi)

        xr, xi = self.conv2(xr, xi)
        xr = self.tanh(xr)
        xi = self.tanh(xi)

        xr = torch.flatten(xr, 1)
        xi = torch.flatten(xi, 1)

        xr, xi = self.linear1(xr, xi)
        xr = self.tanh(xr)
        xi = self.tanh(xi)

        xr, xi = self.linear2(xr, xi)
        return xr, xi
