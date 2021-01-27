import torch.nn as nn
from complexLayers import ComplexConv2d, ComplexLinear
from complexModules import *

class ProcessingUnit(nn.Module):
    def __init__(self):
        super().__init__()
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
        # print('c1', xi.shape)
        xr = self.avgpool2d(xr)
        xi = self.avgpool2d(xi)
        # print('avgpool', xi.shape)

        xr, xi = self.conv2(xr, xi)
        xr = self.tanh(xr)
        xi = self.tanh(xi)

        # print('c2', xi.shape)

        xr = torch.flatten(xr, 1)
        xi = torch.flatten(xi, 1)

        xr, xi = self.linear1(xr, xi)
        xr = self.tanh(xr)
        xi = self.tanh(xi)
        # print('lin1', xr.shape)

        xr, xi = self.linear2(xr, xi)
        # print('lin2', xr.shape)
        print('\n')
        return xr, xi
