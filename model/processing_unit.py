import torch
import torch.nn as nn
from complexLayers import ComplexConv2d, ComplexLinear
from complexModules import InvariantReLU, invmaxpool2d, _weights_init


class ProcessingUnit(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ComplexConv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = InvariantReLU()
        self.conv2 = ComplexConv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = ComplexConv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = ComplexConv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5 = ComplexConv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6 = ComplexConv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv7 = ComplexConv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, xr, xi):
        xr, xi = self.conv1(xr, xi)
        cr = torch.mean(xr, dim=(0, 2, 3)).view(1, 256, 1, 1)
        ci = torch.mean(xi, dim=(0, 2, 3)).view(1, 256, 1, 1)
        xr, xi = self.relu(xr, xi, cr, ci)
        xr, xi = invmaxpool2d(xr, xi, kernel_size=2, stride=2)
        xr, xi = self.conv2(xr, xi)
        cr = torch.mean(xr, dim=(0, 2, 3)).view(1, 512, 1, 1)
        ci = torch.mean(xi, dim=(0, 2, 3)).view(1, 512, 1, 1)
        xr, xi = self.relu(xr, xi, cr, ci)
        xr, xi = self.conv3(xr, xi)
        cr = torch.mean(xr, dim=(0, 2, 3)).view(1, 512, 1, 1)
        ci = torch.mean(xi, dim=(0, 2, 3)).view(1, 512, 1, 1)
        xr, xi = self.relu(xr, xi, cr, ci)
        xr, xi = self.conv4(xr, xi)
        cr = torch.mean(xr, dim=(0, 2, 3)).view(1, 512, 1, 1)
        ci = torch.mean(xi, dim=(0, 2, 3)).view(1, 512, 1, 1)
        xr, xi = self.relu(xr, xi, cr, ci)
        xr, xi = invmaxpool2d(xr, xi, kernel_size=2, stride=2)
        xr, xi = self.conv5(xr, xi)
        cr = torch.mean(xr, dim=(0, 2, 3)).view(1, 512, 1, 1)
        ci = torch.mean(xi, dim=(0, 2, 3)).view(1, 512, 1, 1)
        xr, xi = self.relu(xr, xi, cr, ci)
        xr, xi = self.conv6(xr, xi)
        cr = torch.mean(xr, dim=(0, 2, 3)).view(1, 512, 1, 1)
        ci = torch.mean(xi, dim=(0, 2, 3)).view(1, 512, 1, 1)
        xr, xi = self.relu(xr, xi, cr, ci)
        xr, xi = self.conv7(xr, xi)
        cr = torch.mean(xr, dim=(0, 2, 3)).view(1, 512, 1, 1)
        ci = torch.mean(xi, dim=(0, 2, 3)).view(1, 512, 1, 1)
        xr, xi = self.relu(xr, xi, cr, ci)
        xr, xi = invmaxpool2d(xr, xi, kernel_size=2, stride=2)
        return xr, xi
