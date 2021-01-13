import math
import torch
import torch.nn as nn
import numpy as np
from resnet import BasicBlock, make_layers


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.layers = make_layers(BasicBlock, 16, 16, 3, stride=1)

    def forward(self, x, I_prime, theta):
        x = self.relu(self.bn1(self.conv1(x)))
        a = self.layers(x)
        I_prime = self.relu(self.bn1(self.conv1(I_prime)))
        b = self.layers(I_prime)
        rotated_r = torch.cos(theta)*a - torch.sin(theta)*b
        rotated_i = torch.sin(theta)*a + torch.cos(theta)*b
        return rotated_r, rotated_i, a
