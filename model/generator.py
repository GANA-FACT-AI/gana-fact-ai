import math
import torch
import torch.nn as nn
import numpy as np
from resnet import BasicBlock, make_layers, _weights_init


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.layers = make_layers(BasicBlock, 16, 16, 5, stride=1)

    def forward(self, x, I_prime, theta):
        if (x != x).any():
            print("error")
        mean1 = torch.mean(x)
        if (I_prime != I_prime).any():
            print("error")
        meani1 = torch.mean(I_prime)
        weights = self.conv1.weight.data
        mean_weights = torch.mean(self.conv1.weight.data)
        mean_var = torch.var(self.conv1.weight.data)
        shape = x.shape
        x = self.conv1(x)
        if (x != x).any():
            print("error")
        mean = torch.mean(x)
        var = torch.var(x)
        var_biased = torch.var(x, unbiased=False)
        x = self.bn1(x)
        if (x != x).any():
            print("error")
        x = self.relu(x)
        if (x != x).any():
            print("error")
        a = self.layers(x)
        if (a != a).any():
            print("error")
        I_prime = self.relu(self.bn1(self.conv1(I_prime)))
        b = self.layers(I_prime)
        rotated_r = torch.cos(theta)*a - torch.sin(theta)*b
        rotated_i = torch.sin(theta)*a + torch.cos(theta)*b
        return rotated_r, rotated_i, a
