import math
import torch
import torch.nn as nn
import numpy as np


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.relu = nn.ReLU()
        self.maxpool2d = nn.MaxPool2d(2)

    def forward(self, x, I_prime, theta):
        x = self.conv1(x)
        #a = self.relu(a)
        a = self.maxpool2d(x)
        I_prime = self.conv1(I_prime)
        #a = self.relu(a)
        b = self.maxpool2d(I_prime)
        rotated_r = torch.cos(theta)*a - torch.sin(theta)*b
        rotated_i = torch.sin(theta)*a + torch.cos(theta)*b
        return rotated_r, rotated_i, a
