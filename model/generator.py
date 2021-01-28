import torch
import torch.nn as nn
from random import random


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        layers = list()
        layers.append(nn.LayerNorm([3, 32, 32]))
        layers.append(nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
        layers.append(nn.LayerNorm([16, 16, 16]))
        layers.append(nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1))
        self.layers = nn.Sequential(*layers)

    def encode(self, x):
        return self.layers(x)

    def rotate(self, a, b, theta):
        rotated_r = torch.cos(theta)*a - torch.sin(theta)*b
        rotated_i = torch.sin(theta)*a + torch.cos(theta)*b
        return rotated_r, rotated_i

    def forward(self, x, I_prime, theta):
        a = self.encode(x)
        b = self.encode(I_prime)
        theta_add = 0
        if random() < 0.5:
            theta_add = -0.5 * 3.1414
            rotated_r, rotated_i = self.rotate(b, a, theta)
        else:
            rotated_r, rotated_i = self.rotate(a, b, theta)
        return rotated_r, rotated_i, a, theta_add
