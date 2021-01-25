import torch
import numpy as np
import math
import torch.nn as nn
from resnet import make_layers, LayerNormBlock


class Angle(nn.Module):
    def __init__(self):
        super().__init__()
        layers = list()
        layers.append(nn.Conv2d(32, 64, 3))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.MaxPool2d(2))

        layers.append(make_layers(LayerNormBlock, 64, 128, input_size=[1, 64, 8, 8], num_blocks=1, stride=2))

        layers.append(nn.Conv2d(128, 256, 3))
        layers.append(nn.ReLU())

        layers.append(nn.AvgPool2d(6))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(256, 1))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)


    @staticmethod
    def rotate_back(xr, xi):
        thetas = torch.rand(xr.shape[0], 1).to(xr.device) * 2 * math.pi
        thetas = thetas.view([thetas.shape[0]] + (len(xr.shape)-1) * [1])
        return torch.cos(-thetas) * xr - torch.sin(-thetas) * xi

    def forward(self, x):
        angle = self.layers(x) * 2 * math.pi
        return angle.view(128,1,1,1)
