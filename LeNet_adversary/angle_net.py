import torch
import numpy as np
import math
import torch.nn as nn


class AngleNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers = list()
        layers.append(nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3, stride=1))
        # added artificially
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU())

        layers.append(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1))
        # added artificially
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU())

        layers.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1))
        #
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.AvgPool2d(kernel_size=2))
        layers.append(nn.ReLU())


        layers.append(nn.Flatten())
        #  1 layer added artificially
        layers.append(nn.Linear(3200, 3200))
        layers.append(nn.BatchNorm1d(3200))
        layers.append(nn.ReLU())

        layers.append(nn.Linear(3200, 3200))
        layers.append(nn.BatchNorm1d(3200))
        layers.append(nn.ReLU())

        layers.append(nn.Linear(3200, 256))
        layers.append(nn.BatchNorm1d(256))
        layers.append(nn.ReLU())


        layers.append(nn.Linear(256, 1))
        #
        # layers.append(nn.Linear(800, 1))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)


    @staticmethod
    def rotate_back(xr, xi):
        thetas = torch.rand(xr.shape[0], 1).to(xr.device) * 2 * math.pi
        thetas = thetas.view([thetas.shape[0]] + (len(xr.shape)-1) * [1])
        return torch.cos(-thetas) * xr - torch.sin(-thetas) * xi

    def forward(self, x):
        angle = self.layers(x) * 2 * math.pi
        return angle.view(x.shape[0],1,1,1)

#
# if __name__ == "__main__":
#     a = torch.rand(1, 3, 32, 32)
#     xr = torch.rand(3, 3, 32, 32)
#     xi = torch.rand(3, 3, 32, 32)
#     x =  torch.rand(3, 6, 14, 14)
#
#     theta_critic = AngleNet()
#     angle = theta_critic(x)
#     print(angle.shape)