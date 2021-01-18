import torch
import numpy as np
import math
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, size, k):
        super().__init__()
        layers = list()
        layers.append(nn.Conv2d(16, 32, 3))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
        layers.append(nn.Conv2d(32, 64, 3))
        layers.append(nn.ReLU())
        layers.append(nn.AvgPool2d(13))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(64, 1))
        self.score = nn.Sequential(*layers)
        self.k = k

    def forward(self, xr, xi, a):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        #xr = torch.flatten(xr, start_dim=1)
        #xi = torch.flatten(xi, start_dim=1)
        #a = torch.flatten(a, start_dim=1)

        real_score = self.score(a).squeeze()

        # for each sample, we want to generate k negative examples
        xr = xr.repeat(self.k, 1, 1, 1)
        xi = xi.repeat(self.k, 1, 1, 1)

        thetas = torch.rand(xr.shape[0], 1).to(device) * 2 * math.pi
        thetas = thetas.view([thetas.shape[0]] + (len(xr.shape)-1) * [1])

        # rotate every sample by a random angle, the result should be a senseless feature vector
        a_prime = torch.cos(-thetas) * xr - torch.sin(-thetas) * xi
        #print("a_prime: ", torch.mean(a_prime))
        fake_scores = self.score(a_prime)

        return real_score, fake_scores.squeeze()
