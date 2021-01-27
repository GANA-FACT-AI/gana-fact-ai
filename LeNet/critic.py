import torch
import numpy as np
import math
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self,  k):
        super().__init__()
        layers = list()
        # self.score = nn.Linear(size, 1)
        layers.append(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1))
        layers.append(nn.Tanh())
        layers.append(nn.AvgPool2d(kernel_size=2))

        layers.append(nn.Flatten())
        layers.append(nn.Linear(800, 1))
        self.score = nn.Sequential(*layers)
        self.k = k

    def forward(self, xr, xi, a):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # xr = torch.flatten(xr, start_dim=1)
        # xi = torch.flatten(xi, start_dim=1)
        # a = torch.flatten(a, start_dim=1)

        real_score = self.score(a).squeeze()
        # for each sample, we want to generate k negative examples
        xr = xr.repeat(self.k - 1, 1, 1, 1)
        xi = xi.repeat(self.k - 1, 1, 1, 1)

        thetas = torch.rand(xr.shape[0], 1, 1, 1).to(device) * 2 * math.pi

        # rotate every sample by a random angle, the result should be a senseless feature vector
        a_prime = torch.cos(-thetas) * xr - torch.sin(-thetas) * xi
        # print("a_prime: ", torch.mean(a_prime))

        fake_scores = self.score(a_prime)

        return real_score, torch.mean(fake_scores)

if __name__ == "__main__":
    a = torch.rand(1, 3, 32, 32)
    xr = torch.rand(3, 3, 32, 32)
    xi = torch.rand(3, 3, 32, 32)

    critic = Critic(3072, 32)
    realscore, fakescore = critic(xr, xi, a)
    print(realscore)
    print(fakescore)