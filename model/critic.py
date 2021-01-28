import torch
import numpy as np
import math
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, k):
        super().__init__()
        layers = list()
        layers.append(nn.Conv2d(16, 32, 3))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.LayerNorm([32, 14, 14]))
        layers.append(nn.MaxPool2d(2))

        layers.append(nn.Conv2d(32, 64, 3))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.LayerNorm([64, 5, 5]))

        layers.append(nn.AvgPool2d(5))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(64, 1))
        self.score = nn.Sequential(*layers)
        self.k = k

    @staticmethod
    def rotate_back(xr, xi):
        thetas = torch.rand(xr.shape[0], 1).to(xr.device) * 2 * math.pi
        thetas = thetas.view([thetas.shape[0]] + (len(xr.shape)-1) * [1])
        return torch.cos(-thetas) * xr - torch.sin(-thetas) * xi

    def compute_gradient_penalty(self, xr, xi, a):
        """Calculates the gradient penalty loss for WGAN GP"""
        device = xr.device
        real_samples = a
        fake_samples = Critic.rotate_back(xr, xi)

        # Random weight term for interpolation between real and fake samples
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        interpolates = interpolates.to(device)
        d_interpolates = self.score(interpolates)
        fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0).to(device)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1).to(device)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def forward(self, xr, xi, a):
        real_score = self.score(a)

        # for each sample, we want to generate k negative examples
        xr = xr.repeat(self.k-1, 1, 1, 1)
        xi = xi.repeat(self.k-1, 1, 1, 1)

        # rotate every sample by a random angle, the result should be a senseless feature vector
        a_prime = Critic.rotate_back(xr, xi)
        fake_scores = self.score(a_prime)

        return real_score.squeeze(), fake_scores.squeeze()
