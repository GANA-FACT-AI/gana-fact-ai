import torch.nn as nn
import torch
from model.critique import Critique
from model.generator import Generator


class WGAN(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.generator = Generator(12)  # TODO: remove magic numbers
        self.critique = Critique(2880, k)  # TODO: remove magic numbers

    def generate(self, a, b, theta):
        xr, xi = self.generator(a, b, theta)

        return xr, xi

    def forward(self, a, b, theta, *args, **kwargs):
        xr, xi = self.generate(a, b, theta)
        real_score, fake_score = self.critique(xr, xi, a)
        critique_loss = torch.mean(real_score) - torch.mean(fake_score)
        generator_loss = -torch.mean(fake_score)

        for p in self.critique.parameters():
            p.data.clamp_(-0.01, 0.01)

        return xr, xi, critique_loss, generator_loss
