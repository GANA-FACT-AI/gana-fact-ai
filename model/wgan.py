import torch.nn as nn
import torch
from model.critic import Critic
from model.generator import Generator


class WGAN(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.generator = Generator()
        self.critic = Critic(1176, k)  # TODO: remove magic numbers

    def generate(self, a, b, theta):
        xr, xi, a = self.generator(a, b, theta)

        return xr, xi, a

    def forward(self, x, I_prime, theta, *args, **kwargs):
        xr, xi, a = self.generate(x, I_prime, theta)
        real_score, fake_score = self.critic(xr, xi, a)
        critic_loss = torch.mean(real_score) - torch.mean(fake_score)
        generator_loss = -torch.mean(real_score) + torch.mean(fake_score)

        for p in self.critic.parameters():
            p.data.clamp_(-0.01, 0.01)

        return xr, xi, critic_loss, generator_loss
