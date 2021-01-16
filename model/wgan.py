import torch.nn as nn
import torch
from model.critic import Critic
from model.generator import Generator


class WGAN(nn.Module):
    def __init__(self, k, log_fn):
        super().__init__()
        self.generator = Generator()  # TODO: remove magic numbers
        self.critic = Critic(16384, k)  # TODO: remove magic numbers
        self.log = log_fn

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

        self.log("real_score_mean", torch.mean(real_score))
        self.log("fake_score_mean", torch.mean(fake_score))
        self.log("real_score_var", torch.var(real_score))
        self.log("fake_score_var", torch.var(fake_score))
        return xr, xi, critic_loss, generator_loss
