import torch.nn as nn
import torch
from model.critic import Critic
from model.generator import Generator


class WGAN(nn.Module):
    def __init__(self, k, log_fn, random_swap):
        super().__init__()
        self.generator = Generator(random_swap)
        self.critic = Critic(k)
        self.log = log_fn

    def forward(self, x, I_prime, theta):
        xr, xi, a, theta_add = self.generator(x, I_prime, theta)
        real_score, fake_score = self.critic(xr, xi, a)
        critic_loss = -torch.mean(real_score) + torch.mean(fake_score)
        generator_loss = torch.mean(real_score) - torch.mean(fake_score)

        self.log("real_score_mean", torch.mean(real_score))
        self.log("fake_score_mean", torch.mean(fake_score))
        self.log("real_score_var", torch.std(real_score))
        self.log("fake_score_var", torch.std(fake_score))
        return xr, xi, critic_loss, generator_loss, theta_add
