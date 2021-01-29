import torch.nn as nn
import torch
from resnet.critic import Critic
from resnet.generator import Generator


class WGAN(nn.Module):
    def __init__(self, k, log_fn, blocks):
        super().__init__()
        self.generator = Generator(blocks)
        self.critic = Critic(k)
        self.log = log_fn

    def forward(self, x, I_prime, theta):
        xr, xi, a = self.generator(x, I_prime, theta)
        real_score, fake_score = self.critic(xr, xi, a)
        critic_loss = torch.mean(real_score) - torch.mean(fake_score)
        generator_loss = -torch.mean(real_score) + torch.mean(fake_score)

        self.log("real_score_mean", torch.mean(real_score))
        self.log("fake_score_mean", torch.mean(fake_score))
        self.log("real_score_var", torch.var(real_score))
        self.log("fake_score_var", torch.var(fake_score))
        return xr, xi, critic_loss, generator_loss
