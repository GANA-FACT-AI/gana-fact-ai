import torch.nn as nn
import model.critic


class Critic(model.critic.Critic):
    def __init__(self,  k):
        super().__init__(k)
        layers = list()
        layers.append(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1))
        layers.append(nn.Tanh())
        layers.append(nn.AvgPool2d(kernel_size=2))

        layers.append(nn.Flatten())
        layers.append(nn.Linear(800, 1))
        self.score = nn.Sequential(*layers)
