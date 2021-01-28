import torch.nn as nn

from resnet import make_layers, LayerNormBlock
import model.critic


class Critic(model.critic.Critic):
    def __init__(self, k):
        super().__init__(k)
        layers = list()
        layers.append(nn.Conv2d(16, 32, 3))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.MaxPool2d(2))

        layers.append(make_layers(LayerNormBlock, 32, 64, input_size=[1, 64, 8, 8], num_blocks=1, stride=2))

        layers.append(nn.Conv2d(64, 128, 3))
        layers.append(nn.ReLU())

        layers.append(nn.AvgPool2d(6))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(128, 1))
        self.score = nn.Sequential(*layers)
