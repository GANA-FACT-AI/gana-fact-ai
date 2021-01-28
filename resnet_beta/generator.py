import torch.nn as nn
from resnet import BasicBlock, make_layers
import model.generator


class Generator(model.generator.Generator):
    def __init__(self, blocks):
        super().__init__()
        layers = list()
        layers.append(nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(16))
        layers.append(nn.ReLU())
        layers.append(make_layers(BasicBlock, 16, 16, blocks, stride=1))
        self.layers = nn.Sequential(*layers)
