import torch.nn as nn
import model.generator


class Generator(model.generator.Generator):
    def __init__(self):
        super().__init__(False)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2)
        )
