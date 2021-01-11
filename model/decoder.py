import torch.nn as nn
from complexLayers import ComplexLinear

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = self.fc3(x)
        return x
