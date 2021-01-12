import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = self.fc3(torch.flatten(x, start_dim=1))
        return x
