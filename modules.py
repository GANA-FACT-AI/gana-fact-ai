import torch
import torch.nn as nn
import numpy as np

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.score = nn.Linear(2880, 2880) # TODO: Change from hardcoded to adaptable to encoder output

    def forward(self, a):
        a = a.reshape((a.shape[0], a.shape[1]*a.shape[2]*a.shape[3]))
        score = self.score(a)
        score = torch.mean(a, dim=1)
        return score