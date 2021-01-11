import torch
import torch.nn as nn
import numpy as np

class Discriminator(nn.Module):

    def __init__(self, size):
        super(Discriminator, self).__init__()
        self.batch_size = size[0]
        self.img_size = size[1] * size[2] * size[3]
        self.score = nn.Linear(self.img_size, self.img_size)

    def forward(self, a):
        a = a.reshape((self.batch_size, self.img_size))
        score = self.score(a)
        score = torch.mean(a, dim=1)
        return score