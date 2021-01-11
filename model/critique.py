import torch
import torch.nn as nn


class Critique(nn.Module):
    def __init__(self, size):
        super().__init__()
        #self.batch_size = size[0]
        #self.img_size = size[1] * size[2] * size[3]
        self.score = nn.Linear(size, size)

    def forward(self, a):
        a = a.reshape((a.shape[0], a.shape[1]*a.shape[2]*a.shape[3]))
        score = self.score(a)
        score = torch.mean(a, dim=1)
        return score