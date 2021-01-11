import torch
import numpy as np
import math
import torch.nn as nn


class Critique(nn.Module):
    def __init__(self, size, k):
        super().__init__()
        #self.batch_size = size[0]
        #self.img_size = size[1] * size[2] * size[3]
        self.score = nn.Linear(size, 1)
        self.k = k

    def forward(self, xr, xi, a):
        batch_size = xr.shape[0]
        img_size = xr.shape[1]*xr.shape[2]*xr.shape[3]
        xr = xr.reshape((batch_size, img_size))
        xi = xi.reshape((batch_size, img_size))
        a = a.reshape((batch_size, img_size))

        real_score = self.score(a).squeeze()
        fake_scores = []
        for i in range(self.k-1):
            theta = torch.from_numpy(np.array(np.random.uniform(0, 2*math.pi)))
            a_prime = torch.cos(-theta)*xr - torch.sin(-theta)*xi
            fake_scores.append(self.score(a_prime))
        fake_scores = torch.stack(fake_scores, 0).squeeze()
        return real_score, torch.mean(fake_scores, dim=0)