import math
import torch
import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.output = nn.Linear(size, size)

    def forward(self, a, b, theta):
        #batch_size = a.shape[0]
        #img_size = a.shape[1]*a.shape[2]*a.shape[3]
        #a = a.reshape((batch_size, img_size))
        a = self.output(a)
        b = self.output(b)
        rotated_r = torch.cos(theta)*a - torch.sin(theta)*b
        rotated_i = torch.sin(theta)*a + torch.cos(theta)*b
        return rotated_r, rotated_i
