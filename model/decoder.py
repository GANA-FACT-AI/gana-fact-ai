import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, xr, xi, thetas):
        thetas = thetas.view([thetas.shape[0]] + (len(xr.shape)-1) * [1]) 
        x_orig_r = torch.cos(-thetas)*xr - torch.sin(-thetas)*xi
        output = self.softmax(x_orig_r)

        return output
