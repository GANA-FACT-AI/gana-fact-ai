import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc3 = nn.Linear(200, 10)

    def forward(self, xr, xi, thetas):
        thetas = thetas.view([thetas.shape[0]] + (len(xr.shape)-1) * [1]) 
        x_orig_r = torch.cos(-thetas)*xr - torch.sin(-thetas)*xi
        x = self.fc3(torch.flatten(x_orig_r, start_dim=1))
        output = F.log_softmax(x, dim=1)

        return output
