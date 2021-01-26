import torch
import torch.nn as nn
from complexModules import _weights_init

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 10),
            nn.LogSoftmax(dim=1)
        )
        #self.apply(_weights_init)

    def forward(self, xr, xi, thetas):
        thetas = thetas.view([thetas.shape[0]] + (len(xr.shape)-1) * [1]) 
        x_orig_r = torch.cos(-thetas)*xr - torch.sin(-thetas)*xi
        x = torch.flatten(xr, 1)
        output = self.layers(x)

        return output
