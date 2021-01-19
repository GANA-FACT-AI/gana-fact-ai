import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.layers = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 40),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, xr, xi, thetas):
        thetas = thetas.view([thetas.shape[0]] + (len(xr.shape)-1) * [1]) 
        x_orig_r = torch.cos(-thetas)*xr - torch.sin(-thetas)*xi
        x = self.avgpool(x_orig_r)
        x = torch.flatten(x, 1)
        output = self.layers(x)

        return output
