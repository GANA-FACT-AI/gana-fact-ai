import model.decoder
import torch
import torch.nn as nn


class Decoder(model.decoder.Decoder):
    def __init__(self):
        super().__init__(make_layers_=False)
        self.linear = nn.LogSoftmax(dim=1)

    def forward(self, xr, xi, thetas, thetas_add):
        thetas = thetas.view([thetas.shape[0]] + (len(xr.shape)-1) * [1])
        x_orig_r = torch.cos(-thetas)*xr - torch.sin(-thetas)*xi
        output = self.linear(x_orig_r)

        return output
