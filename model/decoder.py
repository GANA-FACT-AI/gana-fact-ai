import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet.resnet_blocks import BasicBlock, SpecialBlock, make_layers


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xr, xi, thetas):
        thetas = thetas.view([thetas.shape[0]] + (len(xr.shape)-1) * [1]) 
        x_orig_r = torch.cos(-thetas)*xr - torch.sin(-thetas)*xi
        out = self.layers(x_orig_r)
        out = F.avg_pool2d(out, (out.size()[2], out.size()[3]))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        output = F.log_softmax(out, dim=1)

        return output
