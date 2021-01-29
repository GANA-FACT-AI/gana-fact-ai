import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet.resnet_blocks import BasicBlock, make_layers


class Decoder(nn.Module):
    def __init__(self, blocks=3, make_layers_=True):
        super().__init__()
        self.layers = make_layers(BasicBlock, 32, 64, blocks, stride=2) if make_layers_ else None
        self.linear = nn.Linear(64, 10)

    def forward(self, xr, xi, thetas, theta_add):
        thetas = thetas.view([thetas.shape[0]] + (len(xr.shape)-1) * [1]) 
        x_orig_r = torch.cos(-thetas + theta_add)*xr - torch.sin(-thetas + theta_add)*xi
        out = self.layers(x_orig_r) if self.layers else x_orig_r
        out = F.avg_pool2d(out, (out.size()[2], out.size()[3]))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        output = F.log_softmax(out, dim=1)

        return output
