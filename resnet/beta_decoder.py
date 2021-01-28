import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet.resnet_blocks import BasicBlock, make_layers


# class BetaDecoder(Decoder):
#     def __init__(self, blocks, classes):
#         super().__init__()
#         self.layers = make_layers(BasicBlock, 32, 64, blocks, stride=2)
#         self.linear = nn.Linear(64, classes)

class BetaDecoder(nn.Module):
    def __init__(self, blocks, classes):
        super().__init__()
        self.layer3 = make_layers(BasicBlock, 32, 64, blocks, stride=2)
        self.linear = nn.Linear(64, classes)
        
    def forward(self, xr, xi, thetas):
        thetas = thetas.view([thetas.shape[0]] + (len(xr.shape)-1) * [1]) 
        x_orig_r = torch.cos(-thetas)*xr - torch.sin(-thetas)*xi
        out = self.layer3(x_orig_r)
        out = F.avg_pool2d(out, (out.size()[2], out.size()[3]))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        output = F.log_softmax(out, dim=1)

        return output
