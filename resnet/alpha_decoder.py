import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet.resnet_blocks import BasicBlock, SpecialBlock, make_layers

# class AlphaDecoder(Decoder):
#     def __init__(self, blocks, classes):
#         super().__init__()
#         self.special_layer = make_layers(SpecialBlock, 64, 64, 1, stride=1)
#         self.layer3 = make_layers(BasicBlock, 64, 64, blocks-1, stride=1)
#         self.layers = nn.Sequential(self.special_layer, self.layer3)
#         self.linear = nn.Linear(64, classes)


class AlphaDecoder(nn.Module):
    def __init__(self, blocks, classes):
        super().__init__()
        self.special_layer = make_layers(SpecialBlock, 64, 64, 1, stride=1)
        self.layer3 = make_layers(BasicBlock, 64, 64, blocks-1, stride=1)
        self.linear = nn.Linear(64, classes)

    def forward(self, xr, xi, thetas):
        thetas = thetas.view([thetas.shape[0]] + (len(xr.shape)-1) * [1]) 
        x_orig_r = torch.cos(-thetas)*xr - torch.sin(-thetas)*xi
        out = self.special_layer(x_orig_r)
        out = self.layer3(out)
        out = F.avg_pool2d(out, (out.size()[2], out.size()[3]))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        output = F.log_softmax(out, dim=1)

        return output
