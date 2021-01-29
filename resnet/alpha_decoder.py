import torch.nn as nn
from resnet.resnet_blocks import BasicBlock, SpecialBlock, make_layers
import model.decoder


class AlphaDecoder(model.decoder.Decoder):
    def __init__(self, blocks):
        super().__init__()
        special_layer = make_layers(SpecialBlock, 64, 64, 1, stride=1)
        layer = make_layers(BasicBlock, 64, 64, blocks - 1, stride=1)
        self.layers = nn.Sequential(special_layer, layer)
        self.linear = nn.Linear(64, 10)
