import torch.nn as nn
from resnet import BasicBlock, SpecialBlock, make_layers
import model.decoder


class Decoder(model.decoder.Decoder):
    def __init__(self, blocks):
        super().__init__()
        self.special_layer = make_layers(SpecialBlock, 64, 64, 1, stride=1)
        self.layer = make_layers(BasicBlock, 64, 64, blocks - 1, stride=1)
        self.layers = nn.Sequential(self.special_layer, self.layer)
        self.linear = nn.Linear(64, 10)
