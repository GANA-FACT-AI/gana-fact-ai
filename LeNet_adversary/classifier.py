import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision
from torch.optim import Adam

import torch.nn.functional as F

from adversary.angle_net import AngleNet
from resnet import ResNet, BasicBlock


class Classifier(pl.LightningModule):
    def __init__(self):
        super(Classifier, self).__init__()
        self.random_batch = None
        self.resnet56 = ResNet(BasicBlock, [9, 9, 9])
        # self.softmax = nn.LogSoftmax()

    def forward(self, img):
        return self.resnet56(img)

    def training_step(self, batch, batch_idx):
        x, targets = batch
        pred = self.resnet56(x)

        loss = F.cross_entropy(pred, targets)
        return loss

    def configure_optimizers(self):
        return Adam(self.resnet56.parameters(), lr=1e-4)  # given in the UNet-paper