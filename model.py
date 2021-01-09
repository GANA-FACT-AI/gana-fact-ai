import torch
import pytorch_lightning as pl


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass

    def training_step(self, *args, **kwargs):
        pass