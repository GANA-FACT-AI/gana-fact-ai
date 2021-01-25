import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision
from torch.optim import Adam

from adversary.AngleNet import AngleNet


class AnglePred(pl.LightningModule):
    def __init__(self, privacy_model):
        super(AnglePred, self).__init__()
        self.angle_net = AngleNet()
        self.privacy_model = privacy_model
        self.random_batch = None

    def forward(self, xr, xi):
        angle = self.angle_net(torch.cat((xr, xi), 1))
        xr = torch.cos(-angle) * xr - torch.sin(-angle) * xi
        xi = torch.sin(-angle) * xr + torch.cos(-angle) * xi
        return xr, xi


    def training_step(self, batch, batch_idx):
        x, _ = batch
        I_prime = x if self.random_batch is None else self.random_batch
        self.random_batch = x
        thetas = self.privacy_model.thetas(x)

        with torch.no_grad():
            xr, xi, a = self.privacy_model.wgan.generator(x, I_prime, thetas)

        angle = self.angle_net(torch.cat((xr, xi), 1))

        loss = torch.mean(torch.abs(thetas - angle))

        self.log("angle loss", loss)
        #if batch_idx % 30 == 0:  # TODO: move this code entirely into validation_step
            #self.validation_step_c(batch, batch_idx, I_prime)

        return loss

    def configure_optimizers(self):
        return Adam(self.angle_net.parameters(), lr=1e-4)  # given in the UNet-paper
