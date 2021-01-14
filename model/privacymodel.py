import torch
import math
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F

from model.decoder import Decoder
from model.processing_unit import ProcessingUnit
from model.wgan import WGAN


class PrivacyModel(pl.LightningModule):
    def __init__(self, train_loader, *args):
        super().__init__()
        self.wgan = WGAN(k=8)
        self.decoder = Decoder()
        self.processing_unit = ProcessingUnit()
        self.random_batch = None
        self.accuracy = pl.metrics.Accuracy()

    def forward(self, batch):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        x, _ = batch
        I_prime = x if self.random_batch is None else self.random_batch  # TODO: set random batches accordingly for inference

        thetas = torch.rand(x.shape[0]).to(device) * 2 * math.pi
        thetas = thetas.view([thetas.shape[0]] + (len(x.shape)-1) * [1])

        # Encoder/GAN
        xr, xi = self.wgan.generate(x, I_prime, thetas)

        # Processing Unit
        xr, xi = self.processing_unit(xr, xi)

        # Decoder
        output = self.decoder(xr, xi, thetas)

        return output

    def training_step(self, batch, batch_idx, optimizer_idx, *args, **kwargs):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        x, target = batch
        I_prime = x if self.random_batch is None else self.random_batch
        self.random_batch = x

        thetas = torch.rand(x.shape[0]).to(device) * 2 * math.pi
        thetas = thetas.view([thetas.shape[0]] + (len(x.shape)-1) * [1])

        # Encoder/GAN
        xr, xi, crit_loss, gen_loss = self.wgan.forward(x, I_prime, thetas)

        if optimizer_idx == 0:
            # Processing Unit
            xr, xi = self.processing_unit(xr, xi)

            # Decoder
            output = self.decoder(xr, xi, thetas)

            # Loss
            loss = F.nll_loss(output, target)
            total_loss = loss + gen_loss
            self.logger.experiment.add_scalar("generator_loss", gen_loss)
            self.logger.experiment.add_scalar("classifier_loss", loss)
            self.logger.experiment.add_scalar("total_loss", total_loss)
            self.logger.experiment.add_scalar("critic_loss", crit_loss)
            self.logger.experiment.add_scalar("accuracy", self.accuracy(output, target))
            if batch_idx % 50 == 0:
                print("Generator Loss: ", gen_loss)
                print("Total Loss: ", total_loss)  # TODO: move to logger
                print('Train Acc', self.accuracy(output, target))
            return total_loss
        else:
            if batch_idx % 50 == 0:
                print("Critic Loss: ", crit_loss)  # TODO: move to logger
            return crit_loss

    def configure_optimizers(self):
        optimizer_gen = torch.optim.RMSprop(list(self.wgan.generator.parameters()) + list(self.processing_unit.parameters()) + list(self.decoder.parameters()), lr=0.00005)
        optimizer_crit = torch.optim.RMSprop(self.wgan.critic.parameters(), lr=0.00005)
        return (
            {'optimizer': optimizer_gen, 'frequency': 1},
            {'optimizer': optimizer_crit, 'frequency': 5}
        )

    def validation_step(self):
        pass

    def test_step(self):
        pass
