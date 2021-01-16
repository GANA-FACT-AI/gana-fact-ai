import torch
import math
import pytorch_lightning as pl
import torch.nn.functional as F

from model.decoder import Decoder
from model.processing_unit import ProcessingUnit
from model.wgan import WGAN


class PrivacyModel(pl.LightningModule):
    def __init__(self, train_loader, *args):
        super().__init__()
        self.wgan = WGAN(k=8, log_fn=self.log)
        self.decoder = Decoder()
        self.processing_unit = ProcessingUnit()
        self.random_batch = None
        self.accuracy = pl.metrics.Accuracy()

        # needed to fix logging bug
        self.loss = None
        self.total_loss = None
        self.last_accuracy = None

    def forward(self, x):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        I_prime = x if self.random_batch is None else self.random_batch  # TODO: set random batches accordingly for inference

        thetas = torch.rand(x.shape[0]).to(device) * 2 * math.pi
        thetas = thetas.view([thetas.shape[0]] + (len(x.shape)-1) * [1])

        # Encoder/GAN
        xr, xi, _ = self.wgan.generate(x, I_prime, thetas)

        # Processing Unit
        xr, xi = self.processing_unit(xr, xi)

        # Decoder
        output = self.decoder(xr, xi, thetas)

        return output

    def training_step(self, batch, batch_idx, optimizer_idx, *args, **kwargs):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if self.current_epoch == 0 and batch_idx == 0:
            self.logger.experiment.add_graph(self, torch.rand(1, 3, 32, 32).to(device))
        x, target = batch
        I_prime = x if self.random_batch is None else self.random_batch
        self.random_batch = x

        thetas = torch.rand(x.shape[0]).to(device) * 2 * math.pi
        thetas = thetas.view([thetas.shape[0]] + (len(x.shape)-1) * [1])

        # Encoder/GAN
        xr, xi, crit_loss, gen_loss = self.wgan.forward(x, I_prime, thetas)

        self.log("generator_loss", gen_loss)
        self.log("critic_loss", crit_loss)

        if optimizer_idx == 0:
            # Processing Unit
            xr, xi = self.processing_unit(xr, xi)

            # Decoder
            output = self.decoder(xr, xi, thetas)

            # Loss
            self.loss = F.nll_loss(output, target)
            self.total_loss = self.loss + gen_loss
            self.last_accuracy = self.accuracy(output, target)

            self.log("classifier_loss", self.loss)
            self.log("total_loss", self.total_loss)
            self.log("accuracy", self.last_accuracy)
            return self.total_loss
        else:
            self.log("classifier_loss", self.loss)
            self.log("total_loss", self.total_loss)
            self.log("accuracy", self.last_accuracy)
            return crit_loss

    def configure_optimizers(self):
        optimizer_gen = torch.optim.RMSprop(list(self.wgan.generator.parameters()) + list(self.processing_unit.parameters()) + list(self.decoder.parameters()), lr=0.00005)
        optimizer_crit = torch.optim.RMSprop(self.wgan.critic.parameters(), lr=0.00005)
        return (
            {'optimizer': optimizer_gen, 'frequency': 1},
            {'optimizer': optimizer_crit, 'frequency': 5}
        )

    # def validation_step(self):
    #     pass

    # def test_step(self):
    #     pass
