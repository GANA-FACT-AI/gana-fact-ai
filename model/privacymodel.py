import torch
import math
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F

from model.decoder import Decoder
from model.encoder import Encoder
from model.processing_unit import ProcessingUnit
from model.wgan import WGAN


class PrivacyModel(pl.LightningModule):
    def __init__(self, train_iter, *args):
        super().__init__()
        #self.save_hyperparameters()
        self.wgan = WGAN(k=8)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.processing_unit = ProcessingUnit()
        self.train_iter = train_iter

    def forward(self, batch):
        x = batch[0]  # TODO: use whole batch
        I_prime,_ = self.train_iter.next()

        # Encoder
        a = self.encoder(x)
        b = self.encoder(I_prime)  # TODO: maybe feed also through GAN
        theta = torch.from_numpy(np.array(np.random.uniform(0, 2*math.pi)))  # TODO: Why use numpy? Use Torch instead!

        # GAN
        xr, xi = self.wgan(a, b, theta)

        # Processing Unit
        xr, xi = self.processing_unit(xr, xi)

        # Decoder
        x_orig_r = torch.cos(-theta)*xr - torch.sin(-theta)*xi  # TODO: Move this code into the decoder
        x = self.decoder(x_orig_r)
        output = F.log_softmax(x, dim=1)

        return output

    def training_step(self, batch, batch_idx, optimizer_idx, *args, **kwargs):
        x = batch[0]  # TODO: Use full batch
        target = batch[1]  # TODO: Really?
        I_prime, _ = self.train_iter.next()

        # Encoder
        a = self.encoder(x)
        b = self.encoder(I_prime)
        theta = torch.from_numpy(np.array(np.random.uniform(0, 2*math.pi)))  # TODO: remove Numpy

        # GAN
        xr, xi, crit_loss, gen_loss = self.wgan.training_step(a, b, theta)

        if optimizer_idx == 0:
            # Processing Unit
            xr, xi = self.processing_unit(xr, xi)

            # Decoder
            x_orig_r = torch.cos(-theta)*xr - torch.sin(-theta)*xi  # TODO: Move to decoder
            x = self.decoder(x_orig_r)
            output = F.log_softmax(x, dim=1)

            # Loss
            loss = F.nll_loss(output, target)
            total_loss = loss + gen_loss  # TODO: Do we really want to train both losses with the same optimizer? Also, is gen and crit loss with the right sign?
            print("Total Loss: ", total_loss)  # TODO: move to logger
            return total_loss
        elif optimizer_idx == 1:
            print("Critique Loss: ", crit_loss)  # TODO: move to logger
            return crit_loss

    def configure_optimizers(self):
        optimizer_gen = torch.optim.RMSprop(list(self.encoder.parameters()) + list(self.wgan.generator.parameters()) + list(self.processing_unit.parameters()) + list(self.decoder.parameters()), lr=0.00005)
        optimizer_crit = torch.optim.RMSprop(self.wgan.critique.parameters(), lr=0.00005)
        return [optimizer_gen, optimizer_crit]

    def validation_step(self):
        pass

    def test_step(self):
        pass
