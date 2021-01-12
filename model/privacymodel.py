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
    def __init__(self, train_loader, *args):
        super().__init__()
        self.wgan = WGAN(k=8)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.processing_unit = ProcessingUnit()
        self.train_loader = train_loader
        self.train_iter = iter(train_loader)
        self.accuracy = pl.metrics.Accuracy()

    def get_random_batch(self):
        try:
            x = next(self.train_iter)
        except Exception as e:
            self.train_iter = iter(self.train_loader)
            x = next(self.train_iter)
        return x

    def forward(self, batch):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        x = batch[0]  # TODO: use whole batch
        I_prime, _ = self.get_random_batch()

        # Encoder
        a = self.encoder(x)
        b = self.encoder(I_prime)  # TODO: maybe feed also through GAN
        thetas = torch.rand(x.shape[0]).to(device) * 2 * math.pi
        thetas = thetas.view([thetas.shape[0]] + (len(x.shape)-1) * [1])

        # GAN
        xr, xi = self.wgan.generate(a, b, thetas)

        # Processing Unit
        xr, xi = self.processing_unit(xr, xi)

        # Decoder
        x_orig_r = torch.cos(-thetas)*xr - torch.sin(-thetas)*xi  # TODO: Move this code into the decoder
        x = self.decoder(x_orig_r)
        output = F.log_softmax(x, dim=1)

        return output

    def training_step(self, batch, batch_idx, optimizer_idx, *args, **kwargs):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        x, target = batch
        I_prime, _ = self.get_random_batch()
        I_prime = I_prime.to(device)

        # Encoder
        a = self.encoder(x)
        b = self.encoder(I_prime)
        thetas = torch.tensor(x.shape[0] * [math.pi]) #torch.rand(x.shape[0]).to(device) * 2
        thetas = thetas.view([thetas.shape[0]] + (len(x.shape)-1) * [1])

        # GAN
        xr, xi, crit_loss, gen_loss = self.wgan.forward(a, b, thetas)


        if optimizer_idx == 0:
            # Processing Unit
            xr, xi = self.processing_unit(xr, xi)

            # Decoder
            thetas = thetas.view([thetas.shape[0]] + (len(xr.shape)-1) * [1]) 
            x_orig_r = torch.cos(-thetas)*xr - torch.sin(-thetas)*xi  # TODO: Move to decoder
            x = self.decoder(x_orig_r)
            output = F.log_softmax(x, dim=1)

            #print('train_acc', self.accuracy(output, target))

            # Loss
            loss = F.nll_loss(output, target)
            #print("Generator Loss: ", gen_loss)
            total_loss = loss + gen_loss  # TODO: Do we really want to train both losses with the same optimizer? Also, is gen and crit loss with the right sign?
            #print("Total Loss: ", total_loss)  # TODO: move to logger
            return total_loss
        else:
            #print("Critique Loss: ", crit_loss)  # TODO: move to logger
            return crit_loss

    def configure_optimizers(self):
        optimizer_gen = torch.optim.RMSprop(list(self.encoder.parameters()) + list(self.wgan.generator.parameters()) + list(self.processing_unit.parameters()) + list(self.decoder.parameters()), lr=0.00005)
        optimizer_crit = torch.optim.RMSprop(self.wgan.critique.parameters(), lr=0.00005)
        return (
            {'optimizer': optimizer_gen, 'frequency': 1},
            {'optimizer': optimizer_crit, 'frequency': 5}
        )

    def validation_step(self):
        pass

    def test_step(self):
        pass
