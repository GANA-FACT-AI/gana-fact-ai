from typing import List, Any

import torch
import math
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F

from model.decoder import Decoder
from model.processing_unit import ProcessingUnit
from model.wgan import WGAN


class PrivacyModel(pl.LightningModule):
    def __init__(self, *args):
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
        xr, xi, _ = self.wgan.generate(x, I_prime, thetas)

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

        self.log("Critic Loss: ", crit_loss)
        self.log("Gen Loss: ", gen_loss)

        if optimizer_idx == 0:
            # Processing Unit
            xr, xi = self.processing_unit(xr, xi)

            # Decoder
            output = self.decoder(xr, xi, thetas)

            # Loss
            loss = F.nll_loss(output, target)
            total_loss = loss + gen_loss

            accuracy = self.accuracy(output, target)

            self.logger.experiment.add_scalar("Accuracy", accuracy)
            self.logger.experiment.add_scalar("Regular Loss", loss)
            
            if batch_idx % 50 == 0:
                print("Generator Loss: ", gen_loss)
                print("Total Loss: ", total_loss)  # TODO: move to logger
                print('Train Acc', accuracy)
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

    '''
    def training_epoch_end(self, outputs: List[Any]) -> None:
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)
    '''
    def on_after_backward(self):
        # example to inspect gradient information in tensorboard
        if self.trainer.global_step % 389 == 0:  # don't make the tf file huge
            params = self.state_dict()
            for k, v in params.items():
                grads = v
                name = k
                self.logger.experiment.add_histogram(tag=name, values=grads, global_step=self.trainer.global_step)
                #self.logger.experiment.add_histogram(tag="{}_gradients".format(name), values=grads.grad, global_step=self.trainer.global_step)
            #self.logger.experiment.add_histogram(tag="conv1_weight_grads", values=self.wgan.generator.conv1.weight.grad, global_step=self.trainer.global_step)


    def validation_step(self):
        pass

    def test_step(self):
        pass
