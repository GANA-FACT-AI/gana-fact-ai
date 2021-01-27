from typing import List, Any

import torch
import math
import pytorch_lightning as pl
import torch.nn.functional as F

from torch.nn.modules.module import ModuleAttributeError

from resnet_beta.decoder import Decoder
from resnet_beta.processing_unit import ProcessingUnit
from resnet_beta.wgan import WGAN


class PrivacyModel(pl.LightningModule):
    def __init__(self, hyperparams):
        super().__init__()
        self.blocks = self.get_block_amount(hyperparams.model)
        self.wgan = WGAN(k=8, log_fn=self.log, blocks=self.blocks)
        self.processing_unit = ProcessingUnit(self.blocks)
        self.decoder = Decoder(self.blocks)

        self.random_batch = None
        self.accuracy = pl.metrics.Accuracy()
        self.lr_gen = hyperparams.lr_gen
        self.lr_crit = hyperparams.lr_crit
        self.lr_model = hyperparams.lr_model
        self.plot_graph = hyperparams.plot_graph

        # needed to fix logging bug
        self.loss = None
        self.last_accuracy = None
        self.log_grads = True
        self.log_critic_gradients = False
        self.crit_loss = None
        self.gen_loss = None

    @staticmethod
    def thetas(x):
        thetas = torch.rand(x.shape[0]).to(x.device) * 2 * math.pi
        return thetas.view([thetas.shape[0]] + (len(x.shape)-1) * [1])

    @staticmethod
    def get_block_amount(model):
        if model == 'resnet20b':
            blocks = 3
        elif model == 'resnet32b':
            blocks = 5
        elif model == 'resnet44b':
            blocks = 7
        elif model == 'resnet56b':
            blocks = 9
        else:
            blocks = 18
        print("Initializing {} with {} blocks".format(model, blocks))
        return blocks

    def forward(self, x):
        I_prime = x if self.random_batch is None else self.random_batch  # TODO: set random batches accordingly for inference

        thetas = PrivacyModel.thetas(x)

        # Encoder/GAN
        xr, xi, _ = self.wgan.generator(x, I_prime, thetas)

        # Processing Unit
        xr, xi = self.processing_unit(xr, xi)

        # Decoder
        output = self.decoder(xr, xi, thetas)

        return output

    def training_step(self, batch, batch_idx, optimizer_idx, *args, **kwargs):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        try:
            self.gradient_penalty
        except ModuleAttributeError:
            self.gradient_penalty = torch.tensor(0).to(device)
        if self.plot_graph:
            self.logger.experiment.add_graph(self, torch.rand(1, 3, 32, 32).to(device))
            self.plot_graph = False

        x, target = batch
        I_prime = x if self.random_batch is None else self.random_batch
        self.random_batch = x

        thetas = PrivacyModel.thetas(x)

        # Encoder/GAN
        xr, xi, self.crit_loss, self.gen_loss = self.wgan(x, I_prime, thetas)

        if optimizer_idx == 0:
            # Processing Unit
            xr, xi = self.processing_unit(xr, xi)

            # Decoder
            output = self.decoder(xr, xi, thetas)

            # Loss
            self.loss = F.nll_loss(output, target)
            self.last_accuracy = self.accuracy(output, target)

            self.log_values()
            return self.loss
        elif optimizer_idx == 1:
            self.log_values()
            return self.gen_loss
        else:
            self.log_critic_gradients = True

            a = self.wgan.generator.encode(x)
            self.gradient_penalty = self.wgan.critic.compute_gradient_penalty(xr, xi, a)
            self.log_values()
            return self.crit_loss + 10 * self.gradient_penalty

    def configure_optimizers(self):
        optimizer_all = torch.optim.Adam(list(self.wgan.generator.parameters()) + list(self.processing_unit.parameters())
                                         + list(self.decoder.parameters()), lr=self.lr_model)
        optimizer_generator = torch.optim.Adam(self.wgan.generator.parameters(), lr=self.lr_gen)
        optimizer_crit = torch.optim.Adam(self.wgan.critic.parameters(), lr=self.lr_crit)
        return (
            {'optimizer': optimizer_all, 'frequency': 1},
            {'optimizer': optimizer_generator, 'frequency': 1},
            {'optimizer': optimizer_crit, 'frequency': 5}
        )

    def log_values(self):
        self.log("generator_loss", self.gen_loss)
        self.log("critic_loss", self.crit_loss)
        self.log("classifier_loss", self.loss)
        self.log("accuracy", self.last_accuracy)
        self.log("gradient_penalty", self.gradient_penalty)

    def training_epoch_end(self, outputs: List[Any]) -> None:
        self.log_grads = True
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def on_after_backward(self) -> None:
        if self.log_critic_gradients:
            for name, params in self.wgan.critic.named_parameters():
                if params.grad is not None:
                    self.logger.experiment.add_scalar("critic_grad_norm", torch.norm(params.grad)) # TODO: use self.log
            self.log_critic_gradients = False
        else:
            for name, params in self.named_parameters():
                if params.grad is not None:
                    self.logger.experiment.add_scalar("grad_norm", torch.norm(params.grad))

        if self.log_grads:
            for name, params in self.named_parameters():
                if params.grad is not None:
                    self.logger.experiment.add_histogram(name + " Gradients", params.grad, self.current_epoch)
            self.log_grads = False
