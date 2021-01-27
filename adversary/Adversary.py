import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision
from torch.optim import Adam

from adversary.Unet import UNet


class Adversary(pl.LightningModule):
    def __init__(self, privacy_model):
        super(Adversary, self).__init__()
        self.unet = UNet(32, 3)  # TODO: right number of input channels
        self.privacy_model = privacy_model
        self.loss = nn.MSELoss()  # TODO: is this the right loss function? In the paper, they used a different one, but they did segmentation and we image generation
        self.random_batch = None
        self.gen_img_number = 0

    def forward(self, xr, xi):
        concats = torch.cat((xr, xi), 1)
        return torch.tanh(self.unet(concats))

    def gen_imgs(self, x, I_prime):
        with torch.no_grad():
            xr, xi, a, _ = self.privacy_model.wgan.generator(x, I_prime, self.privacy_model.thetas(x))
            return self(xr, xi)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        I_prime = x if self.random_batch is None else self.random_batch
        self.random_batch = x
        thetas = self.privacy_model.thetas(x)

        with torch.no_grad():
            xr, xi, a, theta_add = self.privacy_model.wgan.generator(x, I_prime, thetas)
        output = self(xr, xi)
        loss = self.loss(output, x)

        self.log("reconstruction error", self.reconstruction_error(output, x))
        self.log("image_reconstruction_loss", loss)

        return loss

    def configure_optimizers(self):
        return Adam(self.unet.parameters(), lr=1e-4)  # given in the UNet-paper

    def plot_images(self, x, I_prime):
        imgs = self.gen_imgs(x, I_prime)
        img_grid_orig = torchvision.utils.make_grid(x/2 + 0.5)
        img_grid_gen = torchvision.utils.make_grid(imgs/2 + 0.5)
        self.logger.experiment.add_image('original images', img_grid_orig, self.gen_img_number)
        self.logger.experiment.add_image('reconstructed images', img_grid_gen, self.gen_img_number)
        self.gen_img_number += 1

    @staticmethod
    def reconstruction_error(a, b):
        """
        Returns the official evaluation metric from the paper. Scales pixels between [0, 1]
        and calculates E(I-Î)
        """
        a = a / 2 + 0.5
        b = b / 2 + 0.5
        return torch.mean(torch.abs(a-b))

    def test_step(self, batch, batch_idx):
        x, _ = batch
        I_prime = x if self.random_batch is None else self.random_batch
        self.random_batch = x
        self.plot_images(x, I_prime)
        thetas = self.privacy_model.thetas(x)

        with torch.no_grad():
            xr, xi, a, _ = self.privacy_model.wgan.generator(x, I_prime, thetas)

            output = self(xr, xi)

        return self.reconstruction_error(output, x)

    def test_epoch_end(self, outs):
        mean_reconstruction_error = sum(outs) / len(outs)
        self.log("test_reconstruction_error", mean_reconstruction_error)
