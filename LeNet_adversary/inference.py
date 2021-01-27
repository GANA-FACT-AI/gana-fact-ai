import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision
from torch.optim import Adam

# from adversary.unet import UNet
from resnet import ResNet, BasicBlock
import torch.nn.functional as F

class Inference(pl.LightningModule):
    def __init__(self, inversion=None):
        super(Inference, self).__init__()
        self.random_batch = None
        self.resnet56 = ResNet(BasicBlock, [9, 9, 9], num_classes=100)
        self.inversion = inversion
        self.accuracy = pl.metrics.Accuracy()

    def forward(self, batch):
        x, _ = batch
        I_prime = x if self.random_batch is None else self.random_batch
        self.random_batch = x

        rec_img = self.inversion.gen_imgs(x, I_prime)


        return self.resnet56(rec_img)

    def training_step(self, batch, batch_idx):
        x, target = batch
        out = self.resnet56(x)              
        pred = F.log_softmax(out, dim=1)

        loss = F.nll_loss(pred, target)
        last_accuracy = self.accuracy(pred, target)

        if batch_idx % 50 == 0:
            print("Accuracy: {}".format(last_accuracy))
        return loss

    def configure_optimizers(self):
        return Adam(self.resnet56.parameters(), lr=1e-4)  # given in the UNet-paper

    # def validation_step_c(self, batch, batch_idx, I_prime):
    #     x, _ = batch
    #     imgs = self.gen_imgs(x, I_prime)
    #     img_grid_orig = torchvision.utils.make_grid(x/2 + 0.5)
    #     img_grid_gen = torchvision.utils.make_grid(imgs/2 + 0.5)
    #     self.logger.experiment.add_image('original images', img_grid_orig, batch_idx)
    #     self.logger.experiment.add_image('reconstructed images', img_grid_gen, batch_idx)

    # def validation_step(self, batch, batch_idx):
    #     x, _ = batch
    #     I_prime = x if self.random_batch is None else self.random_batch
    #     self.random_batch = x
    #     imgs = self.gen_imgs(x, I_prime)
    #     img_grid_orig = torchvision.utils.make_grid(x/2 + 0.5)
    #     img_grid_gen = torchvision.utils.make_grid(imgs/2 + 0.5)
    #     self.logger.experiment.add_image('original images {}'.format(self.img_counter), img_grid_orig, batch_idx)
    #     self.logger.experiment.add_image('reconstructed images {}'.format(self.img_counter), img_grid_gen, batch_idx)
    #     self.img_counter += 1
