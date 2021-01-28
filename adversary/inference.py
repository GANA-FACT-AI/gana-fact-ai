import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision
from torch.optim import Adam

# from adversary.unet import UNet
from resnet import ResNet, BasicBlock
from alexnet import AlexNet
import torch.nn.functional as F

class Inference(pl.LightningModule):
    def __init__(self, dataset, inversion=None):
        super(Inference, self).__init__()
        self.random_batch = None
        self.inversion = inversion
        self.accuracy = pl.metrics.Accuracy()

        if dataset == 'cifar100':
            self.model = ResNet(BasicBlock, [9, 9, 9], num_classes=100)
        else:   # dataset == 'celeba'
            self.model = AlexNet(num_classes=40)

    def forward(self, batch):
        x, target = batch
        I_prime = x if self.random_batch is None else self.random_batch
        self.random_batch = x

        rec_img = self.inversion.gen_imgs(x, I_prime)
        out = self.model(rec_img)
        pred = F.log_softmax(out, dim=1)
        rec_error = 1.0 - self.accuracy(pred, target)

        return rec_error

    def training_step(self, batch, batch_idx):
        x, target = batch
        out = self.model(x)   
        pred = F.log_softmax(out, dim=1)

        loss = F.nll_loss(pred, target)
        accuracy = self.accuracy(pred, target)

        if batch_idx % 50 == 0:
            print(" Accuracy: {}".format(accuracy))
        return loss

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=1e-2)  # given in the UNet-paper

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
