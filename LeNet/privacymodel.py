import pytorch_lightning as pl
import torch
from LeNet.decoder import Decoder
from LeNet.processing_unit import ProcessingUnit
from LeNet.wgan import WGAN
import model.privacymodel


class LeNetPrivacyModel(model.privacymodel.PrivacyModel):
    def __init__(self, hyperparams):
        super().__init__(hyperparams)

        self.wgan = WGAN(k=8)
        self.decoder = Decoder()
        self.processing_unit = ProcessingUnit()
        self.random_batch = None
        self.accuracy = pl.metrics.Accuracy()
        self.test_accuracy = 0
        self.test_counter = 0

    def configure_optimizers(self):
        optimizer_gen = torch.optim.RMSprop(list(self.wgan.generator.parameters()) + list(self.processing_unit.parameters()) + list(self.decoder.parameters()), lr=0.00005)
        optimizer_crit = torch.optim.RMSprop(self.wgan.critic.parameters(), lr=0.00005)
        return (
            {'optimizer': optimizer_gen, 'frequency': 1},
            {'optimizer': optimizer_crit, 'frequency': 5}
        )
