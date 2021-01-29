import pytorch_lightning as pl

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
