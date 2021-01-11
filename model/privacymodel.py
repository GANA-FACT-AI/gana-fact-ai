import pytorch_lightning as pl

from model.decoder import Decoder
from model.encoder import Encoder
from model.processing_unit import ProcessingUnit
from model.wgan import WGAN


class PrivacyModel(pl.LightningModule):
    def __init__(self, *args):
        super().__init__()
        self.save_hyperparameters()
        self.wgan = WGAN()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.processing_unit = ProcessingUnit()

    def forward(self):
        pass

    def training_step(self, x, *args, **kwargs):
        x = self.encoder(x)
        x, _ = self.wgan(x)
        x = self.processing_unit(x)
        x = self.decoder(x)
        return x

    def configure_optimizers(self):
        pass

    def validation_step(self):
        pass

    def test_step(self):
        pass
