import pytorch_lightning as pl

from model.critique import Critique
from model.generator import Generator


class WGAN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.generator = Generator()
        self.critique = Critique()

    def forward(self):
        pass

    def training_step(self, *args, **kwargs):
        generated_features = self.generator()
        realness = self.critique()
        return generated_features, realness

    def configure_optimizers(self):
        pass

    def validation_step(self):
        pass

    def test_step(self):
        pass
