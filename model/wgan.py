import pytorch_lightning as pl


class WGAN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        pass

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
