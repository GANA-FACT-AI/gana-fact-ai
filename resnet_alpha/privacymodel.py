from resnet_alpha.decoder import Decoder
from resnet_alpha.processing_unit import ProcessingUnit
from resnet_alpha.wgan import WGAN

import model.privacymodel


class PrivacyModel(model.privacymodel.PrivacyModel):
    def __init__(self, hyperparams):
        super().__init__(hyperparams)
        self.blocks = self.get_block_amount(hyperparams.model)
        self.wgan = WGAN(k=8, log_fn=self.log, blocks=self.blocks)
        self.processing_unit = ProcessingUnit(self.blocks)
        self.decoder = Decoder(self.blocks)

    @staticmethod
    def get_block_amount(model):
        if model == 'resnet20a':
            blocks = 3
        elif model == 'resnet32a':
            blocks = 5
        elif model == 'resnet44a':
            blocks = 7
        elif model == 'resnet56a':
            blocks = 9
        else:
            blocks = 18
        print("Initializing {} with {} blocks".format(model, blocks))
        return blocks
