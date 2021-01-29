from resnet.alpha_decoder import AlphaDecoder
from resnet.alpha_processing_unit import AlphaProcessingUnit
from resnet.beta_processing_unit import BetaProcessingUnit
from resnet.beta_decoder import BetaDecoder
from resnet.wgan import WGAN

import model.privacymodel


class ResNetPrivacyModel(model.privacymodel.PrivacyModel):
    def __init__(self, hyperparams):
        super().__init__(hyperparams)
        self.blocks = self.get_block_amount(hyperparams.model)
        self.wgan = WGAN(k=8, log_fn=self.log, blocks=self.blocks, random_swap=hyperparams.random_swap,
                         add_gen_conv=hyperparams.add_gen_conv)
        if 'a' in hyperparams.model:
            self.processing_unit = AlphaProcessingUnit(self.blocks)
            self.decoder = AlphaDecoder(self.blocks)
        elif 'b' in hyperparams.model:
            self.processing_unit = BetaProcessingUnit(self.blocks)
            self.decoder = BetaDecoder(self.blocks)

    @staticmethod
    def get_block_amount(model):
        if '20' in model:
            blocks = 3
        elif '32' in model:
            blocks = 5
        elif '44' in model:
            blocks = 7
        elif '56' in model:
            blocks = 9
        else:
            blocks = 18
        print("Initializing {} with {} blocks".format(model, blocks))
        return blocks
