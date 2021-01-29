from resnet.alpha_processing_unit import AlphaProcessingUnit
from resnet.beta_processing_unit import BetaProcessingUnit
from resnet.alpha_decoder import AlphaDecoder
from resnet.beta_decoder import BetaDecoder
from resnet.wgan import WGAN

from model.privacy_model import PrivacyModel


class ResNetPrivacyModel(PrivacyModel):
    def __init__(self, hyperparams):
        super().__init__(hyperparams)
        self.classes = self.get_class_amount(hyperparams.dataset)
        self.blocks = self.get_block_amount(hyperparams.model)
        # self.wgan = WGAN(k=8, log_fn=self.log, blocks=self.blocks, random_swap=hyperparams.random_swap)
        self.wgan = WGAN(k=8, log_fn=self.log, blocks=self.blocks)
        if 'a' in hyperparams.model:
            self.processing_unit = AlphaProcessingUnit(self.blocks)
            self.decoder = AlphaDecoder(self.blocks, self.classes)
        elif 'b' in hyperparams.model:
            self.processing_unit = BetaProcessingUnit(self.blocks)
            self.decoder = BetaDecoder(self.blocks, self.classes)

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

    @staticmethod
    def get_class_amount(dataset):
        return int(dataset.replace('cifar', ''))