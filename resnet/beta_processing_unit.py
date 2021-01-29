from resnet.resnet_blocks import ComplexBlock, make_layers
import model.processing_unit


class BetaProcessingUnit(model.processing_unit.ProcessingUnit):
    def __init__(self, blocks):
        super().__init__(blocks=blocks, init_layers=False)
        self.layers = []
