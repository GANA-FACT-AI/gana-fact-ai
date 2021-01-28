from resnet.resnet_blocks import ComplexBlock, make_layers
import model.processing_unit


class BetaProcessingUnit(model.processing_unit.ProcessingUnit):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = make_layers(ComplexBlock, 16, 32, blocks, stride=2)
        self.layers = []
