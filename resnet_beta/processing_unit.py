from resnet import ComplexBlock, make_layers
import model.processing_unit


class ProcessingUnit(model.processing_unit.ProcessingUnit):
    def __init__(self, blocks):
        super().__init__()
        self.layers = make_layers(ComplexBlock, 16, 32, blocks, stride=2)
