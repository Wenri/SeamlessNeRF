from models.colorRF import ColorVMSplit, PoissonMLPRender
from models.loss import PLTLoss
from models.renderBase import PLTRender, MultiplePLTRender, SHRender, RGBRender, MLPRender_Fea, MLPRender_PE, MLPRender
from models.tensoRF import TensorVM, TensorCP, TensorVMSplit


class ClassCollection(dict):
    def __init__(self, *classes):
        super().__init__((cls.__name__, cls) for cls in classes)
        self.aliases = {alias: cls for cls in classes for alias in getattr(cls, '_aliases', ())}

    def __contains__(self, item):
        return item in self.keys() or item in self.values()

    def get(self, name):
        return super().get(name, self.aliases.get(name, name))


MODEL_ZOO = ClassCollection(TensorVM, TensorCP, TensorVMSplit, ColorVMSplit)
LOSS_ZOO = ClassCollection(PLTLoss)
RENDER_ZOO = ClassCollection(MLPRender_PE, MLPRender_Fea, MLPRender, SHRender, RGBRender, PLTRender, MultiplePLTRender,
                             PoissonMLPRender)
