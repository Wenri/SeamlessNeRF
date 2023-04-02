from models.colorRF import ColorVMSplit
from models.loss import PLTLoss
from models.renderBase import PLTRender, MultiplePLTRender
from models.tensoRF import TensorVM, TensorCP, TensorVMSplit
from models.tensorBase import MLPRender, MLPRender_PE, MLPRender_Fea, SHRender, RGBRender


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
RENDER_ZOO = ClassCollection(MLPRender_PE, MLPRender_Fea, MLPRender, SHRender, RGBRender, PLTRender, MultiplePLTRender)
