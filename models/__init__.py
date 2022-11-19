from models.colorRF import ColorVMSplit
from models.tensoRF import TensorVM, TensorCP, TensorVMSplit

MODEL_ZOO = {a.__name__: a for a in (TensorVM, TensorCP, TensorVMSplit, ColorVMSplit)}
