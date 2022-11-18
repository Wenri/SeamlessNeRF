from .render import PLTRender
from .tensoRF import TensorVMSplit


class ColorVMSplit(TensorVMSplit):
    def init_render_func(self):
        match self.shadingMode:
            case 'PLT_Fea':
                return PLTRender(self.app_dim, self.view_pe, self.fea_pe, self.featureC).to(self.device)
            case _:
                return super(ColorVMSplit, self).init_render_func()
