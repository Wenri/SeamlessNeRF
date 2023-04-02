from .renderBase import PLTRender, MultiplePLTRender
from .tensoRF import TensorVMSplit


class ColorVMSplit(TensorVMSplit):
    def init_render_func(self, shadingMode, pos_pe=6, view_pe=6, fea_pe=6, featureC=128, palette=None,
                         **kwargs):
        match shadingMode.__name__:
            case PLTRender.__name__:
                return shadingMode(self.app_dim, view_pe, fea_pe, featureC, palette, **kwargs).to(self.device)
            case MultiplePLTRender.__name__:
                return shadingMode(self.app_dim, view_pe, fea_pe, featureC, *palette, **kwargs).to(self.device)
            case _:
                return super().init_render_func(shadingMode, pos_pe, view_pe, fea_pe, featureC, **kwargs)
