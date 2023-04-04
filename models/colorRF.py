import torch.autograd.functional

from .renderBase import MLPRender_Fea
from .tensoRF import TensorVMSplit


class ColorVMSplit(TensorVMSplit):
    def init_render_func(self, shadingMode, pos_pe=6, view_pe=6, fea_pe=6, featureC=128, *args, **kwargs):
        return super().init_render_func(shadingMode, pos_pe, view_pe, fea_pe, featureC, **kwargs)

    def compute_radiance(self, pts, viewdirs):
        def batch_func(x):
            return super(ColorVMSplit, self).compute_radiance(x, viewdirs).sum(dim=0)

        j = torch.autograd.functional.jacobian(batch_func, pts, create_graph=True, strict=True)
        return j.sum(dim=0)


class PoissonMLPRender(MLPRender_Fea):
    def __init__(self, inChanel, viewpe=6, feape=6, featureC=128, **kwargs):
        super().__init__(inChanel, viewpe, feape, featureC)
