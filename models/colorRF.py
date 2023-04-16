import torch

from .renderBase import MLPRender_Fea
from .tensoRF import TensorVMSplit


class NormalizeCoord:
    def __init__(self, xyz_sampled):
        super().__init__()
        self.xyz_sampled = xyz_sampled
        self.valid_mask = None
        self.idx = torch.empty(*xyz_sampled.shape[:-1], dtype=torch.int64, device=xyz_sampled.device)
        self.idx.fill_(-1)

    def __getitem__(self, item):
        self.valid_mask = item
        return self

    def get_masked_array(self):
        return self.xyz_sampled[self.valid_mask]

    def get_masked_index(self):
        return self.idx[self.valid_mask]

    def set_masked_index(self, idx):
        self.idx[self.valid_mask] = idx


class ColorVMSplit(TensorVMSplit):
    def __init__(self, *args, **kwargs):
        self.merge_target = []
        super().__init__(*args, **kwargs)

    def init_render_func(self, shadingMode, pos_pe=6, view_pe=6, fea_pe=6, featureC=128, *args, **kwargs):
        return super().init_render_func(shadingMode, pos_pe, view_pe, fea_pe, featureC, **kwargs)

    def normalize_coord(self, xyz_sampled):
        return NormalizeCoord(xyz_sampled)

    def compute_validmask(self, xyz_sampled: torch.Tensor):
        ray_valid = super().compute_validmask(xyz_sampled)
        for model in self.merge_target:
            ray_valid |= model.compute_validmask(xyz_sampled)
        return ray_valid

    def compute_densityfeature(self, xyz_sampled: NormalizeCoord):
        if len(self.merge_target) == 0:
            return super().compute_densityfeature(super().normalize_coord(xyz_sampled.get_masked_array()))

        density = [model.compute_densityfeature(xyz_sampled) for model in self.merge_target]
        density = [super().compute_densityfeature(super().normalize_coord(xyz_sampled.get_masked_array()))] + density
        density, idx = torch.stack(density, dim=0).max(dim=0)
        xyz_sampled.set_masked_index(idx)
        return density

    def compute_radiance(self, pts: NormalizeCoord, viewdirs):
        if len(self.merge_target) == 0:
            return super().compute_radiance(super().normalize_coord(pts.get_masked_array()), viewdirs)

        tgt, = [model.compute_radiance(pts, viewdirs) for model in self.merge_target]
        rgb = super().compute_radiance(super().normalize_coord(pts.get_masked_array()), viewdirs)
        idx = pts.get_masked_index()
        rgb = torch.where((idx == 0).unsqueeze(-1), rgb, tgt)
        return rgb


class PoissonMLPRender(MLPRender_Fea):
    def __init__(self, inChanel, viewpe=6, feape=6, featureC=128, **kwargs):
        super().__init__(inChanel, viewpe, feape, featureC)

    def forward(self, pts, viewdirs, features):
        rgb = super().forward(pts, viewdirs, features)
        return rgb
