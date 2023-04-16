import torch

from .renderBase import MLPRender_Fea
from .tensoRF import TensorVMSplit


class NormalizeCoordMasked:
    def __init__(self, pts, valid_mask):
        super().__init__()
        self._pts = pts
        self.valid_mask = valid_mask

    def get_array(self):
        return self._pts.get_array()[self.valid_mask]

    def adj_coord(self, func):
        return NormalizeCoordMasked(self._pts.adj_coord(func), self.valid_mask)

    def get_index(self):
        return self._pts.idx[self.valid_mask]

    def set_index(self, idx):
        self._pts.idx[self.valid_mask] = idx


class NormalizeCoord:
    def __init__(self, func, xyz_sampled):
        super().__init__()
        self.func = func
        if isinstance(xyz_sampled, NormalizeCoord):
            self.xyz_sampled = xyz_sampled.xyz_sampled
            self.idx = xyz_sampled.idx
        else:
            self.xyz_sampled = xyz_sampled
            self.idx = torch.empty(*xyz_sampled.shape[:-1], dtype=torch.int64, device=xyz_sampled.device)
            self.idx.fill_(-1)

    def __getitem__(self, item):
        return NormalizeCoordMasked(self, item)

    def get_array(self):
        return self.func(self.xyz_sampled)

    def adj_coord(self, func):
        return NormalizeCoord(func, self)


class ColorVMSplit(TensorVMSplit):
    def __init__(self, *args, **kwargs):
        self.merge_target = []
        super().__init__(*args, **kwargs)

    def init_render_func(self, shadingMode, pos_pe=6, view_pe=6, fea_pe=6, featureC=128, *args, **kwargs):
        return super().init_render_func(shadingMode, pos_pe, view_pe, fea_pe, featureC, **kwargs)

    def normalize_coord(self, xyz_sampled):
        return NormalizeCoord(super().normalize_coord, xyz_sampled)

    def adjust_coord(self, xyz_sampled):
        xyz_sampled = self.shift_and_scale(xyz_sampled)
        return super().normalize_coord(xyz_sampled)

    def shift_and_scale(self, xyz_sampled):
        from scipy.spatial.transform import Rotation as R
        r = R.from_euler('zyx', [90, 0, 0], degrees=True)
        r = torch.as_tensor(r.as_matrix(), dtype=xyz_sampled.dtype, device=xyz_sampled.device)
        xyz_sampled = xyz_sampled / 0.65
        xyz_sampled = xyz_sampled - torch.tensor([0.1, 1.0, -0.5], dtype=xyz_sampled.dtype, device=xyz_sampled.device)
        return xyz_sampled @ r

    def compute_validmask(self, xyz_sampled: torch.Tensor):
        ray_valid = super().compute_validmask(xyz_sampled)
        for model in self.merge_target:
            ray_valid |= model.compute_validmask(model.shift_and_scale(xyz_sampled))
            # ray_valid |= torch.ones_like(ray_valid, dtype=torch.bool, device=ray_valid.device)
        return ray_valid

    def compute_densityfeature(self, xyz_sampled: NormalizeCoordMasked):
        density = super().compute_densityfeature(xyz_sampled.get_array())
        if len(self.merge_target) == 0:
            return density

        tgt = [model.compute_densityfeature(xyz_sampled.adj_coord(model.adjust_coord)) for model in self.merge_target]
        density = [density] + tgt
        density, idx = torch.stack(density, dim=0).max(dim=0)
        xyz_sampled.set_index(idx)
        return density

    def compute_radiance(self, pts: NormalizeCoordMasked, viewdirs):
        rgb = super().compute_radiance(pts.get_array(), viewdirs)
        if len(self.merge_target) == 0:
            return rgb

        tgt, = [model.compute_radiance(pts.adj_coord(model.adjust_coord), viewdirs) for model in self.merge_target]
        idx = pts.get_index()
        rgb = torch.where((idx == 0).unsqueeze(-1), rgb, tgt)
        return rgb


class PoissonMLPRender(MLPRender_Fea):
    def __init__(self, inChanel, viewpe=6, feape=6, featureC=128, **kwargs):
        super().__init__(inChanel, viewpe, feape, featureC)

    def forward(self, pts, viewdirs, features):
        rgb = super().forward(pts, viewdirs, features)
        return rgb
