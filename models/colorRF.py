from collections import UserList
from functools import partial

import torch

from .renderBase import MLPRender_Fea
from .tensoRF import TensorVMSplit


class NormalizeCoordMasked:
    def __init__(self, pts, valid_mask):
        super().__init__()
        self.pts = pts
        self.valid_mask = valid_mask

    def get_array(self):
        return self.pts.get_array()[self.valid_mask]

    def adj_coord(self, func):
        return NormalizeCoordMasked(self.pts.adj_coord(func), self.valid_mask)

    def get_index(self):
        return self.pts.idx[self.valid_mask]

    def set_index(self, idx):
        self.pts.idx[self.valid_mask] = idx


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

    def get_index(self):
        return self.idx

    def set_index(self, idx):
        self.idx = idx


class DensityFeature(UserList):
    def __init__(self, pts: NormalizeCoord | NormalizeCoordMasked, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pts = pts

    def set_index(self, idx):
        self.pts.set_index(idx)


class ColorVMSplit(TensorVMSplit):
    def __init__(self, *args, **kwargs):
        self.merge_target = []
        self.args = None
        super().__init__(*args, **kwargs)

    def init_render_func(self, shadingMode, pos_pe=6, view_pe=6, fea_pe=6, featureC=128, *args, **kwargs):
        return super().init_render_func(shadingMode, pos_pe, view_pe, fea_pe, featureC, **kwargs)

    def add_merge_target(self, model):
        if isinstance(model, type(self)):
            self.merge_target.extend(model.merge_target)
            model = super(type(self), model)
        self.merge_target.append(model)

    def normalize_coord(self, xyz_sampled):
        return NormalizeCoord(super().normalize_coord, xyz_sampled)

    def adjust_coord(self, xyz_sampled, func):
        xyz_sampled = self.shift_and_scale(xyz_sampled)
        return func(xyz_sampled)

    def shift_and_scale(self, xyz_sampled):
        from scipy.spatial.transform import Rotation as R
        r = R.from_euler('zyx', self.args.tgt_rot, degrees=True)
        r = torch.as_tensor(r.as_matrix(), dtype=xyz_sampled.dtype, device=xyz_sampled.device)
        xyz_sampled = xyz_sampled - torch.tensor(self.args.tgt_trans, dtype=xyz_sampled.dtype,
                                                 device=xyz_sampled.device)
        xyz_sampled = xyz_sampled / self.args.tgt_scale
        return xyz_sampled @ r

    def compute_validmask(self, xyz_sampled: torch.Tensor):
        ray_valid = super().compute_validmask(xyz_sampled)
        for model in self.merge_target:
            ray_valid |= model.compute_validmask(self.shift_and_scale(xyz_sampled))
            # ray_valid |= torch.ones_like(ray_valid, dtype=torch.bool, device=ray_valid.device)
        return ray_valid

    def compute_densityfeature(self, xyz_sampled: NormalizeCoord | NormalizeCoordMasked):
        gen = (model.compute_densityfeature(xyz_sampled.adj_coord(
            partial(self.adjust_coord, func=model.normalize_coord)).get_array()) for model in self.merge_target)
        density = DensityFeature(xyz_sampled, gen)
        density.append(super().compute_densityfeature(xyz_sampled.get_array()))
        return density

    def feature2density(self, feature: DensityFeature):
        density = [model.feature2density(feat) for model, feat in zip(self.merge_target, feature)]
        density.append(super().feature2density(feature[-1]))
        tgt_density = density[0]
        density, idx = torch.stack(density, dim=0).max(dim=0)
        feature.set_index(idx)
        return density

    def compute_radiance(self, pts: NormalizeCoord | NormalizeCoordMasked, viewdirs):
        rgb = super().compute_radiance(pts.get_array(), viewdirs)
        if len(self.merge_target) == 0:
            return rgb

        tgt, = [model.compute_radiance(pts.adj_coord(partial(
            self.adjust_coord, func=model.normalize_coord)).get_array(), viewdirs) for model in self.merge_target]
        idx = pts.get_index()
        rgb = torch.where((idx > 0.5).unsqueeze(-1), rgb, tgt)
        # rgb = tgt
        return rgb

    def forward(self, *params, args=None, **kwargs):
        if self.args is None and args:
            self.args = args
        return super().forward(*params, **kwargs)


class PoissonMLPRender(MLPRender_Fea):
    def __init__(self, inChanel, viewpe=6, feape=6, featureC=128, **kwargs):
        super().__init__(inChanel, viewpe, feape, featureC)

    def forward(self, pts, viewdirs, features):
        rgb = super().forward(pts, viewdirs, features)
        return rgb
