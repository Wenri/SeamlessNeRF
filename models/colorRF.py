import io
from collections import UserList
from contextlib import nullcontext
from functools import partial
from itertools import zip_longest

import torch
from torch import nn

from .renderBase import MLPRender_Fea
from .tensoRF import TensorVMSplit
from .tensorBase import AlphaGridMask


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


class MultipleGridMask(torch.nn.ModuleList):
    def __init__(self, matrix, *masks: AlphaGridMask):
        super().__init__(masks)
        self.matrix = torch.as_tensor(matrix).view(4, 4)

    def sample_alpha(self, xyz_sampled):
        alpha_vals = [m.sample_alpha(f(xyz_sampled)) for m, f in
                      zip_longest(self, (torch.nn.Identity(),), fillvalue=self.shift_and_scale)]
        alpha_vals, idx = torch.stack(alpha_vals, dim=0).max(dim=0)
        return alpha_vals

    def shift_and_scale(self, xyz_sampled):
        ones = torch.ones(xyz_sampled.shape[:-1], dtype=xyz_sampled.dtype, device=xyz_sampled.device)
        xyz_sampled = torch.cat((xyz_sampled, ones.unsqueeze(-1)), dim=-1)
        t_matrix = self.matrix.T.to(dtype=xyz_sampled.dtype, device=xyz_sampled.device)
        xyz_sampled = torch.linalg.solve(t_matrix, xyz_sampled, left=False)
        return xyz_sampled[..., :3]


class ColorVMSplit(TensorVMSplit):
    def __init__(self, *args, **kwargs):
        self.merge_target = []
        self.args = None
        self.tgt_rgb = None
        super().__init__(*args, **kwargs)

    def init_render_func(self, shadingMode, pos_pe=6, view_pe=6, fea_pe=6, featureC=128, *args, **kwargs):
        return super().init_render_func(shadingMode, pos_pe, view_pe, fea_pe, featureC, **kwargs)

    def add_merge_target(self, model):
        if isinstance(self.alphaMask, AlphaGridMask):
            self.alphaMask = MultipleGridMask(self.args.matrix, self.alphaMask)
        elif self.alphaMask is None:
            self.alphaMask = MultipleGridMask(self.args.matrix)

        if isinstance(model.alphaMask, type(self.alphaMask)):
            self.alphaMask.extend(model.alphaMask)
        else:
            self.alphaMask.append(model.alphaMask)

        if isinstance(model, type(self)):
            self.merge_target.extend(model.merge_target)
            model = super(type(self), model)
        self.merge_target.append(model)

    def normalize_coord(self, xyz_sampled):
        return NormalizeCoord(super().normalize_coord, xyz_sampled)

    def adjust_coord(self, xyz_sampled, func):
        shift_and_scale = getattr(self.alphaMask, 'shift_and_scale', torch.nn.Identity())
        xyz_sampled = shift_and_scale(xyz_sampled)
        return func(xyz_sampled)

    def compute_validmask(self, xyz_sampled: torch.Tensor, bitmap=False):
        ray_valid = super().compute_validmask(xyz_sampled).int()
        shift_and_scale = getattr(self.alphaMask, 'shift_and_scale', torch.nn.Identity())
        for model in self.merge_target:
            tgt_mask = model.compute_validmask(shift_and_scale(xyz_sampled))
            ray_valid = torch.bitwise_left_shift(ray_valid, 1)
            ray_valid = torch.bitwise_or(ray_valid, tgt_mask.int())
            # ray_valid |= torch.ones_like(ray_valid, dtype=torch.bool, device=ray_valid.device)
        return ray_valid if bitmap else ray_valid.bool()

    def compute_densityfeature(self, xyz_sampled: NormalizeCoord | NormalizeCoordMasked):
        gen = (model.compute_densityfeature(xyz_sampled.adj_coord(
            partial(self.adjust_coord, func=model.normalize_coord)).get_array()) for model in self.merge_target)
        density = DensityFeature(xyz_sampled, gen)
        density.append(super().compute_densityfeature(xyz_sampled.get_array()))
        return density

    def feature2density(self, feature: DensityFeature):
        density = [model.feature2density(feat) for model, feat in
                   zip_longest(self.merge_target, feature, fillvalue=super())]
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
        self.tgt_rgb = tgt
        return rgb

    def forward(self, *params, args=None, **kwargs):
        if self.args is None and args:
            self.args = args
        return super().forward(*params, **kwargs)


class PoissonMLPRender(MLPRender_Fea):
    def __init__(self, inChanel, viewpe=6, feape=6, featureC=128, **kwargs):
        super().__init__(inChanel, viewpe, feape, featureC)
        self.orig_rgb = None

    def forward(self, pts, viewdirs, features, ignore_control=False):
        add_control = hasattr(self, 'mlp_control') and not ignore_control
        with self.mlp.register_forward_hook(self.mlp_forward_hook) if add_control else nullcontext():
            rgb = super().forward(pts, viewdirs, features)
        return rgb

    def mlp_forward_hook(self, mlp, data_in, output: torch.Tensor):
        self.orig_rgb = output
        return output + self.mlp_control(*data_in)

    def enable_trainable_control(self):
        buffer = io.BytesIO()  # read from buffer
        torch.save(self.mlp, buffer)
        buffer.seek(0)  # <--- must see to origin every time before reading
        new_mlp = torch.load(buffer)
        zero_conv = nn.Linear(3, 3)
        torch.nn.init.constant_(zero_conv.bias, 0)
        torch.nn.init.constant_(zero_conv.weight, 0)
        new_mlp.append(zero_conv)
        new_mlp.cuda()
        self.add_module('mlp_control', new_mlp)
