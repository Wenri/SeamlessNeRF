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


def torch_copy(obj):
    buffer = io.BytesIO()  # read from buffer
    torch.save(obj, buffer)
    buffer.seek(0)  # <--- must see to origin every time before reading
    return torch.load(buffer)


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
    def __init__(self, pts, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pts = pts

    def set_index(self, idx):
        self.pts.set_index(idx)


class MultipleGridMask(torch.nn.ModuleList):
    def __init__(self, matrix, *masks: AlphaGridMask):
        super().__init__(masks)
        self.matrix = torch.linalg.inv(torch.as_tensor(matrix).view(4, 4))

    def sample_alpha(self, xyz_sampled):
        alpha_vals = [m.sample_alpha(f(xyz_sampled)) for m, f in
                      zip_longest(self, (torch.nn.Identity(),), fillvalue=self.shift_and_scale)]
        alpha_vals, idx = torch.stack(alpha_vals, dim=0).max(dim=0)
        return alpha_vals

    def shift_and_scale(self, xyz_sampled):
        ones = torch.ones(xyz_sampled.shape[:-1], dtype=xyz_sampled.dtype, device=xyz_sampled.device)
        xyz_sampled = torch.cat((xyz_sampled, ones.unsqueeze(-1)), dim=-1)
        t_matrix = self.matrix.T.to(dtype=xyz_sampled.dtype, device=xyz_sampled.device)
        # xyz_sampled = torch.linalg.solve(t_matrix, xyz_sampled, left=False)
        xyz_sampled = xyz_sampled @ t_matrix
        return xyz_sampled[..., :3]


class ColorVMSplit(TensorVMSplit):
    def __init__(self, *args, at_least_aabb=None, **kwargs):
        self.merge_target = []
        self.args = None
        self.tgt_rgb = None
        self.render_gap = 1
        self.density_gain = 1
        self.at_least_aabb = at_least_aabb
        super().__init__(*args, **kwargs)

    def init_render_func(self, shadingMode, pos_pe=6, view_pe=6, fea_pe=6, featureC=128, *args, **kwargs):
        return super().init_render_func(shadingMode, pos_pe, view_pe, fea_pe, featureC, **kwargs)

    def getDenseAlpha(self, gridSize=None, aabb=None):
        if aabb is None:
            aabb = getattr(self, 'merged_aabb', None)
        return super().getDenseAlpha(gridSize, aabb)

    def add_merge_target(self, model, density_gain, render_gap=None):
        if isinstance(self.alphaMask, AlphaGridMask):
            self.alphaMask = MultipleGridMask(self.args.matrix, self.alphaMask)
        elif self.alphaMask is None:
            self.alphaMask = MultipleGridMask(self.args.matrix)

        if isinstance(model.alphaMask, type(self.alphaMask)):
            self.alphaMask.extend(model.alphaMask)
        else:
            self.alphaMask.append(model.alphaMask)

        if render_gap is None:
            render_gap = self.stepSize / model.stepSize
        self.render_gap = render_gap
        self.density_gain = density_gain
        # self.register_buffer(name='merged_aabb', persistent=False, tensor=torch.as_tensor(
        #     (-1.5, -1.5, -1.5, 1.5, 1.5, 1.5), device=self.aabb.device, dtype=self.aabb.dtype).view(-1, 3))
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

    def get_appparam(self):
        if self.renderModule.ignore_control or not hasattr(self, 'app_plane_ctl'):
            return super().get_appparam()
        return self.app_plane_ctl, self.app_line_ctl, self.basis_mat_ctl

    def compute_validmask(self, xyz_sampled: torch.Tensor, bitmap=False):
        ray_valid = super().compute_validmask(xyz_sampled).int()
        shift_and_scale = getattr(self.alphaMask, 'shift_and_scale', torch.nn.Identity())
        for model in self.merge_target:
            tgt_mask = model.compute_validmask(shift_and_scale(xyz_sampled))
            ray_valid = torch.bitwise_left_shift(ray_valid, 1)
            ray_valid = torch.bitwise_or(ray_valid, tgt_mask.int())
            # ray_valid |= torch.ones_like(ray_valid, dtype=torch.bool, device=ray_valid.device)
        return ray_valid if bitmap else ray_valid.bool()

    def compute_densityfeature(self, xyz_sampled):
        gen = (model.compute_densityfeature(xyz_sampled.adj_coord(
            partial(self.adjust_coord, func=model.normalize_coord)).get_array()) for model in self.merge_target)
        density = DensityFeature(xyz_sampled, gen)
        density.append(super().compute_densityfeature(xyz_sampled.get_array()))
        return density

    def feature2density(self, feature: DensityFeature):
        density = [model.feature2density(feat) for model, feat in
                   zip_longest(self.merge_target, feature, fillvalue=super())]
        if len(density) == 2:
            tgt, src = density
            _, idx = torch.stack((tgt * self.render_gap, src * self.density_gain), dim=0).max(dim=0)
            density = torch.where(idx > 0.5, src, tgt)
        else:
            density, idx = torch.stack(density, dim=0).max(dim=0)
        feature.set_index(idx)
        return density

    def compute_radiance(self, pts, viewdirs):
        rgb = super().compute_radiance(pts.get_array(), viewdirs)
        if len(self.merge_target) == 0:
            return rgb

        shift_and_scale = getattr(self.alphaMask, 'shift_and_scale', torch.nn.Identity())
        viewdirs = shift_and_scale(viewdirs) - shift_and_scale(torch.zeros(3, device=viewdirs.device))
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        tgt, = [model.compute_radiance(pts.adj_coord(partial(
            self.adjust_coord, func=model.normalize_coord)).get_array(), viewdirs) for model in self.merge_target]
        idx = pts.get_index()
        rgb = torch.where((idx > 0.5).unsqueeze(-1), rgb, tgt)
        self.tgt_rgb = tgt
        return rgb

    def scale_distance(self, xyz_sampled, dists, scale=None):
        dists = super().scale_distance(xyz_sampled, dists, scale)
        if self.render_gap == 1 or not hasattr(xyz_sampled, 'get_index'):
            return dists

        idx = xyz_sampled.get_index()
        dists[torch.logical_not(idx)] *= self.render_gap
        return dists

    def forward(self, *params, args=None, **kwargs):
        if self.args is None and args:
            self.args = args
        return super().forward(*params, **kwargs)

    def enable_trainable_control(self):
        self.add_module('app_plane_ctl', torch_copy(self.app_plane))
        self.add_module('app_line_ctl', torch_copy(self.app_line))
        self.add_module('basis_mat_ctl', torch_copy(self.basis_mat))

    def update_stepSize(self, gridSize):
        ret = super().update_stepSize(gridSize)
        return ret

    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(200, 200, 200)):
        new_aabb = super().updateAlphaMask(gridSize)
        if self.at_least_aabb:
            at_least = torch.as_tensor(self.at_least_aabb, device=new_aabb.device, dtype=new_aabb.dtype).view(-1, 3)
            new_aabb[0] = torch.minimum(new_aabb[0], at_least[0])
            new_aabb[1] = torch.maximum(new_aabb[1], at_least[1])
        return new_aabb


class PoissonMLPRender(MLPRender_Fea):
    def __init__(self, inChanel, viewpe=6, feape=6, featureC=128, **kwargs):
        super().__init__(inChanel, viewpe, feape, featureC)
        self.orig_rgb = None
        self.ignore_control = False

    def forward(self, pts, viewdirs, features):
        add_control = hasattr(self, 'mlp_control') and not self.ignore_control
        self.orig_rgb = None
        with self.mlp.register_forward_hook(self.mlp_forward_hook) if add_control else nullcontext():
            rgb = super().forward(pts, viewdirs, features)
        return rgb

    def mlp_forward_hook(self, mlp, data_in, output: torch.Tensor):
        self.orig_rgb = output
        return output + self.mlp_control(*data_in)

    def enable_trainable_control(self):
        new_mlp = torch_copy(self.mlp)
        zero_conv = nn.Linear(3, 3)
        torch.nn.init.constant_(zero_conv.bias, 0)
        torch.nn.init.constant_(zero_conv.weight, 0)
        new_mlp.append(zero_conv)
        new_mlp.cuda()
        self.add_module('mlp_control', new_mlp)
