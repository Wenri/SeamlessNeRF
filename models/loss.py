from collections import namedtuple
from operator import itemgetter

import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.ops import knn_points
from scipy.spatial import ConvexHull, Delaunay
from tqdm import tqdm

from .palette.Additive_mixing_layers_extraction import DCPPointTriangle


def recon_with_palette(palette, points):
    hull = ConvexHull(palette)
    de = Delaunay(hull.points[hull.vertices].clip(0.0, 1.0))
    ind, = np.nonzero(de.find_simplex(points, tol=1e-8) < 0)
    for i in tqdm(ind, desc='recon_with_palette'):
        points[i] = min((DCPPointTriangle(points[i], hull.points[j]) for j in hull.simplices),
                        key=itemgetter('distance'))['closest']
    return points


class LossBase:
    RegWeights_t = namedtuple('RegWeights_t', 'BLACK')

    def __init__(self, **reg_weights):
        self.reg_weights = self.RegWeights_t(**reg_weights)

    def apply_weights(self, reg_term):
        return sum(getattr(self.reg_weights, k) * v for k, v in reg_term.items())


class PLTLoss(LossBase):
    RegWeights_t = namedtuple('RegWeights_PLT_t', 'E_opaque PD BLACK', defaults=(.1, 1., -1. / 375))

    def __init__(self, hull_vertices, device, **reg_weights):
        if hull_vertices.size:
            self.hull = ConvexHull(hull_vertices)
            self.de = Delaunay(hull_vertices)
        else:
            self.hull = self.de = None
        self.hull_vertices = torch.from_numpy(hull_vertices).to(device, dtype=torch.float32)
        self.ones = torch.ones((1, 1), device=device)
        super().__init__(**reg_weights)

    def outsidehull_points_distance(self, inp_points, w_in=1e-3, w_out=1.):
        points = inp_points.detach().to('cpu', dtype=torch.double).numpy()
        simplex = self.de.find_simplex(points, tol=1e-8)
        loss = knn_points(inp_points[simplex >= 0].unsqueeze(0), self.hull_vertices[None],
                          K=1, return_sorted=False).dists
        loss = w_in / n_in * loss.sum() if (n_in := loss.nelement()) else loss.sum()
        ind, = np.nonzero(simplex < 0)
        points = [min((DCPPointTriangle(pts, self.hull.points[j]) for j in self.hull.simplices),
                      key=itemgetter('distance'))['closest'] for pts in points[ind]]
        if points:
            points = torch.asarray(points, device=inp_points.device, dtype=inp_points.dtype)
            loss = loss + w_out * F.mse_loss(inp_points[ind], points, reduction='none').sum(dim=-1).max()

        assert torch.isfinite(loss)
        return loss

    def plt_loss(self, plt_map, gt_train, palette, weight=1.):  # palette in 3xN
        loss = F.mse_loss(plt_map[..., :3], gt_train, reduction='mean')
        if palette is not None:
            plt_map = plt_map[..., 3:]
            E_opaque = F.mse_loss(plt_map, self.ones.expand_as(plt_map), reduction='mean')
            reg_term = {'E_opaque': E_opaque,
                        'PD': self.outsidehull_points_distance(palette.T),
                        'BLACK': torch.linalg.vector_norm(palette[:, -1])}
        else:
            reg_term = {}
        return loss * weight, reg_term
