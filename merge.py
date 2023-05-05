import os
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from numpy.lib.format import open_memmap
from pytorch3d.ops import knn_points, ball_query
from torch import from_numpy
from trimesh import PointCloud

from dataLoader import dataset_dict
from eval import Evaluator
from utils import convert_sdf_samples_to_ply


class Merger(Evaluator):
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # init dataset
        dataset = dataset_dict[args.dataset_name]
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_test, is_stack=False,
                                semantic_type=args.semantic_type)  # if not args.render_only else None
        test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_test, is_stack=True,
                               semantic_type=args.semantic_type, pca=getattr(train_dataset, 'pca', None))
        super().__init__(self.build_network(), args, test_dataset, train_dataset)

    def build_network(self, ckpt=None):
        args = self.args
        if not ckpt:
            ckpt = Path(args.basedir, args.expname) / f'{args.expname}.th'

        if not os.path.exists(ckpt):
            raise RuntimeError(f'the ckpt {ckpt} does not exists!!')

        ckpt = torch.load(ckpt, map_location=self.device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device': self.device})
        tensorf = args.model_name(**kwargs)
        tensorf.load(ckpt)

        return tensorf

    @torch.no_grad()
    def export_mesh(self, tensorf=None):
        args = self.args
        savePath = Path(args.basedir, args.expname)
        if tensorf is None:
            tensorf = self.tensorf
            savePath = savePath / args.expname
        else:
            savePath = savePath / os.path.basename(args.ckpt)
        savePath = savePath.with_suffix('.ply')
        alpha, _ = tensorf.getDenseAlpha()
        convert_sdf_samples_to_ply(alpha.cpu(), savePath, bbox=tensorf.aabb.cpu(), level=0.005)

    @torch.no_grad()
    def export_pointcloud(self, tensorf=None):
        args = self.args
        savePath = Path(args.basedir, args.expname)
        if tensorf is None:
            tensorf = self.tensorf
            savePath = savePath / args.expname
        else:
            savePath = savePath / os.path.basename(args.ckpt)
        savePath = savePath.with_stem(savePath.stem + '_pc').with_suffix('.ply')
        alpha, xyz = tensorf.getDenseAlpha()
        pts = xyz[alpha > 0.005]
        pc = PointCloud(pts.cpu().numpy())
        pc.export(savePath)
        return pts

    @torch.no_grad()
    def sample_filter_dataset(self, chunk=64 * 1024):
        rays = torch.concat((self.alt_dataset.all_rays.view(-1, 6), self.test_dataset.all_rays.view(-1, 6)), dim=0)
        N_rays_all = rays.shape[0]
        aval_pts = []
        aval_id = []
        aval_rep = []
        for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
            rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(self.device)
            xyz_sampled, z_vals, dists, viewdirs, ray_valid = self.tensorf.sample_and_filter_rays(
                rays_chunk, is_train=False, ndc_ray=self.args.ndc_ray, N_samples=512)
            app_mask = torch.zeros_like(ray_valid)
            app_alpha = torch.zeros_like(ray_valid, dtype=torch.float32)
            if ray_valid.any():
                sigma_feature = self.tensorf.compute_densityfeature(
                    self.tensorf.normalize_coord(xyz_sampled)[ray_valid])
                validsigma = self.tensorf.feature2density(sigma_feature)
                alpha = 1. - torch.exp(-validsigma * dists[ray_valid])
                app_mask.masked_scatter_(ray_valid, alpha > 20 * self.tensorf.rayMarch_weight_thres)
                app_alpha.masked_scatter_(ray_valid, alpha)
            if app_mask.any():
                aval_pts.append(torch.concat((xyz_sampled[app_mask], viewdirs[app_mask]), dim=-1).cpu())
                _, ind = app_alpha.max(dim=-1)
                chunk_mask = app_mask.any(dim=-1)
                ind = ind[chunk_mask]
                chunk_rep, = chunk_mask.nonzero(as_tuple=True)
                aval_rep.append(torch.concat((xyz_sampled[chunk_rep, ind], viewdirs[chunk_rep, ind]), dim=-1).cpu())
            aval_id.append(app_mask.count_nonzero(dim=-1).cpu())

        del rays
        aval_pts = torch.concat(aval_pts, dim=0)
        aval_id = torch.concat(aval_id, dim=0)
        ind, = torch.nonzero(aval_id, as_tuple=True)
        aval_id = torch.stack((ind, aval_id[ind]), dim=-1)
        aval_rep = torch.concat(aval_rep, dim=0)

        return aval_pts, aval_id, aval_rep

    @torch.no_grad()
    def generate_grad(self, pts):
        ptsPath = Path(self.args.basedir, self.args.expname, 'cache')
        ptsPath.mkdir(exist_ok=True)

        ptsPath = ptsPath / 'all_query_pts.npy'
        if ptsPath.exists():
            all_query_pts = torch.from_numpy(open_memmap(ptsPath, mode='r')).to(self.device)
            assert torch.allclose(pts, all_query_pts.view(-1, 7, 3)[:, 0])
        else:
            dx = torch.tensor((self.tensorf.stepSize, 0., 0.), device=self.device)
            dy = torch.tensor((0., self.tensorf.stepSize, 0.), device=self.device)
            dz = torch.tensor((0., 0., self.tensorf.stepSize), device=self.device)
            pts_diff = torch.stack((pts + dx, pts + dy, pts + dz, pts - dx, pts - dy, pts - dz), dim=1)
            all_query_pts = torch.concat((pts.unsqueeze(dim=1), pts_diff), dim=1).view(-1, 3)
            np.save(ptsPath, all_query_pts.cpu().numpy())

        ptsPath = ptsPath.with_stem('aval_pts')
        if ptsPath.exists():
            aval_pts = torch.from_numpy(open_memmap(ptsPath, mode='r'))
            aval_id = torch.from_numpy(open_memmap(ptsPath.with_stem('aval_id'), mode='r'))
            aval_rep = torch.from_numpy(open_memmap(ptsPath.with_stem('aval_rep'), mode='r'))
        else:
            aval_pts, aval_id, aval_rep = self.sample_filter_dataset()
            np.save(ptsPath, aval_pts.numpy())
            np.save(ptsPath.with_stem('aval_id'), aval_id.numpy())
            np.save(ptsPath.with_stem('aval_rep'), aval_rep.numpy())

        ptsPath = ptsPath.with_stem('ball_ret_dists.7')
        if ptsPath.exists():
            ball_dists = rearrange(from_numpy(open_memmap(ptsPath, mode='r')), '1 (n d) k -> n d k', n=pts.shape[0])
            ball_idx = from_numpy(open_memmap(ptsPath.with_stem('ball_ret_idx.7'), mode='r')).view(*ball_dists.shape)
            ball_knn = from_numpy(open_memmap(ptsPath.with_stem('ball_ret_knn.7'), mode='r')).view(*ball_dists.shape, 3)
        else:
            aval_rep_pts = aval_rep[None, ..., :3].cuda()
            knn_ret = knn_points(pts[None].cuda(), aval_rep_pts, K=1, return_sorted=False)
            median_dist = torch.sqrt(knn_ret.dists).median().item()
            print(knn_ret.dists.min().item(), knn_ret.dists.max().item(), median_dist)
            median_dist = max(median_dist, self.tensorf.stepSize) * 2
            ball_ret = ball_query(all_query_pts[None].cuda(), aval_rep_pts, K=40, radius=median_dist)
            np.save(ptsPath.with_stem('ball_ret_dists.7'), ball_ret.dists.cpu().numpy())
            np.save(ptsPath.with_stem('ball_ret_idx.7'), ball_ret.idx.cpu().numpy())
            np.save(ptsPath.with_stem('ball_ret_knn.7'), ball_ret.knn.cpu().numpy())

        ptsPath = ptsPath.with_stem('knn_ret_dists')
        if ptsPath.exists():
            knn_dists = torch.from_numpy(open_memmap(ptsPath, mode='r')).squeeze(0)
            knn_idx = torch.from_numpy(open_memmap(ptsPath.with_stem('knn_ret_idx'), mode='r')).squeeze(0)
        else:
            knn_ret = knn_points(pts[None].cuda(), aval_rep[None, ..., :3].cuda(), K=100, return_sorted=True)
            np.save(ptsPath, knn_ret.dists.cpu().numpy())
            np.save(ptsPath.with_stem('knn_ret_idx'), knn_ret.idx.cpu().numpy())

        pts_mask = (ball_idx[:, 0] < 0).all(dim=-1)

        return pts_diff

    @torch.no_grad()
    def merge(self):
        self.tensorf.args = self.args
        self.tensorf.add_merge_target(target := self.build_network(self.args.ckpt))
        if self.args.export_mesh:
            self.export_mesh(target)
            self.export_mesh()
        pts = self.export_pointcloud()
        self.generate_grad(pts)
        self.render_test()


def config_parser(parser):
    from dataLoader import dataset_dict
    from models import MODEL_ZOO

    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='./log', help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', help='input data directory')

    parser.add_argument('--downsample_test', type=float, default=1.0)
    parser.add_argument('--tgt_rot', type=float, nargs='+', default=[0., 0., 0.])
    parser.add_argument('--tgt_trans', type=float, nargs='+', default=[0., 0., 0.])
    parser.add_argument('--tgt_scale', type=float, default=1.0)

    parser.add_argument('--model_name', type=MODEL_ZOO.get, default='TensorVMSplit', choices=MODEL_ZOO)
    parser.add_argument('--dataset_name', type=str, default='blender', choices=dataset_dict.keys())

    # network decoder
    parser.add_argument("--semantic_type", type=str, default='None', help='semantic type')

    parser.add_argument("--ckpt", type=str, default=None, help='specific weights npy file to reload for coarse network')
    parser.add_argument("--render_only", action="store_true")
    parser.add_argument("--render_test", action="store_true")
    parser.add_argument("--render_train", action="store_true")
    parser.add_argument("--render_path", action="store_true")
    parser.add_argument("--export_mesh", action="store_true")

    # rendering options
    parser.add_argument('--ndc_ray', type=int, default=0)

    # logging/saving options
    parser.add_argument("--N_vis", type=int, default=5, help='N images to vis')


def main(args):
    assert args.render_only
    merger = Merger(args)
    merger.merge()
