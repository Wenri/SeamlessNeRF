import os
from itertools import count
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap
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
        aval_viewdir = []
        aval_id = []
        ray_id = count()
        for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
            rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(self.device)
            rays_id = torch.from_numpy(np.fromiter(ray_id, dtype=np.int64, count=rays_chunk.shape[0]))
            xyz_sampled, z_vals, dists, viewdirs, ray_valid = self.tensorf.sample_and_filter_rays(
                rays_chunk, is_train=False, ndc_ray=self.args.ndc_ray, N_samples=512)
            app_mask = torch.zeros_like(ray_valid)
            if ray_valid.any():
                sigma_feature = self.tensorf.compute_densityfeature(
                    self.tensorf.normalize_coord(xyz_sampled)[ray_valid])
                validsigma = self.tensorf.feature2density(sigma_feature)
                alpha = 1. - torch.exp(-validsigma * dists[ray_valid])
                app_mask.masked_scatter_(ray_valid, alpha > 20 * self.tensorf.rayMarch_weight_thres)
            if app_mask.any():
                aval_pts.append(xyz_sampled[app_mask].cpu())
                aval_viewdir.append(viewdirs[app_mask].cpu())
                aval_id.append(rays_id.unsqueeze(dim=-1).expand(-1, app_mask.shape[1])[app_mask.cpu()])

        del rays
        aval_pts = torch.concat(aval_pts, dim=0)
        aval_viewdir = torch.concat(aval_viewdir, dim=0)
        aval_id = torch.concat(aval_id, dim=0)

        return aval_pts, aval_viewdir, aval_id

    @torch.no_grad()
    def generate_grad(self, pts):
        dx = torch.tensor((self.tensorf.stepSize, 0., 0.), device=self.device)
        dy = torch.tensor((0., self.tensorf.stepSize, 0.), device=self.device)
        dz = torch.tensor((0., 0., self.tensorf.stepSize), device=self.device)
        pts_diff = torch.stack((pts + dx, pts + dy, pts + dz, pts - dx, pts - dy, pts - dz), dim=1)
        all_query_pts = torch.concat((pts.unsqueeze(dim=1), pts_diff), dim=1).view(-1, 3)

        ptsPath = Path(self.args.basedir, self.args.expname, 'aval_pts.npy')
        if ptsPath.exists():
            aval_pts = torch.from_numpy(open_memmap(ptsPath, mode='r'))
            aval_viewdir = torch.from_numpy(open_memmap(ptsPath.with_stem('aval_viewdir'), mode='r'))
            aval_id = torch.from_numpy(open_memmap(ptsPath.with_stem('aval_id'), mode='r'))
        else:
            aval_pts, aval_viewdir, aval_id = self.sample_filter_dataset()
            np.save(ptsPath, aval_pts.numpy())
            np.save(ptsPath.with_stem('aval_viewdir'), aval_viewdir.numpy())
            np.save(ptsPath.with_stem('aval_id'), aval_id.numpy())

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
