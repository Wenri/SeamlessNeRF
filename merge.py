import os
import shutil
from contextlib import suppress
from itertools import repeat, chain, count
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from einops import rearrange
from numpy.lib.format import open_memmap
from pytorch3d.ops import knn_points, ball_query
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm, trange
from trimesh import PointCloud

from dataLoader import dataset_dict
from eval import Evaluator
from models.colorRF import DensityFeature
from utils import convert_sdf_samples_to_ply, cal_n_samples


class Merger(Evaluator):
    def __init__(self, args, pool):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target = self.build_network(self.args.ckpt)

        # init dataset
        dataset = dataset_dict[args.dataset_name]
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_test, is_stack=False,
                                semantic_type=args.semantic_type)  # if not args.render_only else None
        test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_test, is_stack=True,
                               semantic_type=args.semantic_type, pca=getattr(train_dataset, 'pca', None))
        super().__init__(self.build_network(), args, test_dataset, train_dataset, pool=pool)
        self.tensorf.args = self.args
        self.optimizer = None

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
    def export_mesh(self, tensorf=None, prefix=None):
        args = self.args
        if tensorf is None:
            tensorf = self.tensorf
            if not prefix:
                prefix = args.expname
        elif not prefix:
            prefix = os.path.basename(args.ckpt)
        save_path = Path(args.basedir, args.expname, prefix).with_suffix('.ply')
        alpha, _ = tensorf.getDenseAlpha()
        bbox = getattr(tensorf, 'merged_aabb', tensorf.aabb)
        convert_sdf_samples_to_ply(alpha.cpu(), save_path, bbox=bbox.cpu(), level=0.005)

    @torch.no_grad()
    def export_pointcloud(self, tensorf=None):
        args = self.args
        save_path = Path(args.basedir, args.expname)
        if tensorf is None:
            tensorf = self.tensorf
            save_path = save_path / args.expname
        else:
            save_path = save_path / os.path.basename(args.ckpt)
        save_path = save_path.with_stem(save_path.stem + '_pc').with_suffix('.ply')
        gridSize = tensorf.gridSize * 2
        alpha, xyz = tensorf.getDenseAlpha(gridSize)
        pts = xyz[alpha > 0.005]
        pc = PointCloud(pts.cpu().numpy())
        pc.export(save_path)
        return pts

    @torch.no_grad()
    def filter_pts(self, xyz_sampled, dists, ray_valid):
        app_mask = torch.zeros_like(ray_valid)
        app_alpha = torch.zeros_like(ray_valid, dtype=torch.float32)
        if ray_valid.any():
            xyz_sampled = self.tensorf.normalize_coord(xyz_sampled)
            sigma_feature = self.tensorf.compute_densityfeature(xyz_sampled[ray_valid])
            validsigma = self.tensorf.feature2density(sigma_feature)
            dists = self.tensorf.scale_distance(xyz_sampled, dists, scale=1)
            alpha = 1. - torch.exp(-validsigma * dists[ray_valid])
            app_mask.masked_scatter_(ray_valid, alpha > 20 * self.tensorf.rayMarch_weight_thres)
            app_alpha.masked_scatter_(ray_valid, alpha)
        return app_mask, app_alpha

    @torch.no_grad()
    def sample_filter_dataset(self, pts_path, chunk=8192):
        rays = torch.concat((self.alt_dataset.all_rays.view(-1, 6), self.test_dataset.all_rays.view(-1, 6)), dim=0)
        N_rays_all = rays.shape[0]
        cnt = count()
        aval_id = []
        aval_rep = []
        prefix = f'{Path(pts_path).stem}_'
        nSamples = cal_n_samples(self.tensorf.gridSize.cpu().numpy(), self.tensorf.step_ratio)
        nSamples = min(self.args.nSamples, nSamples / self.args.delta_scale)
        with TemporaryDirectory(dir=os.path.dirname(pts_path)) as tmpdir:
            for chunk_idx in trange(N_rays_all // chunk + int(N_rays_all % chunk > 0), desc='sample_filter_dataset'):
                rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(self.device)
                xyz_sampled, z_vals, dists, viewdirs, ray_valid = self.tensorf.sample_and_filter_rays(
                    rays_chunk, is_train=False, ndc_ray=self.args.ndc_ray, N_samples=nSamples)
                app_mask, app_alpha = self.filter_pts(xyz_sampled, dists, ray_valid)
                if app_mask.any():
                    np.save(os.path.join(tmpdir, f'{prefix}{next(cnt)}.npy'),
                            torch.concat((xyz_sampled[app_mask], viewdirs[app_mask]), dim=-1).cpu().numpy())
                    _, ind = app_alpha.max(dim=-1)
                    chunk_mask = app_mask.any(dim=-1)
                    ind = ind[chunk_mask]
                    chunk_rep, = chunk_mask.nonzero(as_tuple=True)
                    aval_rep.append(torch.cat((xyz_sampled[chunk_rep, ind], viewdirs[chunk_rep, ind]), dim=-1).cpu())
                aval_id.append(app_mask.count_nonzero(dim=-1).cpu())
            del rays
            self.logger.warn('sample_filter_dataset done. Concatenating files...')
            aval_id = torch.cat(aval_id, dim=0)
            id_cnt = aval_id.sum().item()
            ind, = torch.nonzero(aval_id, as_tuple=True)
            aval_id = torch.stack((ind, aval_id[ind]), dim=-1)
            aval_rep = torch.cat(aval_rep, dim=0)
            aval_pts = [open_memmap(os.path.join(tmpdir, f'{prefix}{cur}.npy'), mode='r') for cur in range(next(cnt))]
            aval_out = open_memmap(pts_path, mode='w+', shape=(id_cnt, 6), dtype=np.float32)
            aval_pts = np.concatenate(aval_pts, axis=0, out=aval_out)
        aval_out.flush()
        return torch.from_numpy(aval_pts), aval_id, aval_rep

    @torch.no_grad()
    def load_all_query_pts(self, pts_path: Path, pts):
        if pts_path.exists():
            self.logger.info('Loading all_query_pts from cached into GPU...')
            with suppress(Exception):
                all_query_pts = torch.from_numpy(open_memmap(pts_path, mode='r')).to(self.device)
                if torch.allclose(pts, all_query_pts.view(-1, 7, 3)[:, 0]):
                    return all_query_pts
            self.logger.warn('Cached query_pts mismatch. Regenerating...')
            parent = pts_path.parent
            shutil.rmtree(parent, ignore_errors=True)
            parent.mkdir(exist_ok=True)

        self.logger.warn('Generating all_query_pts using CUDA...')
        delta = self.tensorf.stepSize * self.args.delta_scale
        dx = torch.tensor((delta, 0., 0.), device=self.device)
        dy = torch.tensor((0., delta, 0.), device=self.device)
        dz = torch.tensor((0., 0., delta), device=self.device)
        pts_diff = torch.stack((pts + dx, pts + dy, pts + dz, pts - dx, pts - dy, pts - dz), dim=1)
        all_query_pts = torch.concat((pts.unsqueeze(dim=1), pts_diff), dim=1).view(-1, 3)
        np.save(pts_path, all_query_pts.cpu().numpy())
        return all_query_pts

    @torch.no_grad()
    def load_aval_pts(self, pts_path: Path):
        pts_path = pts_path.with_stem('aval_pts')
        if pts_path.exists():
            self.logger.info('Loading aval_pts from cached...')
            with suppress(Exception):
                aval_pts = torch.from_numpy(open_memmap(pts_path, mode='r'))
                aval_id = torch.from_numpy(open_memmap(pts_path.with_stem('aval_id'), mode='r'))
                aval_rep = torch.from_numpy(open_memmap(pts_path.with_stem('aval_rep'), mode='r'))
                return aval_pts, aval_id, aval_rep

        self.logger.warn('Calc aval_pts using CUDA...')
        aval_pts, aval_id, aval_rep = self.sample_filter_dataset(pts_path, self.args.batch_size)
        np.save(pts_path.with_stem('aval_id'), aval_id.numpy())
        np.save(pts_path.with_stem('aval_rep'), aval_rep.numpy())
        return aval_pts, aval_id, aval_rep

    @torch.no_grad()
    def load_ball_pts(self, ptsPath: Path, pts, all_query_pts, aval_rep):
        ptsPath = ptsPath.with_stem('ball_dists')
        if ptsPath.exists():
            self.logger.info('Loading ball_pts from cached...')
            ball_dists = torch.from_numpy(open_memmap(ptsPath, mode='r'))
            ball_idx = torch.from_numpy(open_memmap(ptsPath.with_stem('ball_idx'), mode='r'))
            ball_knn = torch.from_numpy(open_memmap(ptsPath.with_stem('ball_knn'), mode='r'))
        else:
            self.logger.warn('Calc ball_pts using CUDA...')
            aval_rep_pts = aval_rep[None, ..., :3].cuda()
            knn_ret = knn_points(pts[None].cuda(), aval_rep_pts, K=1, return_sorted=False)
            mdn = torch.sqrt(knn_ret.dists).median().item()
            self.logger.warn('Min %f, Max %f, Mdn %f', knn_ret.dists.min().item(), knn_ret.dists.max().item(), mdn)
            mdn = max(mdn, self.tensorf.stepSize) * 2
            self.logger.warn('Using Mdn = %f', mdn)
            ball_dists, ball_idx, ball_knn = ball_query(all_query_pts[None].cuda(), aval_rep_pts, K=10, radius=mdn)
            np.save(ptsPath, (ball_dists := ball_dists.cpu()).numpy())
            np.save(ptsPath.with_stem('ball_idx'), (ball_idx := ball_idx.cpu()).numpy())
            np.save(ptsPath.with_stem('ball_knn'), (ball_knn := ball_knn.cpu()).numpy())

        ball_dists = rearrange(ball_dists, '1 (n d) k -> n d k', n=pts.shape[0])
        return ball_dists, ball_idx.view(*ball_dists.shape), ball_knn.view(*ball_dists.shape, 3)

    @torch.no_grad()
    def load_knn_pts(self, pts_path: Path, pts, aval_rep):
        pts_path = pts_path.with_stem('knn_dists')
        if pts_path.exists():
            self.logger.info('Loading knn_pts from cached...')
            knn_dists = torch.from_numpy(open_memmap(pts_path, mode='r'))
            knn_idx = torch.from_numpy(open_memmap(pts_path.with_stem('knn_idx'), mode='r'))
        else:
            self.logger.warn('Calc knn_pts using CUDA...')
            knn_dists, knn_idx, _ = knn_points(pts[None].cuda(), aval_rep[None, ..., :3].cuda(), K=100,
                                               return_sorted=True)
            np.save(pts_path, (knn_dists := knn_dists.cpu()).numpy())
            np.save(pts_path.with_stem('knn_idx'), (knn_idx := knn_idx.cpu()).numpy())
        return knn_dists.squeeze(0), knn_idx.squeeze(0)

    @torch.no_grad()
    def generate_grad(self, pts):
        ptsPath = Path(self.args.basedir, self.args.expname, 'cache', 'all_query_pts.npy')
        ptsPath.parent.mkdir(exist_ok=True)

        all_query_pts = self.load_all_query_pts(ptsPath, pts)
        aval_pts, aval_id, aval_rep = self.load_aval_pts(ptsPath)
        # ball_dists, ball_idx, ball_knn = self.load_ball_pts(ptsPath, pts, all_query_pts, aval_rep)
        knn_dists, knn_idx = self.load_knn_pts(ptsPath, pts, aval_rep)

        mask = self.tensorf.compute_validmask(all_query_pts, bitmap=True)
        dists = torch.full_like(mask, self.tensorf.stepSize, dtype=torch.float32)
        app_mask, _ = self.filter_pts(all_query_pts, dists, mask.bool())
        mask[~app_mask] = 0
        mask = rearrange(mask, '(n d) -> n d', n=pts.shape[0]).cpu()
        all_query_pts = rearrange(all_query_pts, '(n d) c -> n d c', n=pts.shape[0]).cpu()
        pts_viewdir = aval_rep[knn_idx[:, 0], None, 3:].cpu().expand(-1, all_query_pts.shape[1], -1)
        all_query_pts = torch.cat((all_query_pts, pts_viewdir), dim=-1)
        return all_query_pts, mask, dists.view(*mask.shape).cpu()

    def compute_diff_loss(self, sigma_feature: DensityFeature, rgb, orig_rgb, bit_mask):
        pts = sigma_feature.pts
        mask = pts.valid_mask
        diff = rgb[:, 0:1, :] - rgb[:, 1:, :]
        diff_ogt = orig_rgb[:, 0:1, :] - orig_rgb[:, 1:, :]

        loss = {}
        valid = torch.zeros_like(mask)
        valid.masked_scatter_(mask, torch.logical_not(pts.get_index()))
        valid = torch.logical_and(valid[:, 0:1], valid[:, 1:])
        if valid.any():
            loss['loss_diff'] = torch.nn.functional.l1_loss(diff[valid], diff_ogt[valid], reduction='mean')

        valid = torch.zeros_like(mask)
        valid.masked_scatter_(mask, pts.get_index() > 0)
        valid = torch.logical_and(valid[:, 0], torch.bitwise_and(bit_mask[:, 0], 1))
        if valid.any():
            orig_rgb = torch.zeros_like(orig_rgb)
            orig_rgb[mask] = self.tensorf.tgt_rgb
            loss['loss_pin'] = torch.nn.functional.l1_loss(orig_rgb[valid, 0], rgb[valid, 0], reduction='mean')

        return loss

    def compute_diff_loss2(self, sigma_feature: DensityFeature, rgb, orig_rgb, bit_mask):
        pts = sigma_feature.pts
        mask = pts.valid_mask
        diff = rgb[:, 0:1, :] - rgb[:, 1:, :]
        diff_ogt = orig_rgb[:, 0:1, :] - orig_rgb[:, 1:, :]

        valid = torch.logical_and(torch.bitwise_and(bit_mask[:, 0:1], 1), mask[:, 1:])
        loss = {'loss_diff': torch.nn.functional.mse_loss(diff[valid], diff_ogt[valid], reduction='mean')}

        return loss

    def train_one_batch(self, bit_mask, cur_pts):
        bit_mask = bit_mask.to(self.device)
        cur_mask = bit_mask.bool()
        sigma = torch.zeros(cur_mask.shape, device=self.device)
        rgb = torch.zeros((*cur_mask.shape, 3), device=self.device)
        orig_rgb = torch.zeros((*cur_mask.shape, 3), device=self.device)

        with torch.no_grad():
            cur_dirs = cur_pts[..., 3:].to(self.device)
            cur_pts = cur_pts[..., :3].to(self.device)
            xyz_sampled = self.tensorf.normalize_coord(cur_pts)
            sigma_feature = self.tensorf.compute_densityfeature(xyz_sampled[cur_mask])
            validsigma = self.tensorf.feature2density(sigma_feature)
            sigma.masked_scatter_(cur_mask, validsigma)
            self.target.renderModule.ignore_control = True
            orig_rgb[cur_mask] = self.tensorf.compute_radiance(xyz_sampled[cur_mask], cur_dirs[cur_mask])
            assert self.tensorf.renderModule.orig_rgb is None

        self.target.renderModule.ignore_control = False
        rgb[cur_mask] = self.tensorf.compute_radiance(xyz_sampled[cur_mask], cur_dirs[cur_mask])
        loss_dict = self.compute_diff_loss(sigma_feature, rgb, orig_rgb, bit_mask)
        loss_weight = {k: v for k in loss_dict.keys() if (v := getattr(self.args, k, None)) is not None}
        loss_total = sum(v * loss_weight.get(k, 1) for k, v in loss_dict.items())
        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()
        return {k: v.item() for k, v in loss_dict.items()}

    def poisson_editing(self, pts, mask, dists, batch_size=65536):
        args = self.args
        dataset = TensorDataset(pts, mask, dists)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        save_path = Path(args.basedir, args.expname, 'imgs_test_iters')
        pbar = tqdm(chain.from_iterable(repeat(data_loader)))
        loss = None

        grad_vars = chain(
            self.target.basis_mat_ctl.parameters(),
            self.target.app_plane_ctl.parameters(),
            self.target.app_line_ctl.parameters(),
            self.target.renderModule.mlp_control.parameters(),
        )
        self.optimizer = torch.optim.Adam(grad_vars, lr=args.lr_basis, betas=(0.9, 0.99))
        save_path.mkdir(exist_ok=True)
        (save_path / 'rgbd').mkdir(exist_ok=True)

        batch_counter = count()
        for cur_pts, cur_mask, cur_dist in pbar:
            loss_dict = self.train_one_batch(cur_mask, cur_pts)
            pbar.set_description(', '.join(f'{k}: {v:.4f}' for k, v in loss_dict.items()))

            cur_idx = next(batch_counter)
            total = self.test_dataset.all_rays.shape[0]
            test_idx = cur_idx % total
            with torch.no_grad():
                self.eval_sample(test_idx, self.test_dataset.all_rays[test_idx], save_path, f'it{cur_idx:06d}_',
                                 N_samples=-1, white_bg=self.test_dataset.white_bg, save_GT=False)

        return loss

    def merge(self):
        if self.args.export_mesh:
            self.export_mesh(prefix=os.path.basename(self.args.datadir))
        self.target.renderModule.enable_trainable_control()
        self.target.enable_trainable_control()
        self.tensorf.add_merge_target(self.target, self.args.density_gain, self.args.render_gap)
        if self.args.export_mesh:
            self.export_mesh(self.target)
            self.export_mesh()
        pts = self.export_pointcloud()
        pts, mask, dists = self.generate_grad(pts)
        if self.args.export_mesh:
            self.render_test()
        self.poisson_editing(pts, mask, dists)


def config_parser(parser):
    from dataLoader import dataset_dict
    from models import MODEL_ZOO

    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='./log', help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', help='input data directory')

    parser.add_argument('--downsample_test', type=float, default=1.0)
    parser.add_argument('--matrix', type=float, nargs='+', default=())
    parser.add_argument("--lr_basis", type=float, default=1e-3, help='learning rate')
    parser.add_argument("--batch_size", type=int, default=8192)

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

    # logging/saving options
    parser.add_argument("--render_gap", type=float, help='render step size gap for target')
    parser.add_argument("--density_gain", type=float, default=1, help='density gain for source')
    parser.add_argument("--delta_scale", type=float, default=1, help='weight for loss_diff')
    parser.add_argument('--nSamples', type=int, default=1e6, help='sample point each ray, pass 1e6 if automatic adjust')

    # loss weight
    parser.add_argument("--loss_diff", type=float, help='weight for loss_diff')


def main(args):
    assert args.render_only
    with ThreadPool() as pool:
        merger = Merger(args, pool)
        merger.merge()
