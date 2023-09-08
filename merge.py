import argparse
import logging
import os
import shutil
from contextlib import suppress
from itertools import repeat, chain, count
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tempfile import TemporaryDirectory

import imageio
import mmengine
import numpy as np
import torch
from einops import rearrange
from numpy.lib.format import open_memmap
from pytorch3d.ops import knn_points, ball_query
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm, trange
from trimesh import PointCloud

from lib import utils, dvgo, dcvgo, dmpigo
from lib.colorRF import DensityFeature, ColorVoxE
from lib.dvgo import DirectVoxGO
from lib.utils import convert_sdf_samples_to_ply
from run import config_parser, seed_everything, load_everything, render_viewpoints


class Evaluator:
    def __init__(self, args, cfg, device):
        self.args = args
        self.cfg = cfg
        self.device = device

        # load images / poses / camera settings / data split
        self.data_dict = load_everything(args=args, cfg=cfg)

        # load model for rendring
        if args.render_test or args.render_train or args.render_video:
            self.render_viewpoints_kwargs, self.ckpt_name = self.build_network()

    def build_network(self, ckpt_path=None, expname=None):
        args = self.args
        cfg = self.cfg
        model_class = ColorVoxE if expname is None else DirectVoxGO
        expname = expname or cfg.expname

        if not ckpt_path:
            if args.ft_path:
                ckpt_path = args.ft_path
            else:
                ckpt_path = os.path.join(cfg.basedir, expname, 'fine_last.tar')

        ckpt_name = ckpt_path.split('/')[-1][:-4]

        assert not (cfg.data.ndc or cfg.data.unbounded_inward)
        model = utils.load_model(model_class, ckpt_path).to(self.device)
        stepsize = cfg.fine_model_and_render.stepsize
        render_viewpoints_kwargs = {
            'model': model,
            'ndc': cfg.data.ndc,
            'render_kwargs': {
                'near': self.data_dict['near'],
                'far': self.data_dict['far'],
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': stepsize,
                'inverse_y': cfg.data.inverse_y,
                'flip_x': cfg.data.flip_x,
                'flip_y': cfg.data.flip_y,
                'render_depth': True,
            },
        }
        return render_viewpoints_kwargs, ckpt_name

    def render_test(self):
        args = self.args
        data_dict = self.data_dict
        cfg = self.cfg
        ckpt_name = self.ckpt_name
        render_viewpoints_kwargs = self.render_viewpoints_kwargs

        # render trainset and eval
        if args.render_train:
            testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_train_{ckpt_name}')
            os.makedirs(testsavedir, exist_ok=True)
            print('All results are dumped into', testsavedir)
            rgbs, depths, bgmaps = render_viewpoints(
                render_poses=data_dict['poses'][data_dict['i_train']],
                HW=data_dict['HW'][data_dict['i_train']],
                Ks=data_dict['Ks'][data_dict['i_train']],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_train']],
                savedir=testsavedir, dump_images=args.dump_images,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                **render_viewpoints_kwargs)
            imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(1 - depths / np.max(depths)),
                             fps=30, quality=8)

        # render testset and eval
        if args.render_test:
            testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_test_{ckpt_name}')
            os.makedirs(testsavedir, exist_ok=True)
            print('All results are dumped into', testsavedir)
            rgbs, depths, bgmaps = render_viewpoints(
                render_poses=data_dict['poses'][data_dict['i_test']],
                HW=data_dict['HW'][data_dict['i_test']],
                Ks=data_dict['Ks'][data_dict['i_test']],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
                savedir=testsavedir, dump_images=args.dump_images,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                **render_viewpoints_kwargs)
            imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(1 - depths / np.max(depths)),
                             fps=30, quality=8)

        # render video
        if args.render_video:
            testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_video_{ckpt_name}')
            os.makedirs(testsavedir, exist_ok=True)
            print('All results are dumped into', testsavedir)
            rgbs, depths, bgmaps = render_viewpoints(
                render_poses=data_dict['render_poses'],
                HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                render_factor=args.render_video_factor,
                render_video_flipy=args.render_video_flipy,
                render_video_rot90=args.render_video_rot90,
                savedir=testsavedir, dump_images=args.dump_images,
                **render_viewpoints_kwargs)
            imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
            import matplotlib.pyplot as plt
            depths_vis = depths * (1 - bgmaps) + bgmaps
            dmin, dmax = np.percentile(depths_vis[bgmaps < 0.1], q=[5, 95])
            depth_vis = plt.get_cmap('rainbow')(1 - np.clip((depths_vis - dmin) / (dmax - dmin), 0, 1)
                                                ).squeeze()[..., :3]
            imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(depth_vis), fps=30, quality=8)

        print('Done')


class Merger(Evaluator):
    def __init__(self, args, pool):
        self.args = args

        # init enviroment
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        seed_everything(args.seed)
        cfg = mmengine.Config.fromfile(args.config)

        # self.target = self.build_network(self.args.ckpt)
        self.optimizer = None

        self.logger = logging.getLogger(type(self).__name__)
        super().__init__(args, cfg, device)
        kwargs, _ = self.build_network(expname=cfg.target)
        self.tensorf.add_merge_target(kwargs['model'], cfg.matrix)

    @property
    def tensorf(self):
        return self.render_viewpoints_kwargs['model']

    @property
    def target(self):
        return self.tensorf.target

    @torch.no_grad()
    def export_mesh(self, tensorf=None, prefix=None):
        args = self.args
        cfg = self.cfg
        if tensorf is None:
            tensorf = self.render_viewpoints_kwargs['model']
            if not prefix:
                prefix = os.path.basename(args.config)
        elif not prefix:
            prefix = os.path.basename(args.ckpt)
        save_path = Path(cfg.basedir, cfg.expname, prefix).with_suffix('.ply')
        gridSize = tensorf.world_size * 2
        alpha, _ = tensorf.getDenseAlpha(gridSize, **self.render_viewpoints_kwargs['render_kwargs'])
        bbox = getattr(tensorf, 'merged_aabb', tensorf.aabb)
        convert_sdf_samples_to_ply(alpha.cpu(), save_path, bbox=bbox.cpu(), level=0.005)

    @torch.no_grad()
    def export_pointcloud(self, tensorf=None):
        args = self.args
        cfg = self.cfg
        save_path = Path(cfg.basedir, cfg.expname, f'render_test_{self.ckpt_name}')
        if tensorf is None:
            tensorf = self.render_viewpoints_kwargs['model']
            save_path = save_path / os.path.basename(args.config)
        else:
            save_path = save_path / os.path.basename(args.ckpt)
        save_path = save_path.with_stem(save_path.stem + '_pc').with_suffix('.ply')
        gridSize = tensorf.world_size  # * 2
        alpha, xyz = tensorf.getDenseAlpha(gridSize, **self.render_viewpoints_kwargs['render_kwargs'])
        pts = xyz[alpha > 0.005]
        pc = PointCloud(pts.cpu().numpy())
        pc.export(save_path)
        return pts

    @torch.no_grad()
    def load_all_query_pts(self, pts_path: Path, pts):
        assert pts_path.exists()

        self.logger.info('Loading all_query_pts from cached into GPU...')
        all_query_pts = torch.from_numpy(open_memmap(pts_path, mode='r')).to(self.device)
        with suppress(Exception):
            if torch.allclose(pts, all_query_pts.view(-1, 7, 3)[:, 0]):
                return all_query_pts
        self.logger.warn('Cached query_pts mismatch. Regenerating...')
        return all_query_pts

    @torch.no_grad()
    def load_aval_pts(self, pts_path: Path):
        pts_path = pts_path.with_stem('aval_pts')
        assert pts_path.exists()
        self.logger.info('Loading aval_pts from cached...')
        with suppress(Exception):
            aval_pts = torch.from_numpy(open_memmap(pts_path, mode='r'))
            aval_id = torch.from_numpy(open_memmap(pts_path.with_stem('aval_id'), mode='r'))
            aval_rep = torch.from_numpy(open_memmap(pts_path.with_stem('aval_rep'), mode='r'))
            return aval_pts, aval_id, aval_rep

    @torch.no_grad()
    def load_ball_pts(self, ptsPath: Path, pts, all_query_pts, aval_rep):
        ptsPath = ptsPath.with_stem('ball_dists')
        assert ptsPath.exists()
        self.logger.info('Loading ball_pts from cached...')
        ball_dists = torch.from_numpy(open_memmap(ptsPath, mode='r'))
        ball_idx = torch.from_numpy(open_memmap(ptsPath.with_stem('ball_idx'), mode='r'))
        ball_knn = torch.from_numpy(open_memmap(ptsPath.with_stem('ball_knn'), mode='r'))

        ball_dists = rearrange(ball_dists, '1 (n d) k -> n d k', n=pts.shape[0])
        return ball_dists, ball_idx.view(*ball_dists.shape), ball_knn.view(*ball_dists.shape, 3)

    @torch.no_grad()
    def load_knn_pts(self, pts_path: Path, pts, aval_rep):
        pts_path = pts_path.with_stem('knn_dists')
        assert pts_path.exists()
        self.logger.info('Loading knn_pts from cached...')
        knn_dists = torch.from_numpy(open_memmap(pts_path, mode='r'))
        knn_idx = torch.from_numpy(open_memmap(pts_path.with_stem('knn_idx'), mode='r'))

        return knn_dists.squeeze(0), knn_idx.squeeze(0)

    @torch.no_grad()
    def generate_grad(self, pts):
        ptsPath = Path(self.cfg.basedir, self.cfg.expname, 'cache', 'all_query_pts.npy')
        ptsPath.parent.mkdir(exist_ok=True)

        all_query_pts = self.load_all_query_pts(ptsPath, pts)
        aval_pts, aval_id, aval_rep = self.load_aval_pts(ptsPath)
        # ball_dists, ball_idx, ball_knn = self.load_ball_pts(ptsPath, pts, all_query_pts, aval_rep)
        knn_dists, knn_idx = self.load_knn_pts(ptsPath, pts, aval_rep)

        mask = torch.from_numpy(open_memmap(ptsPath.with_stem('mask'), mode='r'))
        stepSize = self.render_viewpoints_kwargs['render_kwargs']['stepsize'] * self.tensorf.voxel_size_ratio
        dists = torch.full_like(mask, stepSize, dtype=torch.float32)
        all_query_pts = rearrange(all_query_pts, '(n d) c -> n d c', d=7).cpu()
        pts_viewdir = aval_rep[knn_idx[:, 0], None, 3:].cpu().expand(-1, all_query_pts.shape[1], -1)
        all_query_pts = torch.cat((all_query_pts, pts_viewdir), dim=-1)
        return all_query_pts, mask, dists.view(*mask.shape).cpu()

    def compute_diff_loss(self, mask, rgb, orig_rgb, bit_mask):
        idx = self.tensorf.idx
        diff = rgb[:, 0:1, :] - rgb[:, 1:, :]
        diff_ogt = orig_rgb[:, 0:1, :] - orig_rgb[:, 1:, :]

        loss = {}
        valid = torch.zeros_like(mask)
        valid.masked_scatter_(mask, torch.logical_not(idx))
        valid = torch.logical_and(valid[:, 0:1], valid[:, 1:])
        if valid.any():
            loss['loss_diff'] = torch.nn.functional.l1_loss(diff[valid], diff_ogt[valid], reduction='mean')

        valid = torch.zeros_like(mask)
        valid.masked_scatter_(mask, idx > 0)
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
        # sigma = torch.zeros(cur_mask.shape, device=self.device)
        rgb = torch.zeros((*cur_mask.shape, 3), device=self.device)
        orig_rgb = torch.zeros((*cur_mask.shape, 3), device=self.device)

        with torch.no_grad():
            cur_dirs = cur_pts[..., 3:].to(self.device)
            cur_pts = cur_pts[..., :3].to(self.device)
            # xyz_sampled = self.tensorf.normalize_coord(cur_pts)
            # sigma_feature = self.tensorf.compute_densityfeature(xyz_sampled[cur_mask])
            # validsigma = self.tensorf.feature2density(sigma_feature)
            # sigma.masked_scatter_(cur_mask, validsigma)
            self.tensorf.compute_alpha(cur_pts[cur_mask])
            orig_rgb[cur_mask] = self.tensorf.compute_radiance(cur_pts[cur_mask], cur_dirs[cur_mask], orig=True)
            # assert self.tensorf.renderModule.orig_rgb is None

        rgb[cur_mask] = self.tensorf.compute_radiance(cur_pts[cur_mask], cur_dirs[cur_mask], orig=False)
        loss_dict = self.compute_diff_loss(cur_mask, rgb, orig_rgb, bit_mask)
        loss_weight = {k: v for k in loss_dict.keys() if (v := getattr(self.args, k, None)) is not None}
        loss_total = sum(v * loss_weight.get(k, 1) for k, v in loss_dict.items())
        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()
        return {k: v.item() for k, v in loss_dict.items()}

    def poisson_editing(self, pts, mask, dists, batch_size=65536):
        args = self.args
        cfg = self.cfg
        dataset = TensorDataset(pts, mask, dists)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        save_path = Path(cfg.basedir, cfg.expname, 'imgs_test_iters')
        pbar = tqdm(chain.from_iterable(repeat(data_loader)))
        loss = None

        grad_vars = chain(
            self.target.k0_new.parameters(),
            self.target.rgbnet_new.parameters(),
        )
        self.optimizer = torch.optim.Adam(grad_vars, lr=args.lr_basis, betas=(0.9, 0.99))
        save_path.mkdir(exist_ok=True)
        (save_path / 'rgbd').mkdir(exist_ok=True)

        batch_counter = count()
        for cur_pts, cur_mask, cur_dist in pbar:
            loss_dict = self.train_one_batch(cur_mask, cur_pts)
            pbar.set_description(', '.join(f'{k}: {v:.4f}' for k, v in loss_dict.items()))

            # cur_idx = next(batch_counter)
            # total = self.test_dataset.all_rays.shape[0]
            # test_idx = cur_idx % total
            # with torch.no_grad():
            #     self.eval_sample(test_idx, self.test_dataset.all_rays[test_idx], save_path, f'it{cur_idx:06d}_',
            #                      N_samples=-1, white_bg=self.test_dataset.white_bg, save_GT=False)

        return loss

    def merge(self):
        export_mesh = getattr(self.args, 'export_mesh', None)
        # if export_mesh:
        #     self.export_mesh(prefix=os.path.basename(self.args.datadir))
        # self.target.renderModule.enable_trainable_control()
        # self.target.enable_trainable_control()
        # self.tensorf.add_merge_target(self.target, self.args.density_gain, self.args.render_gap)
        if export_mesh:
            # self.export_mesh(self.target)
            self.export_mesh()
        pts = self.export_pointcloud()
        pts, mask, dists = self.generate_grad(pts)
        if self.args.export_mesh:
            self.render_test()
        self.poisson_editing(pts, mask, dists)


def config_parser():
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true',
                        help='do not reload optimizer state from saved ckpt')
    parser.add_argument("--ft_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--export_bbox_and_cams_only", type=str, default='',
                        help='export scene bbox and camera poses for debugging and 3d visualization')
    parser.add_argument("--export_coarse_only", type=str, default='')

    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_train", action='store_true')
    parser.add_argument("--render_video", action='store_true')
    parser.add_argument("--export_mesh", action="store_true")
    parser.add_argument("--render_video_flipy", action='store_true')
    parser.add_argument("--render_video_rot90", default=0, type=int)
    parser.add_argument("--render_video_factor", type=float, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--dump_images", action='store_true')
    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=500,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=100000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--lr_basis", type=float, default=1e-3, help='learning rate')
    return parser


def config_parser_merge(parser):
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='./log', help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', help='input data directory')

    parser.add_argument('--downsample_test', type=float, default=1.0)
    parser.add_argument('--matrix', type=float, nargs='+', default=())
    parser.add_argument("--batch_size", type=int, default=8192)

    # network decoder
    parser.add_argument("--semantic_type", type=str, default='None', help='semantic type')

    parser.add_argument("--ckpt", type=str, default=None, help='specific weights npy file to reload for coarse network')

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


if __name__ == '__main__':
    # load setup
    main(config_parser().parse_args())
