import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from scipy.spatial import ConvexHull, Delaunay
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm

from dataLoader import dataset_dict
from models import MODEL_ZOO
from models.palette.Additive_mixing_layers_extraction import Hull_Simplification_determined_version, \
    Hull_Simplification_old, _DCP_PT_RET
from models.palette.GteDistPointTriangle import DCPPointTriangle
from opt import config_parser
from renderer import OctreeRender_trilinear_fast, evaluation, evaluation_path
from utils import convert_sdf_samples_to_ply, N_to_reso, cal_n_samples, TVLoss, sort_palette


class SimpleSampler:
    def __init__(self, train_dataset, batch):
        total = train_dataset.all_rays.shape[0]
        w, h = train_dataset.img_wh
        self.rgb_shape = (train_dataset.all_sems.shape[0], h, w) if len(train_dataset.all_sems) else None
        self.dataset = train_dataset
        self.batch = batch
        self.curr = total
        self.ids = np.random.permutation(total)

    def apply_filter(self, func, *args, **kwargs):
        mask_filtered = func(self.dataset.all_rays[self.ids], *args, **kwargs)
        self.ids = self.ids[mask_filtered]
        self.curr = self.ids.shape[0]

    def nextids(self):
        total = self.ids.shape[0]
        self.curr += self.batch
        if self.curr + self.batch > total:
            np.random.shuffle(self.ids)
            self.curr = 0
        return self.ids[self.curr:self.curr + self.batch]

    def getbatch(self, device):
        ids = self.nextids()
        if self.rgb_shape is not None:
            sems = torch.from_numpy(np.stack(np.unravel_index(ids, self.rgb_shape), axis=-1)).to(device)
            sems = self.dataset.sample_sems(sems)
            return self.dataset.all_rays[ids].to(device), self.dataset.all_rgbs[ids].to(device), sems
        else:
            return self.dataset.all_rays[ids].to(device), self.dataset.all_rgbs[ids].to(device)


class Trainer:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.renderer = OctreeRender_trilinear_fast

        # init dataset
        dataset = dataset_dict[args.dataset_name]
        self.train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False,
                                     semantic_type=args.semantic_type)
        self.test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True,
                                    semantic_type=args.semantic_type, pca=self.train_dataset.pca)

        # init parameters
        # tensorVM, renderer = init_parameters(args, train_dataset.scene_bbox.to(device), reso_list[0])
        self.aabb = self.train_dataset.scene_bbox.to(self.device)
        self.reso_cur = N_to_reso(args.N_voxel_init, self.aabb)
        self.reso_mask = None
        self.nSamples = min(args.nSamples, cal_n_samples(self.reso_cur, args.step_ratio))
        self.palette, hull_vertices = self.build_palette(args.datadir)
        self.hull = ConvexHull(hull_vertices)
        self.de = Delaunay(hull_vertices)
        if len(self.train_dataset.all_sems):
            self.palette_sem = self.build_sem_palette(len(self.palette))
            print(self.palette_sem)
        else:
            self.palette_sem = None

        # linear in logrithmic space
        self.N_voxel_list = torch.round(torch.exp(torch.linspace(
            np.log(args.N_voxel_init), np.log(args.N_voxel_final), len(args.upsamp_list) + 1))).long().tolist()[1:]
        self.tvreg = TVLoss()
        self.Ortho_reg_weight = args.Ortho_weight
        print("initial Ortho_reg_weight", self.Ortho_reg_weight)
        self.L1_reg_weight = args.L1_weight_inital
        print("initial L1_reg_weight", self.L1_reg_weight)
        self.TV_weight_density, self.TV_weight_app = args.TV_weight_density, args.TV_weight_app
        print(f"initial TV_weight density: {self.TV_weight_density} appearance: {self.TV_weight_app}")

        if args.lr_decay_iters > 0:
            self.lr_factor = args.lr_decay_target_ratio ** (1 / args.lr_decay_iters)
        else:
            args.lr_decay_iters = args.n_iters
            self.lr_factor = args.lr_decay_target_ratio ** (1 / args.n_iters)
        print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)

        self.args = args
        self.optimizer = None
        self.summary_writer = None
        self.trainingSampler = None
        self.logger = logging.getLogger(type(self).__name__)
        self.ones = torch.ones((1, 1), device=self.device)

    def recon_with_palette(self, palette, points):
        hull = ConvexHull(palette)
        de = Delaunay(hull.points[hull.vertices].clip(0.0, 1.0))
        ind = de.find_simplex(points, tol=1e-8)
        for i in tqdm(np.nonzero(ind < 0)[0], desc='recon_with_palette'):
            dist_list = [_DCP_PT_RET(**DCPPointTriangle(points[i], hull.points[j])) for j in hull.simplices]
            idx = np.fromiter((j.distance for j in dist_list), dtype=np.float_, count=len(dist_list)).argmin()
            points[i] = dist_list[idx].closest
        return points

    def build_palette(self, filepath):
        filepath = Path(filepath)
        rgbs = self.train_dataset.all_rgbs
        if self.train_dataset.white_bg:
            fg = torch.lt(rgbs, 1.).any(dim=-1)
            rgbs = rgbs[fg]
        rgbs = rgbs.to(device='cpu', dtype=torch.double).numpy()
        hull = ConvexHull(rgbs)
        return sort_palette(rgbs, Hull_Simplification_determined_version(
            rgbs, filepath.stem + "-convexhull_vertices", error_thres=1. / 256.)), hull.points[hull.vertices]

    def build_sem_palette(self, E_vertice_num=10):
        w, h = self.train_dataset.img_wh
        if self.train_dataset.white_bg:
            fg = torch.lt(self.train_dataset.all_rgbs, 1.).any(dim=-1)
        else:
            fg = torch.ones(self.train_dataset.all_rgbs.shape[:-1], dtype=torch.bool)
        coord = rearrange(fg, '(n h w) -> n h w', h=h, w=w).nonzero()
        rgbs = self.train_dataset.sample_sems(coord).to(device='cpu', dtype=torch.double).numpy()
        return sort_palette(rgbs, Hull_Simplification_old(rgbs, E_vertice_num=E_vertice_num, error_thres=1. / 256.))

    def pick_palette(self, root=None):
        from tkcolorpicker import askcolor
        if root is None:
            import tkinter as tk
            import tkinter.ttk as ttk
            root = tk.Tk()
            style = ttk.Style(root)
            style.theme_use('clam')
        palette = (np.asarray(self.palette) * 255).astype(np.uint8)
        palette = [askcolor(p.tolist(), root)[0] for p in palette]
        palette = np.asarray(palette) / 255.
        return [tuple(p) for p in palette.tolist()]

    def build_network(self):
        args = self.args
        n_lamb_sigma = args.n_lamb_sigma
        n_lamb_sh = args.n_lamb_sh
        near_far = self.train_dataset.near_far
        if args.ckpt is not None:
            ckpt = torch.load(args.ckpt, map_location=self.device)
            kwargs = ckpt['kwargs']
            kwargs.update({'device': self.device,
                           'palette': self.pick_palette()})
            tensorf = MODEL_ZOO[args.model_name](**kwargs)
            tensorf.load(ckpt)
        else:
            palette = self.palette
            if self.palette_sem is not None:
                palette = (palette, self.palette_sem)
            tensorf = MODEL_ZOO[args.model_name](
                self.aabb, self.reso_cur, self.device,
                density_n_comp=n_lamb_sigma, appearance_n_comp=n_lamb_sh,
                app_dim=args.data_dim_color, near_far=near_far,
                shadingMode=args.shadingMode, alphaMask_thres=args.alpha_mask_thre,
                density_shift=args.density_shift, distance_scale=args.distance_scale,
                pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe,
                featureC=args.featureC, step_ratio=args.step_ratio,
                fea2denseAct=args.fea2denseAct, palette=palette)

        return tensorf

    def outsidehull_points_distance(self, inp_points):
        points = inp_points.detach().to('cpu', dtype=torch.double, copy=True).numpy()
        ind = self.de.find_simplex(points, tol=1e-8)
        for i in np.nonzero(ind < 0)[0]:
            dist_list = [_DCP_PT_RET(**DCPPointTriangle(points[i], self.hull.points[j])) for j in self.hull.simplices]
            idx = np.fromiter((j.distance for j in dist_list), dtype=np.float_, count=len(dist_list)).argmin()
            points[i] = dist_list[idx].closest
        points = torch.from_numpy(points).to(inp_points.device, dtype=inp_points.dtype)
        return F.mse_loss(inp_points, points, reduction='sum')

    def plt_loss(self, plt_map, gt_train, palette, weight=1.):  # palette in 3xN
        pix, opq = plt_map[..., :3], plt_map[..., 3:]
        E_opaque = F.mse_loss(opq, self.ones.expand_as(opq), reduction='mean')
        loss = F.mse_loss(pix, gt_train, reduction='mean')
        E_opaque = E_opaque - 1e3 * self.outsidehull_points_distance(palette.T)
        E_opaque = E_opaque - 5e3 * torch.linalg.vector_norm(palette[:, -1])
        return loss * weight, E_opaque * weight

    def train_one_batch(self, tensorf, iteration, rays_train, *batch_gt):
        args = self.args
        white_bg = self.train_dataset.white_bg
        ndc_ray = args.ndc_ray
        n_palette = getattr(tensorf.renderModule, 'n_palette', 1)
        palette = (render.palette for render in tensorf.renderModule
                   ) if n_palette > 1 else (getattr(tensorf.renderModule, 'palette'),)

        # rgb_map, alphas_map, depth_map, weights, uncertainty
        rgb_map, alphas_map, depth_map, weights, uncertainty = self.renderer(
            rays_train, tensorf, chunk=args.batch_size, N_samples=self.nSamples, white_bg=white_bg,
            ndc_ray=ndc_ray, device=self.device, is_train=True)

        rgb_map = torch.tensor_split(rgb_map, n_palette, dim=-1)
        # batch_gt = (batch_gt[0],)
        loss_weight = (1., 0.1)
        loss, E_opaque = zip(*map(self.plt_loss, rgb_map, batch_gt, palette, loss_weight))

        # loss
        total_loss = sum(loss) - sum(E_opaque) / 375
        loss, *_ = loss
        if self.Ortho_reg_weight > 0:
            loss_reg = tensorf.vector_comp_diffs()
            total_loss += self.Ortho_reg_weight * loss_reg
            self.summary_writer.add_scalar('train/reg', loss_reg.detach().item(), global_step=iteration)
        if self.L1_reg_weight > 0:
            loss_reg_L1 = tensorf.density_L1()
            total_loss += self.L1_reg_weight * loss_reg_L1
            self.summary_writer.add_scalar('train/reg_l1', loss_reg_L1.detach().item(), global_step=iteration)

        if self.TV_weight_density > 0:
            self.TV_weight_density *= self.lr_factor
            loss_tv = tensorf.TV_loss_density(self.tvreg) * self.TV_weight_density
            total_loss = total_loss + loss_tv
            self.summary_writer.add_scalar('train/reg_tv_density', loss_tv.detach().item(), global_step=iteration)
        if self.TV_weight_app > 0:
            self.TV_weight_app *= self.lr_factor
            loss_tv = tensorf.TV_loss_app(self.tvreg) * self.TV_weight_app
            total_loss = total_loss + loss_tv
            self.summary_writer.add_scalar('train/reg_tv_app', loss_tv.detach().item(), global_step=iteration)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        loss = loss.detach().item()
        return loss

    def update_grid_resolution(self, tensorf, iteration):
        args = self.args
        # init resolution
        upsamp_list = args.upsamp_list
        update_AlphaMask_list = args.update_AlphaMask_list

        if iteration in update_AlphaMask_list:
            if self.reso_cur[0] * self.reso_cur[1] * self.reso_cur[2] < 256 ** 3:  # update volume resolution
                self.reso_mask = self.reso_cur
            new_aabb = tensorf.updateAlphaMask(tuple(self.reso_mask))
            if iteration == update_AlphaMask_list[0]:
                tensorf.shrink(new_aabb)
                # tensorVM.alphaMask = None
                self.L1_reg_weight = args.L1_weight_rest
                print("continuing L1_reg_weight", self.L1_reg_weight)

            if not args.ndc_ray and iteration == update_AlphaMask_list[1]:
                # filter rays outside the bbox
                self.trainingSampler.apply_filter(tensorf.filtering_rays)

        if iteration in upsamp_list:
            n_voxels = self.N_voxel_list.pop(0)
            self.reso_cur = N_to_reso(n_voxels, tensorf.aabb)
            self.nSamples = min(args.nSamples, cal_n_samples(self.reso_cur, args.step_ratio))
            tensorf.upsample_volume_grid(self.reso_cur)

            if args.lr_upsample_reset:
                print("reset lr to initial")
                lr_scale = 1  # 0.1 ** (iteration / args.n_iters)
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
            grad_vars = tensorf.get_optparam_groups(args.lr_init * lr_scale, args.lr_basis * lr_scale)
            self.optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    def create_summary_writer(self, logfolder=None):
        args = self.args

        if logfolder is None:
            logfolder = Path(args.basedir, args.expname)
            if args.add_timestamp:
                logfolder = logfolder / datetime.now().strftime("-%Y%m%d-%H%M%S")
        else:
            logfolder = Path(logfolder)

        # init log file
        os.makedirs(logfolder, exist_ok=True)
        os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
        os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
        os.makedirs(f'{logfolder}/rgba', exist_ok=True)
        self.summary_writer = SummaryWriter(os.fspath(logfolder))

    def reconstruction(self):
        args = self.args
        white_bg = self.train_dataset.white_bg

        self.create_summary_writer()
        tensorf = self.build_network()

        grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
        self.optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

        torch.cuda.empty_cache()
        PSNRs, PSNRs_test = [], [0]

        self.trainingSampler = SimpleSampler(self.train_dataset, args.batch_size)
        if not args.ndc_ray:
            self.trainingSampler.apply_filter(tensorf.filtering_rays, bbox_only=True)

        pbar = trange(args.n_iters, miniters=args.progress_refresh_rate, file=sys.stdout)
        for iteration in pbar:
            batch_train = self.trainingSampler.getbatch(device=self.device)
            loss = self.train_one_batch(tensorf, iteration, *batch_train)

            PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
            self.summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=iteration)
            self.summary_writer.add_scalar('train/mse', loss, global_step=iteration)

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * self.lr_factor

            # Print the current values of the losses.
            if iteration % args.progress_refresh_rate == 0:
                pbar.set_description(
                    f'Iteration {iteration:05d}:'
                    + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                    + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                    + f' mse = {loss:.6f}'
                )
                PSNRs = []

            if iteration % args.vis_every == args.vis_every - 1 and args.N_vis != 0:
                try:
                    savePath = Path(self.summary_writer.log_dir, 'imgs_vis')
                    PSNRs_test = evaluation(self.test_dataset, tensorf, args, self.renderer, os.fspath(savePath),
                                            N_vis=args.N_vis, prtx=f'{iteration:06d}_', N_samples=self.nSamples,
                                            white_bg=white_bg, ndc_ray=args.ndc_ray, compute_extra_metrics=False)
                    self.summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)
                except Exception as e:
                    self.logger.warning(f'Evaluation failed: {e}')

            self.update_grid_resolution(tensorf, iteration)

        tensorf.save(f'{self.summary_writer.log_dir}/{args.expname}.th')
        PSNRs_test = self.render_test(tensorf)
        self.summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=pbar.total)

    @torch.no_grad()
    def export_mesh(self):
        args = self.args
        ckpt = torch.load(args.ckpt, map_location=self.device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device': self.device})
        tensorf = MODEL_ZOO[args.model_name](**kwargs)
        tensorf.load(ckpt)

        alpha, _ = tensorf.getDenseAlpha()
        convert_sdf_samples_to_ply(alpha.cpu(), f'{args.ckpt[:-3]}.ply', bbox=tensorf.aabb.cpu(), level=0.005)

    @torch.no_grad()
    def render_test(self, tensorf):
        args = self.args
        white_bg = self.test_dataset.white_bg
        ndc_ray = args.ndc_ray

        if self.summary_writer is not None:
            logfolder = Path(self.summary_writer.log_dir)
        else:
            logfolder = Path(os.path.dirname(args.ckpt), args.expname)
            if not os.path.exists(args.ckpt):
                self.logger.warning('the ckpt path does not exists!!')

        PSNRs_test = None
        if args.render_train:
            filePath = logfolder / 'imgs_train_all'
            PSNRs_test = evaluation(self.train_dataset, tensorf, args, self.renderer, os.fspath(filePath),
                                    N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=self.device)
            print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

        if args.render_test:
            filePath = logfolder / 'imgs_test_all'
            PSNRs_test = evaluation(self.test_dataset, tensorf, args, self.renderer, os.fspath(filePath),
                                    N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=self.device)
            print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

        PSNRs_path = None
        if args.render_path:
            filePath = logfolder / 'imgs_path_all'
            c2ws = self.test_dataset.render_path
            # c2ws = test_dataset.poses
            print('========>', c2ws.shape)
            PSNRs_path = evaluation_path(self.test_dataset, tensorf, c2ws, self.renderer, os.fspath(filePath),
                                         N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=self.device)

        return PSNRs_test or PSNRs_path


def main(args):
    print(args)

    trainer = Trainer(args)
    if args.export_mesh:
        trainer.export_mesh()

    if args.render_only and (args.render_test or args.render_path):
        trainer.render_test(trainer.build_network())
    else:
        trainer.reconstruction()


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    np.random.seed(np.bitwise_xor(*np.atleast_1d(np.asarray(torch.seed(), dtype=np.uint64)).view(np.uint32)).item())
    main(config_parser())
