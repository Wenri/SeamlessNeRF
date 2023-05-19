import logging
import os
import sys
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Optional

import imageio
import numpy as np
import torch
from einops import rearrange
from tqdm.asyncio import tqdm

from dataLoader.ray_utils import get_rays, ndc_rays_blender
from extra.compute_metrics import rgb_ssim, rgb_lpips
from renderer import visualize_rgb, visualize_palette, OctreeRender_trilinear_fast
from utils import visualize_depth_numpy


class Evaluator:
    def __init__(self, tensorf, args, test_dataset, train_dataset=None, summary_writer=None, pool=None):
        self.tensorf = tensorf
        self.renderer = OctreeRender_trilinear_fast
        self.args = args
        self.pool = pool
        self.test_dataset = test_dataset
        self.alt_dataset = train_dataset
        self.summary_writer = summary_writer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_extra_metrics = False

        self.n_palette = getattr(tensorf.renderModule, 'n_palette', 1)
        if self.n_palette > 1:
            palette = [render.palette for render in tensorf.renderModule]
            plt_names = tensorf.renderModule.PLT_NAMES
        else:
            palette = (getattr(tensorf.renderModule, 'palette', None),)
            plt_names = ('RGB',)

        self.palette, self.plt_names = palette, plt_names
        self.PSNRs, self.rgb_maps, self.depth_maps = [], [], []
        self.ssims, self.l_alex, self.l_vgg = [], [], []
        self.logger = logging.getLogger(type(self).__name__)

    def eval_sample(self, idx, samples, savePath: Path, prtx='', N_samples=-1, white_bg=False, ndc_ray=False,
                    save_GT=True):
        test_dataset = self.test_dataset
        tensorf = self.tensorf

        W, H = test_dataset.img_wh
        rays = samples.view(-1, samples.shape[-1])

        plt_map, _, depth_map, _, _ = self.renderer(rays, tensorf, chunk=4096, N_samples=N_samples, ndc_ray=ndc_ray,
                                                    white_bg=white_bg, device=self.device, args=self.args)
        rgb_map, depth_map = self.clamp_rgb_depth(plt_map[..., :3], depth_map)
        gt_vis = []

        if len(test_dataset.all_rgbs):
            gt_rgb = test_dataset.all_rgbs[idx].view(H, W, 3)
            if save_GT:
                gt_vis.append(gt_rgb)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            self.PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))
            self.compute_metrics(gt_rgb, rgb_map)

        if save_GT and len(getattr(test_dataset, 'all_sems', ())):
            c1, c2 = torch.ones(test_dataset.all_rgbs.shape[1:-1], dtype=torch.bool).nonzero(as_tuple=True)
            c0 = torch.full_like(c1, idx)
            coord = torch.stack((c0, c1, c2), dim=-1)
            gt_sem = test_dataset.sample_sems(coord).view(H, W, 3)
            gt_vis.append(gt_sem)

        if gt_vis:
            gt_vis = (torch.cat(gt_vis, dim=1).numpy() * 255).astype('uint8')
            imageio.imwrite(savePath / f'{prtx}{idx:03d}_GT.png', gt_vis)

        for rgb, plt, name in zip(
                torch.tensor_split(plt_map.cpu(), self.n_palette, dim=-1), self.palette, self.plt_names):
            visualize_rgb(depth_map, rgb[..., :3].clamp(0.0, 1.0).reshape(H, W, 3), savePath,
                          prtx=f'{prtx}{idx:03d}_{name}', pool=self.pool)
            if plt is not None:
                visualize_palette(rearrange(rgb[..., 3:], '(h w) c-> h w c', h=H, w=W), plt.T, savePath,
                                  prtx=f'{prtx}{idx:03d}_{name}')

        return rgb_map, depth_map

    @torch.no_grad()
    def evaluation(self, savePath=None, N_vis=5, prtx='', N_samples=-1, white_bg=False, ndc_ray=False):
        for x in (self.PSNRs, self.rgb_maps, self.depth_maps, self.ssims, self.l_alex, self.l_vgg):
            x.clear()

        os.makedirs(savePath, exist_ok=True)
        os.makedirs(os.path.join(savePath, "rgbd"), exist_ok=True)
        os.makedirs(os.path.join(savePath, "palette"), exist_ok=True)
        savePath = Path(savePath)
        test_dataset = self.test_dataset

        with suppress(Exception):
            tqdm._instances.clear()

        img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis, 1)
        idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))
        for idx, samples in tqdm(enumerate(test_dataset.all_rays[0::img_eval_interval]), file=sys.stdout):
            self.eval_sample(idxs[idx], samples, savePath, prtx, N_samples, white_bg, ndc_ray)
            torch.cuda.empty_cache()

        self.save_video_mean(savePath, prtx)

        return self.PSNRs

    @torch.no_grad()
    def evaluation_path(self, c2ws, savePath=None, N_vis=5, prtx='', N_samples=-1, white_bg=False, ndc_ray=False):
        for x in (self.PSNRs, self.rgb_maps, self.depth_maps, self.ssims, self.l_alex, self.l_vgg):
            x.clear()
        os.makedirs(savePath, exist_ok=True)
        os.makedirs(os.path.join(savePath, "rgbd"), exist_ok=True)
        test_dataset = self.test_dataset
        tensorf = self.tensorf
        renderer = self.renderer
        W, H = test_dataset.img_wh

        with suppress(Exception):
            tqdm._instances.clear()

        for idx, c2w in tqdm(enumerate(c2ws)):
            c2w = torch.FloatTensor(c2w)
            rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
            if ndc_ray:
                rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
            rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

            rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=1024, N_samples=N_samples, ndc_ray=ndc_ray,
                                                   white_bg=white_bg, device=self.device, args=self.args)
            self.clamp_rgb_depth(rgb_map, depth_map, savePath, f'{prtx}{idx:03d}')

        self.save_video_mean(savePath, prtx)

        return self.PSNRs

    def compute_metrics(self, gt_rgb, rgb_map):
        if self.compute_extra_metrics:
            ssim = rgb_ssim(rgb_map, gt_rgb, 1)
            l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', self.tensorf.device)
            l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', self.tensorf.device)
            self.ssims.append(ssim)
            self.l_alex.append(l_a)
            self.l_vgg.append(l_v)

    def clamp_rgb_depth(self, rgb_map, depth_map, savePath: Optional[os.PathLike] = None, prtx=''):
        W, H = self.test_dataset.img_wh
        near_far = self.test_dataset.near_far

        rgb_map = rgb_map[..., :3].clamp(0.0, 1.0)
        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()
        depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)
        ret = rgb_map, depth_map

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        self.rgb_maps.append(rgb_map)
        self.depth_maps.append(depth_map)

        if savePath is not None:
            savePath = Path(savePath)
            imageio.imwrite(savePath / f'{prtx}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            savePath = savePath / 'rgbd'
            imageio.imwrite(savePath / f'{prtx}.png', rgb_map)

        return ret

    def save_video_mean(self, savePath: os.PathLike, prtx=''):
        savePath = Path(savePath)

        kwargs = {
            'fps': min(len(self.rgb_maps) / 5, 30),
            'quality': 10,
            'macro_block_size': 8,
        }
        imageio.mimwrite(savePath / f'{prtx}video.mp4', self.rgb_maps, **kwargs)
        imageio.mimwrite(savePath / f'{prtx}depthvideo.mp4', self.depth_maps, **kwargs)

        if self.PSNRs:
            psnr = np.mean(np.asarray(self.PSNRs))
            if self.compute_extra_metrics:
                ssim = np.mean(np.asarray(self.ssims))
                l_a = np.mean(np.asarray(self.l_alex))
                l_v = np.mean(np.asarray(self.l_vgg))
                np.savetxt(savePath / f'{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
            else:
                np.savetxt(savePath / f'{prtx}mean.txt', np.asarray([psnr]))

    @torch.no_grad()
    def render_test(self):
        args = self.args
        white_bg = self.test_dataset.white_bg
        ndc_ray = args.ndc_ray

        if self.summary_writer is not None:
            logfolder = Path(self.summary_writer.log_dir)
        else:
            logfolder = Path(getattr(args, 'basedir', os.path.dirname(args.ckpt)), args.expname)
            if getattr(args, 'add_timestamp', None):
                logfolder = logfolder / datetime.now().strftime("-%Y%m%d-%H%M%S")

        PSNRs_test = None
        if args.render_test:
            filePath = logfolder / 'imgs_test_all'
            PSNRs_test = self.evaluation(os.fspath(filePath), N_vis=-1, N_samples=-1, white_bg=white_bg,
                                         ndc_ray=ndc_ray)
            print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

        PSNRs_path = None
        if args.render_path:
            filePath = logfolder / 'imgs_path_all'
            c2ws = self.test_dataset.render_path
            # c2ws = test_dataset.poses
            print('========>', c2ws.shape)
            PSNRs_path = self.evaluation_path(c2ws, os.fspath(filePath), N_vis=-1, N_samples=-1, white_bg=white_bg,
                                              ndc_ray=ndc_ray)

        PSNRs_train = None
        if args.render_train:
            filePath = logfolder / 'imgs_train_all'
            self.test_dataset, self.alt_dataset = self.alt_dataset, self.test_dataset
            PSNRs_train = self.evaluation(os.fspath(filePath), N_vis=-1, N_samples=-1, white_bg=white_bg,
                                          ndc_ray=ndc_ray)
            print(f'======> {args.expname} test(alt) all psnr: {np.mean(PSNRs_train)} <========================')

        return PSNRs_test or PSNRs_path or PSNRs_train
