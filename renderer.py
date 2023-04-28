import os
import sys
from contextlib import suppress
from pathlib import Path

import imageio
import numpy as np
import torch
from einops import rearrange
from tqdm.auto import tqdm

from dataLoader.ray_utils import get_rays
from dataLoader.ray_utils import ndc_rays_blender
from extra.compute_metrics import rgb_ssim, rgb_lpips
from utils import visualize_depth_numpy


def OctreeRender_trilinear_fast(rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False,
                                device='cuda', **kwargs):
    rgbs, alphas, depth_maps, weights, uncertainties = [], [], [], [], []
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)

        rgb_map, depth_map = tensorf(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray,
                                     N_samples=N_samples, **kwargs)

        rgbs.append(rgb_map)
        depth_maps.append(depth_map)

    return torch.cat(rgbs), None, torch.cat(depth_maps), None, None


def visualize_rgb(depth_map, rgb_map, savePath, prtx):
    if savePath is None:
        return
    rgb_map = (rgb_map.numpy() * 255).astype('uint8')
    imageio.imwrite(savePath / f'{prtx}.png', rgb_map)
    rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
    imageio.imwrite(savePath / 'rgbd' / f'{prtx}.png', rgb_map)


def visualize_palette(opaque, palette, savePath, prtx):
    if savePath is None:
        return
    opaque = opaque[..., None] * palette[:opaque.shape[-1]]
    for j, opq_map in enumerate(torch.split(opaque, 1, dim=2)):
        opq_map = (opq_map.cpu().squeeze(dim=2).numpy() * 255).astype('uint8')
        imageio.imwrite(savePath / 'palette' / f'{prtx}_{j}.png', opq_map)
    bg = rearrange(palette[-1].cpu() * 255, 'c -> 1 1 c').expand(*opaque.shape[:2], -1).to(torch.uint8).numpy()
    imageio.imwrite(savePath / 'palette' / f'{prtx}_BG.png', bg)


@torch.no_grad()
def evaluation(test_dataset, tensorf, args, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims, l_alex, l_vgg = [], [], []
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(os.path.join(savePath, "rgbd"), exist_ok=True)
    os.makedirs(os.path.join(savePath, "palette"), exist_ok=True)
    savePath = Path(savePath)
    n_palette = getattr(tensorf.renderModule, 'n_palette', 1)
    if n_palette > 1:
        palette = [render.palette for render in tensorf.renderModule]
        plt_names = tensorf.renderModule.PLT_NAMES
    else:
        palette = (getattr(tensorf.renderModule, 'palette', None),)
        plt_names = ('RGB',)

    with suppress(Exception):
        tqdm._instances.clear()

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis, 1)
    idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))
    for idx, samples in tqdm(enumerate(test_dataset.all_rays[0::img_eval_interval]), file=sys.stdout):

        W, H = test_dataset.img_wh
        rays = samples.view(-1, samples.shape[-1])

        plt_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=4096, N_samples=N_samples,
                                               ndc_ray=ndc_ray, white_bg=white_bg, device=device, args=args)
        rgb_map = plt_map[..., :3].clamp(0.0, 1.0)
        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()
        gt_vis = []

        if len(test_dataset.all_rgbs):
            gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)
            gt_vis.append(gt_rgb)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', tensorf.device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', tensorf.device)
                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        if len(getattr(test_dataset, 'all_sems', ())):
            c1, c2 = torch.ones(test_dataset.all_rgbs.shape[1:-1], dtype=torch.bool).nonzero(as_tuple=True)
            c0 = torch.full_like(c1, idxs[idx])
            coord = torch.stack((c0, c1, c2), dim=-1)
            gt_sem = test_dataset.sample_sems(coord).view(H, W, 3)
            gt_vis.append(gt_sem)

        gt_vis = (torch.cat(gt_vis, dim=1).numpy() * 255).astype('uint8')
        imageio.imwrite(savePath / f'{prtx}{idx:03d}_GT.png', gt_vis)

        rgb_maps.append((rgb_map.numpy() * 255).astype('uint8'))
        depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)
        depth_maps.append(depth_map)

        for rgb_map, plt, name in zip(torch.tensor_split(plt_map, n_palette, dim=-1), palette, plt_names):
            visualize_rgb(depth_map, rgb_map[..., :3].clamp(0.0, 1.0).reshape(H, W, 3).cpu(), savePath,
                          prtx=f'{prtx}{idx:03d}_{name}')
            if plt is not None:
                visualize_palette(rearrange(rgb_map[..., 3:], '(h w) c-> h w c', h=H, w=W), plt.T, savePath,
                                  prtx=f'{prtx}{idx:03d}_{name}')

    fps = min(len(rgb_maps) / 5, 30)
    imageio.mimwrite(savePath / f'{prtx}video.mp4', np.stack(rgb_maps), fps=fps, quality=10)
    imageio.mimwrite(savePath / f'{prtx}depthvideo.mp4', np.stack(depth_maps), fps=fps, quality=10)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(savePath / f'{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(savePath / f'{prtx}mean.txt', np.asarray([psnr]))

    return PSNRs


@torch.no_grad()
def evaluation_path(test_dataset, tensorf, c2ws, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                    white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims, l_alex, l_vgg = [], [], []
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(os.path.join(savePath, "rgbd"), exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    for idx, c2w in tqdm(enumerate(c2ws)):

        W, H = test_dataset.img_wh

        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=8192, N_samples=N_samples,
                                               ndc_ray=ndc_ray, white_bg=white_bg, device=device)
        rgb_map = rgb_map[..., :3].clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    fps = min(len(rgb_maps) / 5, 30)
    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=fps, quality=8)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=fps, quality=8)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))

    return PSNRs
