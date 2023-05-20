from collections import namedtuple

import imageio
import numpy as np
import torch
from einops import rearrange

OctreeRender_trilinear_fast_ret_t = namedtuple('OctreeRender_trilinear_fast_ret_t',
                                               ('rgbs', 'alphas', 'depth_map', 'weights', 'uncertainties'),
                                               defaults=(None, None, None, None, None))


def OctreeRender_trilinear_fast(rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False,
                                device='cuda', **kwargs):
    rgbs, alphas, depth_maps, weights, uncertainties = [], [], [], [], []
    N_rays_all = rays.shape[0]
    render_only = (args := kwargs.get('args', None)) and args.render_only
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)

        rgb_map, depth_map = tensorf(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray,
                                     N_samples=N_samples, **kwargs)

        if render_only or not is_train:
            rgb_map, depth_map = rgb_map.cpu(), depth_map.cpu()

        rgbs.append(rgb_map)
        depth_maps.append(depth_map)

    return OctreeRender_trilinear_fast_ret_t(rgbs=torch.cat(rgbs), depth_map=torch.cat(depth_maps))


def visualize_rgb(depth_map, rgb_map, savePath, prtx, apply=lambda f, args: f(*args)):
    if savePath is None:
        return
    rgb_map = (rgb_map.numpy() * 255).astype('uint8')
    apply(imageio.imwrite, (savePath / f'{prtx}.png', rgb_map))
    rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
    apply(imageio.imwrite, (savePath / 'rgbd' / f'{prtx}.png', rgb_map))


def visualize_palette(opaque, palette, savePath, prtx):
    if savePath is None:
        return
    opaque = opaque[..., None] * palette[:opaque.shape[-1]]
    for j, opq_map in enumerate(torch.split(opaque, 1, dim=2)):
        opq_map = (opq_map.cpu().squeeze(dim=2).numpy() * 255).astype('uint8')
        imageio.imwrite(savePath / 'palette' / f'{prtx}_{j}.png', opq_map)
    bg = rearrange(palette[-1].cpu() * 255, 'c -> 1 1 c').expand(*opaque.shape[:2], -1).to(torch.uint8).numpy()
    imageio.imwrite(savePath / 'palette' / f'{prtx}_BG.png', bg)
