import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from dataLoader import dataset_dict
from renderer import OctreeRender_trilinear_fast, evaluation, evaluation_path
from utils import convert_sdf_samples_to_ply


class Merger:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.renderer = OctreeRender_trilinear_fast

        # init dataset
        dataset = dataset_dict[args.dataset_name]
        self.train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False,
                                     semantic_type=args.semantic_type)
        self.test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True,
                                    semantic_type=args.semantic_type, pca=getattr(self.train_dataset, 'pca', None))

        self.args = args
        self.logger = logging.getLogger(type(self).__name__)

    def build_network(self):
        args = self.args
        ckpt = args.ckpt
        if not ckpt and args.render_only:
            ckpt = Path(args.basedir, args.expname) / f'{args.expname}.th'

        if not os.path.exists(ckpt):
            raise RuntimeError('the ckpt path does not exists!!')

        ckpt = torch.load(ckpt, map_location=self.device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device': self.device})
        tensorf = args.model_name(**kwargs)
        tensorf.load(ckpt)

        return tensorf

    @torch.no_grad()
    def export_mesh(self):
        args = self.args
        ckpt = torch.load(args.ckpt, map_location=self.device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device': self.device})
        tensorf = args.model_name(**kwargs)
        tensorf.load(ckpt)

        alpha, _ = tensorf.getDenseAlpha()
        convert_sdf_samples_to_ply(alpha.cpu(), f'{args.ckpt[:-3]}.ply', bbox=tensorf.aabb.cpu(), level=0.005)

    @torch.no_grad()
    def render_test(self, tensorf):
        args = self.args
        white_bg = self.test_dataset.white_bg
        ndc_ray = args.ndc_ray

        if args.ckpt is not None:
            logfolder = Path(os.path.dirname(args.ckpt), args.expname)
            if not os.path.exists(args.ckpt):
                self.logger.warning('the ckpt path does not exists!!')
        else:
            logfolder = Path(args.basedir, args.expname)
            if args.add_timestamp:
                logfolder = logfolder / datetime.now().strftime("-%Y%m%d-%H%M%S")

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

    merger = Merger(args)
    if args.export_mesh:
        merger.export_mesh()

    assert args.render_only

    merger.render_test(merger.build_network())
