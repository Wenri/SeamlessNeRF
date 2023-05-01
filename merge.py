import logging
import os
from pathlib import Path

import numpy as np
import torch

from dataLoader import dataset_dict
from renderer import OctreeRender_trilinear_fast
from eval import Evaluator
from utils import convert_sdf_samples_to_ply


class Merger:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.renderer = OctreeRender_trilinear_fast

        # init dataset
        dataset = dataset_dict[args.dataset_name]
        self.train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_test, is_stack=False,
                                     semantic_type=args.semantic_type)
        self.test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_test, is_stack=True,
                                    semantic_type=args.semantic_type, pca=getattr(self.train_dataset, 'pca', None))

        self.args = args
        self.logger = logging.getLogger(type(self).__name__)

    def build_network(self, ckpt=None):
        args = self.args
        if not ckpt:
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

        logfolder = Path(args.basedir, args.expname)

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

    @torch.no_grad()
    def merge(self):
        tensorf = self.build_network()
        tensorf.add_merge_target(self.build_network(self.args.ckpt))
        self.render_test(tensorf)


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
    merger = Merger(args)
    if args.export_mesh:
        merger.export_mesh()

    assert args.render_only

    merger.merge()
