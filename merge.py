import os
from pathlib import Path

import torch
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
                                semantic_type=args.semantic_type) if not args.render_only else None
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
        pc = PointCloud(xyz[alpha > 0.005].cpu().numpy())
        pc.export(savePath)

    @torch.no_grad()
    def merge(self):
        self.tensorf.args = self.args
        self.tensorf.add_merge_target(target := self.build_network(self.args.ckpt))
        if self.args.export_mesh:
            self.export_mesh(target)
            self.export_mesh()
            self.export_pointcloud()
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
