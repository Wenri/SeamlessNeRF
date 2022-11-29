from collections import namedtuple

import torch
from einops import rearrange
from sklearn.decomposition import PCA
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models import VGG16_Weights


class VGGSemantic(nn.Module):
    IMAGE_SIZE_T = namedtuple('IMAGE_SIZE_T', 'H W')

    def __init__(self):
        super().__init__()
        self.vgg = models.vgg16(weights=VGG16_Weights.DEFAULT)
        self.preprocess = VGG16_Weights.DEFAULT.transforms()
        for layer in self.vgg.features:
            if isinstance(layer, nn.MaxPool2d):
                layer.register_forward_pre_hook(self._call_back)
        self.layer_features = {}
        self.target_size = self.IMAGE_SIZE_T(224, 224)

    def __enter__(self):
        self.layer_features.clear()
        return self.layer_features

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.layer_features.clear()

    def _call_back(self, layer, args):
        x = args[0]
        t = self.layer_features.setdefault(tuple(x.shape), x)
        assert x is t

    def forward(self, x):
        x = self.preprocess(x).unsqueeze(0)
        self.layer_features.clear()
        with self as features:
            self.vgg.features(x)
            x = []
            for out in features.values():
                if out.shape[-2:] != self.target_size:
                    out = F.interpolate(out, self.target_size, mode='nearest')
                x.append(out)
        return torch.cat(x, dim=1)


class PCASemantic:
    def __init__(self, dataset, n_components=3):
        self.dataset = dataset
        self.pca = PCA(n_components=n_components)

    def build_semantic(self):
        sems = self.dataset.all_sems
        w, h = self.dataset.img_wh
        all_imgs = rearrange(self.dataset.all_rgbs, '(n h w) c -> n c h w', h=w, w=w)
        all_imgs = F.interpolate(all_imgs, size=sems.shape[1:-1], mode='bilinear')
        fg = torch.lt(all_imgs, 1.).any(dim=1)
        sems = sems[fg].numpy()
        return self.pca.fit(sems)
