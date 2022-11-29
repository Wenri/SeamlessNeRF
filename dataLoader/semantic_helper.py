import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models import VGG16_Weights


class VGGSemantic(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = models.vgg16(weights=VGG16_Weights.DEFAULT)
        self.preprocess = VGG16_Weights.DEFAULT.transforms()
        for layer in self.vgg.features:
            if isinstance(layer, nn.MaxPool2d):
                layer.register_forward_pre_hook(self._call_back)
        self.layer_features = {}
        self.target_size = (224, 224)

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
        return torch.cat(x[-2:], dim=1)
