from torch import nn
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

    def _call_back(self, layer, args):
        x = args[0]
        t = self.layer_features.setdefault(tuple(x.shape), x)
        assert x is t

    def forward(self, x):
        x = self.preprocess(x).unsqueeze(0)

        self.layer_features.clear()
        self.vgg.features(x)
        x = list(self.layer_features.values())
        self.layer_features.clear()

        return x
