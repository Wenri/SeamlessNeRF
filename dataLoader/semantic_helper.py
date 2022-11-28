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

    def _call_back(self, layer, args):
        print(args[0].shape)

    def forward(self, x):
        x = self.preprocess(x).unsqueeze(0)
        x = self.vgg.features(x)
        return x
