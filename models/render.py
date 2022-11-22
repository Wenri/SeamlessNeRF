import torch
import torch.nn.functional as F

from .tensorBase import positional_encoding


class PLTRender(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, feape=6, featureC=128, palette=None):
        super().__init__()

        self.in_mlpC = 2 * viewpe * 3 + 2 * feape * inChanel + 3 + inChanel
        self.viewpe = viewpe
        self.feape = feape
        self.n_palette = len(palette)
        self.n_dim = 3 + self.n_palette - 1
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, self.n_palette - 1)
        torch.nn.init.constant_(layer3.bias, 0)
        self.register_buffer('palette', torch.as_tensor(palette, dtype=torch.float32), persistent=False)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.LeakyReLU(inplace=True),
                                       layer2, torch.nn.LeakyReLU(inplace=True),
                                       layer3)

    def rgb_from_palette(self, logits, opaque):
        log_csum = torch.cumsum(F.logsigmoid(torch.neg(logits)), dim=-1)
        log_w0 = log_csum[..., -1:None]
        w_0 = torch.exp(log_w0)
        w_a = torch.exp(log_w0 - log_csum[..., :-1]) * opaque[..., :-1]
        w_last = opaque[..., -1:None]
        bary_coord = torch.cat((w_0, w_a, w_last), dim=-1)
        # assert torch.allclose(bary_coord.sum(dim=-1), torch.ones(()), atol=1e-3, rtol=1e-3)
        return bary_coord @ self.palette

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.feape > 0:
            indata.append(positional_encoding(features, self.feape))
        if self.viewpe > 0:
            indata.append(positional_encoding(viewdirs, self.viewpe))
        logits = self.mlp(torch.cat(indata, dim=-1))
        opaque = F.sigmoid(logits)
        rgb = self.rgb_from_palette(logits, opaque)

        return torch.cat((rgb, opaque), dim=-1)
