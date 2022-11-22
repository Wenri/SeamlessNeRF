import torch

from .tensorBase import positional_encoding


class PLTRender(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, feape=6, featureC=128, palette=None):
        super().__init__()

        self.in_mlpC = 2 * viewpe * 3 + 2 * feape * inChanel + 3 + inChanel
        self.viewpe = viewpe
        self.feape = feape
        self.n_palette = len(palette)
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, self.n_palette - 1)
        torch.nn.init.constant_(layer3.bias, 0)
        self.register_buffer('palette', torch.as_tensor(palette, dtype=torch.float32), persistent=False)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True),
                                       layer2, torch.nn.ReLU(inplace=True),
                                       layer3, torch.nn.LogSigmoid())

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        log_alpha = self.mlp(mlp_in)
        wlast = 1 - torch.exp(log_alpha[..., -1:None])
        log_alpha = torch.cumsum(log_alpha, dim=-1)
        log_w0 = log_alpha[..., -1:None]
        log_alpha_a = log_w0 - log_alpha[..., :-1]
        log_alpha_b = log_w0 - log_alpha[..., :-2]
        log_alpha_b = torch.concat((log_w0, log_alpha_b), dim=-1)
        bary_coord = (torch.exp(log_w0), torch.exp(log_alpha_a) - torch.exp(log_alpha_b), wlast)
        bary_coord = torch.cat(bary_coord, dim=-1)
        rgb = bary_coord @ self.palette

        return rgb
