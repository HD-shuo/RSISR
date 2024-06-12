import math
import torch
import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size, dilation=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        dilation=dilation,
        padding=(kernel_size-1)//2, bias=bias)


class CCA(nn.Module):
    def __init__(self, C, ratio=16):
        super(CCA, self).__init__()
        self.squeeze = nn.Conv2d(C, 1, 1, padding=0)
        self.squeeze_fn = nn.Softmax(dim=-1)
        self.excitation = nn.Sequential(*[
            nn.Conv2d(C, C // ratio, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // ratio, C, 1, padding=0),
            nn.Sigmoid()
        ])

    def spatial_squeeze(self, x):
        b, c, h, w = x.size()
        # squeeze
        input_x = x
        input_x = input_x.view(b, c, h * w)
        input_x = input_x.unsqueeze(1)
        var = x - x.mean(dim=[2, 3], keepdim=True)
        mask = self.squeeze(var)
        mask = mask.view(b, 1, h * w)
        mask = self.squeeze_fn(mask)
        mask = mask.unsqueeze(-1)
        squeeze = torch.matmul(input_x, mask)
        squeeze = squeeze.view(b, c, 1, 1)

        return squeeze
    
    def forward(self, x):
        # squeeze
        att = self.spatial_squeeze(x)
        # excitation
        att = self.excitation(att)
        x = x * att
        return x


class ESA(nn.Module):
    def __init__(self, Cin, KSize=1, ratio=16):
        super(ESA, self).__init__()
        self.ESA = nn.Sequential(*[
            nn.Conv2d(Cin, Cin // ratio, KSize, padding=(KSize - 1) // 2, stride=1),
            nn.Conv2d(Cin // ratio, Cin // ratio, 3, padding=2, stride=1, groups=Cin // ratio, dilation=2),
            nn.Conv2d(Cin // ratio, Cin // ratio, 3, padding=2, stride=1, groups=Cin // ratio, dilation=2),
            nn.Conv2d(Cin // ratio, 1, KSize, padding=(KSize - 1) // 2, stride=1),
            nn.Sigmoid()
        ])

    def forward(self, x):
        mean = x.mean(dim=[2, 3], keepdim=True)
        x = x - mean
        x = self.ESA(x)
        return x


class PSA(nn.Module):
    def __init__(self, Cin, SA_in):
        super(PSA, self).__init__()
        self.SA = ESA(Cin)
        self.SA_in = SA_in
        if SA_in > 0:
            self.SA_squeeze = SA_squeeze(SA_in)

    def forward(self, x, prior_SAs):
        current_SA = self.SA(x)
        if self.SA_in > 0:
            prior_SA = self.SA_squeeze(prior_SAs)
            PSA = current_SA + prior_SA
        else:
            PSA = current_SA
        x = x * PSA
        return x, PSA
    

class SA_squeeze(nn.Module):
    def __init__(self, SA_in, KSize=1, fu_by_conv=True):
        super(SA_squeeze, self).__init__()
        self.conv = nn.Sequential(*[
            nn.Conv2d(SA_in, 1, KSize, padding=(KSize - 1) // 2, stride=1),
            nn.ReLU(inplace=True)
        ])
        self.fu_by_conv = fu_by_conv

    def forward(self, x):
        x = torch.cat(x, dim=1)
        x = self.conv(x)
        return x
    

class EARB(nn.Module):
    def __init__(
            self, n_feat, SA_in, kernel_size=3, ratio=16,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(EARB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        self.CA = CCA(n_feat, ratio)
        self.SA = PSA(n_feat, SA_in)
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x, prior_SAs):
        res = self.body(x)
        res_CA = self.CA(res)
        res, sa_f = self.SA(res_CA, prior_SAs)
        res += x
        return res, sa_f
    

class RIRGroup(nn.Module):
    def __init__(self, n_feat, kernel_size, n_resblocks):
        super(RIRGroup, self).__init__()
        self.n_resblocks = n_resblocks
        self.body = nn.ModuleList()
        for i in range(n_resblocks):
            self.body.append(
                EARB(n_feat=n_feat, SA_in=i, kernel_size=kernel_size)
            )

    def forward(self, x):
        SA_maps = []
        res = x
        for i in range(self.n_resblocks):
            res, sa_f = self.body[i](res, SA_maps)
            SA_maps.append(sa_f)
        res += x
        return res


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(default_conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(default_conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)