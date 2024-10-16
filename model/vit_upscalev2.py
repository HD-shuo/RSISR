from typing import Optional, Tuple, Union, List
from omegaconf import DictConfig, OmegaConf
from timm.models.layers import to_2tuple
from PIL import Image
import numpy as np
import cv2
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import Compose,ToTensor, ToPILImage, Resize
from torchvision.transforms.functional import InterpolationMode
from torchvision.models import ViT_B_16_Weights

from model.ddpm_model import DdpmModel
from model.swin_transformer_v2 import BasicLayer, PatchEmbed
from model.easr import RIRGroup, MeanShift, Upsampler, default_conv
from model.ddpm.gaussian_diffusion import get_named_beta_schedule


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if isinstance(arr, torch.Tensor):
        res = arr.to(device=timesteps.device)[timesteps].float()
    if isinstance(arr, np.ndarray):
        res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


class ChannelAttention(nn.Module):
    """
        init channel attention       
    """
    def __init__(self, in_channels:int, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = self.avg_pool(x).view(batch_size, channels)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        y = y.view(batch_size, channels, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    """
        init spatial attention       
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        pool = torch.cat([avg_pool, max_pool], dim=1)
        attention = self.conv(pool)
        attention = self.sigmoid(attention)
        return x * attention


class UpsampleModule(nn.Module):
    """
        init upsamle module, use convelution transpose and channel attention     
    """
    def __init__(self, conf):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(conf.in_channels, conf.out_channels, kernel_size=4, stride=conf.upscale_factor, padding=0, output_padding=0)
        self.attn = conf.attn
        self.bn = nn.BatchNorm2d(conf.out_channels)  
        self.relu = nn.ReLU(inplace=True)
        if self.attn == 'ch_att':
            self.attention = ChannelAttention(conf.out_channels)
        if self.attn == 'sp_att':
            self.attention = SpatialAttention()
        else:
            self.attention = nn.Identity()

    def forward(self, x):
        x = self.upsample(x)
        # x = self.bn(x)
        # x = self.relu(x)
        # x = self.attention(x)
        return x
    

class UpsampleModulev2(nn.Module):
    """
        init upsamle module, use convelution transpose and channel attention     
    """
    def __init__(self, scale, n_color, bn=False, act=False, bias=True):
        super(Upsampler, self).__init__()
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(default_conv(n_color, 4 * n_color, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_color))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_color))

        elif scale == 3:
            m.append(default_conv(n_color, 9 * n_color, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_color))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_color))
        else:
            raise NotImplementedError
        self.body = nn.Sequential(*m)

    def forward(self, x):
        return self.body(x)


class STB_layer(nn.Module):
    def __init__(self, img_size=256, patch_size=4, in_chans=3,
                 embed_dim=64, depths=[2, 2, 6, 2], num_heads=[2, 4, 8, 16],
                 window_size=8, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0], **kwargs):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=embed_dim,
                               input_resolution=(img_size,img_size),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size)
            self.layers.append(layer)
        self.norm = norm_layer(embed_dim)
    
    def forward(self, x):
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


class Decoder(nn.Module):
    """
        init img restore decoder    
    """
    def __init__(self, conf):
        super().__init__()
        self.config = conf
        self.n_group = conf.n_group
        self.img_size = to_2tuple(conf.img_size)
        
        # define head module
        modules_IFE =[default_conv(conf.hidden_dim, conf.n_feats, kernel_size=3)]
        modules_head = [
            default_conv(conf.n_feats, conf.n_feats, kernel_size=3)]

        # build attn block
        self.attn_blocks = nn.ModuleList([
            RIRGroup(conf.n_feats, kernel_size=3, n_resblocks=conf.n_resblocks)
            for _ in range(conf.n_resgroups)
        ])
        
        # build stb layer
        self.stb_layer = STB_layer(
            img_size=conf.img_size,
            in_chans=conf.n_feats,
            embed_dim = conf.stb.embed_dim,
            depths=conf.stb.depths,
            num_heads=conf.stb.num_heads,
            window_size=conf.stb.window_size
        )
        
        # build decoder tail
        # modules_tail = [
        #     Upsampler(conf.upsacle_factor, conf.n_feats, act=False),
        #     default_conv(conf.n_feats, conf.n_colors, kernel_size=3)
        # ]
        modules_tail = [
            default_conv(conf.n_feats, conf.n_colors, kernel_size=3)
        ]
        
        self.add_mean = MeanShift(conf.rgb_range, sign=1)
        self.IFE = nn.Sequential(*modules_IFE)
        self.head = nn.Sequential(*modules_head)
        self.tail = nn.Sequential(*modules_tail)
    
    def forward(self, x):
        IFE_x = self.IFE(x)
        x = self.head(IFE_x)
        for _ in range(self.n_group):
            x = self.stb_layer(x)
            B, _, C = x.shape
            H, W = self.img_size
            x = x.view(B, C, H, W)
            for anblk in self.attn_blocks:
                x = anblk(x)
        x += IFE_x
        x = self.tail(x)
        x = self.add_mean(x)
        return x


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, alphas, betas, T):
        super().__init__()

        self.model = model
        self.T = T
        self.betas = torch.from_numpy(betas)
        self.alphas = alphas
        self.betas = torch.from_numpy(betas)
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, x_c):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        # loss = F.mse_loss(self.model(torch.cat((x_t, x_c), dim=1), t), noise, reduction='mean')
        loss = F.mse_loss(self.model(x_t, x_c, t), noise, reduction='mean')
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, alphas, betas, T):
        super().__init__()

        self.model = model
        self.T = T
        self.betas = torch.from_numpy(betas)
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, x_c, t):
        # below: only log_variance is used in the KL computations
        self.betas = self.betas.to(self.posterior_var.device)
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        eps = self.model(x_t, x_c, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    @torch.no_grad()
    def forward(self, x_T, x_c):
        """
        Algorithm 2.
        """
        # 确定性去噪过程
        x_t = x_T
        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, _ = self.p_mean_variance(x_t=x_t, x_c=x_c, t=t)
            # Remove stochastic sampling; use mean directly for deterministic denoising
            x_t = mean
            # No need to check for NaNs as we're not introducing any randomness
        x_0 = x_t
        return torch.clamp(x_0, -1, 1)
        # x_t = x_T
        # for time_step in reversed(range(self.T)):
        #     # print(time_step)
        #     t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
        #     mean, var= self.p_mean_variance(x_t=x_t, t=t)
        #     # no noise when t == 0
        #     if time_step > 0:
        #         noise = torch.randn_like(x_t)
        #     else:
        #         noise = 0
        #     x_t = mean + torch.sqrt(var + 1e-6) * noise
        #     assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        # x_0 = x_t
        # return torch.clip(x_0, -1, 1)   


class VitUpscalev2(nn.Module):
    """
        init upscale model     
    """
    def __init__(self, model_config):
        super().__init__()
        self.configs = model_config
        # noise and ddpm caculation schedule
        self.T = model_config.ddpm.T
        self.sample_step = model_config.ddpm.sample_step
        self.beta_schedule = model_config.ddpm.beta_schedule
        self.ddpm_ch = model_config.ddpm.in_ch
        self.sample_betas = get_named_beta_schedule(self.beta_schedule, self.sample_step)
        self.betas = get_named_beta_schedule(self.beta_schedule, self.T)

        self.alphas = torch.from_numpy(1. - self.betas)
        self.sample_alphas = torch.from_numpy(1. - self.sample_betas)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.T,)

        # 初始化time_embedding
        # 初始化模型
        self.vit = torchvision.models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.unet = DdpmModel(n_channels=self.ddpm_ch)
        self.ddpm = GaussianDiffusionTrainer(model=self.unet, alphas=self.alphas, betas=self.betas, T=model_config.ddpm.T)
        self.sampler = GaussianDiffusionSampler(model=self.unet, alphas=self.sample_alphas, betas=self.sample_betas, T=model_config.ddpm.sample_step)
        self.upsample = UpsampleModule(model_config.upsample)
        self.proj = nn.Linear(1000, 3*model_config.img_size * model_config.img_size)
        self.decoder = Decoder(model_config.decoder)
        self.conv_trans = nn.ConvTranspose2d(model_config.upsample.in_channels, model_config.upsample.out_channels, kernel_size=4, stride=model_config.upsample.upscale_factor, padding=0, output_padding=0)
        # self.bn = nn.BatchNorm2d(model_config.upsample.out_channels)  

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

    def q_sample(self, x_start, t, noise):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            x_start = x_start.float()
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def forward(self, x):
        conf = self.configs
        # noise scheduel
        # t = torch.randint(self.T, size=(x.shape[0], ), device=x.device)
        x = self.upsample(x)
        # x_up = F.interpolate(x, size=conf.img_size, mode='bicubic', align_corners=False)
        # x_up = self.bn(x_up)
        # save feature
        # feature_map_img = ToPILImage()(x[0].detach().cpu().squeeze())  
        # feature_map_img.save('feature_map.png')  

        x = self.vit(x)
        x = self.proj(x)
        bs = x.size(0)
        x = x.view(bs, conf.ch, conf.img_size, conf.img_size)
        # noise = torch.randn_like(x)
        # ddpm
        # xt = self.q_sample(x, t, noise)
        ddpm_loss = self.ddpm(x).sum() / 1000.
        x = self.sampler(x)
        # x_res = x_up - x
        # x += x_res
        x = self.decoder(x)
        #x = self.conv_trans(x)
        return x, ddpm_loss

def train_lr_transform(crop_size, upscale_factor, images):
    transform = Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=InterpolationMode.BICUBIC),
        ToTensor()
    ])
    for i in range(len(images)):
        img = cv2.resize(images[i], (crop_size, crop_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform(img)
        images[i] = img.clone().detach() 
    images = torch.stack(images)
    return images


if __name__ == "__main__":
    # load conf
    confdir = "/share/program/dxs/RSISR/configs/model_2.yaml"
    conf = OmegaConf.load(confdir)
    # load model
    upscale_model = VitUpscale(conf)
    # test
    # time embedding
    t = torch.randint(1000, (2, ))
    test_images = []
    test_img1 = '/share/program/dxs/RSISR/test_demo/airplane08.png'
    test_img2 = '/share/program/dxs/RSISR/test_demo/freeway26.png'
    test_img1 = Image.open(test_img1).convert("RGB")
    test_img1 = np.array(test_img1)
    test_images.append(test_img1)
    test_img2 = Image.open(test_img2).convert("RGB")
    test_img2 = np.array(test_img2)
    test_images.append(test_img2)
    test_tensors = train_lr_transform(224, 4, test_images)
    output_features = upscale_model(test_tensors)
    print(output_features.shape)