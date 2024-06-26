from typing import Optional, Tuple, Union, List
from omegaconf import DictConfig, OmegaConf
from timm.models.layers import to_2tuple
from PIL import Image
import numpy as np
import cv2

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose,ToTensor, ToPILImage, Resize
from torchvision.transforms.functional import InterpolationMode

from model.ddpm_model import DdpmModel
from model.swin_transformer_v2 import SwinTransformerBlock
from model.easr import RIRGroup, MeanShift, Upsampler, default_conv
from model.ddpm.gaussian_diffusion import get_named_beta_schedule


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
        x = self.bn(x)
        x = self.relu(x)
        x = self.attention(x)
        return x


class Decoder(nn.Module):
    """
        init img restore decoder    
    """
    def __init__(self, conf):
        super().__init__()
        self.config = conf
        self.n_group = conf.n_group
        img_size = to_2tuple(conf.img_size)
        patch_size = to_2tuple(conf.patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        # define head module
        modules_IFE =[default_conv(conf.hidden_dim, conf.n_feats, kernel_size=3)]
        modules_head = [
            default_conv(conf.n_feats, conf.n_feats, kernel_size=3)]

        # build attn block
        self.attn_blocks = nn.ModuleList([
            RIRGroup(conf.n_feats, kernel_size=3, n_resblocks=conf.n_resblocks)
            for _ in range(conf.n_resgroups)
        ])
        # build stb blocks
        self.stb_blocks = nn.ModuleList()
        for i_blk in range(conf.n_stb):
            stb_block = SwinTransformerBlock(
                dim = int(conf.stb.embed_dim *2 ** i_blk),
                input_resolution=(patches_resolution[0] // (2 ** i_blk),
                                  patches_resolution[1] // (2 ** i_blk)),
                num_heads=conf.stb.num_heads[i_blk],
                window_size=conf.stb.window_size
            )
            self.stb_blocks.append(stb_block)
        
        # build decoder tail
        modules_tail = [
            Upsampler(conf.upsacle_factor, conf.n_feats, act=False)
        ]
        
        self.add_mean = MeanShift(conf.rgb_range, sign=1)
        self.IFE = nn.Sequential(*modules_IFE)
        self.head = nn.Sequential(*modules_head)
        self.tail = nn.Sequential(*modules_tail)
    
    def forward(self, x):
        IFE_x = self.IFE(x)
        x = self.head(IFE_x)
        for _ in range(self.n_group):
            for anblk in self.attn_blocks:
                x = anblk(x)
            for blk in self.stb_blocks:
                x = blk(x)
        x += IFE_x
        x = self.tail(x)
        x = self.add_mean(x)
        return x



class VitUpscale(nn.Module):
    """
        init upscale model     
    """
    def __init__(self, model_config):
        super().__init__()
        self.configs = model_config
        # 初始化time_embedding
        # 初始化模型
        self.vit = torchvision.models.vit_b_16(pretrained=True)
        self.ddpm = DdpmModel()
        self.upsample = UpsampleModule(model_config.upsample)
        self.proj = nn.Linear(1000, 3*model_config.img_size * model_config.img_size)
        self.decoder = Decoder(model_config.decoder)

        # noise and ddpm caculation schedule
        self.T = model_config.ddpm.T
        self.beta_schedule = model_config.ddpm.beta_schedule
        self.betas = get_named_beta_schedule(self.beta_schedule, self.T)

        self.alphas = torch.from_numpy(1. - self.betas)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.T,)

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
        t = torch.randint(self.T, size=(x.shape[0], ), device=x.device)
        x = self.upsample(x)
        # save feature
        # feature_map_img = ToPILImage()(x[0].detach().cpu().squeeze())  
        # feature_map_img.save('feature_map.png')  

        x = self.vit(x)
        x = self.proj(x)
        bs = x.size(0)
        x = x.view(bs, conf.ch, conf.img_size, conf.img_size)
        noise = torch.randn_like(x)
        # ddpm
        xt = self.q_sample(x, t, noise)
        x = self.ddpm(xt, t)
        x = self.decoder(x) 
        return x

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
    test_images.append(test_img1)
    test_img2 = Image.open(test_img2).convert("RGB")
    test_img2 = np.array(test_img2)
    test_images.append(test_img2)
    test_images.append(test_img2)
    test_tensors = train_lr_transform(224, 4, test_images)
    output_features = upscale_model(test_tensors)
    print(output_features.shape)