from typing import Optional, Tuple, Union, List
from omegaconf import DictConfig, OmegaConf
from PIL import Image
import numpy as np
import cv2

import torch
import torch.nn as nn
from torchvision.transforms import Compose,ToTensor, ToPILImage, Resize

from model.encoder import Encoder
from model.ddpm_model import DdpmModel
from model.decoder import Decoder


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
        self.relu = nn.ReLU()
        if self.attn == 'ch_att':
            self.attention = ChannelAttention(conf.out_channels)
        if self.attn == 'sp_att':
            self.attention = SpatialAttention()
        else:
            self.attention = nn.Identity()

    def forward(self, x):
        x = self.upsample(x)
        x = self.relu(x)
        x = self.attention(x)
        return x


class UpscaleModel(nn.Module):
    """
        init upscale model     
    """
    def __init__(self, model_config):
        super().__init__()
        self.configs = model_config
        # 初始化time_embedding
        # 初始化模型
        self.encoder = Encoder(model_config.encoder)
        self.ddpm = DdpmModel()
        self.decoder = Decoder(model_config.decoder)
        self.upsample = UpsampleModule(model_config.upsample)
    
    def forward(self, x, t):
        x = self.upsample(x)
        x = self.encoder(x)
        posterior = x.latent_dist
        z = posterior.mode()
        x = self.ddpm(z, t)
        x = self.decoder(x)
        return x

def train_lr_transform(crop_size, upscale_factor, images):
    transform = Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])
    for i in range(len(images)):
        img = cv2.resize(images[i], (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform(img)
        images[i] = torch.tensor(img)
    images = torch.stack(images)
    return images


if __name__ == "__main__":
    # load conf
    confdir = "/home/work/daixingshuo/RSISR/configs/model.yaml"
    conf = OmegaConf.load(confdir)
    # load model
    encoder = Encoder(conf.encoder)
    upscale_model = UpscaleModel(conf)
    # test
    # time embedding
    t = torch.randint(1000, (2, ))
    test_images = []
    test_img1 = '/home/work/daixingshuo/RSISR/test_demo/intersection94.png'
    test_img2 = '/home/work/daixingshuo/RSISR/test_demo/agricultural08.png'
    test_img1 = Image.open(test_img1).convert("RGB")
    test_img1 = np.array(test_img1)
    test_images.append(test_img1)
    test_img2 = Image.open(test_img2).convert("RGB")
    test_img2 = np.array(test_img2)
    test_images.append(test_img2)
    test_tensors = train_lr_transform(256, 4, test_images)
    output_features = upscale_model(test_tensors, t)
    print(output_features.shape)