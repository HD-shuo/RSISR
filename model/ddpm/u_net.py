import torch
from torch import nn
from typing import Optional, Tuple, Union, List

from model.ddpm_module import TimeEmbedding
from model.ddpm_module import ResBlock
from model.ddpm_module import AttnBlock
from model.ddpm_model import Upsample

class UNet(nn.Module):
    """
        desc:定义扩散模型UNet
        params:
            T (int): 时间序列长度
            img_ch: 输入图像通道数
            n_ch (int): 输入特征通道数
            ch_mult (list): 每个block的通道数
            attn (list): 每个block的注意力模块
    """
    def __init__(self, T:int, img_ch, n_ch, ch_mult, attn):
        super().__init__()
        assert all(i < len(ch_mult) for i in attn), 'attn index out of bound'
        # 将时间嵌入维度设置为输入维度的四倍，通过拓展维度特征增强表达能力，本质上是将时间信息编码为高维特征。这个倍数是可选的
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        # Project image into feature map
        self.image_proj = nn.Conv2d(img_ch, n_channels, kernel_size=(3, 3), padding=(1, 1))
        # Downsampling
        self.downblocks = nn.ModuleList()
        self.attnblocks = nn.ModuleList()
        self.upblocks = nn.ModuleList()
    
    def forward(self, x, t):
        # Timestep embedding
        temb = self.time_embedding(t)
        # DownSampling
        # Middle
        # Upsampling
        