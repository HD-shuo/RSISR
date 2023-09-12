import torch
import torch.nn as nn
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

unet = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)