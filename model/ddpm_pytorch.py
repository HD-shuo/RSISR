import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

class DdpmPytorch(object):
    """
        desc:
        params:
            object (_type_): _description_        
    """
    def __init__(self, conf):
        super().__init__()
        self.unet_conf = conf.unet
        self.diffusion_conf = conf.diffusion
        self.model()
        self.diffusion()

    def model(self):
        """
            desc:
            return:
                _type_: _description_
        """
        uet = Unet(
            dim=self.unet_conf.dim,
            dim_mults=self.unet_conf.dim_mults,
            flash_attn=self.unet_conf.flash_attn
        )
        return uet
    
    def diffusion(self):
        """
            desc:
            return:
                _type_: _description_
        """
        model = self.model()
        diffusion = GaussianDiffusion(
            model,
            image_size=self.diffusion_conf.image_size,
            timesteps=self.diffusion_conf.timesteps,
            sampling_timesteps=self.diffusion_conf.sampling_timesteps
        )
        return diffusion
    