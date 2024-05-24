# uncompyle6 version 3.9.1
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.8.5 (default, Sep  4 2020, 07:30:14) 
# [GCC 7.3.0]
# Embedded file name: /share/program/dxs/RSISR/model/encoder.py
# Compiled at: 2024-05-23 07:14:30
# Size of source mod 2**32: 5924 bytes
from diffusers import AutoencoderKL
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchsummary import summary
from typing import List
import traceback, numpy as np, cv2
from PIL import Image
from omegaconf import OmegaConf
from log import logger
from model.vae import DiagonalGaussianDistribution
from model.utils.modeling_output import AutoencoderKLOutput

class Encoder(nn.Module):
    __doc__ = "\n        init AutoencoderKL encoder model\n    "

    def __init__(self, conf):
        """
            desc:
            params:
        """
        super().__init__()
        self.model_path = conf.model_path
        self.model = None
        self.hooks = []
        self._init_encoder_model(self.model_path)
        self._init_quant_conv(self.model_path)

    def _init_encoder_model(self, model_path):
        """
            desc: init encoder model, inject it into the class
            params:
                model_path: the path of model file
            retrun:
                None
        """
        try:
            vae_model = AutoencoderKL.from_pretrained(model_path)
            self.model = vae_model.encoder
        except Exception as e:
            try:
                logger.error(f"init pre-train encoder model error: {str(e)}, {traceback.format_exc()}")
                self.model = None
            finally:
                e = None
                del e

    def _init_quant_conv(self, model_path):
        """
            desc: init quant layer, inject it into the class
            params:
                None
            retrun:
                None
        """
        try:
            vae_model = AutoencoderKL.from_pretrained(model_path)
            self.quant_conv = vae_model.quant_conv
        except Exception as e:
            try:
                logger.error(f"init pre-train quant layer error: {str(e)}, {traceback.format_exc()}")
                self.quant_conv = None
            finally:
                e = None
                del e

    def preprocess(self, images: List[np.ndarray]):
        """
            desc: preprocess a batch of images before model forward
            params:
                images: the image to be preprocessed
            return:
                images: torch.Tensor 
        """
        transform = transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        for i in range(len(images)):
            img = cv2.resize(images[i], (256, 256))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = transform(img)
            images[i] = img.clone().detach()
        else:
            images = torch.stack(images)
            return images

    def register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Module):
                self.hooks.append(module.register_forward_hook(hook_function))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def forward(self, images: torch.Tensor, return_dict: bool=True):
        """
            desc: model forward function
            params:
                images (torch.Tensor): _description_        
            return:
                _type_: _description_
        """
        for param in self.model.parameters():
            param.requires_grad = False
        else:
            features = self.model(images)
            moments = self.quant_conv(features)
            posterior = DiagonalGaussianDistribution(moments)
            if not return_dict:
                return (
                 moments, posterior)
            return AutoencoderKLOutput(latent_dist=posterior)


def hook_function(module, input, output):
    print(f"Module name: {module.__class__.__name__}")
    print(f"Output shape: {output.shape}")
    print("Output stats - mean: {}, std: {}, min: {}, max: {}".format(output.mean().item(), output.std().item(), output.min().item(), output.max().item()))
    if torch.isnan(output).any():
        print(f"NaN detected after layer: {module.__class__.__name__}")


if __name__ == "__main__":
    confdir = "/home/work/daixingshuo/RSISR/configs/model.yaml"
    conf = OmegaConf.load(confdir)
    encoder = Encoder(conf)
    test_images = []
    test_img1 = "/home/work/daixingshuo/RSISR/test_demo/intersection94.png"
    test_img2 = "/home/work/daixingshuo/RSISR/test_demo/agricultural08.png"
    test_img1 = Image.open(test_img1).convert("RGB")
    test_img1 = np.array(test_img1)
    test_images.append(test_img1)
    test_img2 = Image.open(test_img2).convert("RGB")
    test_img2 = np.array(test_img2)
    test_images.append(test_img2)
    test_tensors = encoder.preprocess(test_images)
    features = encoder(test_tensors)

# okay decompiling /share/program/dxs/RSISR/model/__pycache__/encoder.cpython-38.pyc
