from diffusers import AutoencoderKL
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import List
import traceback
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

from log import logger

class Decoder(nn.Module):
    """
        init AutoencoderKL encoder model
    """
    def __init__(self, conf):
        """
            desc:
            params:
        """
        super().__init__()
        self.model_path = conf.decoder.model_path
        self.model = None
        self.post_quant = None
        self._init_decoder_model(self.model_path)
        self._init_post_quant_layer(self.model_path)

    def _init_decoder_model(self, model_path):
        """
            desc: init encoder model, inject it into the class
            params:
                model_path: the path of model file
            retrun:
                None
        """
        try:
            vae_model = AutoencoderKL.from_pretrained(model_path)
            self.model = vae_model.decoder
        except Exception as e:
            logger.error(f"init pre-train encoder model error: {str(e)}, {traceback.format_exc()}")
            self.model = None

    def _init_post_quant_layer(self, model_path):
        """
            desc: init post quant layer, inject it into the class
            params:
                None
            retrun:
                None
        """
        try:
            vae_model = AutoencoderKL.from_pretrained(model_path)
            self.post_quant = vae_model.post_quant_conv
        except Exception as e:
            logger.error(f"init pre-train encoder model error: {str(e)}, {traceback.format_exc()}")
            self.model = None

    def forward(self, features: torch.Tensor):
        """
            desc: model forward function
            params:
                images (torch.Tensor): _description_        
            return:
                _type_: _description_
        """
        latents = self.post_quant(features)
        output = self.model(latents)
        return output

if __name__ == "__main__":
    # load conf
    # load model
    confdir = "/home/work/daixingshuo/RSISR/configs/model.yaml"
    conf = OmegaConf.load(confdir)
    decoder = Decoder(conf)
    # test
    from model.encoder import Encoder
    encoder = Encoder(conf)
    test_images = []
    test_img1 = '/home/work/daixingshuo/RSISR/test_demo/intersection94.png'
    test_img2 = '/home/work/daixingshuo/RSISR/test_demo/agricultural08.png'
    test_img1 = Image.open(test_img1).convert("RGB")
    test_img1 = np.array(test_img1)
    test_images.append(test_img1)
    test_img2 = Image.open(test_img2).convert("RGB")
    test_img2 = np.array(test_img2)
    test_images.append(test_img2)
    test_tensors = encoder.preprocess(test_images)
    posterior = encoder(test_tensors).latent_dist
    z = posterior.mode()
    output_features = decoder(z)
    print(output_features.shape)