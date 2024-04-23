from diffusers import AutoencoderKL
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import List
import traceback
import numpy as np
import cv2
from PIL import Image
from omegaconf import OmegaConf

from log import logger
from model.vae import DiagonalGaussianDistribution
from model.utils.modeling_output import AutoencoderKLOutput

class Encoder(nn.Module):
    """
        init AutoencoderKL encoder model
    """
    def __init__(self, conf):
        """
            desc:
            params:
        """
        super().__init__()
        self.model_path = conf.model_path
        self.model = None
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
            logger.error(f"init pre-train encoder model error: {str(e)}, {traceback.format_exc()}")
            self.model = None

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
            logger.error(f"init pre-train quant layer error: {str(e)}, {traceback.format_exc()}")
            self.quant_conv = None

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
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        for i in range(len(images)):
            img = cv2.resize(images[i], (256, 256))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = transform(img)
            images[i] = torch.tensor(img)
        images = torch.stack(images)
        return images

    def forward(self, images: torch.Tensor, return_dict: bool = True):
        """
            desc: model forward function
            params:
                images (torch.Tensor): _description_        
            return:
                _type_: _description_
        """
        features = self.model(images)
        moments = self.quant_conv(features)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return moments, posterior

        return AutoencoderKLOutput(latent_dist=posterior)

if __name__ == "__main__":
    # load conf
    # load model
    confdir = "/home/work/daixingshuo/RSISR/configs/model.yaml"
    conf = OmegaConf.load(confdir)
    encoder = Encoder(conf)
    # test
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
    # print(test_tensors.shape)
    features = encoder(test_tensors)
    #print(features.shape)