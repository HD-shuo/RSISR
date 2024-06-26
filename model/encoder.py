from diffusers import AutoencoderKL
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchsummary import summary
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
            images[i] = img.clone().detach() 
        images = torch.stack(images)
        return images
    # hook函数
    def register_hooks(self):  
        # 遍历预训练模型的所有模块，为它们注册前向钩子  
        for name, module in self.model.named_modules():  
            if isinstance(module, nn.Module):  # 确保是模块，而非其他（如参数）  
                self.hooks.append(module.register_forward_hook(hook_function))  
  
    def remove_hooks(self):  
        # 移除之前注册的所有钩子  
        for hook in self.hooks:  
            hook.remove()  

    def forward(self, images: torch.Tensor, return_dict: bool = True):
        """
            desc: model forward function
            params:
                images (torch.Tensor): _description_        
            return:
                _type_: _description_
        """
        # summary(self.model, (3,256,256))
        # nan test
        #print("Input stats - mean: {}, std: {}, min: {}, max: {}".format(images.mean().item(), images.std().item(), images.min().item(), images.max().item()))
        # self.register_hooks()  
        # for name, layer in self.model.named_children():
        #     x = layer(x)
        #     print(f"Output of layer {name} - mean: {x.mean().item()}, std: {x.std().item()}, min: {x.min().item()}, max: {x.max().item()}")
        #     if torch.isnan(x).any():
        #         print(f"NaN detected after layer: {name}")
        #         break
        for param in self.model.parameters():
            param.requires_grad = False
        
        # for name, param in self.model.named_parameters():
            # print(f"参数名称: {name}")
            # print(f"参数值: {param.data}")
            # if torch.isnan(param.data).any():
                # print(f"NaN detected after layer: {name}")
        features = self.model(images)
        #self.remove_hooks()
        moments = self.quant_conv(features)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return moments, posterior

        return AutoencoderKLOutput(latent_dist=posterior)

def hook_function(module, input, output):  
    print(f"Module name: {module.__class__.__name__}")  
    print(f"Output shape: {output.shape}")  
    print("Output stats - mean: {}, std: {}, min: {}, max: {}".format(output.mean().item(), output.std().item(), output.min().item(), output.max().item()))
    if torch.isnan(output).any():
        print(f"NaN detected after layer: {module.__class__.__name__}")
        
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