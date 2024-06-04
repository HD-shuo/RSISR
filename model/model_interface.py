import inspect
import torch
import numpy as np
from PIL import Image
import os
import cv2
import importlib
from torch import nn
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
import torchvision.models as models
from torchvision.models.vgg import VGG16_Weights

import pytorch_lightning as pl
from model.metrics import tensor_accessment
from model.utils.utils import quantize
from model.utils.loss_function import PerceptualLoss, mixed_loss
from model.fid_score import calculate_fid_score


class MInterface(pl.LightningModule):
    def __init__(self, model_name, loss, lr, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()
        #self.configure_optimizers()

        # Project-Specific Definitions

    def forward(self, batch):
        #lr, hr, _ = batch
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        lr, hr  = batch
        input_tensor = lr.clone()
        sr_rgb = self(input_tensor)  # 使用模型进行高分辨率重建得到 RGB 图像
        loss = self.loss_function(sr_rgb, hr)  # 计算损失函数
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        lr, hr, _ = batch
        input_tensor = lr.clone().detach()
        #input_tensor = torch.tensor(lr)
        sr_rgb = self(input_tensor)
        #print(sr_rgb.size())
        #sr_rgb = self(lr, hr)  # 使用模型进行高分辨率重建得到 RGB 图像
        if sr_rgb.dtype == torch.bfloat16:
            sr_rgb = sr_rgb.float()
        sr_rgb = quantize(sr_rgb, self.hparams.color_range)  # 对重建的图像进行量化处理（可选）
        mpsnr, mssim, lpips, _, _ = tensor_accessment(
            x_pred=sr_rgb.cpu().numpy(),
            x_true=hr.cpu().numpy(),
            data_range=self.hparams.color_range,
            multi_dimension=False)
        #new fid score calculation
        fid_score = calculate_fid_score(hr.cpu().numpy(), sr_rgb.cpu().numpy(), hr.size(0))
        self.log("fid_score", fid_score,on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('mpsnr', mpsnr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('mssim', mssim, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('lpips', lpips, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        lr, hr, _ = batch
        input_tensor = lr.clone().detach()
        #input_tensor = torch.tensor(lr)
        sr_rgb = self(input_tensor)
        #print(sr_rgb.size())
        #sr_rgb = self(lr, hr)  # 使用模型进行高分辨率重建得到 RGB 图像
        if sr_rgb.dtype == torch.bfloat16:
            sr_rgb = sr_rgb.float()
        sr_rgb = quantize(sr_rgb, self.hparams.color_range)  # 对重建的图像进行量化处理（可选）
        mpsnr, mssim, lpips, _, _ = tensor_accessment(
            x_pred=sr_rgb.cpu().numpy(),
            x_true=hr.cpu().numpy(),
            data_range=self.hparams.color_range,
            multi_dimension=False)
        #new fid score calculation
        fid_score = calculate_fid_score(hr.cpu().numpy(), sr_rgb.cpu().numpy(), hr.size(0))
        self.log("fid_score", fid_score,on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('mpsnr', mpsnr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('mssim', mssim, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('lpips', lpips, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        save_path = "/share/program/dxs/Database/data/pngimage_restore"
        for i in range(sr_rgb.size(0)):
            img = sr_rgb[i].cpu().numpy().transpose((1,2,0))
            img = Image.fromarray((img * 255).astype(np.uint8))
            filename = f'restore_{i}.png'
            path = os.path.join(save_path, filename)
            img.save(path)  

    def on_validation_epoch_end(self):
        # Make the Progress Bar leave there
        self.print('')
        # self.print(self.get_progress_bar_dict())

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                print('Creating StepLR scheduler')
                print('Step size:', self.hparams.lr_decay_steps)
                print('Gamma:', self.hparams.lr_decay_rate)
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_T_max,
                                                  eta_min=self.hparams.min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            #return [optimizer], [scheduler]
            return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'metric_to_monitor',
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def configure_loss(self):
        loss = self.hparams.loss.lower()
        perceptual_loss = PerceptualLoss()
        if loss == 'mse':
            self.loss_function = F.mse_loss
        elif loss == 'l1':
            self.loss_function = F.l1_loss
        elif loss == 'p_loss':
            self.loss_function = perceptual_loss
        elif loss == 'mix':
            self.loss_function = mixed_loss
        else:
            raise ValueError("Invalid Loss Type!")

    def load_model(self):
        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.model = self.instancialize(Model)

    def instancialize(self, Model, class_name="Model", **other_args):
        init_sig = inspect.signature(Model.__init__)
        class_params = {}
        for param in init_sig.parameters.values():
            if param.default != inspect.Parameter.empty:
                class_params[param.name] = param.default
            else:
                class_params[param.name] = None
        class_args = list(class_params.keys())
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)
    
# class VGGFeatures(nn.Module):
#     def __init__(self):
#         super(VGGFeatures, self).__init__()
#         vgg16 = models.vgg16(pretrained=True)
#         self.features = nn.Sequential(*list(vgg16.features.children())[:-1])
        
#     def forward(self, x):
#         x = self.features(x)
#         return x
        
# def perceptual_loss(img1, img2):
#     vgg = VGGFeatures().cuda()
#     vgg.eval()
    
#     img1_feat = vgg(img1)
#     img2_feat = vgg(img2)
    
#     loss = F.l1_loss(img1_feat, img2_feat)
#     return loss

# def mixed_loss(img1, img2):
#     p_loss = perceptual_loss(img1, img2)
#     mse_loss = F.mse_loss(img1, img2)
#     mix_loss = p_loss + mse_loss
#     return mix_loss