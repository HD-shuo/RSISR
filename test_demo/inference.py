import torch
import torch.nn as nn
from torchvision.transforms import Compose,ToTensor, ToPILImage, Resize
from safetensors import safe_open

from omegaconf import OmegaConf
from PIL import Image
import numpy as np
import cv2

from model.upscale_model import UpscaleModel


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
        images[i] = img.clone().detach() 
    images = torch.stack(images)
    return images

def get_keys(model_home):
    with safe_open(model_home, framework="pt", device = 0) as f:
        for k in f.keys():
            print(k)
        keys = list(f.keys())
        return keys

if __name__ == "__main__":
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    # load conf
    confdir = "/share/program/dxs/RSISR/configs/model.yaml"
    conf = OmegaConf.load(confdir)
    ckpt_path = "/share/program/dxs/RSISR/checkpoint/best-epoch=17-mpsnr=62.72-mssim=0.994--fid_score=2436.29--lpips=0.72.ckpt"
    pretrain_path = "/share/program/dxs/RSISR/pretrain_weights/diffusion_pytorch_model.fp16.safetensors"
    # load model
    upscale_model = UpscaleModel(conf)
    # merge dict
    ckpt_params = torch.load(ckpt_path)
    all_params = {}
    for k, v in upscale_model.state_dict().items():
        if k in list(ckpt_params.keys()):
            all_params[k] = ckpt_params[k]
        else:
            with safe_open(pretrain_path, framework="pt", device = 0) as f:
                if k in f.keys():
                    all_params[k] = f[k]

    upscale_model.load_state_dict(all_params)
    upscale_model.to(device)
    upscale_model.eval()
    # test
    # time embedding
    test_images = []
    test_img1 = '/share/program/dxs/RSISR/test_demo/airplane08.png'
    test_img2 = '/share/program/dxs/RSISR/test_demo/freeway26.png'
    test_img1 = Image.open(test_img1).convert("RGB")
    test_img1 = np.array(test_img1)
    test_images.append(test_img1)
    test_img2 = Image.open(test_img2).convert("RGB")
    test_img2 = np.array(test_img2)
    test_images.append(test_img2)
    test_tensors = train_lr_transform(256, 4, test_images)
    with torch.no_grad():
        outputs = upscale_model(test_tensors)
    outputs = (outputs.detach().cpu().numpy() * 0.5) + 0.5 
    for idx, img in enumerate(outputs):
        image_tensor = img.transpose((1, 2, 0))  # 变换到PIL需要的[H, W, C]格式  
        image = Image.fromarray((image_tensor * 255).astype(np.uint8))  
        image.save('test_demo/output_image_{}.png'.format(idx))
