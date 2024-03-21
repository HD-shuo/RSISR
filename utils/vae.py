from diffusers import AutoencoderKL
from safetensors import safe_open
from safetensors.torch import save_file

from PIL import Image
import torchvision.transforms as transforms
import torch
import cv2
import numpy as np

model_path = '/home/work/daixingshuo/RSISR/pretrain_weights'
model_home = '/home/work/daixingshuo/RSISR/pretrain_weights/diffusion_pytorch_model.fp16.safetensors'

def load_vae_model(model_path):
    # load vae model
    vae_model = AutoencoderKL.from_pretrained(model_path)
    # print(vae_model)
    return vae_model

# load encoder
def test_models(vae_model):
    test_images = []
    test_img1 = '/home/work/daixingshuo/RSISR/test_demo/intersection94.png'
    test_img2 = '/home/work/daixingshuo/RSISR/test_demo/agricultural08.png'
    test_img1 = Image.open(test_img1).convert("RGB")
    test_img1 = np.array(test_img1)
    test_images.append(test_img1)
    test_img2 = Image.open(test_img2).convert("RGB")
    test_img2 = np.array(test_img2)
    test_images.append(test_img2)

    for i in range(len(test_images)):
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
        img = cv2.resize(test_images[i], (512, 512))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform(img)
        test_images[i] = torch.tensor(img)
    images = torch.stack(test_images)

    # img = vae_model.encoder(images)
    img = vae_model.decoder(images)
    idx = 0
    #for i in img:
        # print(i.size())
    for i in img.sample:
        i = i.detach().numpy()
        i = i * 255
        i = cv2.cvtColor(np.transpose(i, (1, 2, 0)), cv2.COLOR_BGR2RGB)
        idx += 1
        cv2.imwrite(f"/home/work/daixingshuo/RSISR/test_demo/{idx}.png", i)
    print(img.sample.size())

def get_keys(model_home):
    with safe_open(model_home, framework="pt", device = 0) as f:
        for k in f.keys():
            print(k)
        keys = list(f.keys())
        return keys

def get_model_part(model_home):
    tensors = {}
    with safe_open(model_home, framework="pt", device = 0) as f:
        keys = list(f.keys())
        for k in keys:
            if k.startswith("decoder"):
                tensors[k] = f.get_tensor(k)
                save_file(tensors, "/home/work/daixingshuo/RSISR/pretrain_weights/decoder.safetensors")

def get_model_slice(model_home):
    with safe_open(model_home, framework="pt", device = 0) as f:
        decoder_slice_1 = f.get_slice("decoder.conv_in.bias")
        #decoder_slice_2 = f.get_slice("decoder.conv_in.weight")
        print(decoder_slice_1.get_shape())
        #print(decoder_slice_2.get_shape())

if __name__ == "__main__":
    model_path = '/home/work/daixingshuo/RSISR/pretrain_weights'
    model_home = '/home/work/daixingshuo/RSISR/pretrain_weights/diffusion_pytorch_model.fp16.safetensors'
    encoder = '/home/work/daixingshuo/RSISR/pretrain_weights/encoder.safetensors'
    decoder = '/home/work/daixingshuo/RSISR/pretrain_weights/decoder.safetensors'
    #get_keys(encoder)
    get_model_slice(model_home)