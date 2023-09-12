from os import listdir
from os.path import join

from PIL import Image
import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize,transforms

def train_hr_transform(crop_size):
    return Compose([
        Resize((crop_size,crop_size), interpolation=Image.BICUBIC),
        #RandomCrop(crop_size),
        ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def train_lr_transform(crop_size, upscale_factor):
    transformed = Compose([
        
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])
    print("After resizing: size =", crop_size // upscale_factor)
    return transformed

if __name__ == "__main__":
    #dataset_dir = "/share/program/dxs/Database/data/DIV2K_test_HR"
    #image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
    img_path = "/share/program/dxs/RSISR/test_demo/intersection94.png"
    save_path = "/share/program/dxs/RSISR/test_demo/lr_intersection94.png"
    crop_size = 256
    upscale_factor = 4
    transform_function = train_lr_transform(crop_size, upscale_factor)
    img = Image.open(img_path)
    lr_img = transform_function(img).numpy()
    lr_img = (lr_img*255).astype(np.uint8)
    lr_img = np.transpose(lr_img, (1, 2, 0))
    image = Image.fromarray(lr_img)
    image.save(save_path)