from os import listdir
from os.path import join

from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, transforms, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
from torchvision.transforms.functional import InterpolationMode


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def is_type_image_file(filename, type_name):
    return filename.startswith(type_name) and any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def calculate_valid_crop_size(crop_size, upscale_factor):
    new_crop_size = [n - (n % upscale_factor) for n in crop_size]
    return new_crop_size

def img_pre_process(image: Image, crop_size):
    w, h = image.size
    if h != w:
        if h > w:
            image = image.rotate(90)
        w, h = image.size
        H, W = crop_size
        # 大于就padding
        if H > h or W > w:
            padh1 = int((H - h) / 2)
            padh2 = H - h - padh1
            padw1 = int((W - w) / 2)
            padw2 = W - w - padw1
            #  (left, top, right, bottom)
            image = ImageOps.expand(image, border=(padw1,padh1,padw2,padh2))
            # reflect 填充
            reflect = False
            if reflect:
                to_tensor = transforms.ToTensor()
                image_tensor = to_tensor(image)
                # 定义padding参数，例如：padding_left, padding_right, padding_top, padding_bottom
                padding = (padw1,padw2,padh1,padh2) 
                padded_image_tensor = torch.nn.functional.pad(image_tensor, padding, mode='reflect', value=0)  # 填充0，默认模式是'constant'
                # 如果需要，将结果转换回PIL Image以便显示或保存
                to_pil = transforms.ToPILImage()
                image = to_pil(padded_image_tensor)
    return image

# 将输入的高分辨率图像调整到指定的大小（crop_size x crop_size），使用BICUBIC插值进行插值，以保留图像的质量和细节
# 将处理后的图像转换为PyTorch张量的格式
def train_hr_transform(original_size):
    w, h = original_size
    return Compose([
        Resize((w,h), interpolation=InterpolationMode.BICUBIC),
        CenterCrop(original_size),
        #RandomCrop(crop_size),
        ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def train_lr_transform(crop_size, upscale_factor):
    """
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])
    """
    w, h = crop_size
    transformed = Compose([
        ToPILImage(),
        Resize((w // upscale_factor, h // upscale_factor), interpolation=InterpolationMode.BICUBIC),
        ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # print("After resizing: size =", crop_size // upscale_factor)
    return transformed


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, original_size, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        self.crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(original_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        # self.crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        hr_image = img_pre_process(hr_image,self.crop_size)
        hr_image = self.hr_transform(hr_image)
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir,crop_size, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        self.crop_size = crop_size
    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        hr_image = img_pre_process(hr_image,self.crop_size)
        w, h = self.crop_size
        lr_scale = Resize((w // self.upscale_factor, h // self.upscale_factor), interpolation=InterpolationMode.BICUBIC)
        hr_scale = Resize((w,h), interpolation=InterpolationMode.BICUBIC)
        # hr_image = hr_scale(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image),  ToTensor()(hr_restore_img)

    def __len__(self):
        return len(self.image_filenames)

class ValDatasetFromFolder2(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(ValDatasetFromFolder2, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        # crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)
        self.crop_size = crop_size
    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        hr_scale = Resize(self.crop_size, interpolation=InterpolationMode.BICUBIC)
        hr_restore_img = hr_scale(lr_image)       
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)

class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir
        self.hr_path = dataset_dir
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=InterpolationMode.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        #返回一个包含图像名称、低分辨率图像Tensor、恢复后的高分辨率图像Tensor和原始高分辨率图像Tensor的元组。
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)
        

    def __len__(self):
        return len(self.lr_filenames)

class TestDatasetFromFolder2(Dataset):
    def __init__(self, dataset_dir,crop_size, original_size, upscale_factor):
        super(TestDatasetFromFolder2, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        self.crop_size = crop_size
        self.original_size = original_size
    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        # self.crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        hr_image = img_pre_process(hr_image,self.crop_size)
        lr_scale = Resize((w // self.upscale_factor, h // self.upscale_factor), interpolation=InterpolationMode.BICUBIC)
        hr_scale = Resize((w,h), interpolation=InterpolationMode.BICUBIC)
        hr_image = hr_scale(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_image), ToTensor()(hr_restore_img)
    def __len__(self):
        return len(self.image_filenames)
    

class TestDatasetFromFolderinType1(Dataset):
    def __init__(self, dataset_dir,crop_size, original_size, upscale_factor, type_name):
        super(TestDatasetFromFolder2, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_type_image_file(x, type_name)]
        self.crop_size = crop_size
        self.original_size = original_size
    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        lr_scale = Resize(self.crop_size // self.upscale_factor, interpolation=InterpolationMode.BICUBIC)
        hr_scale = Resize((self.original_size,self.original_size), interpolation=InterpolationMode.BICUBIC)
        hr_image = hr_scale(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_image), ToTensor()(hr_restore_img)
    def __len__(self):
        return len(self.image_filenames)
    

class TestDatasetFromFolderinType(Dataset):
    def __init__(self, dataset_dir, crop_size, original_size, upscale_factor, type_name, n_augmentations, augment=False, inference_mode=False):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_type_image_file(x, type_name)]
        self.crop_size = crop_size
        self.original_size = original_size
        self.augment = augment
        self.inference_mode = inference_mode
        self.n_augmentations = n_augmentations
        # 定义扩增变换
        if augment:
            self.augment_transform = Compose([
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                RandomRotation(degrees=90)
            ])

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        lr_scale = Resize(self.crop_size // self.upscale_factor, interpolation=InterpolationMode.BICUBIC)
        hr_scale = Resize((self.original_size,self.original_size), interpolation=InterpolationMode.BICUBIC)

        if self.inference_mode:
            augmented_hr_images = [self.augment_transform(hr_image) for _ in range(self.n_augmentations)]  # N_AUGMENTATIONS为扩增次数
            augmented_lr_images = [lr_scale(aug_hr_img) for aug_hr_img in augmented_hr_images]
            hr_images = torch.stack(augmented_hr_images)
            lr_images = torch.stack(augmented_lr_images)
        else:
            hr_images = hr_image.unsqueeze(0)
            lr_scale = Resize(self.crop_size // self.upscale_factor, interpolation=InterpolationMode.BICUBIC)
            lr_images = lr_scale(hr_images).unsqueeze(0)

        hr_restore_imgs = hr_scale(lr_images)
        # 返回处理后的图像
        return ToTensor()(lr_images), ToTensor()(hr_images), ToTensor()(hr_restore_imgs)

    def __len__(self):
        return len(self.image_filenames)
