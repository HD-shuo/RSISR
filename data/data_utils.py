from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, transforms
from torchvision.transforms.functional import InterpolationMode


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

# 将输入的高分辨率图像调整到指定的大小（crop_size x crop_size），使用BICUBIC插值进行插值，以保留图像的质量和细节
# 将处理后的图像转换为PyTorch张量的格式
def train_hr_transform(crop_size):
    return Compose([
        Resize((crop_size,crop_size), interpolation=InterpolationMode.BICUBIC),
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
    transformed = Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=InterpolationMode.BICUBIC),
        ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print("After resizing: size =", crop_size // upscale_factor)
    return transformed


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
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
        w, h = hr_image.size
        #crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(self.crop_size // self.upscale_factor, interpolation=InterpolationMode.BICUBIC)
        hr_scale = Resize((self.crop_size,self.crop_size), interpolation=InterpolationMode.BICUBIC)
        hr_image = hr_scale(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)

class ValDatasetFromFolder2(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(ValDatasetFromFolder2, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
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
        self.lr_path = dataset_dir + '/DIV2K_test_HR/'
        self.hr_path = dataset_dir + '/DIV2K_valid_HR/'
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
