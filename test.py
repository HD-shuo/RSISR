import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import torch
from torchvision.transforms import ToTensor
from PIL import Image
from pytorch_lightning import LightningModule, Trainer

import os
from omegaconf import OmegaConf
import numpy as np
import scipy.misc
from model import MInterface


def test(conf, importpath, savepath):
    # 加载模型权重
    model = MInterface(**conf.model)

    #指定显卡
    # device = torch.device('cuda:2')
    # model.to(device)

    model.load_state_dict(torch.load('/share/program/dxs/RSISR/checkpoint/last.ckpt'), strict=False)


    # 预处理图像
    image_path = importpath
    image = Image.open(image_path)
    image = ToTensor()(image)  # 转换为Tensor

    # 将图像移动到指定的显卡设备
    #image = image.to(device)

    # 执行推断
    model.eval()
    with torch.no_grad():
        output = model(image.unsqueeze(0))  # 添加批次维度

    # 后处理输出
    result = output.squeeze().numpy()  # 移除批次维度并转换为NumPy数组
    result = (result*255).astype(np.uint8)
    """
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            for k in range(result.shape[2]):
                element = result[i, j, k]
                if element > 255:
                    print("error")
    """
    img_np = np.transpose(result, (1, 2, 0))
    print(img_np.shape)
    image_pil = Image.fromarray(img_np)

    img_name = os.path.basename(importpath)
    savepath = savepath + 'hr_' + img_name

    # 查看结果
    #scipy.misc.imsave(savepath, result)
    image_pil.save(savepath)

if __name__ == "__main__":
    lr_path = "/share/program/dxs/RSISR/test_demo/intersection94.png"
    hr_path = "/share/program/dxs/RSISR/test_demo/"
    configdir = "/share/program/dxs/RSISR/configs/ptp.yaml"
    conf = OmegaConf.load(configdir)
    test(conf, lr_path, hr_path)