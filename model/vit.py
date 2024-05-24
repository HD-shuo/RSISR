import torch
import torchvision
from PIL import Image
import numpy as np
from typing import List
import torchvision.transforms as transforms
import cv2


def preprocess(images: List[np.ndarray]):
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
            img = cv2.resize(images[i], (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = transform(img)
            images[i] = img.clone().detach() 
        images = torch.stack(images)
        return images

if __name__ == "__main__":
    model = torchvision.models.vit_b_16(pretrained=True)
    image = '/share/program/dxs/RSISR/test_demo/airplane08.png'
    test_images = []
    test_img2 = Image.open(image).convert("RGB")
    test_img2 = np.array(test_img2)
    test_images.append(test_img2)
    test_tensors = preprocess(test_images)
    # print(test_tensors.shape)
    features = model(test_tensors)
    print(features.size())
