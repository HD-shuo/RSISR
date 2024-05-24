import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
import numpy as np

def calculate_activation_statistics(images, model, batch_size=50, dims=2048):
    model.eval()
    dataloader = torch.utils.data.DataLoader(images, batch_size=batch_size)
    act_values = np.empty((len(images), dims))

    start_idx = 0
    for batch in dataloader:
        batch = batch.cuda()
        with torch.no_grad():
            act = model(batch)

        if act.dtype == torch.bfloat16:
            act = act.float()
        act = act.cpu().numpy()
        batch_size = act.shape[0]
        act_values = np.resize(act_values, act.shape)
        act_values[start_idx:start_idx+batch_size] = act
        start_idx += batch_size

    mu = np.mean(act_values, axis=0)
    sigma = np.cov(act_values, rowvar=False)

    return mu, sigma

def calculate_fid_score(real_images: np.array, generated_images, batch_size=50):
    # 调整批次维度为 [batch_size * num_images, channels, height, width]
    # Load Inception-v3 model pretrained on ImageNet
    inception_model = inception_v3(pretrained=True)
    inception_model.cuda()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    realimage_list = []
    for image in real_images:
        assert np.all((image >= 0)&(image<=1))
        image = (image * 255).astype(np.uint8)
        image = np.transpose(image, (1, 2, 0))
        image = TF.to_pil_image(image)
        image = transform(image)
        realimage_list.append(image)

    real_tensor = np.stack(realimage_list, axis=0)
    real_images = torch.from_numpy(real_tensor)
    #real_images = torch.cat(realimage_list, dim=1)

    genimage_list = []
    for image in generated_images:
        image = (image * 255).astype(np.uint8)
        image = np.transpose(image, (1, 2, 0))
        image = TF.to_pil_image(image)
        image = transform(image)
        genimage_list.append(image)

    generated_tensor = np.stack(genimage_list, axis=0)
    generated_images = torch.from_numpy(generated_tensor)
    #generated_images = torch.cat(genimage_list, dim=0)


    # Calculate activation statistics for real and generated images
    real_mu, real_sigma = calculate_activation_statistics(real_images, inception_model, batch_size)
    generated_mu, generated_sigma = calculate_activation_statistics(generated_images, inception_model, batch_size)

    # Calculate FID score
    epsilon = 1e-6
    cov_sqrt = sqrtm(np.dot(real_sigma, generated_sigma))
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    fid_score = np.sum((real_mu - generated_mu)**2) + np.trace(real_sigma + generated_sigma - 2 * cov_sqrt)
    fid_score += epsilon

    return fid_score
