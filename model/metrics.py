import numpy as np
from scipy.signal import convolve2d
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
import torch
import torchvision.transforms as transforms

def compare_rmse(x_true, x_pred):
    """
    Calculate Root mean squared error
    :param x_true:
    :param x_pred:
    :return:
    """
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    return np.linalg.norm(x_true - x_pred) / (np.sqrt(x_true.shape[0] * x_true.shape[1] * x_true.shape[2]))


def compare_mpsnr(x_true, x_pred, data_range, detail=False):
    """
    :param x_true: Input image must have three dimension (H, W, C)
    :param x_pred:
    :return:
    """
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    channels = x_true.shape[2]
    total_psnr = [peak_signal_noise_ratio(image_true=x_true[:, :, k], image_test=x_pred[:, :, k], data_range=data_range)
                  for k in range(channels)]
    if detail:
        return np.mean(total_psnr), total_psnr
    else:
        return np.mean(total_psnr)


def compare_mssim(x_true, x_pred, data_range, multidimension, detail=False):
    """
    :param x_true:
    :param x_pred:
    :param data_range:
    :param multidimension:
    :return:
    """
    mssim = [structural_similarity(x_true[:, :, i], x_pred[:, :, i], data_range=data_range, multidimension=multidimension)
             for i in range(x_true.shape[2])]
    if detail:
        return np.mean(mssim), mssim
    else:
        return np.mean(mssim)


def compare_mare(x_true, x_pred):
    """

    :param x_true:
    :param x_pred:
    :return:
    """
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    diff = x_true - x_pred
    abs_diff = np.abs(diff)
    # added epsilon to avoid division by zero.
    relative_abs_diff = np.divide(abs_diff, x_true + 1)
    return np.mean(relative_abs_diff)


def img_qi(img1, img2, block_size=8):
    N = block_size ** 2
    sum2_filter = np.ones((block_size, block_size))

    img1_sq = img1 * img1
    img2_sq = img2 * img2
    img12 = img1 * img2

    img1_sum = convolve2d(img1, np.rot90(sum2_filter), mode='valid')
    img2_sum = convolve2d(img2, np.rot90(sum2_filter), mode='valid')
    img1_sq_sum = convolve2d(img1_sq, np.rot90(sum2_filter), mode='valid')
    img2_sq_sum = convolve2d(img2_sq, np.rot90(sum2_filter), mode='valid')
    img12_sum = convolve2d(img12, np.rot90(sum2_filter), mode='valid')

    img12_sum_mul = img1_sum * img2_sum
    img12_sq_sum_mul = img1_sum * img1_sum + img2_sum * img2_sum
    numerator = 4 * (N * img12_sum - img12_sum_mul) * img12_sum_mul
    denominator1 = N * (img1_sq_sum + img2_sq_sum) - img12_sq_sum_mul
    denominator = denominator1 * img12_sq_sum_mul
    quality_map = np.ones(denominator.shape)
    index = (denominator1 == 0) & (img12_sq_sum_mul != 0)
    quality_map[index] = 2 * img12_sum_mul[index] / img12_sq_sum_mul[index]
    index = (denominator != 0)
    quality_map[index] = numerator[index] / denominator[index]
    return quality_map.mean()


def cacul_lpips(img1, img2):
    # 加载预训练的LPIPS网络
    lpips_net = lpips.LPIPS(net='alex')
    # 转换图像为模型可接受的格式
    transform = transforms.ToTensor()
    # 加载并转换图像
    image1 = transform(img1)  # image1是第一个图像的数据，可以是PIL图像或NumPy数组
    image2 = transform(img2)  # image2是第二个图像的数据，可以是PIL图像或NumPy数组

    # 将图像扩展为四维张量 (batch_size=1)
    image1 = image1.unsqueeze(0)
    image2 = image2.unsqueeze(0)

    # 将图像输入LPIPS网络进行计算
    distance = lpips_net(image1, image2)
    return distance.item()



def quality_assessment(x_true, x_pred, data_range, multi_dimension=False):
    """
    :param multi_dimension:
    :param ratio:
    :param data_range:
    :param x_true:
    :param x_pred:
    :param block_size
    :return:
    """
    result = {'MPSNR': compare_mpsnr(x_true=x_true, x_pred=x_pred, data_range=data_range),
              'MSSIM': compare_mssim(x_true=x_true, x_pred=x_pred, data_range=data_range,
                                     multidimension=multi_dimension),
              #   'ERGAS': compare_ergas(x_true=x_true, x_pred=x_pred, ratio=ratio),
              'RMSE': compare_rmse(x_true=x_true, x_pred=x_pred),
              }
    return result


def baseline_assessment(x_true, x_pred, data_range, multi_dimension=False):
    mpsnr, psnrs = compare_mpsnr(x_true=x_true, x_pred=x_pred, data_range=data_range, detail=True)
    mssim, ssims = compare_mssim(x_true=x_true, x_pred=x_pred, data_range=data_range,
                                     multidimension=multi_dimension, detail=True)
    return mpsnr, mssim, psnrs, ssims


def tensor_accessment(x_true, x_pred, data_range, multi_dimension=False):
    #将输入的Tensor从(batch_size, channels, height, width)的形式转置为(batch_size, height, width, channels)的形式，以便后续的评估函数能够正确处理图像数据
    x_true = x_true.transpose(0, 2, 3, 1)[0]
    x_pred = x_pred.transpose(0, 2, 3, 1)[0]
    lpips = cacul_lpips(x_true, x_pred)
    mpsnr, psnrs = compare_mpsnr(x_true=x_true, x_pred=x_pred, data_range=data_range, detail=True)
    mssim, ssims = compare_mssim(x_true=x_true, x_pred=x_pred, data_range=data_range,
                                     multidimension=multi_dimension, detail=True)
    return mpsnr, mssim, lpips, psnrs, ssims


def batch_accessment(x_true, x_pred, data_range, multi_dimension=False):
    scores = []
    avg_score = {'MPSNR': 0, 'MSSIM': 0, 'SAM': 0,
                 'CrossCorrelation': 0, 'RMSE': 0}
    x_true = x_true.transpose(0, 2, 3, 1)
    x_pred = x_pred.transpose(0, 2, 3, 1)

    for i in range(x_true.shape[0]):
        scores.append(quality_assessment(
            x_true[i], x_pred[i], data_range, multi_dimension))
    for met in avg_score.keys():
        avg_score[met] = np.mean([score[met] for score in scores])
    return avg_score