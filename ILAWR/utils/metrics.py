import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def make_gaussian_kernel(win_size, sigma):
    coords = torch.arange(win_size, dtype=torch.float32)
    g = torch.exp(-((coords - win_size // 2) ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def make_2d_window(win_size, n_channels):
    kernel_1d = make_gaussian_kernel(win_size, 1.5).unsqueeze(1)
    kernel_2d = kernel_1d.mm(kernel_1d.t()).unsqueeze(0).unsqueeze(0)
    window = kernel_2d.expand(n_channels, 1, win_size, win_size).contiguous()
    return Variable(window)


def compute_ssim(img1, img2, win_size=11, size_average=True):
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    n_ch = img1.size(1)
    window = make_2d_window(win_size, n_ch)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    pad = win_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=n_ch)
    mu2 = F.conv2d(img2, window, padding=pad, groups=n_ch)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu12 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=n_ch) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=n_ch) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=n_ch) - mu12

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(1).mean(1).mean(1)


def compute_psnr(pred, gt):
    pred_np = pred.clamp(0, 1).cpu().numpy()
    gt_np = gt.clamp(0, 1).cpu().numpy()
    mse = np.mean((pred_np - gt_np) ** 2)
    if mse == 0:
        return 100.0
    return 20 * math.log10(1.0 / math.sqrt(mse))
