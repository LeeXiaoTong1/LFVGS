#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch.autograd import Variable
from math import exp
import math
import torch.nn.functional as F
from torchmetrics.functional.regression import pearson_corrcoef


# def compute_patchwise_pearson(midas_depth, rendered_depth, patch_size=(5, 17), stride=(5, 17)):
#     H, W = 504, 378
#     patch_H, patch_W = patch_size
#     stride_H, stride_W = stride

#     # 确保数据大小正确
#     if midas_depth.shape[0] != H * W or rendered_depth.shape[0] != H * W:
#         raise ValueError(f"Input size mismatch: expected {H*W} elements, but got {midas_depth.shape[0]} and {rendered_depth.shape[0]} elements")
    
#     # Reshape 成 (H, W) 的二维张量
#     midas_depth = midas_depth.view(1, 1, H, W)  # (N=1, C=1, H, W)
#     rendered_depth = rendered_depth.view(1, 1, H, W)

#     # 使用 unfold 获取所有 patches
#     midas_patches = F.unfold(midas_depth, kernel_size=(patch_H, patch_W), stride=(stride_H, stride_W))
#     rendered_patches = F.unfold(rendered_depth, kernel_size=(patch_H, patch_W), stride=(stride_H, stride_W))

#     # 展平 patches 为 2D 张量 (num_patches, patch_size * patch_size)
#     midas_patches = midas_patches.view(midas_patches.size(1), -1)  # (num_patch_elements, num_patches)
#     rendered_patches = rendered_patches.view(rendered_patches.size(1), -1)  # (num_patch_elements, num_patches)

#     # 计算 Pearson 相关性
#     pcc_1 = pearson_corrcoef(-midas_patches, rendered_patches)
#     pcc_2 = pearson_corrcoef(1 / (midas_patches + 200.), rendered_patches)

#     # 计算最小损失
#     loss = torch.min(1 - pcc_1, 1 - pcc_2).mean()
#     return loss

def patchify(input, patch_size):
    patches = F.unfold(input, kernel_size=patch_size, stride=patch_size).permute(0,2,1).view(-1, 1*patch_size*patch_size)
    return patches

def margin_l2_loss(network_output, gt, margin, return_mask=False):
    mask = (network_output - gt).abs() > margin
    if not return_mask:
        return ((network_output - gt)[mask] ** 2).mean()
    else:
        return ((network_output - gt)[mask] ** 2).mean(), mask

def normalize(input, mean=None, std=None):
    input_mean = torch.mean(input, dim=1, keepdim=True) if mean is None else mean
    input_std = torch.std(input, dim=1, keepdim=True) if std is None else std
    return (input - input_mean) / (input_std + 1e-2*torch.std(input.reshape(-1)))

def patch_norm_mse_loss(input, target, patch_size, margin, return_mask=False):
    input_patches = normalize(patchify(input, patch_size))
    target_patches = normalize(patchify(target, patch_size))
    return margin_l2_loss(input_patches, target_patches, margin, return_mask)

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l1_loss_mask(network_output, gt, mask = None):
    if mask is None:
        return l1_loss(network_output, gt)
    else:
        return torch.abs((network_output - gt) * mask).sum() / mask.sum()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, mask=None, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if mask is not None:
        img1 = img1 * mask + (1 - mask)
        img2 = img2 * mask + (1 - mask)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)




