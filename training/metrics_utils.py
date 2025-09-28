#!/usr/bin/env python3
"""
Utility functions for computing image quality metrics like PSNR and SSIM
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Union


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: Union[float, torch.Tensor] = 1.0) -> torch.Tensor:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between predicted and target images.
    
    Args:
        pred: Predicted images tensor of shape (B, C, H, W) or (B*V, C, H, W)
        target: Target images tensor of same shape as pred
        max_val: Maximum possible pixel value (1.0 for normalized images, 255.0 for uint8)
    
    Returns:
        PSNR value as a scalar tensor
    """
    # Compute MSE
    mse = F.mse_loss(pred, target)
    
    # Handle edge case where MSE is 0 (perfect match)
    if mse == 0:
        return torch.tensor(float('inf'), device=pred.device, dtype=pred.dtype)
    
    # Ensure max_val is a tensor on the same device
    if not isinstance(max_val, torch.Tensor):
        max_val = torch.tensor(max_val, device=pred.device, dtype=pred.dtype)
    
    # Compute PSNR
    psnr = 20.0 * torch.log10(max_val) - 10.0 * torch.log10(mse)
    return psnr


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, 
                 window_size: int = 11, max_val: Union[float, torch.Tensor] = 1.0) -> torch.Tensor:
    """
    Compute Structural Similarity Index (SSIM) between predicted and target images.
    
    Args:
        pred: Predicted images tensor of shape (B, C, H, W) or (B*V, C, H, W)
        target: Target images tensor of same shape as pred
        window_size: Size of the sliding window (should be odd)
        max_val: Maximum possible pixel value (1.0 for normalized images, 255.0 for uint8)
    
    Returns:
        SSIM value as a scalar tensor (average across all pixels and channels)
    """
    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    # Create Gaussian window
    sigma = 1.5
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    gauss = gauss / gauss.sum()
    
    # Create 2D Gaussian kernel
    gauss_kernel = gauss[:, None] @ gauss[None, :]
    gauss_kernel = gauss_kernel.unsqueeze(0).unsqueeze(0)
    
    # Move kernel to same device as input
    gauss_kernel = gauss_kernel.to(pred.device)
    
    # Get number of channels
    num_channels = pred.shape[1]
    
    # Replicate kernel for each channel
    gauss_kernel = gauss_kernel.repeat(num_channels, 1, 1, 1)
    
    # Ensure max_val is a tensor on the same device
    if not isinstance(max_val, torch.Tensor):
        max_val = torch.tensor(max_val, device=pred.device, dtype=pred.dtype)
    
    # Constants for SSIM calculation
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    
    # Compute local means
    mu1 = F.conv2d(pred, gauss_kernel, padding=window_size//2, groups=num_channels)
    mu2 = F.conv2d(target, gauss_kernel, padding=window_size//2, groups=num_channels)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # Compute local variances and covariance
    sigma1_sq = F.conv2d(pred * pred, gauss_kernel, padding=window_size//2, groups=num_channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, gauss_kernel, padding=window_size//2, groups=num_channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, gauss_kernel, padding=window_size//2, groups=num_channels) - mu1_mu2
    
    # Compute SSIM
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim_map = numerator / denominator
    
    # Return mean SSIM across all pixels and channels
    return ssim_map.mean()


def compute_image_metrics(pred: torch.Tensor, target: torch.Tensor, 
                         max_val: Union[float, torch.Tensor] = 1.0) -> dict:
    """
    Compute multiple image quality metrics.
    
    Args:
        pred: Predicted images tensor of shape (B, C, H, W) or (B*V, C, H, W)
        target: Target images tensor of same shape as pred
        max_val: Maximum possible pixel value (1.0 for normalized images, 255.0 for uint8)
    
    Returns:
        Dictionary containing computed metrics
    """
    metrics = {}
    
    # Ensure tensors are on the same device
    if pred.device != target.device:
        target = target.to(pred.device)
    
    # Compute PSNR
    psnr = compute_psnr(pred, target, max_val)
    metrics['psnr'] = psnr
    
    # Compute SSIM
    ssim = compute_ssim(pred, target, max_val=max_val)
    metrics['ssim'] = ssim
    
    # Compute MSE for reference
    mse = F.mse_loss(pred, target)
    metrics['mse'] = mse
    
    # Compute L1 loss for reference
    l1 = F.l1_loss(pred, target)
    metrics['l1'] = l1
    
    return metrics


def batch_compute_metrics(pred_batch: torch.Tensor, target_batch: torch.Tensor,
                         max_val: Union[float, torch.Tensor] = 1.0) -> dict:
    """
    Compute metrics for a batch of images, returning per-sample and average metrics.
    
    Args:
        pred_batch: Batch of predicted images (B, C, H, W)
        target_batch: Batch of target images (B, C, H, W)
        max_val: Maximum possible pixel value
    
    Returns:
        Dictionary with per-sample metrics and batch averages
    """
    batch_size = pred_batch.shape[0]
    
    # Initialize lists to store per-sample metrics
    psnr_values = []
    ssim_values = []
    mse_values = []
    l1_values = []
    
    # Compute metrics for each sample in the batch
    for i in range(batch_size):
        pred_sample = pred_batch[i:i+1]  # Keep batch dimension
        target_sample = target_batch[i:i+1]
        
        sample_metrics = compute_image_metrics(pred_sample, target_sample, max_val)
        
        psnr_values.append(sample_metrics['psnr'].item())
        ssim_values.append(sample_metrics['ssim'].item())
        mse_values.append(sample_metrics['mse'].item())
        l1_values.append(sample_metrics['l1'].item())
    
    # Compute batch averages
    return {
        'psnr_per_sample': psnr_values,
        'ssim_per_sample': ssim_values,
        'mse_per_sample': mse_values,
        'l1_per_sample': l1_values,
        'avg_psnr': np.mean(psnr_values),
        'avg_ssim': np.mean(ssim_values),
        'avg_mse': np.mean(mse_values),
        'avg_l1': np.mean(l1_values),
    }