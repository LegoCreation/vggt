#!/usr/bin/env python3
"""
Simple script to run inference on a single configuration
Usage: python run_single_inference.py <config_name>
"""

import os
import sys
import torch
import torch.distributed as dist
from hydra import initialize, compose
from hydra.utils import instantiate
import argparse
import logging
import time
import numpy as np

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import metrics utilities
from metrics_utils import compute_image_metrics, batch_compute_metrics

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Run inference on validation dataset')
    parser.add_argument('config', help='Configuration name (e.g., uco3d_full_cam_nvs)')
    parser.add_argument('--checkpoint', help='Checkpoint path (optional, will use default from config)')
    parser.add_argument('--max-batches', type=int, default=None, help='Maximum number of batches to process')
    
    args = parser.parse_args()
    logger = setup_logging()
    
    # Initialize distributed training
    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("gloo", rank=0, world_size=1)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    logger.info(f"Config: {args.config}")
    
    # Load configuration first to get checkpoint path from config
    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name=args.config)
    
    # Determine checkpoint path: command line arg > config > fallback to save_dir
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        # Try to get from config first
        if hasattr(cfg.checkpoint, 'resume_checkpoint_path') and cfg.checkpoint.resume_checkpoint_path:
            checkpoint_path = cfg.checkpoint.resume_checkpoint_path
            logger.info(f"Using checkpoint from config: {checkpoint_path}")
        else:
            # Fallback: look for checkpoint.pt in the save_dir
            from train_utils.general import get_resume_checkpoint
            checkpoint_path = get_resume_checkpoint(cfg.checkpoint.save_dir)
            if checkpoint_path:
                logger.info(f"Found checkpoint in save_dir: {checkpoint_path}")
    
    if not checkpoint_path:
        logger.error(f"No checkpoint found for config {args.config}. Please specify --checkpoint or set resume_checkpoint_path in config")
        return
    
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return
    
    logger.info(f"Checkpoint: {checkpoint_path}")
    
    # Load model (handle base_checkpoint separately)
    logger.info("Loading model...")
    base_checkpoint = getattr(cfg.model, 'base_checkpoint', None)
    
    # Create model config - use 518 to match checkpoint
    model_kwargs = {
        'img_size': 518,  # Use 518 to match the trained checkpoints
        'patch_size': 14,
        'embed_dim': 1024,
        'enable_camera': getattr(cfg.model, 'enable_camera', True),
        'enable_depth': getattr(cfg.model, 'enable_depth', True),
        'enable_point': getattr(cfg.model, 'enable_point', True),
        'enable_track': getattr(cfg.model, 'enable_track', True),
    }
    
    logger.info(f"Model config: img_size={model_kwargs['img_size']}, patch_size={model_kwargs['patch_size']}")
    
    from vggt.models.vggt import VGGT
    model = VGGT(**model_kwargs)
    model = model.to(device)
    
    # Skip base checkpoint loading when using different img_size
    # The trained checkpoint should already contain the fine-tuned weights
    logger.info("Skipping base checkpoint loading - using trained checkpoint directly")
    
    # Load checkpoint
    logger.info("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model' in checkpoint:
        model_state = checkpoint['model']
    else:
        model_state = checkpoint
    
    # Try loading with strict=False to handle size mismatches
    try:
        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
        if missing_keys:
            logger.warning(f"Missing keys in checkpoint: {missing_keys[:5]}...")  # Show first 5
        if unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys[:5]}...")  # Show first 5
        logger.info("Checkpoint loaded successfully (with warnings if any)")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        logger.info("Continuing with randomly initialized model...")
    model.eval()
    
    if 'prev_epoch' in checkpoint:
        logger.info(f"Loaded checkpoint from epoch {checkpoint['prev_epoch']}")
    
    # Create validation dataloader
    logger.info("Creating validation dataloader...")
    val_dataset = instantiate(cfg.data.val, _recursive_=False)
    val_dataloader = val_dataset.get_loader(0)
    
    logger.info(f"Validation dataset: {len(val_dataloader)} batches")
    
    # Run inference
    logger.info("Starting inference...")
    model.eval()
    total_batches = 0
    total_samples = 0
    
    # Initialize metric accumulators
    all_psnr_values = []
    all_ssim_values = []
    all_mse_values = []
    all_l1_values = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            if args.max_batches and batch_idx >= args.max_batches:
                break
            
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            try:
                # Forward pass - VGGT expects specific arguments
                if 'images' in batch and 'intrinsics' in batch and 'extrinsics' in batch:
                    

                    target_intr = batch['intrinsics'][:, [-1], ...]
                    target_extr = batch['extrinsics'][:, [-1], ...]

                    outputs = model(
                        images=batch['images'],
                        target_intrinsics=target_intr,
                        target_extrinsics=target_extr
                    )
                else:
                    logger.error(f"Missing required keys in batch. Available keys: {list(batch.keys())}")
                    continue
                
                # Compute image quality metrics if predicted image is available
                if 'predicted_image' in outputs and 'target_images' in batch:
                    pred_img = outputs['predicted_image']
                    target_img = batch['target_images']
                    
                    # Reshape if necessary (handle multi-view case)
                    if len(pred_img.shape) == 5:  # B, V, C, H, W
                        b, v, c, h, w = pred_img.shape
                        pred_img = pred_img.reshape(b * v, c, h, w)
                        target_img = target_img.reshape(b * v, c, h, w)
                    
                    # Compute metrics for this batch
                    try:
                        batch_metrics = batch_compute_metrics(pred_img, target_img, max_val=1.0)
                        
                        # Accumulate metrics
                        all_psnr_values.extend(batch_metrics['psnr_per_sample'])
                        all_ssim_values.extend(batch_metrics['ssim_per_sample'])
                        all_mse_values.extend(batch_metrics['mse_per_sample'])
                        all_l1_values.extend(batch_metrics['l1_per_sample'])
                        
                        # Log batch averages every 10 batches
                        if batch_idx % 10 == 0:
                            logger.info(f"Batch {batch_idx} metrics - PSNR: {batch_metrics['avg_psnr']:.3f}, SSIM: {batch_metrics['avg_ssim']:.4f}")
                            
                    except Exception as e:
                        logger.warning(f"Failed to compute metrics for batch {batch_idx}: {str(e)}")
                
                total_batches += 1
                total_samples += batch['images'].shape[0] if 'images' in batch else 1
                
                if batch_idx % 10 == 0:
                    logger.info(f"Processed batch {batch_idx}/{len(val_dataloader)}")
                    
                # Log some outputs for the first batch
                if batch_idx == 0 and isinstance(outputs, dict):
                    logger.info("Sample outputs from first batch:")
                    for key, value in outputs.items():
                        if isinstance(value, torch.Tensor) and value.numel() == 1:
                            logger.info(f"  {key}: {value.item():.6f}")
                        elif isinstance(value, torch.Tensor):
                            logger.info(f"  {key}: shape {value.shape}")
                    
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                continue
    
    inference_time = time.time() - start_time
    
    logger.info(f"\nInference completed!")
    logger.info(f"Total batches processed: {total_batches}")
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Inference time: {inference_time:.2f}s")
    if total_batches > 0:
        logger.info(f"Average time per batch: {inference_time/total_batches:.3f}s")
    else:
        logger.warning("No batches were successfully processed")
    
    # Log final metrics if available
    if all_psnr_values:
        avg_psnr = np.mean(all_psnr_values)
        avg_ssim = np.mean(all_ssim_values)
        avg_mse = np.mean(all_mse_values)
        avg_l1 = np.mean(all_l1_values)
        
        std_psnr = np.std(all_psnr_values)
        std_ssim = np.std(all_ssim_values)
        
        logger.info(f"\n{'='*50}")
        logger.info("FINAL IMAGE QUALITY METRICS")
        logger.info(f"{'='*50}")
        logger.info(f"PSNR: {avg_psnr:.3f} ± {std_psnr:.3f}")
        logger.info(f"SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")
        logger.info(f"MSE:  {avg_mse:.6f}")
        logger.info(f"L1:   {avg_l1:.6f}")
        logger.info(f"Total image pairs evaluated: {len(all_psnr_values)}")
    else:
        logger.warning("No image quality metrics were computed (no predicted images found)")

if __name__ == "__main__":
    main()