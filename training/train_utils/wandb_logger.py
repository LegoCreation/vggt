import wandb
import torch
import numpy as np

class WandbLogger:
    def __init__(self, project, name, log_freq=50):
        """Initialize WandbLogger.
        
        Args:
            project: Wandb project name
            name: Run name
            log_freq: How often to log metrics
        """
        self.project = project
        self.name = name
        self.log_freq = log_freq
        
        wandb.init(
            project=project,
            name=name,
            config={
                "architecture": "VGGT-NVS",
            }
        )
        
    def log(self, key, value, step):
        """Log a scalar metric.
        
        Args:
            key: Metric name
            value: Metric value
            step: Current step
        """
        if step % self.log_freq == 0:
            wandb.log({key: value}, step=step)
            
    def log_images(self, key, images, step, dataformats="NCHW"):
        """Log images to wandb.
        
        Args:
            key: Image group name
            images: Tensor of images
            step: Current step
            dataformats: Format of image tensor
        """
        if step % self.log_freq == 0:
            if isinstance(images, torch.Tensor):
                images = images.detach().cpu().numpy()
                
            if dataformats == "NCHW":
                images = np.transpose(images, (0, 2, 3, 1))
                
            wandb.log({
                key: [wandb.Image(img) for img in images]
            }, step=step)
            
    def log_visuals(self, name, visuals, step, fps=None):
        """Log visualization grid.
        
        Args:
            name: Visual group name
            visuals: Numpy array of visualization grid
            step: Current step
            fps: Frames per second for video
        """
        if step % self.log_freq == 0:
            wandb.log({
                name: wandb.Image(visuals)
            }, step=step) 