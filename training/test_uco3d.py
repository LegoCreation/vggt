# Script for debugigng the dataloader

from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch.distributed as dist

import torch
from einops import rearrange

@torch.no_grad()
def compute_rays_gt(c2w, fxfycxcy, h, w, device="cuda"):
    """
    Args:
        c2w (torch.tensor): [b, v, 4, 4]
        fxfycxcy (torch.tensor): [b, v, 4]
        h (int): height of the image
        w (int): width of the image
    Returns:
        ray_o (torch.tensor): [b, v, 3, h, w]
        ray_d (torch.tensor): [b, v, 3, h, w]
    """

    b, v = c2w.size()[:2]
    c2w = c2w.reshape(b * v, 4, 4)

    fxfycxcy = fxfycxcy.reshape(b * v, 4)
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    y, x = y.to(device), x.to(device)
    x = x[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
    y = y[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
    x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
    y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
    z = torch.ones_like(x)
    ray_d = torch.stack([x, y, z], dim=2)  # [b*v, h*w, 3]
    ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))  # [b*v, h*w, 3]
    ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)  # [b*v, h*w, 3]
    ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)  # [b*v, h*w, 3]
    

    ray_o = rearrange(ray_o, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)
    ray_d = rearrange(ray_d, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)

    return torch.cat([ray_o, ray_d], dim=2)

@torch.no_grad()
def compute_rays_mine(extr: torch.Tensor, intrinsics: torch.Tensor, target_image_shape: torch.Size, device="cuda"):
    """
    Args:
        extr (torch.Tensor): [b, s, 3, 4] OpenCV Extrinsics matrix
        intrinsics (torch.Tensor): [b, s, 3, 3] OpenCV Intrinsics matrix
        target_image_shape (torch.Size): [b, s, c, h, w]
    Returns:
        [ray_o, ray_d] (torch.tensor): [b, s, 6, h, w]
    """
    B, S, _, _ = extr.shape
    _, _, _, H, W = target_image_shape
    R_w2c = extr[..., :3, :3]
    t_wc = extr[..., :3, 3:]
    R_c2w = R_w2c.transpose(-1, -2)
    cam_center = -torch.matmul(R_c2w, t_wc).squeeze(-1)

    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    x = x + 0.5
    y = y + 0.5

    x = x.unsqueeze(0).unsqueeze(0).expand(B, S, H, W)
    y = y.unsqueeze(0).unsqueeze(0).expand(B, S, H, W)
    
    fx = intrinsics[..., 0:1, 0:1]
    fy = intrinsics[..., 1:2, 1:2]
    cx = intrinsics[..., 0:1, 2:3]
    cy = intrinsics[..., 1:2, 2:3]

    x = (x - cx) / fx
    y = (y - cy) / fy
    z = torch.ones_like(x)

    ray_d = torch.stack([x, y, z], dim=-1)
    ray_d = ray_d.view(B, S, H * W, 3) @ R_w2c
    ray_d = ray_d.view(B, S, H, W, 3)
    ray_d = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)

    ray_o = cam_center[..., None, None, :].expand(B, S, H, W, 3)

    ray_o = rearrange(ray_o, "b s h w c -> b s c h w")
    ray_d = rearrange(ray_d.view(B, S, H, W, 3), "b s h w c -> b s c h w")
    
    rays_cat = torch.cat([ray_o, ray_d], dim=2)
    return rays_cat


DATASET_DIR = "/storage/group/dataset_mirrors/uco3d/uco3d_preprocessed_new"
SPLIT_FILE = "./test.json"
split = 'test'
config = 'uco3d_nvs_training'

dist.init_process_group("gloo", rank=0, world_size=1)
with initialize(version_base=None, config_path="config"):
    cfg = compose(config_name=config)

train_dataset = instantiate(cfg.data.val, _recursive_=False)
dloader = train_dataset.get_loader(0)

for data_iter, batch in enumerate(dloader):
    extr = batch['extrinsics'][:, [-1], ...].cuda()
    intr = batch['intrinsics'][:, [-1], ...].cuda()
    rays_torch = compute_rays_mine(extr, intr, batch['images'].shape)
    B, S, _, _ = extr.shape
    _, _, _, H, W = batch['images'].shape
    zr = torch.zeros((B, S, 1, 4))
    zr[..., -1] = 1.0
    c2w = torch.cat([extr, zr.cuda()], dim=2)
    torch_inversion = torch.linalg.inv(c2w)

    R_w2c = c2w[..., :3, :3]
    t_wc = c2w[..., :3, 3:]
    R_c2w = R_w2c.transpose(-1, -2)
    cam_center = -torch.matmul(R_c2w, t_wc)
    c2w[..., :3, :3] = R_c2w
    c2w[..., :3, 3:] = cam_center
    c2w = torch_inversion
    fxfycxcx = intr[:, :, [0, 1, 0, 1], [0, 1, 2, 2]]
    rays_gt = compute_rays_gt(c2w, fxfycxcx, H, W)
    print(torch.dist(rays_gt, rays_torch))
    print(torch.dist(torch_inversion, c2w))

