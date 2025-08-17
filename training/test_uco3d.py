# Script for debugigng the dataloader

from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch.distributed as dist

import torch
from einops import rearrange

@torch.no_grad()
def _compute_rays(extr: torch.Tensor, intrinsics: torch.Tensor, target_image_shape: torch.Size, device="cuda"):
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
config = 'uco3d_full_training'

dist.init_process_group("gloo", rank=0, world_size=1)
with initialize(version_base=None, config_path="config"):
    cfg = compose(config_name=config)

train_dataset = instantiate(cfg.data.val, _recursive_=False)
dloader = train_dataset.get_loader(0)

for data_iter, batch in enumerate(dloader):
    B, _, _, _ = batch['extrinsics'].shape
    i = batch['target_views']

    print(batch['seq_name'])
    if i.shape[-1] == 1:
        i = i.squeeze(-1)
    else:
        i = i.squeeze()
    rays_torch = _compute_rays(batch['extrinsics'][:, i, ...].cuda(), batch['intrinsics'][:, i, ...].cuda(), batch['images'].shape)

# dataset = UCO3dDataset(common_conf=common_config, DATASET_DIR=DATASET_DIR, SPLIT_FILE=SPLIT_FILE, split=split)
# dataset.get_data(0, 4)
# dataset.export_dataset_json('./')
