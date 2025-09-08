# Script for debugigng the dataloader

from hydra import initialize, compose
from hydra.utils import instantiate
import torch.distributed as dist
from vggt.models.vggt import VGGT

import torch
from einops import rearrange

@torch.no_grad()
def compute_rays_gt(c2w, fxfycxcy, h=None, w=None, device="cuda"):
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

    return ray_o, ray_d

def _plucker_ray_embeddings(ray_o: torch.Tensor, ray_d: torch.Tensor):
    '''
    Args:
        ray_o: [B, S, 3, h, w]
        ray_d: [B, S, 3, h, w]
    Returns:
        posed_images: [B, S, 6, h, w]
    '''
    o_cross_d = torch.cross(ray_o, ray_d, dim=2)
    pose_cond = torch.cat([o_cross_d, ray_d], dim=2)
    return pose_cond

def invert_pose(intr, extr):
    B, S, _, _ = extr.shape
    zr = torch.zeros((B, S, 1, 4))
    zr[..., -1] = 1.0
    c2w = torch.cat([extr, zr.cuda()], dim=2)
    
    R_w2c = c2w[..., :3, :3]
    t_wc = c2w[..., :3, 3:]
    R_c2w = R_w2c.transpose(-1, -2)
    cam_center = -torch.matmul(R_c2w, t_wc)
    c2w[..., :3, :3] = R_c2w
    c2w[..., :3, 3:] = cam_center
    fxfycxcx = intr[:, :, [0, 1, 0, 1], [0, 1, 2, 2]]
    return c2w, fxfycxcx

DATASET_DIR = "/storage/group/dataset_mirrors/uco3d/uco3d_preprocessed_new"
SPLIT_FILE = "./test.json"
split = 'test'
config = 'uco3d_debugging_full_batch'

dist.init_process_group("gloo", rank=0, world_size=1)
with initialize(version_base=None, config_path="config"):
    cfg = compose(config_name=config)

train_dataset = instantiate(cfg.data.val, _recursive_=False)
dloader = train_dataset.get_loader(0)

for data_iter, batch in enumerate(dloader):
    _, _, _, H, W = batch['images'].shape
    extr = batch['extrinsics'][:, [-1], ...].cuda()
    intr = batch['intrinsics'][:, [-1], ...].cuda()
    ray_o, ray_d = VGGT.compute_rays(extr, intr, batch['images'].shape)
    plucekr_rays_mine = _plucker_ray_embeddings(ray_o, ray_d)

    c2w, fxfycxcx = invert_pose(intr, extr)

    gt_ray_o, gt_ray_d = compute_rays_gt(c2w, fxfycxcx, H, W)
    plucekr_rays_gt = _plucker_ray_embeddings(gt_ray_o, gt_ray_d)
    print(torch.dist(plucekr_rays_gt, plucekr_rays_mine))
    print(plucekr_rays_gt[0, ..., 0, 0].data_ptr())
    print(plucekr_rays_gt[1, ..., 0, 0].data_ptr())
    for b in range(1, 8):
        for h in range(H):
            for w in range(W):
                if gt_ray_o[b, ..., h, w].data_ptr() == gt_ray_o[0, ..., h, w].data_ptr() or \
                    plucekr_rays_gt[b, ..., h, w].data_ptr() == plucekr_rays_gt[0, ..., h, w].data_ptr():
                    print("DP equal", h, w, b)
    print(plucekr_rays_gt.shape)
    print(plucekr_rays_mine.shape)
    exit()
