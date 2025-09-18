# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub
from einops import rearrange

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        enable_camera=True,
        enable_depth=True,
        enable_point=True,
        enable_track=True,
    ):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        
        # Initialize heads based on enabled features
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None
        
        # NVS head is always enabled for our task
        self.nvs_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="sigmoid", conf_activation="expp1")

    @torch.no_grad()
    def compute_rays(extr: torch.Tensor, intrinsics: torch.Tensor, target_image_shape: torch.Size, device="cuda"):
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

        return ray_o, ray_d

    # @torch.no_grad()
    def _plucker_ray_embeddings(self, ray_o: torch.Tensor, ray_d: torch.Tensor):
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

    def forward(
        self,
        images: torch.Tensor,
        target_intrinsics: torch.Tensor,
        target_extrinsics: torch.Tensor,
        query_points: torch.Tensor = None,
    ):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            target_view (torch.Tensor): Target view pose [B, S].
                B: batch size, S: sequence length
            intrinsics: (torch.Tensor), OpenCV Camera intrinsics [B, S, 3, 3]
            extrinsics: (torch.Tensor), OpenCV Camera Extrinsics [B, S, 3, 4]
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """

        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)
        
        ray_o, ray_d = VGGT.compute_rays(target_extrinsics, target_intrinsics, images.shape)
        target_plucker = self._plucker_ray_embeddings(ray_o, ray_d)
        aggregated_tokens_list, patch_start_idx, target_tokens = self.aggregator(images, target_plucker)
        predictions = {}

        with torch.amp.autocast('cuda', enabled=False):
            if self.nvs_head is not None:
                B, S, C, H, W = images.shape
                n = torch.zeros((B, S+1, C, H, W))
                predicted_image, predicted_image_conf = self.nvs_head(
                    aggregated_tokens_list, images=n, patch_start_idx=patch_start_idx
                )
                predictions['predicted_image'] = predicted_image[:, [-1]].permute(0, 1, 4, 2, 3)
                predictions['predicted_image_conf'] = predicted_image_conf[:, [-1]]
                images = torch.cat([images, predictions['predicted_image']], dim=1)

                # Here, not passing the tokens sequentially results
                # In different predictions (l2 distnace of 3E-3 on average, so approx. 9e-6 per pixel)
                # Negligible, but might produce slightly different results.
                # image_predictions = []
                # confidence_predictions = []
                # n = torch.zeros((1, S+1, C, H, W))
                # for b in range(B):
                #     predicted_image, predicted_image_conf = self.nvs_head(
                #     [x[b:b+1] for x in aggregated_tokens_list], images=n, patch_start_idx=patch_start_idx
                # )
                #     image_predictions.append(predicted_image[:, [-1]].permute(0, 1, 4, 2, 3))
                #     confidence_predictions.append(predicted_image_conf[:, [-1]])
                # predictions['predicted_image'] = torch.cat(image_predictions, dim=0)
                # predictions['predicted_image_conf'] = torch.cat(confidence_predictions, dim=0)
                # images = torch.cat([images, predictions['predicted_image']], dim=1)

            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

            if self.track_head is not None and query_points is not None:
                track_list, vis, conf = self.track_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
                )
                predictions["track"] = track_list[-1]  # track of the last iteration
                predictions["vis"] = vis
                predictions["conf"] = conf

            predictions["images"] = images
        return predictions

    def deactivate_heads(self, model_conf):
        """
        Removes heads that we are disabled in the model config.
        """
        if not model_conf.enable_camera:
            self.camera_head = None
        if not model_conf.enable_depth:
            self.depth_head = None
        if not model_conf.enable_point:
            self.point_head = None
        if not model_conf.enable_track:
            self.track_head = None
