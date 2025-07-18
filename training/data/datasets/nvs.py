import os
import os.path as osp
import logging
import random
import json
import numpy as np
import cv2

from data.dataset_util import *
from data.base_dataset import BaseDataset

class NVSDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        DATA_DIR: str = None,
        min_num_images: int = 24,
        len_train: int = 100000,
        len_test: int = 10000,
    ):
        """
        Initialize the NVS Dataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            DATA_DIR (str): Directory path to RE10K dataset.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        if DATA_DIR is None:
            raise ValueError("DATA_DIR must be specified.")

        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        self.data_store = {}
        self.seqlen = None
        self.min_num_images = min_num_images
        self.DATA_DIR = DATA_DIR

        # Load sequence list from full_list.txt
        with open(osp.join(DATA_DIR, "full_list.txt"), "r") as f:
            metadata_files = [line.strip() for line in f.readlines()]

        # Load metadata for each sequence
        total_frame_num = 0
        for metadata_file in metadata_files:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            
            scene_name = metadata["scene_name"]
            frames = metadata["frames"]
            
            if len(frames) < min_num_images:
                continue
            
            # Store metadata for each sequence
            self.data_store[scene_name] = {
                "frames": frames,
                "metadata": metadata
            }
            total_frame_num += len(frames)

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        self.total_frame_num = total_frame_num

        status = "Training" if self.training else "Test"
        logging.info(f"{status}: NVS Data size: {self.sequence_list_len}")
        logging.info(f"{status}: NVS Data dataset length: {len(self)}")

    def get_rays(self, H: int, W: int, fxfycxcy: list, w2c: list) -> tuple:
        """
        Generate rays for a given camera.
        
        Args:
            H, W: Image height and width
            fxfycxcy: Camera intrinsics [fx, fy, cx, cy]
            w2c: World to camera transformation matrix
            
        Returns:
            tuple: (ray_origins, ray_directions)
        """
        fx, fy, cx, cy = fxfycxcy
        
        # Generate pixel coordinates
        i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
        
        # Convert to normalized camera coordinates
        x = (i - cx) / fx
        y = (j - cy) / fy
        z = np.ones_like(x)
        
        # Stack coordinates
        rays_d = np.stack([x, y, z], axis=0)  # [3, H, W]
        
        # Convert from camera to world space
        c2w = np.linalg.inv(np.array(w2c))  # [4, 4]
        R = c2w[:3, :3]  # [3, 3]
        
        # Rotate ray directions to world space
        rays_d = np.sum(rays_d[..., None, :] * R, axis=-1)  # [3, H, W]
        
        # Normalize directions
        rays_d = rays_d / np.linalg.norm(rays_d, axis=0, keepdims=True)
        
        # Get ray origins (camera position in world space)
        rays_o = c2w[:3, 3]  # [3]
        rays_o = np.broadcast_to(rays_o[:, None, None], rays_d.shape)  # [3, H, W]
        
        return rays_o, rays_d

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        """
        Retrieve data for a specific sequence.

        Args:
            seq_index (int): Index of the sequence to retrieve.
            img_per_seq (int): Number of images per sequence.
            seq_name (str): Name of the sequence.
            ids (list): Specific IDs to retrieve.
            aspect_ratio (float): Aspect ratio for image processing.

        Returns:
            dict: A batch of data including source images and target views.
        """
        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)
            
        if seq_name is None:
            seq_name = self.sequence_list[seq_index]

        sequence_data = self.data_store[seq_name]
        frames = sequence_data["frames"]

        if ids is None:
            # Select source and target frames
            source_ids = np.random.choice(
                len(frames), img_per_seq - 1, replace=self.allow_duplicate_img
            )
            target_id = np.random.choice(
                [i for i in range(len(frames)) if i not in source_ids]
            )
            ids = np.concatenate([source_ids, [target_id]])

        target_image_shape = self.get_target_shape(aspect_ratio)
        H, W = target_image_shape

        images = []
        image_paths = []
        original_sizes = []
        camera_params = []

        for frame_id in ids:
            frame = frames[frame_id]
            image_path = frame["image_path"]
            image = read_image_cv2(image_path)
            
            original_size = np.array(image.shape[:2])
            image = self.process_image(image, target_image_shape)
            
            images.append(image)
            image_paths.append(image_path)
            original_sizes.append(original_size)
            camera_params.append({
                "fxfycxcy": frame["fxfycxcy"],
                "w2c": frame["w2c"]
            })

        # Split into source and target
        source_images = images[:-1]
        target_image = images[-1]
        target_camera = camera_params[-1]

        # Generate target view rays using actual camera parameters
        ray_o, ray_d = self.get_rays(
            H, W,
            target_camera["fxfycxcy"],
            target_camera["w2c"]
        )

        batch = {
            "images": np.stack(source_images),
            "target_view": {
                "image": target_image[None],
                "ray_o": ray_o[None],
                "ray_d": ray_d[None],
                "camera": target_camera
            },
            "source_cameras": camera_params[:-1],
            "image_paths": image_paths,
            "original_sizes": np.stack(original_sizes),
            "sequence_name": seq_name,
        }

        return batch

    def __len__(self):
        return self.len_train 