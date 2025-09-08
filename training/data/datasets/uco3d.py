# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os.path as osp
import os
import logging

import cv2
import random
import numpy as np
import json
import tqdm

from data.dataset_util import *
from data.base_dataset import BaseDataset


class UCO3dDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        DATASET_DIR: str = None,
        SPLIT_FILE: str = None,
        min_num_images: int = 24,
        len_train: int = 100000,
        len_test: int = 10000,
    ):
        """
        Initialize the UCo3dDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            DATASET_DIR (str): Directory path the processed UCO3D dataset.
            SPLIT_DIR (str): Directory path to UCO3D set lists.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
        Raises:
            ValueError: If DATASET_DIR or SPLIT_DIR is not specified.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.single_sequence = common_conf.single_sequence
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.load_depth = common_conf.load_depth
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        if DATASET_DIR is None or SPLIT_FILE is None:
            raise ValueError("Both DATASET_DIR and SPLIT_FILE must be specified.")

        self.split = split
        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        self.invalid_sequence = [] # set any invalid sequence names here


        self.category_map = {}
        self.data_store = {}
        self.seqlen = None
        self.min_num_images = min_num_images

        logging.info(f"DATASET_DIR is {DATASET_DIR}")

        self.DATASET_DIR = DATASET_DIR
        self.SPLIT_FILE = SPLIT_FILE

        total_frame_num = 0
        # Parse split dictionary
        with open(self.SPLIT_FILE , 'r') as f:
            split_dict = json.load(f)

        ct = 0
        for sequence_id, seq_data in tqdm.tqdm(split_dict.items()):
            if ct > 0 and self.debug:
                break
            ct+=1
            camera_data_path = osp.join(self.DATASET_DIR, seq_data['camera_data'])
            camera_poses = np.load(camera_data_path)
            extrinsics = np.linalg.inv(camera_poses['camera_poses'])
            annotations = []
            for image_path, idx in seq_data['image_paths']:
                annotation = {'filepath': image_path, 'frame_id': idx}
                annotation['extri'] = extrinsics[idx][:3, ...]
                annotation['intri'] = camera_poses['intrinsics'][idx]
                annotations.append(annotation)
            self.data_store[sequence_id] = annotations
            total_frame_num += len(annotations)
        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        self.total_frame_num = total_frame_num

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: Co3D Data size: {self.sequence_list_len}")
        logging.info(f"{status}: Co3D Frame number: {self.total_frame_num}")
        logging.info(f"{status}: Co3D Data dataset length: {len(self)}")

    def export_dataset_json(self, output_path='./'):
        export_json = {}
        for seq_id, data in self.data_store.items():
            path_to_cam_data = osp.join('/'.join(data[0]['filepath'].split('/')[:-2]), 'camera_data.npz')
            export_json[seq_id] = {'camera_data': path_to_cam_data}
            collected_image_paths = []
            for annotation in data:
                collected_image_paths.append((annotation['filepath'], annotation['frame_id']))
            export_json[seq_id]['image_paths'] = collected_image_paths
        
        with open(osp.join(output_path, self.split+'.json'), 'w') as f:
            json.dump(export_json, f, indent=2)

    def compute_rays(self, extr, intrinsics, target_image_shape):
        """
        Args:
            extr (np.ndarray): [3, 4] OpenCV Extrinsics matrix
            intrinsics (np.ndarray): [3, 3] OpenCV Intrinsics matrix
            target_image_shape
        Returns:
            [ray_o, ray_d] (np.ndarray): [6, h, w]
        """
        h, w = target_image_shape
        R_c2w = extr[:3, :3].T
        cam_center = -R_c2w @ extr[:3, 3]

        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        x = (x + 0.5 - intrinsics[0, 2]) / intrinsics[0, 0]
        y = (y + 0.5 - intrinsics[1, 2]) / intrinsics[1, 1]
        z = np.ones_like(x)
        ray_d = np.stack([x, y, z], axis=-1)
        ray_d = ray_d @ R_c2w.T

        ray_d = ray_d / np.linalg.norm(ray_d, axis=1, keepdims=True)
        ray_o = np.broadcast_to(cam_center, ray_d.shape)
        
        rays_cat = np.concatenate([ray_o.T.reshape(3, h, w), ray_d.T.reshape(3, h, w)], axis=0)

        return rays_cat

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
        num_target_images: int = 1
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
            dict: A batch of data including images, depths, and other metadata.
        """
        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)
        if seq_name is None:
            seq_name = self.sequence_list[seq_index]

        metadata = self.data_store[seq_name]

        if ids is None:
            if self.single_sequence:
                ids = np.arange(0, img_per_seq)
            else:
                ids = np.random.choice(
                    len(metadata), img_per_seq, replace=self.allow_duplicate_img
                )
        annos = [metadata[i] for i in ids]


        target_image_shape = self.get_target_shape(aspect_ratio)
        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        image_paths = []
        original_sizes = []
        target_images = []
        for idx, anno in enumerate(annos):
            filepath = anno["filepath"]

            image_path = osp.join(self.DATASET_DIR, filepath)
            image = read_image_cv2(image_path)
            # mask_path = image_path.replace("/images", "/masks").replace('.jpg', '.png')
            # mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) > 0
            # image *= mask[:, :, None]
            if self.load_depth:
                depth_path = image_path.replace("/images", "/depths").replace('.jpg', '.npy')
                depth_map = np.load(depth_path)
                depth_map = threshold_depth_map(depth_map, min_percentile=-1, max_percentile=98)
            else:
                depth_map = None

            original_size = np.array(image.shape[:2])
            extri_opencv = anno["extri"]
            intri_opencv = anno["intri"]

            (
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
            ) = self.process_one_image(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath=filepath,
            )
            if idx >= (len(annos) - num_target_images):
                target_images.append(image)
            else:
                images.append(image)
            
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            image_paths.append(image_path)
            original_sizes.append(original_size)
        set_name = "uco3d"
        batch = {
            "seq_name": set_name + "_" + seq_name,
            "ids": ids,
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
            "target_images": target_images
        }
        return batch
