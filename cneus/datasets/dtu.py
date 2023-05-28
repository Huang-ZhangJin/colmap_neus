# Copyright (c) Gorilla-Lab. All rights reserved.
import os
from typing import Optional, Union

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .dataset_base import DatasetBase
from .utils import (
    GtRays, 
    Intrin, 
    Intrin_and_Inv, 
    select_or_shuffle_rays,
    select_or_shuffle_rays_withmask
)


class DTUDataset(DatasetBase):
    focal: float
    c2w: torch.Tensor        # [n_images, 4, 4]
    gt: torch.Tensor         # [n_images, h, w, 3]
    image_ids: torch.Tensor  # [n_images]
    h: int
    w: int
    n_images: int
    rays: Optional[GtRays]
    split: str

    def __init__(
        self,
        data_root: str,
        mvs_root: str,
        scene: str,
        split: str,
        epoch_size: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        permutation: bool = True,
        factor: int = 1,
        patch_size: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.split = split
        self.data_root = data_root
        self.scene = scene
        self.instance_dir = os.path.join(data_root, scene)
        assert os.path.exists(self.instance_dir)

        image_dir = os.path.join(self.instance_dir, "images")
        image_list = os.listdir(image_dir)
        image_list.sort(key=lambda x: int(x.split(".")[0]))
        self.n_images = len(image_list)

        self.device = device
        self.permutation = permutation
        self.epoch_size = epoch_size
        self.input_normal = True

        # load the camera parameters and rgb images
        intrinsic_all = []
        pose_all = []
        all_gt = []
        all_mask = []

        self.image_names = []
        intrinsic = np.loadtxt(os.path.join(self.instance_dir, "intrinsic.txt"))[:3, :3]

        for imgname in tqdm(
            image_list, desc=f"Loading camera parameters and images ^-^ : {split}"
        ):
            self.image_names.append(imgname)
            c2w = np.loadtxt(os.path.join(self.instance_dir, "pose", f"{imgname[:-4]}.txt"))
            pose_all.append(torch.from_numpy(c2w).float())
            intrinsic_all.append(torch.from_numpy(intrinsic).float())

            rgb = imageio.imread(os.path.join(self.instance_dir, "images", f"{imgname[:-4]}.png"))
            all_gt.append(torch.from_numpy(rgb))

            if split == "train":
                maskone = np.load(
                    os.path.join(
                        self.instance_dir, 
                        "sphere_mask", 
                        f"{imgname[:-4]}.npy"
                    )
                )
                all_mask.append(torch.from_numpy(maskone).to(torch.bool))

        self.c2w = torch.stack(pose_all)
        self.c2w_inv = torch.inverse(self.c2w)
        self.intrins = torch.stack(intrinsic_all)
        self.gt = torch.stack(all_gt).float() / 255.0
        if split == "train":
            self.sphere_mask = torch.stack(all_mask)
        self.n_images, self.h_full, self.w_full, _ = self.gt.shape

        self.intrins_full: Intrin = Intrin(
            self.intrins[..., 0, 0],
            self.intrins[..., 1, 1],
            self.intrins[..., 0, 2],
            self.intrins[..., 1, 2],
        )

        self.intrins_and_inv_full: Intrin_and_Inv = Intrin_and_Inv(
            self.intrins, torch.inverse(self.intrins)
        )

        self.split = split
        self.intrins: Intrin = self.intrins_full
        if self.split == "train":
            assert(patch_size == 1, "TnTOutdoor dataset does not support patch mode")
            self.gen_rays_scannet_neus(factor=factor)
            self.h, self.w = self.h_full, self.w_full
        else:
            # Rays are not needed for testing
            self.h, self.w = self.h_full, self.w_full

    def shuffle_rays(self, epoch_size: Optional[int] = None) -> Optional[torch.Tensor]:
        """
        Shuffle all rays
        """
        if self.split == "train":
            del self.rays
            # self.rays, indexer = select_or_shuffle_rays_withmask(
            #     self.rays_init, self.rays_mask, self.permutation, epoch_size, self.device
            # )
            self.rays, indexer = select_or_shuffle_rays(
                self.rays_init, self.permutation, epoch_size, self.device
            )
            self.indexer = indexer

    def gen_rays_scannet_neus(self, factor: int = 1) -> None:
        # Generate rays
        self.factor = factor
        self.h = self.h_full // factor
        self.w = self.w_full // factor
        true_factor = self.h_full / self.h
        self.intrins_and_inv = self.intrins_and_inv_full.scale(1.0 / true_factor)

        tx = torch.linspace(0, self.w - 1, self.w)
        ty = torch.linspace(0, self.h - 1, self.h)
        pixels_x, pixels_y = torch.meshgrid(tx, ty, indexing="ij")
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3

        intri_inv = self.intrins_and_inv_full.intrinsic_inv[0]
        p_one = torch.matmul(intri_inv[None, None, :3, :3], p[:, :, :, None]).squeeze()

        depth_ratio = torch.linalg.norm(p_one, ord=2, dim=-1, keepdim=True)
        dirs = p_one / depth_ratio
        dirs = dirs.transpose_(0, 1)
        depth_ratio = depth_ratio.transpose_(0, 1)
        dirs = dirs.reshape(1, -1, 3, 1)
        dirs = (self.c2w[:, None, :3, :3] @ dirs)[..., 0]
        dirs = dirs.contiguous().view(-1, 3)
        origins = self.c2w[:, None, :3, 3].expand(-1, self.h * self.w, -1).contiguous().view(-1, 3)
        gt = self.gt.reshape(self.n_images, -1, 3).contiguous().view(-1, 3)
        rays_mask = self.sphere_mask.reshape(self.n_images, -1).contiguous().view(-1)

        self.rays_init = GtRays(
            origins=origins,
            dirs=dirs,
            gt=gt,
            depth_gt=None,
            normal_gt=None,
            vpsplanar_mask=None,
            mask=rays_mask
        )
        self.rays = self.rays_init