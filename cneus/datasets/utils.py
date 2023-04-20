import glob
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch


@dataclass
class GtRays:
    origins: Union[torch.Tensor, List[torch.Tensor]]
    dirs: Union[torch.Tensor, List[torch.Tensor]]
    gt: Union[torch.Tensor, List[torch.Tensor]]
    depth_gt: Optional[Union[torch.Tensor, List[torch.Tensor]]]
    normal_gt: Optional[Union[torch.Tensor, List[torch.Tensor]]]
    vpsplanar_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]]
    mask: Optional[Union[torch.Tensor, List[torch.Tensor]]]

    def to(self, *args, **kwargs) -> "GtRays":
        origins = self.origins.to(*args, **kwargs)
        dirs = self.dirs.to(*args, **kwargs)
        gt = self.gt.to(*args, **kwargs)
        depth_gt = self.depth_gt.to(*args, **kwargs) if self.depth_gt is not None else None
        normal_gt = self.normal_gt.to(*args, **kwargs) if self.normal_gt is not None else None
        vpsplanar_mask = (
            self.vpsplanar_mask.to(*args, **kwargs) if self.vpsplanar_mask is not None else None
        )
        mask = self.mask.to(*args, **kwargs) if self.mask is not None else None
        return GtRays(origins, dirs, gt, depth_gt, normal_gt, vpsplanar_mask, mask)

    def __getitem__(self, key: str) -> "GtRays":
        origins = self.origins[key]
        dirs = self.dirs[key]
        gt = self.gt[key]
        depth_gt = self.depth_gt[key] if self.depth_gt is not None else None
        normal_gt = self.normal_gt[key] if self.normal_gt is not None else None
        vpsplanar_mask = self.vpsplanar_mask[key] if self.vpsplanar_mask is not None else None
        mask = self.mask[key] if self.mask is not None else None
        return GtRays(origins, dirs, gt, depth_gt, normal_gt, vpsplanar_mask, mask)

    def __len__(self) -> int:
        return self.origins.size(0)


@dataclass
class Intrin:
    fx: Union[float, torch.Tensor]
    fy: Union[float, torch.Tensor]
    cx: Union[float, torch.Tensor]
    cy: Union[float, torch.Tensor]

    def scale(self, scaling: float) -> "Intrin":
        return Intrin(self.fx * scaling, self.fy * scaling, self.cx * scaling, self.cy * scaling)

    def get(self, field: str, image_id: int = 0) -> Any:
        val = self.__dict__[field]
        return val if isinstance(val, float) else val[image_id].item()


@dataclass
class Intrin_and_Inv:
    intrinsic: Union[float, torch.Tensor]
    intrinsic_inv: Union[float, torch.Tensor]

    def scale(self, scaling: float) -> "Intrin_and_Inv":
        return Intrin_and_Inv(
            self.intrinsic[..., :3, :3] * scaling,
            torch.inverse(self.intrinsic[..., :3, :3] * scaling),
        )

    def get(self, field: str, image_id: int = 0) -> Any:
        val = self.__dict__[field]
        return val if isinstance(val, float) else val[image_id].item()


# Data
def select_or_shuffle_rays(
    rays_init: GtRays,
    permutation: bool = False,
    epoch_size: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
) -> GtRays:
    n_rays = rays_init.origins.size(0)
    n_samp = n_rays if (epoch_size is None) else int(epoch_size)
    if permutation and not (n_samp > n_rays):
        print(" Shuffling rays")
        indexer = torch.randperm(n_rays, device="cpu")[:n_samp]
    else:
        print(" Selecting random rays")
        indexer = torch.randint(n_rays, (n_samp,), device="cpu")
    return rays_init[indexer].to(device=device), indexer


def select_or_shuffle_rays_withmask(
    rays_init: GtRays,
    rays_mask: torch.Tensor,
    permutation: bool=False,
    epoch_size: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
) -> GtRays:
    valid_index = torch.where(rays_mask==True)[0]
    n_rays = rays_mask.sum()
    assert len(valid_index) == n_rays
    n_samp = n_rays if (epoch_size is None) else int(epoch_size)
    if permutation and not (n_samp > n_rays):
        print(" Shuffling rays")
        indexer = torch.randperm(n_rays, device="cpu")[:n_samp]
    else:
        print(" Selecting random rays")
        indexer = torch.randint(n_rays, (n_samp,), device="cpu")
    mask_indexer = valid_index[indexer]
    return rays_init[indexer].to(device=device), mask_indexer


def load_K_Rt_from_P(filename: str, P: Optional[np.ndarray] = None) -> Tuple[np.ndarray]:
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def lift(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, intrinsics: torch.Tensor
) -> torch.Tensor:
    # parse intrinsics
    intrinsics = intrinsics.cuda()
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    sk = intrinsics[:, 0, 1]

    x_lift = (
        (
            x
            - cx.unsqueeze(-1)
            + cy.unsqueeze(-1) * sk.unsqueeze(-1) / fy.unsqueeze(-1)
            - sk.unsqueeze(-1) * y / fy.unsqueeze(-1)
        )
        / fx.unsqueeze(-1)
        * z
    )
    y_lift = (y - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z

    # homogeneous
    return torch.stack((x_lift, y_lift, z, torch.ones_like(z).cuda()), dim=-1)


def draw_camera(
    pose_all: torch.Tensor,
    intrinsics_all: torch.Tensor,
    save_name: str = "posetest.ply",
    h: int = 800,
    w: int = 800,
) -> None:
    import open3d as o3d  # Note the version: pip install open3d==0.10.0

    ###########################################################
    ########## Code for visualizing cameras ###################
    ###########################################################
    pose_all = pose_all.cuda()
    intrinsics_all = intrinsics_all.cuda()

    uv = np.stack(np.meshgrid([0, h - 1], [0, w - 1], indexing="ij"), axis=-1).reshape(-1, 2)
    uv = torch.from_numpy(uv).type(torch.cuda.FloatTensor)
    uv = uv.unsqueeze(0).expand(pose_all.shape[0], 4, 2)

    batch_size, num_samples, _ = uv.shape
    depth = torch.ones((batch_size, num_samples)).cuda()
    x_cam = uv[:, :, 0].view(batch_size, -1)
    y_cam = uv[:, :, 1].view(batch_size, -1)
    # z_cam = -depth.view(batch_size, -1)
    z_cam = depth.view(batch_size, -1)
    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics_all)

    # permute for batch matrix product
    pixel_points_cam = pixel_points_cam.permute(0, 2, 1)

    cam_loc = pose_all[:, :3, 3]  # [n_imgs, 3]
    world_coords = torch.bmm(pose_all, pixel_points_cam).permute(0, 2, 1)[
        :, :, :3
    ]  # [n_imgs, n_pixels, 3]

    points_all = torch.cat([cam_loc, world_coords.reshape(-1, 3)], dim=0).cpu()
    pixel_corner_idxs = (
        torch.arange(num_samples).type(torch.LongTensor).reshape(1, -1)
        + torch.arange(batch_size).type(torch.LongTensor).reshape(-1, 1) * num_samples
    )
    pixel_corner_idxs = pixel_corner_idxs + batch_size
    cam_loc_idxs_exp = (
        torch.arange(batch_size).type(torch.LongTensor).unsqueeze(1).expand(batch_size, num_samples)
    )
    edges = torch.stack([cam_loc_idxs_exp, pixel_corner_idxs], dim=2)
    edges = edges.reshape(-1, 2)
    pixel_corner_idxs_shift = pixel_corner_idxs[:, [1, 3, 0, 2]]
    img_plane_edges = torch.stack([pixel_corner_idxs, pixel_corner_idxs_shift], dim=2)
    img_plane_edges = img_plane_edges.reshape(-1, 2)
    edges = torch.cat([edges, img_plane_edges], dim=0)

    cam_edgeset = o3d.geometry.LineSet()
    cam_edgeset.points = o3d.utility.Vector3dVector(points_all.numpy())
    cam_edgeset.lines = o3d.utility.Vector2iVector(edges.numpy())

    o3d.io.write_line_set(save_name, cam_edgeset)
    print("Saved Pose in PLY format")
