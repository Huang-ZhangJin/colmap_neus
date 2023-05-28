# Copyright (c) Gorilla-Lab. All rights reserved.

from .dataset_base import DatasetBase
from .scannet import ScanNetDataset
from .tnt_outdoor import TnToutdoorDataset
from .nerf360 import NeRF360Dataset
from .colmap import ColmapDataset
from .dtu import DTUDataset

def scannet_dataset(root: str, *args, **kwargs) -> DatasetBase:
    print("Defaulting to extended ScanNet dataset")
    return ScanNetDataset(root, *args, **kwargs)

def tntoutdoor_dataset(root: str, *args, **kwargs) -> DatasetBase:
    print("Using TnT Outdoor dataset")
    return TnToutdoorDataset(root, *args, **kwargs)

def colmap_dataset(root: str, *args, **kwargs) -> DatasetBase:
    print("Using Colmap dataset")
    return ColmapDataset(root, *args, **kwargs)

def nerf360_dataset(root: str, *args, **kwargs) -> DatasetBase:
    print("Using NeRF 360 dataset")
    return NeRF360Dataset(root, *args, **kwargs)

def dtu_dataset(root: str, *args, **kwargs) -> DatasetBase:
    print("Using DTU dataset")
    return DTUDataset(root, *args, **kwargs)

datasets = {"scannet": scannet_dataset, 
            "tntindoor": scannet_dataset, 
            "tntoutdoor": tntoutdoor_dataset,
            "colmap": colmap_dataset,
            "nerf360": nerf360_dataset,
            "dtu": dtu_dataset}

__all__ = [k for k in globals().keys() if not k.startswith("_")]
