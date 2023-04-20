# Copyright (c) Gorilla-Lab. All rights reserved.

from .dataset_base import DatasetBase
from .scannet import ScanNetDataset
from .tnt_outdoor import TnToutdoorDataset

def scannet_dataset(root: str, *args, **kwargs) -> DatasetBase:
    print("Defaulting to extended ScanNet dataset")
    return ScanNetDataset(root, *args, **kwargs)

def tntoutdoor_dataset(root: str, *args, **kwargs) -> DatasetBase:
    print("Using TnT Outdoor dataset")
    return TnToutdoorDataset(root, *args, **kwargs)

datasets = {"scannet": scannet_dataset, 
            "tntindoor": scannet_dataset, 
            "tntoutdoor": tntoutdoor_dataset}

__all__ = [k for k in globals().keys() if not k.startswith("_")]
