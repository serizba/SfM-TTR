from typing import Tuple
import numpy as np
import torch
import torchvision
from PIL import Image, ImageFile

from sfmttr.data.kitti_utils import generate_depth_map

ImageFile.LOAD_TRUNCATED_IMAGES = True

from pathlib import Path

KITTI_TEST_SEQS = unique_seqs = [
    '2011_09_26_drive_0002_sync',
    '2011_09_26_drive_0009_sync',
    '2011_09_26_drive_0013_sync',
    '2011_09_26_drive_0020_sync',
    '2011_09_26_drive_0023_sync',
    '2011_09_26_drive_0027_sync',
    '2011_09_26_drive_0029_sync',
    '2011_09_26_drive_0036_sync',
    '2011_09_26_drive_0046_sync',
    '2011_09_26_drive_0048_sync',
    '2011_09_26_drive_0052_sync',
    '2011_09_26_drive_0056_sync',
    '2011_09_26_drive_0059_sync',
    '2011_09_26_drive_0064_sync',
    '2011_09_26_drive_0084_sync',
    '2011_09_26_drive_0086_sync',
    '2011_09_26_drive_0093_sync',
    '2011_09_26_drive_0096_sync',
    '2011_09_26_drive_0101_sync',
    '2011_09_26_drive_0106_sync',
    '2011_09_26_drive_0117_sync',
    '2011_09_28_drive_0002_sync',
    '2011_09_29_drive_0071_sync',
    '2011_09_30_drive_0016_sync',
    '2011_09_30_drive_0018_sync',
    '2011_09_30_drive_0027_sync',
    '2011_10_03_drive_0027_sync',
    '2011_10_03_drive_0047_sync',
]

KITTI_NO_SFM_SEQS = [
    '2011_09_26_drive_0020_sync',
    '2011_09_26_drive_0048_sync',
    '2011_09_26_drive_0052_sync',
    '2011_09_26_drive_0056_sync',
    '2011_10_03_drive_0047_sync',
]


class KITTI(torch.utils.data.Dataset):
    r"""KITTI Dataset

    Args:
        path_raw: Path to raw KITTI dataset
        path_gt: Path to ground truth KITTI dataset. If None,
            the ground truth will be computed from the LIDAR points
        split_type: 'eigen' or 'eigen_with_gt'
        split: 'test'
        kb_crop: If true crops the images following supervised works
        sequence: If not None, only uses the specified sequence
        size: Size of the returned input images (gt images are in original size)
        return_prev: If true, also returns the previous image
        
    """

    def __init__(
        self,
        path_raw: str,
        path_gt: str,
        split_type: str = 'eigen_with_gt',
        split: str = 'test',
        sequence: str = None,
        return_prev: bool = False,
        inputs_transform=None,
        y_true_transform=None,
    ):
        """
        Args:
            path_gt
            path_raw
            split
        """

        file_list = np.genfromtxt(Path(__file__).parent / 'splits' / split_type / f'{split}_files.txt', dtype=str)

        if path_gt is not None:
            # Load new ground truth
            self.y_list = [
                Path(path_gt) / y
                for x, y in file_list if ((x.split('/')[1] == sequence) or (sequence is None))
            ]
        else:
            # Use reprojected LIDAR
            self.y_list = None

        self.x_list = [
            Path(path_raw) / x
            for x, y in file_list if ((x.split('/')[1] == sequence) or (sequence is None))
        ]

        self.x_transforms = inputs_transform
        self.y_transforms = y_true_transform

        self.split = split
        self.return_prev = return_prev


    def __len__(self):
        return len(self.x_list)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.y_list is not None:
            y = Image.open(self.y_list[idx])
        else:
            y = self.create_gt(self.x_list[idx])
        
        x = Image.open(self.x_list[idx])

        if self.x_transforms:
            x = self.x_transforms(x)
        if self.y_transforms:
            y = self.y_transforms(y)
      
        if self.return_prev:
            x_2 = str(self.x_list[idx]).replace(
                self.x_list[idx].stem, 
                str(int(self.x_list[idx].stem) - 1).zfill(10)
            )
            x_2 = Image.open(x_2)
            if self.x_transforms:
                x_2 = self.x_transforms(x_2)
            x = torch.cat([x_2, x], dim=0)

        return x, y


    def create_gt(self, path_x):
        *path, day, seq, imgtype, _, img = str(path_x).split('/') 

        calib_dir = '/'.join(path) + '/' + day
        velo_filename = calib_dir + '/' + seq + f'/velodyne_points/data/{img[:-4]}.bin' 

        gt_depth = generate_depth_map(calib_dir, velo_filename, int(imgtype[-2:]), True)
        return Image.fromarray((gt_depth * 256.0).astype(np.int32), mode='I')