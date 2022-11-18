from pathlib import Path

import numpy as np
import statsmodels.api as sm

import torch
import torch.nn as nn
import torch.nn.functional as F

import sfmttr.read_write_model as rwm


class TunerDataset(torch.utils.data.Dataset):
    r"""Dataset for SfM-TTR
    
    From a model, a sparse COLMAP reconstruction and a dataset generates the valid
    aligned data to perform the test-time optimization.

    It aligns the COLMAP and the predicted depth using RANSAC and WLS.

    Args:
        min_depth (float, optional): Minimum depth value. Defaults to 1e-3.
        max_depth (float, optional): Maximum depth value. Defaults to 80.
        median_scaling (bool, optional): Whether to use median scaling using
            the median of y_true and y_pred. Defaults to True.
        crop_factors (list, optional): Crop factors for t, b, l, r

    Inputs:
        y_true (torch.Tensor, [1, 1, H1, W1]): Ground truth depth map
        y_pred (torch.Tensor, [1, 1, H2, W2]): Predicted depth map. 
            y_pred will be resized to match y_true

    """

    def __init__(self, dataset, model, colmap_path, kb_crop=False):

        y_pred, target, errors, coords, dataset_idx = get_colmap_data(
            dataset, model, colmap_path, kb_crop=kb_crop
        )

        s, inliers = align_depths(y_pred, target)

        self.dataset = dataset
        self.y_pred = y_pred[inliers]
        self.target = target[inliers]
        self.errors = errors[inliers]
        self.coords = coords[inliers]
        self.dataset_idx = dataset_idx[inliers]
        self.unique_idx = np.unique(dataset_idx)
        self.scale = s

    def __len__(self):
        return len(self.unique_idx)

    def __getitem__(self, idx):
        
        mask = self.dataset_idx == idx

        target_batch = torch.tensor(self.target[mask], dtype=torch.float32)
        errors_batch = torch.tensor(self.errors[mask], dtype=torch.float32)
        coords_batch = torch.tensor(self.coords[mask], dtype=torch.float32)
        coords_batch = coords_batch[None]
        
        inputs, _ = self.dataset[idx]

        scale = self.scale

        return inputs, target_batch, errors_batch, coords_batch, scale


def get_colmap_data(
    dataset,
    model,
    path_model: str,
    kb_crop: bool = False,
    coords_crops: list = [0.40810811, 0.99189189, 0.03594771, 0.96405229]
):

    cam = rwm.read_cameras_binary(Path(path_model) / 'cameras.bin')
    img = rwm.read_images_binary(Path(path_model) / 'images.bin')
    pts = rwm.read_points3D_binary(Path(path_model) / 'points3D.bin')

    y_pred_list = []
    target_list = []
    errors_list = []
    coords_list = []
    dataset_idx = []

    H, W = next(iter(cam.values())).height, next(iter(cam.values())).width

    for idx in range(len(dataset)):

        inputs, _ = dataset[idx]
        inputs = inputs.unsqueeze(0)

        inputs = inputs.cuda() if torch.cuda.is_available() else inputs

        try:
            v = next(v for v in img.values() if v.name == dataset.x_list[idx].name)
        except:
            continue

        colmap_depths = np.array([(v.qvec2rotmat() @ pts[p3d].xyz + v.tvec)[2] for p3d in v.point3D_ids[v.point3D_ids > -1]])
        colmap_coords = np.array([v.xys[np.where(v.point3D_ids == p3d)][0, ::-1] for p3d in v.point3D_ids[v.point3D_ids > -1]])
        colmap_errors = np.array([pts[p3d].error.item() for p3d in v.point3D_ids[v.point3D_ids > -1]])

        # Kitti Benchmark crop
        if kb_crop:
            t = int(H - 352)
            l = int((W - 1216) / 2)
            colmap_coords = (colmap_coords - [t, l]) / [352, 1216]
        else:
            colmap_coords = colmap_coords / [H, W]
        
        # Garg crop
        crops = np.array(coords_crops)
        mask_h = (colmap_coords[..., :1] > crops[0]) & (colmap_coords[..., :1] < crops[1])
        mask_w = (colmap_coords[..., 1:] > crops[2]) & (colmap_coords[..., 1:] < crops[3])
        mask = mask_h & mask_w

        colmap_depths = colmap_depths[mask[:, 0]]
        colmap_errors = colmap_errors[mask[:, 0]]
        
        # Target coords from [0, 1] to [-1, 1]
        colmap_coords = np.stack([colmap_coords[:, :1][mask], colmap_coords[:, 1:][mask]], axis=1)
        colmap_coords = (2 * colmap_coords - 1)[None, None, ...]

        colmap_coords = torch.tensor(colmap_coords, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')

        # Change order in coords, from XY to YX
        colmap_coords = torch.stack([colmap_coords[..., 1], colmap_coords[..., 0]], dim=-1)

        # Obtain model predictions
        with torch.no_grad():
            y_pred = model(inputs)
            

        y_pred_list += torch.nn.functional.grid_sample(
            y_pred, colmap_coords, mode='nearest', align_corners=True
        )[0, 0, 0].detach().cpu().numpy().tolist()
        
        target_list += colmap_depths.tolist()
        errors_list += colmap_errors.tolist()
        coords_list += colmap_coords[0, 0].detach().cpu().numpy().tolist()
        dataset_idx += [idx] * len(colmap_depths)

    y_pred = np.array(y_pred_list)
    target = np.array(target_list)
    errors = np.array(errors_list)
    coords = np.array(coords_list)
    dataset_idx = np.array(dataset_idx)

    return y_pred, target, errors, coords, dataset_idx


def align_depths(
    y_pred,
    target,
    inlier_threshold: float = 0.5,
    num_trials: int = 20,
):


    # RANSAC
    best_score = 0.0
    best_inliers = None
    for _ in range(num_trials):
        
        subset = np.random.randint(len(target))

        subset_y_pred = y_pred[subset]
        subset_target = target[subset]
        
        s = subset_y_pred / subset_target

        inliers = ((target * s - y_pred) ** 2) / (target * s) < inlier_threshold
        score = np.sum(inliers)

        if best_score < score:
            best_score = score
            best_inliers = inliers

    # Weighted Least Squares
    mod_wls = sm.WLS(
        y_pred[best_inliers][:, None],
        target[best_inliers][:, None],
        weights=1.0 / (y_pred[best_inliers][:, None]) ** 2
    )
    s = mod_wls.fit().params.item()
    best_inliers = (np.abs(target * s - y_pred) / (target * s) < 0.5)

    return s, best_inliers