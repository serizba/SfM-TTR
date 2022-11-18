from collections import defaultdict
from turtle import forward
import torch


class KITTIMetrics(torch.nn.Module):
    r"""KITTI Metrics based on the metrics computed in ManyDepth
    Based on: https://github.com/nianticlabs/manydepth/blob/master/manydepth/evaluate_depth.py

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
    def __init__(
        self,
        min_depth = 1e-3,
        max_depth = 80,
        median_scaling = True,
        crop_factors = [0.40810811, 0.99189189, 0.03594771, 0.96405229]
    ) -> None:
        super(KITTIMetrics, self).__init__()

        self.min_depth = min_depth
        self.max_depth = max_depth
        self.median_scaling = median_scaling
        self.crop_factors = crop_factors

    def forward(self, y_true, y_pred):

        if len(y_true.shape) != 4 or y_true.shape[0] != 1 or y_true.shape[1] != 1:
            raise Exception('Expected Tensor with shape [1, 1, H, W]')

        if len(y_pred.shape) != 4 or y_pred.shape[0] != 1 or y_pred.shape[1] != 1:
            raise Exception('Expected Tensor with shape [1, 1, H, W]')

    
        # Resize y_pred to match y_true
        y_pred = torch.nn.functional.interpolate(
            y_pred, size=y_true.shape[2:], mode='bilinear', align_corners=False
        )

        # Create mask of valid pixels
        mask = torch.zeros_like(y_true, dtype=bool)

        # Crop
        *_, H, W = y_true.shape
        T, B, L, R = self.crop_factors
        mask[..., int(T * H):int(B * H), int(L * W):int(R * W)] = True

        # Range
        mask = mask & (y_true > self.min_depth) & (y_true < self.max_depth)

        y_true = y_true[mask]
        y_pred = y_pred[mask]

        # Median scaling on predictions
        if self.median_scaling:
            scale = (torch.median(y_true) / torch.median(y_pred))
            y_pred = y_pred * scale
        else:
            scale = torch.tensor(1.0, dtype=y_true.dtype, device=y_true.device)

        # Out of range cleaning on predictions
        y_pred[y_pred < self.min_depth] = self.min_depth
        y_pred[y_pred > self.max_depth] = self.max_depth

        threshold = torch.maximum(y_true / y_pred, y_pred / y_true)
        a1 = (threshold < 1.25).float().mean()
        a2 = (threshold < 1.25 ** 2).float().mean()
        a3 = (threshold < 1.25 ** 3).float().mean()

        rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2))
        rmse_log = torch.sqrt(torch.mean((torch.log(y_true) - torch.log(y_pred)) ** 2))

        abs_rel = torch.mean(torch.abs(y_true - y_pred) / y_true)
        sq_rel = torch.mean(((y_true - y_pred) ** 2) / y_true)

        return {
            'a1': a1,
            'a2': a2,
            'a3': a3,
            'rmse': rmse,
            'rmse_log': rmse_log,
            'abs_rel': abs_rel,
            'sq_rel': sq_rel,
        }
