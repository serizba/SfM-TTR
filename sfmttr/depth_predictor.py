import abc
import torch



class DepthPredictor(abc.ABC, torch.nn.Module):
    r"""Monocular Depth Model
    
    Base class for monocular depth estimation models.

    Should implement:
        - forward
        - get_encoder_params
        - get_inputs_transform
        - get_y_true_transform

    """
    def __init__(self):
        super(DepthPredictor, self).__init__()

    @abc.abstractmethod
    def forward(self, inputs):
        r"""Forward pass of the model

        Args:
            inputs (torch.Tensor): Input tensor of shape (B, C, H, W).
                The input tensor is expected to be alrady transformed 
                according to the model's `get_inputs_transform` transform.
        Returns:
            torch.Tensor: Output tensor of shape (B, 1, H, W)

        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_encoder_params(self):
        r"""Get the encoder parameters

        Returns:
            list: List of the parameters to optimize during TTR

        """
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def get_inputs_transform():
        r"""Transformations to apply to the inputs before passing them to the model

        Returns:
            torchvision.transforms.Compose: Transform to apply to the inputs

        """
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def get_y_true_transform():
        r"""Transformations to apply to the ground truth (e.g. y = y / 256.0)

        Returns:
            torchvision.transforms.Compose: Transform to apply to the y_true

        """
        raise NotImplementedError
