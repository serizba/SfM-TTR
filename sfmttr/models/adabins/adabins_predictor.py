from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from sfmttr import DepthPredictor

from .AdaBins.models.unet_adaptive_bins import UnetAdaptiveBins
from .AdaBins.model_io import load_checkpoint


class AdaBinsPredictor(DepthPredictor):

    def __init__(self) -> None:
        super(AdaBinsPredictor, self).__init__()

        self.model = UnetAdaptiveBins.build(
            n_bins=256,
            min_val=1e-3,
            max_val=80
        )

        pretrained_path = Path(__file__).parent / 'weights' / 'AdaBins_kitti.pt'
        self.model, _, _ = load_checkpoint(pretrained_path, self.model)
        self.model = self.model.cuda() if torch.cuda.is_available() else self.model

        self.encoder = self.model.encoder

    
    def get_encoder_params(self):
        encoder_params = list(self.model.encoder.parameters())
        transformer_params = list(self.model.adaptive_bins_layer.patch_transformer.transformer_encoder.parameters())
        return encoder_params + transformer_params


    @staticmethod
    def get_inputs_transform():
        return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            torchvision.transforms.Lambda(
                lambda x: torchvision.transforms.functional.crop(x,
                    int(x.shape[1] - 352), int((x.shape[2] - 1216) / 2), 352, 1216
                )
            ),
        ])


    @staticmethod
    def get_y_true_transform():
        return torchvision.transforms.Compose([
            torchvision.transforms.PILToTensor(),
            torchvision.transforms.Lambda(lambda x: x / 256.0),
            torchvision.transforms.Lambda(
                lambda x: torchvision.transforms.functional.crop(x,
                    int(x.shape[1] - 352), int((x.shape[2] - 1216) / 2), 352, 1216
                )
            ),
        ])

    
    def forward(self, inputs):
        _, y_pred = self.model(inputs)
        return y_pred
