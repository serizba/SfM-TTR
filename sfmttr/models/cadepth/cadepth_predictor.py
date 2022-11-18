from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from sfmttr import DepthPredictor

# Add CADepth files to the path
from .CADepth import networks as cadepth_networks

class CADepthPredictor(DepthPredictor):

    def __init__(self) -> None:
        super(CADepthPredictor, self).__init__()


        encoder_path = Path(__file__).parent / 'weights' / 'M_1024x320' / 'encoder.pth'
        decoder_path = Path(__file__).parent / 'weights' / 'M_1024x320' / 'depth.pth'

        pose_encoder_path = Path(__file__).parent / 'weights' / 'M_1024x320' / 'pose_encoder.pth'
        pose_decoder_path = Path(__file__).parent / 'weights' / 'M_1024x320' / 'pose.pth'

        # Depth
        encoder_dict = torch.load(encoder_path) if torch.cuda.is_available() else torch.load(encoder_path, map_location = 'cpu')
        decoder_dict = torch.load(decoder_path) if torch.cuda.is_available() else torch.load(decoder_path, map_location = 'cpu')
       
        encoder = cadepth_networks.ResnetEncoder(encoder_dict['num_layers'], False)
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in encoder.state_dict()})
        encoder.cuda() if torch.cuda.is_available() else encoder.cpu()
        encoder.eval()
        
        decoder = cadepth_networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
        decoder.load_state_dict(decoder_dict)        
        decoder.cuda() if torch.cuda.is_available() else decoder.cpu()
        decoder.eval()

        # Pose
        pose_encoder = cadepth_networks.ResnetEncoder(18, False, 2)
        pose_encoder.load_state_dict(torch.load(pose_encoder_path, map_location = 'cuda' if torch.cuda.is_available() else 'cpu'))
        pose_encoder.cuda() if torch.cuda.is_available() else pose_encoder.cpu()
        pose_encoder.eval()

        pose_decoder = cadepth_networks.PoseDecoder(pose_encoder.num_ch_enc, 1, 2)
        pose_decoder.load_state_dict(torch.load(pose_decoder_path, map_location = 'cuda' if torch.cuda.is_available() else 'cpu'))
        pose_decoder.cuda() if torch.cuda.is_available() else pose_decoder.cpu()
        pose_decoder.eval()


        self.encoder = encoder
        self.decoder = decoder
        self.pose_encoder = pose_encoder
        self.pose_decoder = pose_decoder

    def get_encoder_params(self):
        return self.encoder.parameters()

    @staticmethod
    def get_inputs_transform():
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize((320, 1024), interpolation=torchvision.transforms.InterpolationMode.LANCZOS),
            torchvision.transforms.ToTensor(),
        ])

    @staticmethod
    def get_y_true_transform():
        return torchvision.transforms.Compose([
            torchvision.transforms.PILToTensor(),
            torchvision.transforms.Lambda(lambda x: x / 256.0),
        ])

    def forward(self, inputs):

        y_pred = self.decoder(self.encoder(inputs))
        y_pred = y_pred[('disp', 0)]

        min_disp = 1 / 100.0
        max_disp = 1 / 0.1
        y_pred = 1.0 / (min_disp + (max_disp - min_disp) * y_pred)

        return y_pred


    


