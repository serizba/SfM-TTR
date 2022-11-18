from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from sfmttr import DepthPredictor

from .ManyDepth.manydepth import networks as manydepth_networks
from .ManyDepth.manydepth.layers import transformation_from_parameters


class ManyDepthPredictor(DepthPredictor):

    def __init__(self) -> None:
        super(ManyDepthPredictor, self).__init__()


        encoder_path = Path(__file__).parent / 'weights' / 'KITTI_HR' / 'encoder.pth'
        decoder_path = Path(__file__).parent / 'weights' / 'KITTI_HR' / 'depth.pth'

        pose_encoder_path = Path(__file__).parent / 'weights' / 'KITTI_HR' / 'pose_encoder.pth'
        pose_decoder_path = Path(__file__).parent / 'weights' / 'KITTI_HR' / 'pose.pth'

        
        # Depth
        encoder_dict = torch.load(encoder_path) if torch.cuda.is_available() else torch.load(encoder_path, map_location = 'cpu')
        decoder_dict = torch.load(decoder_path) if torch.cuda.is_available() else torch.load(decoder_path, map_location = 'cpu')
       
        encoder = encoder = manydepth_networks.ResnetEncoderMatching(
            18, False,
            input_width=encoder_dict['width'],
            input_height=encoder_dict['height'],
            adaptive_bins=True,
            min_depth_bin=0.1,
            max_depth_bin=20.0,
            depth_binning='linear',
            num_depth_bins=96
        )
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in encoder.state_dict()})
        encoder.cuda() if torch.cuda.is_available() else encoder.cpu()
        encoder.eval()
        
        decoder = manydepth_networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
        decoder.load_state_dict(decoder_dict)        
        decoder.cuda() if torch.cuda.is_available() else decoder.cpu()
        decoder.eval()

        
        # Pose
        pose_encoder_dict = torch.load(pose_encoder_path) if torch.cuda.is_available() else torch.load(pose_encoder_path, map_location = 'cpu')
        pose_decoder_dict = torch.load(pose_decoder_path) if torch.cuda.is_available() else torch.load(pose_decoder_path, map_location = 'cpu')
        
        pose_encoder = manydepth_networks.ResnetEncoder(18, False, num_input_images=2)
        pose_encoder.load_state_dict(pose_encoder_dict, strict=True)
        pose_encoder.cuda() if torch.cuda.is_available() else pose_encoder.cpu()

        pose_decoder = manydepth_networks.PoseDecoder(pose_encoder.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)
        pose_decoder.load_state_dict(pose_decoder_dict, strict=True)
        pose_decoder.cuda() if torch.cuda.is_available() else pose_decoder.cpu()

        self.encoder = encoder
        self.decoder = decoder
        self.pose_encoder = pose_encoder
        self.pose_decoder = pose_decoder

        self.feat_pad = nn.ReflectionPad2d(1)


        self.min_depth_bin = encoder_dict['min_depth_bin']
        self.max_depth_bin = encoder_dict['max_depth_bin']


        # Intrinsic
        K = np.array([
            [0.58, 0, 0.5, 0],
            [0, 1.92, 0.5, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], dtype=np.float32
        )


        K[0, :] *= encoder_dict['width'] // (2 ** 2)
        K[1, :] *= encoder_dict['height'] // (2 ** 2)

        invK = np.linalg.pinv(K)

        self.K = torch.from_numpy(K).cuda()
        self.invK = torch.from_numpy(invK).cuda()



    def get_encoder_params(self):
        return list(self.encoder.parameters()) + list(self.pose_encoder.parameters()) 

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

        # Check that input contains 2 images
        assert inputs.shape[1] == 6, 'Input must contain 2 concat images when using ManyDepth'


        inputs_1 = inputs[:, :3] # id -1
        inputs_2 = inputs[:, 3:] # id  0

        a, t = self.pose_decoder([self.pose_encoder(torch.cat([inputs_1, inputs_2], dim=1))])
        pose = transformation_from_parameters(a[:, 0], t[:, 0], invert=True)

        output, _, _ = self.encoder(
            inputs_2, inputs_1[:, None, ...], pose[:, None, ...],
            self.K[None, ...], self.invK[None, ...], self.min_depth_bin, self.max_depth_bin
        )
        y_pred = self.decoder(output)
        y_pred = y_pred[('disp', 0)]

        min_disp = 1 / 100.0
        max_disp = 1 / 0.1
        y_pred = 1.0 / (min_disp + (max_disp - min_disp) * y_pred)
        
        return y_pred
