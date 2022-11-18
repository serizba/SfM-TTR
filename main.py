import argparse
from pathlib import Path

from tqdm.auto import tqdm
from collections import defaultdict

import torch
import pytorch_lightning as pl
import numpy as np

from sfmttr.models import ManyDepthPredictor, DIFFNetPredictor, AdaBinsPredictor, CADepthPredictor
from sfmttr.data.kitti import KITTI, KITTI_TEST_SEQS, KITTI_NO_SFM_SEQS
from sfmttr import SfMTuner, TunerDataset, KITTIMetrics

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


class MetricsLogger():
    def __init__(self):
        self.metrics_prev = defaultdict(list)
        self.metrics_post = defaultdict(list)
    
    def __call__(self, m_prev, m_post):
        for k, v in m_prev.items():
            self.metrics_prev[k].append(v.item())
        for k, v in m_post.items():
            self.metrics_post[k].append(v.item())

    def get_metrics(self):
        m = zip(self.metrics_prev.items(), self.metrics_post.items())
        return {
            k1: (np.mean(v1), np.mean(v2)) for (k1, v1), (k2, v2) in m
        }

    def to_str(self):
        m = self.get_metrics()
        r = f'  {"Metric":<10}: {"Prev.":<5} -> {"Post.":<5}'
        r = f'{r}\n {"-"*len(r)}\n'
        return r + '\n'.join(
            f'  {k:<10}: {v1:<5.3f} -> {v2:<5.3f}' for k, (v1, v2) in m.items()
        )


def main(args):

    model_class = {
        'diffnet': DIFFNetPredictor,
        'manydepth': ManyDepthPredictor,
        'adabins': AdaBinsPredictor,
        'cadepth': CADepthPredictor,
    }[args.model]

    # Select sequences
    if args.sequence is not None:
        if args.sequence in KITTI_TEST_SEQS:
            seqs = [args.sequence]
        else:
            raise ValueError(f'Unknown sequence {args.sequence}')
    else:
        seqs = KITTI_TEST_SEQS

    metrics = KITTIMetrics(median_scaling=True if args.model != 'adabins' else False)

    all_metrics = MetricsLogger()

    # Original model
    model_prev = model_class()
    model_prev = model_prev.cuda()
    model_prev = model_prev.eval()


    for seq in tqdm(seqs):

        seq_metrics = MetricsLogger()
        
        # Load sequence data
        kitti = KITTI(
            args.kitti_raw_path,
            args.kitti_gt_path,
            'eigen_with_gt',
            'test',
            sequence=seq,
            inputs_transform=model_class.get_inputs_transform(),
            y_true_transform=model_class.get_y_true_transform(),
            return_prev=(args.model == 'manydepth'),
        )

        # Refined model
        model_post = SfMTuner(model_class())
        model_post = model_post.cuda()

        if seq not in KITTI_NO_SFM_SEQS:
            # If SfM reconstruction is available, use it to refine the model

            # Create SfM data from the reconstruction
            tuner_dataloader = torch.utils.data.DataLoader(
                TunerDataset(
                    kitti, model_post,
                    args.reconstruction_path / seq / 'sparse',
                    kb_crop=(args.model == 'adabins'),
                ), batch_size=1, shuffle=True, num_workers=8
            )

            trainer = pl.Trainer(
                max_steps=200, accelerator="gpu", devices=1, enable_progress_bar=False,
                enable_checkpointing=False, logger=False, enable_model_summary=False
            )
            trainer.fit(model_post, tuner_dataloader)

        model_post = model_post.cuda()
        model_post = model_post.eval()

        # Evaluate sequence
        for inputs, y_true in torch.utils.data.DataLoader(kitti, batch_size=1, shuffle=False, num_workers=8):

            inputs = inputs.cuda()
            y_true = y_true.cuda()

            with torch.no_grad():
                y_pred_prev = model_prev(inputs)
                y_pred_post = model_post(inputs)
            
            metrics_prev = metrics(y_true, y_pred_prev)
            metrics_post = metrics(y_true, y_pred_post)

            seq_metrics(metrics_prev, metrics_post)
            all_metrics(metrics_prev, metrics_post)

        tqdm.write(f'{seq}: ')
        tqdm.write(seq_metrics.to_str())

    tqdm.write('Global results:')
    tqdm.write(all_metrics.to_str())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run SfM-TTR on the KITTI dataset")

    parser.add_argument(
        "--kitti-raw-path",
        type=Path,
        help="path to the KITTI raw data",
        required=True,
    )
    parser.add_argument(
        '--reconstruction-path',
        type=Path,
        help='path to the COLMAP reconstructions folder',
        required=True,
    )
    parser.add_argument(
        "--kitti-gt-path",
        type=Path,
        help="path to the KITTI ground truth data, if omitted, reprojected LIDAR data will be used",
        required=False,
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['diffnet', 'manydepth', 'cadepth', 'adabins'],
        default='diffnet',
        help='Network to refine',
    )
    parser.add_argument(
        '--sequence',
        type=str,
        help='If set, evaluates only on the specified sequence',
    )

    args = parser.parse_args()
    main(args)
