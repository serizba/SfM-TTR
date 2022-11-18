# SfM-TTR: Using Structure from Motion for Test-Time Refinement of Single-View Depth Networks

Code for refining depth estimation networks using COLMAP reconstructions.

## Setup

Install required dependencies for SfM-TTR (for specific model dependencies check their corresponding repositories):
```python
conda install pytorch==1.12 torchvision -c pytorch
conda install -c conda-forge statsmodels matplotlib yacs
conda install tqdm
pip install pytorch-lightning
```

This code is provided with the nested repositories of [AdaBins](https://github.com/shariqfarooq123/AdaBins), [ManyDepth](https://github.com/nianticlabs/manydepth), [CADepth](https://github.com/kamiLight/CADepth-master) and [DIFFNet](https://github.com/brandleyzhou/DIFFNet). 

We provide the weights of DIFFNet to quickly test our method. Although for the rest of the networks all code is included, you need to manually download their weights. Once downloaded, place them in `SfM-TTR/sfmttr/models/{model_name}/weights/`.

## Data

To quickly test our method, we included the input images, ground truth and sparse reconstruction of one scene within this code (`SfM-TTR/example_sequence/`).

To run and evaluate SfM-TTR with the complete KITTI dataset, please download the [KITTI raw data](https://www.cvlibs.net/datasets/kitti/raw_data.php) and the [KITTI ground truth](https://www.cvlibs.net/datasets/kitti/eval_depth.php). You also need to run COLMAP on each sequence to obtain a sparse reconstruction. 



## Running

You can run the provided example of SfM-TTR with:

```bash
python3 main.py \
  --kitti-raw-path ./example_sequence/kitti_raw/ \
  --kitti-gt-path ./example_sequence/kitti_gt  \
  --reconstruction-path ./example_sequence/colmap_reconstructions/ \
  --sequence 2011_09_26_drive_0002_sync
```
