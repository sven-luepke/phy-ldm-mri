# Physics-Informed Latent Diffusion for Multimodal Brain MRI Synthesis | [Project Page](https://sven-luepke.github.io/phy-ldm-mri/) | [Paper](https://arxiv.org/abs/2409.13532)

Code for the paper "Physics-Informed Latent Diffusion for Multimodal Brain MRI Synthesis"

## Prerequisites
- Python 3
- Pytorch 2.1

## Setup

### Install Packages
```
pip install -r requirements.txt
```

### Download Model Checkpoints
Download the model checkpoints [here](https://drive.google.com/drive/folders/1MmBI_DKFBgfpPQJgUtjH4Y2qVGHu4TMR?usp=drive_link) and place them in the `checkpoints` directory.

### Generate Images
Run the following commands to replicate the generated images shown in the paper:
```
python ./source/generate.py --vae_checkpoint=./checkpoints/phy_vae_20240629_203331.pt --unet_checkpoint=./checkpoints/phy_ldm_20240630_081637.pt --seed=7 --te=0.003 --tr=0.1 --ti=1.0 --model=mprage
python ./source/generate.py --vae_checkpoint=./checkpoints/phy_vae_20240629_203331.pt --unet_checkpoint=./checkpoints/phy_ldm_20240630_081637.pt --seed=7 --te=0.003 --tr=2.4 --ti=0.5 --model=mprage
python ./source/generate.py --vae_checkpoint=./checkpoints/phy_vae_20240629_203331.pt --unet_checkpoint=./checkpoints/phy_ldm_20240630_081637.pt --seed=7 --te=0.003 --tr=2.4 --ti=0.25 --model=mprage
python ./source/generate.py --vae_checkpoint=./checkpoints/phy_vae_20240629_203331.pt --unet_checkpoint=./checkpoints/phy_ldm_20240630_081637.pt --seed=7 --te=0.8 --tr=3.2 --model=se
python ./source/generate.py --vae_checkpoint=./checkpoints/phy_vae_20240629_203331.pt --unet_checkpoint=./checkpoints/phy_ldm_20240630_081637.pt --seed=7 --te=0.2 --tr=3.2 --model=se
python ./source/generate.py --vae_checkpoint=./checkpoints/phy_vae_20240629_203331.pt --unet_checkpoint=./checkpoints/phy_ldm_20240630_081637.pt --seed=7 --te=0.45 --tr=1.0 --model=se
python ./source/generate.py --vae_checkpoint=./checkpoints/phy_vae_20240629_203331.pt --unet_checkpoint=./checkpoints/phy_ldm_20240630_081637.pt --seed=7 --te=0.45 --tr=9.0 --ti=2.5 --model=flair
python ./source/generate.py --vae_checkpoint=./checkpoints/phy_vae_20240629_203331.pt --unet_checkpoint=./checkpoints/phy_ldm_20240630_081637.pt --seed=7 --te=0.091 --tr=9.0 --ti=5.0 --model=flair
python ./source/generate.py --vae_checkpoint=./checkpoints/phy_vae_20240629_203331.pt --unet_checkpoint=./checkpoints/phy_ldm_20240630_081637.pt --seed=7 --te=0.5 --tr=9.0 --ti=4.0 --model=flair
```
Inspect the generated images in tensorboard:
```
tensorboard --logdir=./runs/gen_phy_ldm/
```

### Dataset Preparation
1. Download OASIS-3 MR images
```
./download_oasis_scans.sh <input_file.csv> <directory_name> <nitrc_ir_username> T1w,T2w,flair
```
- See https://github.com/NrgXnat/oasis-scripts/tree/master for details
TODO:
- add input_file.csv
- script to copy files from acq_param.csv into flat directory structure
- preprocess with unires
- crop scans
- extract 2D slices

### Training
1. Variational Autoencoder
```
python source/train_vae.py --data=./data
```
2. Latent Diffusion UNet
```
python source/train_ldm.py --data=./data --vae_checkpoint=<vae_checkpoint_file.pth>
```
TODO:
- add Tensorboard commands

## BibTeX
```
@article{lupke2024physics,
  author    = {L{\"u}pke, Sven and Yeganeh, Yousef and Adeli, Ehsan and Navab, Nassir and Farshad, Azade},
  title     = {Physics-Informed Latent Diffusion for Multimodal Brain MRI Synthesis},
  journal   = {5th International Workshop on Multiscale Multimodal Medical Imaging},
  year      = {2024},
}
```
