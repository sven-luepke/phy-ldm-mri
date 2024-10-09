# Physics-Informed Latent Diffusion for Multimodal Brain MRI Synthesis

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
Coming soon...

### Training
Coming soon...

## BibTeX
```
@article{lupke2024physics,
  author    = {L{\"u}pke, Sven and Yeganeh, Yousef and Adeli, Ehsan and Navab, Nassir and Farshad, Azade},
  title     = {Physics-Informed Latent Diffusion for Multimodal Brain MRI Synthesis},
  journal   = {5th International Workshop on Multiscale Multimodal Medical Imaging},
  year      = {2024},
}
```