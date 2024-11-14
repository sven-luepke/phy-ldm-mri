# Physics-Informed Latent Diffusion for Multimodal Brain MRI Synthesis | [Project Page](https://sven-luepke.github.io/phy-ldm-mri/) | [Paper](https://arxiv.org/abs/2409.13532)

Code for the paper "Physics-Informed Latent Diffusion for Multimodal Brain MRI Synthesis" presented at the [5th International Workshop on Multiscale Multimodal Imaging](https://mmmi2024.github.io/) at MICCAI 2024

## Setup
### Prerequisites
- Python 3
- Pytorch 2.1
- [UniRes](https://github.com/brudfors/UniRes)
### Install Dependencies
```
pip install -r requirements.txt
```

## Generate Images
To generate the images shown in the paper, download the pretrained models [here](https://drive.google.com/drive/folders/1MmBI_DKFBgfpPQJgUtjH4Y2qVGHu4TMR?usp=drive_link), place them in the `./checkpoints` directory, and run the following commands:

```
python generate.py --vae_checkpoint=./checkpoints/phy_vae.pt --unet_checkpoint=./checkpoints/phy_ldm.pt --seed=7 --te=0.003 --tr=0.1 --ti=1.0 --model=mprage
python generate.py --vae_checkpoint=./checkpoints/phy_vae.pt --unet_checkpoint=./checkpoints/phy_ldm.pt --seed=7 --te=0.003 --tr=2.4 --ti=0.5 --model=mprage
python generate.py --vae_checkpoint=./checkpoints/phy_vae.pt --unet_checkpoint=./checkpoints/phy_ldm.pt --seed=7 --te=0.003 --tr=2.4 --ti=0.25 --model=mprage
python generate.py --vae_checkpoint=./checkpoints/phy_vae.pt --unet_checkpoint=./checkpoints/phy_ldm.pt --seed=7 --te=0.8 --tr=3.2 --model=se
python generate.py --vae_checkpoint=./checkpoints/phy_vae.pt --unet_checkpoint=./checkpoints/phy_ldm.pt --seed=7 --te=0.2 --tr=3.2 --model=se
python generate.py --vae_checkpoint=./checkpoints/phy_vae.pt --unet_checkpoint=./checkpoints/phy_ldm.pt --seed=7 --te=0.45 --tr=1.0 --model=se
python generate.py --vae_checkpoint=./checkpoints/phy_vae.pt --unet_checkpoint=./checkpoints/phy_ldm.pt --seed=7 --te=0.45 --tr=9.0 --ti=2.5 --model=flair
python generate.py --vae_checkpoint=./checkpoints/phy_vae.pt --unet_checkpoint=./checkpoints/phy_ldm.pt --seed=7 --te=0.091 --tr=9.0 --ti=5.0 --model=flair
python generate.py --vae_checkpoint=./checkpoints/phy_vae.pt --unet_checkpoint=./checkpoints/phy_ldm.pt --seed=7 --te=0.5 --tr=9.0 --ti=4.0 --model=flair
```
Once the images have been generated, they can be viewed in tensorboard:
```
tensorboard --logdir=./runs/gen_phy_ldm/
```

## Dataset
1. Download the `T1w`, `T2w`, and `FLAIR` MRI scans from the [OASIS-3 dataset](https://sites.wustl.edu/oasisbrains/home/oasis-3/).

TODO:
- add input_file.csv and move csv files to repo

- script to copy files from acq_param.csv into flat directory structure
- preprocess with unires
- crop scans script
- extract 2D slices

### Training
1. Variational Autoencoder
```
python train_vae.py --data=./data
```
2. Latent Diffusion UNet
```
python train_ldm.py --data=./data --vae_checkpoint=<vae_checkpoint_file.pth>
```

## BibTeX
```
@article{lupke2024physics,
  author    = {L{\"u}pke, Sven and Yeganeh, Yousef and Adeli, Ehsan and Navab, Nassir and Farshad, Azade},
  title     = {Physics-Informed Latent Diffusion for Multimodal Brain MRI Synthesis},
  journal   = {5th International Workshop on Multiscale Multimodal Medical Imaging},
  year      = {2024},
}
```
