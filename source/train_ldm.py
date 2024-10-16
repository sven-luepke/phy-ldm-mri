from typing import Tuple, List, Optional, Dict, Union
import datetime
import os
import itertools
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from monai.data import DataLoader
from monai.transforms import ScaleIntensity, ScaleIntensityRangePercentiles

from generative.inferers import LatentDiffusionInferer
from generative.networks.nets import DiffusionModelUNet, AutoencoderKL
from generative.networks.schedulers import DDIMScheduler

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from utils import *
from models import denormalize_log_q_map, PhysicalVAE, replace_groupnorm_with_adaptive_groupnorm, AdaptiveGroupNorm


def main():
    parser = argparse.ArgumentParser("Train the variational autoencoder of a latent diffusion model.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--train_sample_limit", type=int, help="Maximum number of training samples to use. If unspecified, the full training set is used.")
    parser.add_argument("--vae_checkpoint", type=str, help="VAE training checkpoint")
    parser.add_argument("--logdir", type=str, default="physical_ldm_2", help="Tensorboard experiment name.")
    parser.add_argument("--data", type=str, help="Path to the directory containing the data.", required=True)

    # hyperparameters
    parser.add_argument("--adagn", action="store_true", help="Use adaptive group normalization to condition the encoder.")
    parser.add_argument("--use_exp_offset", action="store_true", help="Use the exponential offset.")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility.")

    args = parser.parse_args()

    # python experiments/physical_latent/train_ldm.py --epochs=1 --data=../data/ --logdir=phy_ldm_v2_test --vae_checkpoint=../checkpoints/phy_vae_20240625_233031.pt --adagn --use_exp_offset

    # set random seed
    if args.seed is None:
        seed = random.randint(0, 2**32 - 1)
    else:
        seed = args.seed
    torch.manual_seed(seed)

    acq_param_path = os.path.join(args.data, "oasis-3-acq-params.csv")
    image_path = os.path.join(args.data, "oasis-3-mri-2d-8")

    # load data
    train_session_ids, _ = create_oasis_3_mr_data_split("../json/oasis_3_mri_sessions.json")
    meta_df = load_oasis_3_mr_meta_df(acq_param_path)
    train_dataset, _ = create_datasets(
        image_dataset_path=image_path,
        train_session_ids=train_session_ids,
        val_session_ids=[],
        modalities=["T1w", "T2w", "FLAIR"],
        require_all_modalities=True,
        meta_df=meta_df,
        sample_limit=args.train_sample_limit
    )

    batch_size = args.batch_size
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=oasis_multimodal_collate_fn, shuffle=True)

    latent_channels = 6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup models and optimizers
    autoencoder = AutoencoderKL(
        spatial_dims=2,
        in_channels=1,
        out_channels=3,
        num_channels=(64, 128, 128, 128),
        latent_channels=latent_channels,
        num_res_blocks=3,
        attention_levels=(False, False, False, False),
    )
    if args.adagn:
        replace_groupnorm_with_adaptive_groupnorm(autoencoder.encoder)

    vae_checkpoint = torch.load(args.vae_checkpoint)
    autoencoder.load_state_dict(vae_checkpoint["autoencoder_model_state_dict"])

    acq_param_encoder = nn.Sequential(
        nn.Linear(3, 128),
        nn.GELU(),
        nn.Linear(128, 2)
    ).to(device)
    acq_param_encoder.load_state_dict(vae_checkpoint["acq_param_encoder_model_state_dict"])

    physical_autoencoder = PhysicalVAE(autoencoder=autoencoder).to(device)
    physical_autoencoder.eval()
    acq_param_encoder.eval()

    scale_transform = ScaleIntensityRangePercentiles(lower=0, upper=99.5, b_min=-1, b_max=1, channel_wise=True)

    ## compute latent scale factor
    std_sum = 0
    for train_batch in tqdm(train_loader, desc=f"Computing latent scale factor"):
        with torch.no_grad():
            with autocast(enabled=True):
                train_images = train_batch["images"].to(device)
                N, C, _, _ = train_images.shape

                if args.adagn:
                    train_acq_params_norm = train_batch["acq_params_norm"].to(device).float()
                    adagn_input = acq_param_encoder(train_acq_params_norm.view(N * C, 3))
                    adagn_scale = adagn_input[:, 0]
                    adagn_shift = adagn_input[:, 1]
                    for module in physical_autoencoder.autoencoder.encoder.modules():
                        if isinstance(module, AdaptiveGroupNorm):
                            module.assign_adaptive_paramters(scale=adagn_scale, shift=adagn_shift)
            
                train_inputs = scale_transform(train_images)
                z = physical_autoencoder.encode_stage_2_inputs(train_inputs)

        std_sum += torch.std(z)

    std = std_sum / len(train_loader)
    scale_factor = 1 / std
    print(f"Latent scaling factor set to {scale_factor}")

    pd_prior_median = 50
    t1_prior_median = 1
    t2_prior_median = 0.1
    log_pd_prior_mean = torch.log(torch.tensor(pd_prior_median, device=device))
    log_t1_prior_mean = torch.log(torch.tensor(t1_prior_median, device=device))
    log_t2_prior_mean = torch.log(torch.tensor(t2_prior_median, device=device))

    
    unet = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=latent_channels,
        out_channels=latent_channels,
        num_res_blocks=2,
        num_channels=(256, 512, 768),
        attention_levels=(False, True, True),
        resblock_updown=True,
        num_head_channels=(0, 512, 768),
    ).to(device)
    optimizer = torch.optim.Adam(unet.parameters(), lr=2.5e-5)

    ddim_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        schedule="scaled_linear_beta",
        beta_start=0.0015, 
        beta_end=0.0205,
        clip_sample=False,
    )
    inferer = LatentDiffusionInferer(ddim_scheduler, scale_factor=scale_factor)
    ddim_scheduler.set_timesteps(num_inference_steps=50)


    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = "../checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)
    out_checkpoint = os.path.join(checkpoint_dir, f"phy_ldm_{timestamp}.pt")


    tb_writer = create_tensorboard_writer(experiment_name=args.logdir, log_dir_root="../runs/")
    epochs = args.epochs
    grad_scaler = GradScaler()
    tb_image_scaler = ScaleIntensity(channel_wise=True)

    for epoch in range(epochs):
        unet.train()

        for batch_index, train_batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1} Training")):
            optimizer.zero_grad()

            train_images = train_batch["images"].to(device)
            #train_image_mask = train_batch["image_mask"].to(device)
            train_inputs = scale_transform(train_images)
            N, C, _, _ = train_images.shape

            with autocast(enabled=True):

                if args.adagn:
                    train_acq_params_norm = train_batch["acq_params_norm"].to(device).float()
                    adagn_input = acq_param_encoder(train_acq_params_norm.view(N * C, 3))
                    adagn_scale = adagn_input[:, 0]
                    adagn_shift = adagn_input[:, 1]
                    for module in physical_autoencoder.autoencoder.encoder.modules():
                        if isinstance(module, AdaptiveGroupNorm):
                            module.assign_adaptive_paramters(scale=adagn_scale, shift=adagn_shift)

                noise = torch.randn((train_images.shape[0], latent_channels, 20, 28), device=device)
                timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (train_images.shape[0],), device=device).long()

                train_noise_pred = inferer(
                    inputs=train_inputs, diffusion_model=unet, noise=noise, timesteps=timesteps, autoencoder_model=physical_autoencoder
                )
                train_loss = F.mse_loss(train_noise_pred.float(), noise.float())

            grad_scaler.scale(train_loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            iteration = len(train_loader) * epoch + batch_index
            tb_writer.add_scalars("Loss", {"Training": train_loss.item()}, iteration)


        if epoch == 0 or ((epoch + 1) % 5) == 0:
            unet.eval()
            z = torch.randn((1, latent_channels, 20, 28), device=device)
            with autocast(enabled=True):
                gen_norm_log_q_map = inferer.sample(
                    input_noise=z, diffusion_model=unet, autoencoder_model=physical_autoencoder
                )

                if args.use_exp_offset:
                    gen_log_q_map = denormalize_log_q_map(
                        norm_log_q_map=gen_norm_log_q_map,
                        log_pd_mean=log_pd_prior_mean,
                        log_t1_mean=log_t1_prior_mean,
                        log_t2_mean=log_t2_prior_mean
                    )
                else:
                    gen_log_q_map = gen_norm_log_q_map


            q_map = torch.exp(gen_log_q_map[0]).cpu().numpy()
            fig, axs = plt.subplots(1, 3, figsize=(18, 4))  # 1 row, 3 columns, adjusted size for colorbars
            titles = ['PD', 'T1 [s]', 'T2 [s]']
            upper_percentiles = [99, 99, 99]
            cmaps = ["gray", "inferno", "viridis"]

            for i in range(3):
                # Calculate the lower and upper percentiles
                vmin, vmax = np.percentile(q_map[i], [0, upper_percentiles[i]])
                # Clip the q_map values between the lower and upper percentiles
                clipped_q_map = np.clip(q_map[i], vmin, vmax)
                
                # Plot each clipped image
                im = axs[i].imshow(clipped_q_map, cmap=cmaps[i])
                axs[i].set_title(titles[i], fontsize=24)
                axs[i].axis('off')
                cbar = fig.colorbar(im, ax=axs[i], shrink=0.9)
                cbar.ax.tick_params(labelsize=18)

                cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

            plt.tight_layout()

            tb_writer.add_figure('Q-Maps', figure=fig, global_step=epoch)
            plt.close(fig)

        # checkpoint
        if (epoch + 1) % 5 == 0 or epoch == (epochs - 1):
            torch.save({
                    "epoch": epoch,
                    "unet_model_state_dict": unet.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "latent_scale_factor": scale_factor,
                },
                out_checkpoint
            )
            print(f"Saved LDM checkpoint to {out_checkpoint}")


if __name__ == "__main__":
    main()