import argparse

import torch
from monai.transforms import ScaleIntensity
from generative.inferers import LatentDiffusionInferer
from generative.networks.nets import DiffusionModelUNet, AutoencoderKL
from generative.networks.schedulers import DDIMScheduler

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from utils import *
from physical_model import denormalize_log_q_map, replace_groupnorm_with_adaptive_groupnorm


def main():
    parser = argparse.ArgumentParser("Generate tissue parameter maps and MRI tissue contrast using a configurable signal model.")
    parser.add_argument("--vae_checkpoint", type=str, help="VAE training checkpoint")
    parser.add_argument("--unet_checkpoint", type=str, help="UNet training checkpoint")
    parser.add_argument("--logdir", type=str, default="gen_phy_ldm", help="Tensorboard experiment name.")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--tr", type=float)
    parser.add_argument("--te", type=float)
    parser.add_argument("--ti", type=float)
    parser.add_argument("--model", type=str, choices=["mprage", "se", "flair"])

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set random seed
    if args.seed is None:
        seed = random.randint(0, 2**32 - 1)
    else:
        seed = args.seed

    # setup models and optimizers
    latent_channels = 6
    autoencoder = AutoencoderKL(
        spatial_dims=2,
        in_channels=1,
        out_channels=3,
        num_channels=(64, 128, 128, 128),
        latent_channels=latent_channels,
        num_res_blocks=3,
        attention_levels=(False, False, False, False),
    ).to(device=device)
    replace_groupnorm_with_adaptive_groupnorm(autoencoder.encoder)
    vae_checkpoint = torch.load(args.vae_checkpoint)
    autoencoder.load_state_dict(vae_checkpoint["autoencoder_model_state_dict"])
    autoencoder.eval()

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
    unet_checkpoint = torch.load(args.unet_checkpoint)
    unet.load_state_dict(state_dict=unet_checkpoint["unet_model_state_dict"])
    unet.eval()

    # get latent scale factor
    scale_factor = unet_checkpoint["latent_scale_factor"]

    ddim_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        schedule="scaled_linear_beta",
        beta_start=0.0015, 
        beta_end=0.0205,
        clip_sample=False,
    )
    inferer = LatentDiffusionInferer(ddim_scheduler, scale_factor=scale_factor)
    ddim_scheduler.set_timesteps(num_inference_steps=50)

    tb_writer = create_tensorboard_writer(experiment_name=args.logdir, log_dir_root="./runs/")
    tb_image_scaler = ScaleIntensity(channel_wise=True)

    num_synthetic_images = 1
    generator = torch.Generator(device=device)
    generator = generator.manual_seed(seed)
    noise = torch.randn((num_synthetic_images, 6, 20, 28), generator=generator, device=device)

    with torch.no_grad():
        log_q_maps = inferer.sample(input_noise=noise, diffusion_model=unet, autoencoder_model=autoencoder)

    log_q_maps = denormalize_log_q_map(
        norm_log_q_map=log_q_maps,
        log_pd_mean=log_pd_prior_mean,
        log_t1_mean=log_t1_prior_mean,
        log_t2_mean=log_t2_prior_mean
    )
    
    # log q-maps to tensorboard
    q_map = torch.exp(log_q_maps[0]).cpu().numpy()
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))  # 1 row, 3 columns, adjusted size for colorbars
    titles = ['PD [a.u.]', 'T1 [s]', 'T2 [s]']
    upper_percentiles = [99, 99, 98]
    cmaps = ["gray", "inferno", "viridis"]

    for i in range(3):
        # Calculate the lower and upper percentiles
        vmin, vmax = np.percentile(q_map[i], [0, upper_percentiles[i]])
        # Clip the q_map values between the lower and upper percentiles
        clipped_q_map = np.clip(q_map[i], vmin, vmax)

        if i == 0:
            clipped_q_map = clipped_q_map / vmax
        
        # Plot each clipped image
        im = axs[i].imshow(clipped_q_map, cmap=cmaps[i])
        axs[i].set_title(titles[i], fontsize=24)
        axs[i].axis('off')
        cbar = fig.colorbar(im, ax=axs[i], shrink=0.9)
        cbar.ax.tick_params(labelsize=18)

        cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    plt.tight_layout()

    tb_writer.add_figure('Q-Maps', figure=fig, global_step=seed)
    plt.close(fig)

    def gen_t1w_mprage(q_map, echo_time, repetition_time, inversion_time):
        pd = q_map[0]
        t1 = q_map[1]
        t2 = q_map[2]
        return pd * (1 - (2 * torch.exp(-inversion_time / t1)) / (1 + torch.exp(-repetition_time / t1))) *  torch.exp(-echo_time / t2)
    
    t1w_scan = gen_t1w_mprage(
        q_map=torch.exp(log_q_maps[0]),
        echo_time=0.003,
        repetition_time=2.4,
        inversion_time=1.0
    )
    tb_writer.add_image("T1w", img_tensor=tb_image_scaler(t1w_scan.unsqueeze(0).unsqueeze(0)), global_step=seed, dataformats="NCHW")

    # T2w Spin Echo
    def gen_t2w_se(q_map, echo_time, repetition_time, inversion_time):
        pd = q_map[0]
        t1 = q_map[1]
        t2 = q_map[2]
        return pd * (1 - torch.exp(-repetition_time / t1)) *  torch.exp(-echo_time / t2)
    t2w_scan = gen_t2w_se(
        q_map=torch.exp(log_q_maps[0]),
        echo_time=0.45,
        repetition_time=3.2,
        inversion_time=0.0
    )
    tb_writer.add_image("T2w", img_tensor=tb_image_scaler(t2w_scan.unsqueeze(0).unsqueeze(0)), global_step=seed, dataformats="NCHW")
    
    # FLAIR
    def gen_flair(q_map, echo_time, repetition_time, inversion_time):
        pd = q_map[0]
        t1 = q_map[1]
        t2 = q_map[2]
        return pd * (1 - 2 * torch.exp(-inversion_time / t1) + torch.exp(-repetition_time / t1)) *  torch.exp(-echo_time / t2)
    flair_scan = gen_flair(
        q_map=torch.exp(log_q_maps[0]),
        echo_time=0.091,
        repetition_time=9.0,
        inversion_time=2.5
    )
    tb_writer.add_image("FLAIR", img_tensor=tb_image_scaler(flair_scan.unsqueeze(0).unsqueeze(0)), global_step=seed, dataformats="NCHW")

    if args.model == "mprage":
        custom_scan = gen_t1w_mprage(
            q_map=torch.exp(log_q_maps[0]),
            echo_time=args.te,
            repetition_time=args.tr,
            inversion_time=args.ti,
        )
    elif args.model == "se":
        custom_scan = gen_t2w_se(
            q_map=torch.exp(log_q_maps[0]),
            echo_time=args.te,
            repetition_time=args.tr,
            inversion_time=args.ti,
        )
    elif args.model == "flair":
        custom_scan = gen_flair(
            q_map=torch.exp(log_q_maps[0]),
            echo_time=args.te,
            repetition_time=args.tr,
            inversion_time=args.ti,
        )
    tb_writer.add_image("Custom", img_tensor=tb_image_scaler(custom_scan.unsqueeze(0).unsqueeze(0)), global_step=seed, dataformats="NCHW")

    # add hparams to tensorboard
    tb_writer.add_hparams(
        {
            "model": args.model,
            "TE [s]": args.te,
            "TR [s]": args.tr,
            "TI [s]": args.ti,
        },
        metric_dict={}
    )

    tb_writer.close()


if __name__ == "__main__":
    main()