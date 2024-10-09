import argparse
import random
import datetime
import itertools

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torcheval.metrics import PeakSignalNoiseRatio 

from monai.data import DataLoader
from monai.transforms import ScaleIntensityRangePercentiles, ScaleIntensity
from monai.networks.nets import Regressor

from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL,  PatchDiscriminator
from generative.metrics import FIDMetric, MultiScaleSSIMMetric

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from utils import *
from physical_model import MRSignalModel, denormalize_log_q_map, replace_groupnorm_with_adaptive_groupnorm, AdaptiveGroupNorm


def main():
    parser = argparse.ArgumentParser("Train the variational autoencoder of a latent diffusion model.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--train_sample_limit", type=int, help="Maximum number of training samples to use. If unspecified, the full training set is used.")
    parser.add_argument("--kl_weight", type=float, default=1e-6, help="Kullback-Leibler divergence weight.")
    parser.add_argument("--perceptual_weight", type=float, default=1e-3, help="Weight of the perceptual loss.")
    parser.add_argument("--adversarial_weight", type=float, default=1e-2, help="Weight of the adversarial loss.")
    parser.add_argument("--warmup_epochs", type=int, default=1, help="Number of epoch to train without the adversarial loss.")
    parser.add_argument("--logdir", type=str, default="physical_vae", help="Tensorboard experiment name.")
    parser.add_argument("--checkpoint", type=str, help="VAE training checkpoint.")
    parser.add_argument("--data", type=str, help="Path to the directory containing the data.", required=True)

    # TODO: best parameters
    # --adversarial_weight=100
    # --perceptual_weight=10
    # --warmup_epochs=10
    # --epochs=100

    # hparams
    parser.add_argument("--loss", type=str, default="L2", choices=['L1', 'L2'])
    parser.add_argument("--use_regressor", action="store_true")
    parser.add_argument("--use_modality_dropout", action="store_true")
    parser.add_argument("--use_regularization", action="store_true", help="Use Physical Regularization.")
    parser.add_argument("--use_exp_offset", action="store_true", help="Use the exponential offset.")
    parser.add_argument("--adagn", action="store_true", help="Condition the encoder on acqusition parameters using adaptive group normalization.")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility.")

    args = parser.parse_args()

    # python experiments/physical_latent/train_vae.py --epochs=50 --data=../data/ --use_regressor --use_modality_dropout --use_regularization --use_exp_offset --adagn

    # set random seed
    if args.seed is None:
        seed = random.randint(0, 2**32 - 1)
    else:
        seed = args.seed
    torch.manual_seed(seed)

    print(f"Starting VAE training with seed {seed}:")
    print(args)

    acq_param_path = os.path.join(args.data, "oasis-3-acq-params.csv")
    image_path = os.path.join(args.data, "oasis-3-mri-2d-8")

    # load data
    train_session_ids, val_session_ids = create_oasis_3_mr_data_split("../json/oasis_3_mri_sessions.json")
    meta_df = load_oasis_3_mr_meta_df(acq_param_path)
    train_dataset, val_dataset = create_datasets(
        image_dataset_path=image_path,
        train_session_ids=train_session_ids,
        val_session_ids=val_session_ids,
        modalities=["T1w", "T2w", "FLAIR"],
        require_all_modalities=True,
        meta_df=meta_df,
        sample_limit=args.train_sample_limit
    )

    batch_size = args.batch_size
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=oasis_multimodal_collate_fn, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, collate_fn=oasis_multimodal_collate_fn, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    latent_channels = 6
    # setup models and optimizers
    autoencoder = AutoencoderKL(
        spatial_dims=2,
        in_channels=1,
        out_channels=3,
        num_channels=(64, 128, 128, 128),
        latent_channels=latent_channels,
        num_res_blocks=3,
        attention_levels=(False, False, False, False),
    ).to(device)

    acq_param_encoder = nn.Sequential(
        nn.Linear(3, 128),
        nn.GELU(),
        nn.Linear(128, 2)
    ).to(device)
    if args.adagn:
        replace_groupnorm_with_adaptive_groupnorm(autoencoder.encoder)
    
    pd_prior_median = 50
    t1_prior_median = 1
    t2_prior_median = 0.1
    log_pd_prior_mean = torch.log(torch.tensor(pd_prior_median, device=device))
    log_t1_prior_mean = torch.log(torch.tensor(t1_prior_median, device=device))
    log_t2_prior_mean = torch.log(torch.tensor(t2_prior_median, device=device))

    scanner_gain_regressor = Regressor(
        in_shape=(1, 160, 224),
        out_shape=(1,),
        channels=(4, 8, 16, 32),
        strides=(2, 2, 2, 2),
        num_res_units=1,
    ).to(device)
    default_log_scanner_gain = torch.tensor(0.0, device=device)

    mr_signal_model = MRSignalModel().to(device).float()

    discriminator = PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64, in_channels=1, out_channels=1).to(device)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = "../checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)
    out_checkpoint = os.path.join(checkpoint_dir, f"phy_vae_{timestamp}.pt")

    acq_param_encoder.train()
    autoencoder.train()
    scanner_gain_regressor.train()

    tb_writer = create_tensorboard_writer(experiment_name=args.logdir, log_dir_root="../runs/")

    epochs = args.epochs
    kl_weight = args.kl_weight
    perceptual_weight = args.perceptual_weight
    adversarial_weight = args.adversarial_weight # 1e-2
    generator_parameter_list = list(autoencoder.parameters())
    if args.use_regressor:
        generator_parameter_list += list(scanner_gain_regressor.parameters())
    if args.adagn:
        generator_parameter_list += list(acq_param_encoder.parameters())
    optimizer_generator = torch.optim.Adam(generator_parameter_list, lr=5e-5)
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    # load checkpoint
    if args.checkpoint is not None:
        # load checkpoint only for evaluation
        assert epochs == 0

        checkpoint = torch.load(args.checkpoint)
        autoencoder.load_state_dict(checkpoint["autoencoder_model_state_dict"])
        discriminator.load_state_dict(checkpoint["discriminator_model_state_dict"])
        optimizer_generator.load_state_dict(checkpoint["optimizer_generator_state_dict"])
        optimizer_discriminator.load_state_dict(checkpoint["optimizer_discriminator_state_dict"])
        scanner_gain_regressor.load_state_dict(checkpoint["scanner_gain_regressor_state_dict"])
        acq_param_encoder.load_state_dict(checkpoint["acq_param_encoder_model_state_dict"])
        out_checkpoint = args.checkpoint

    hparams = {
        "epochs": args.epochs,
        "checkpoint": out_checkpoint,
        "batch_size": args.batch_size,
        "loss": args.loss,
        "use_regressor": args.use_regressor,
        "use_modality_dropout": args.use_modality_dropout,
        "use_regularization": args.use_regularization,
        "use_exp_offset": args.use_exp_offset,
        "seed": seed,
        "adagn": args.adagn,
    }

    grad_scaler_generator = torch.cuda.amp.GradScaler()
    grad_scaler_discriminator = torch.cuda.amp.GradScaler()

    autoencoder_warm_up_epochs = args.warmup_epochs
    val_interval = 16
    batches_per_validation = 2
    val_loader_iterator = itertools.cycle(iter(val_loader))

    if args.loss == "L1":
        train_reconstruction_loss_func = nn.L1Loss()
    elif args.loss == "L2":
        train_reconstruction_loss_func = nn.MSELoss()


    perceptual_loss_func = PerceptualLoss(spatial_dims=2, network_type="alex").to(device)
    adversarial_loss_func = PatchAdversarialLoss(criterion="least_squares")

    scale_transform = ScaleIntensityRangePercentiles(lower=0, upper=99.5, b_min=-1, b_max=1, channel_wise=True)
    tb_image_scaler = ScaleIntensity(channel_wise=True)

    for epoch in range(epochs):

        for batch_index, train_batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1} Training")):

            iteration = len(train_loader) * epoch + batch_index

            optimizer_generator.zero_grad()
            
            train_images = train_batch["images"].to(device)
            N, C, H, W = train_images.shape
            train_image_mask = train_batch["image_mask"].to(device)
            # mask with input dropout
            train_image_mask_input = train_batch["image_mask_input"].to(device)
            train_acq_params = train_batch["acq_params"].to(device)
            train_modality_id = train_batch["modality_id"].to(device)

            train_inputs = scale_transform(train_images)
            train_targets = train_images

            if args.use_modality_dropout:
                mask = train_image_mask_input.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            else:
                mask = train_image_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            if args.adagn:
                train_acq_params_norm = train_batch["acq_params_norm"].to(device).float()
                adagn_input = acq_param_encoder(train_acq_params_norm.view(N * C, 3))
                adagn_scale = adagn_input[:, 0]
                adagn_shift = adagn_input[:, 1]
                for module in autoencoder.encoder.modules():
                    if isinstance(module, AdaptiveGroupNorm):
                        module.assign_adaptive_paramters(scale=adagn_scale, shift=adagn_shift)

            z_mu, z_sigma = autoencoder.encode(train_inputs.view(N * C, 1, H, W))
            z_mu = z_mu.view(N, C, latent_channels, 20, 28)
            z_sigma = z_sigma.view(N, C, latent_channels, 20, 28)

            # Product of experts (PoE); equation 8 in https://arxiv.org/pdf/2202.03242.pdf
            # sigma^(-2) = sum(sigma^(-2) * mask)
            # mu = sum((mu * mask) / sigma^(-2)) * sigma^2
            z_sigma_squared = z_sigma * z_sigma
            z_sigma_squared_rcp = 1.0 / z_sigma_squared
            z_sigma_squared_fused_rcp = (z_sigma_squared_rcp * mask).sum(1)
            z_sigma_squared_fused = 1.0 / z_sigma_squared_fused_rcp
            z_sigma_fused = torch.sqrt(z_sigma_squared_fused)
            z_mu_fused = ((z_mu * mask) * z_sigma_squared_rcp).sum(1) * z_sigma_squared_fused

            z = autoencoder.sampling(z_mu_fused, z_sigma_fused)
            train_norm_log_q_map = autoencoder.decode(z)

            regularization_loss = torch.sum(train_norm_log_q_map * train_norm_log_q_map) / batch_size

            if args.use_exp_offset:
                train_log_q_map = denormalize_log_q_map(
                    norm_log_q_map=train_norm_log_q_map,
                    log_pd_mean=log_pd_prior_mean,
                    log_t1_mean=log_t1_prior_mean,
                    log_t2_mean=log_t2_prior_mean
                )
            else:
                train_log_q_map = train_norm_log_q_map

            # predict the scanner gain directly from the input images
            # pass the input images through the shared log scanner gain regression model
            # masked the predictions based on the available modalities
            # and average the masked predictions
            if args.use_regressor:
                log_scanner_gain = (scanner_gain_regressor(train_inputs.view(N * C, 1, H, W)).view(N, C) * train_image_mask_input).sum(1) / train_image_mask_input.sum(1)
                log_scanner_gain = log_scanner_gain.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            else:
                log_scanner_gain = default_log_scanner_gain


            train_reconstruction = mr_signal_model(train_log_q_map, train_acq_params, train_modality_id, log_scanner_gain)

            reconstruction_loss = train_reconstruction_loss_func(train_reconstruction.float(), train_targets.float())

            train_recon_scaled = scale_transform(train_reconstruction)
            perceptual_loss = perceptual_loss_func(
                train_recon_scaled.view(N * C, 1, H, W).contiguous().float(),
                train_inputs.view(N * C, 1, H, W).contiguous().float()
            )

            kl_loss = kl_div(z_mu=z_mu_fused, z_sigma=z_sigma_fused)

            loss_generator = reconstruction_loss
            loss_generator += kl_weight * kl_loss

            if args.use_regularization:
                loss_generator += regularization_loss * 0.0005

            perceptual_loss = perceptual_loss * perceptual_weight
            loss_generator += perceptual_loss

            if epoch >= autoencoder_warm_up_epochs:
                # but since they are just replaced with black images it might not be too bad if we don't
                train_recon_scaled = scale_transform(train_reconstruction)
                logits_fake = discriminator(train_recon_scaled.view(N * C, 1, H, W).contiguous().float())[-1]
                adversarial_loss = adversarial_loss_func(logits_fake, target_is_real=True, for_discriminator=False)
                loss_generator += adversarial_weight * adversarial_loss

            tb_writer.add_scalars("Perceptual Loss", {"Training": perceptual_loss.item()}, iteration)
            tb_writer.add_scalars("Reconstruction Loss", {"Training": reconstruction_loss.item()}, iteration)
            tb_writer.add_scalars("Generator Loss", {"Training": loss_generator.item()}, iteration)

            grad_scaler_generator.scale(loss_generator).backward()
            grad_scaler_generator.step(optimizer_generator)
            grad_scaler_generator.update()

            if epoch >= autoencoder_warm_up_epochs:
                
                optimizer_discriminator.zero_grad(set_to_none=True)

                train_recon_scaled = scale_transform(train_reconstruction)
                logits_fake = discriminator(train_recon_scaled.view(N * C, 1, H, W).contiguous().float().detach())[-1]
                adversarial_loss_fake = adversarial_loss_func(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(train_inputs.view(N * C, 1, H, W).contiguous().float().detach())[-1]
                adversarial_loss_real = adversarial_loss_func(logits_real, target_is_real=True, for_discriminator=True)

                loss_discriminator = adversarial_weight * (adversarial_loss_fake + adversarial_loss_real) * 0.5

                tb_writer.add_scalars("Discriminator Loss", {"Training": loss_discriminator.item()}, iteration)

                grad_scaler_discriminator.scale(loss_discriminator).backward()
                grad_scaler_discriminator.step(optimizer_discriminator)
                grad_scaler_discriminator.update()


            # validation
            if (iteration + 1) % val_interval == 0:
                acq_param_encoder.eval()
                autoencoder.eval()
                scanner_gain_regressor.eval()

                val_loss_sum = 0

                mse_func = nn.MSELoss()
                val_mse_sum = 0
                mae_func = nn.L1Loss()
                val_mae_sum = 0
                ms_ssim_func = MultiScaleSSIMMetric(spatial_dims=2, data_range=1.0, kernel_size=4)
                val_ms_ssim_scores = []

                for _ in range(batches_per_validation):
                    with torch.no_grad():
                        val_batch = next(val_loader_iterator)

                        val_images = val_batch["images"].to(device)
                        N, C, H, W = val_images.shape
                        val_image_mask = val_batch["image_mask"].to(device)
                        val_acq_params = val_batch["acq_params"].to(device)
                        val_modality_id = val_batch["modality_id"].to(device)

                        val_inputs = scale_transform(val_images)
                        val_targets = val_images

                        if args.adagn:
                            val_acq_params_norm = val_batch["acq_params_norm"].to(device).float()
                            adagn_input = acq_param_encoder(val_acq_params_norm.view(N * C, 3))
                            adagn_scale = adagn_input[:, 0]
                            adagn_shift = adagn_input[:, 1]
                            for module in autoencoder.encoder.modules():
                                if isinstance(module, AdaptiveGroupNorm):
                                    module.assign_adaptive_paramters(scale=adagn_scale, shift=adagn_shift)

                        mask = val_image_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                        z_mu, z_sigma = autoencoder.encode(val_inputs.view(N * C, 1, H, W))
                        z_mu = z_mu.view(N, C, latent_channels, 20, 28)
                        z_sigma = z_sigma.view(N, C, latent_channels, 20, 28)

                        # Product of experts (PoE); equation 8 in https://arxiv.org/pdf/2202.03242.pdf
                        # sigma^(-2) = sum(sigma^(-2) * mask)
                        # mu = sum((mu * mask) / sigma^(-2)) * sigma^2
                        z_sigma_squared = z_sigma * z_sigma
                        z_sigma_squared_rcp = 1.0 / z_sigma_squared
                        z_sigma_squared_fused_rcp = (z_sigma_squared_rcp * mask).sum(1)
                        z_sigma_squared_fused = 1.0 / z_sigma_squared_fused_rcp
                        z_sigma_fused = torch.sqrt(z_sigma_squared_fused)
                        z_mu_fused = ((z_mu * mask) * z_sigma_squared_rcp).sum(1) * z_sigma_squared_fused

                        z = autoencoder.sampling(z_mu_fused, z_sigma_fused)
                        val_norm_log_q_map = autoencoder.decode(z)

                        if args.use_exp_offset:
                            val_log_q_map = denormalize_log_q_map(
                                norm_log_q_map=val_norm_log_q_map,
                                log_pd_mean=log_pd_prior_mean,
                                log_t1_mean=log_t1_prior_mean,
                                log_t2_mean=log_t2_prior_mean
                            )
                        else:
                            val_log_q_map = val_norm_log_q_map

                        # predict the scanner gain directly from the input images
                        # pass the input images through the shared log scanner gain regression model
                        # masked the predictions based on the available modalities
                        # and average the masked predictions
                        if args.use_regressor:
                            log_scanner_gain = (scanner_gain_regressor(val_inputs.view(N * C, 1, H, W)).view(N, C) * val_image_mask).sum(1) / val_image_mask.sum(1)
                            log_scanner_gain = log_scanner_gain.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                        else:
                            log_scanner_gain = default_log_scanner_gain

                        val_reconstruction = mr_signal_model(val_log_q_map, val_acq_params, val_modality_id, log_scanner_gain)
                        val_loss_sum += train_reconstruction_loss_func(val_reconstruction.float(), val_targets.float())

                        val_mse_sum += mse_func(val_reconstruction.float(), val_targets.float())
                        val_mae_sum += mae_func(val_reconstruction.float(), val_targets.float())

                        scaling_transform = ScaleIntensity()
                        scaled_val_reconstruction = scaling_transform(val_reconstruction.float())
                        scaled_val_targets = scaling_transform(val_targets.float())
                        val_ms_ssim_scores.append(ms_ssim_func(scaled_val_reconstruction, scaled_val_targets))

                val_loss = val_loss_sum / batches_per_validation
                val_mse = val_mse_sum / batches_per_validation
                val_mae = val_mae_sum / batches_per_validation
                ms_ssim_scores = torch.cat(val_ms_ssim_scores, dim=0)

                tb_writer.add_scalars("Reconstruction Loss", {"Validation": val_loss.item()}, iteration)
                tb_writer.add_scalars("MSE", {"Validation": val_mse.item()}, iteration)
                tb_writer.add_scalars("MAE", {"Validation": val_mae.item()}, iteration)
                tb_writer.add_scalars("MS-SSIM", {"Validation": ms_ssim_scores.mean()}, iteration)

                acq_param_encoder.train()
                autoencoder.train()
                acq_param_encoder.train()

        if (epoch + 1) % 2 == 0 or epoch == 0:
            i = random.randrange(val_images.shape[0])
            tb_writer.add_image("Input", img_tensor=tb_image_scaler(val_inputs[i:i+1].transpose(0, 1)), global_step=epoch, dataformats="NCHW")
            tb_writer.add_image("Reconstruction", img_tensor=tb_image_scaler(val_reconstruction[i:i+1].transpose(0, 1)), global_step=epoch, dataformats="NCHW")

            q_map = torch.exp(val_log_q_map.detach()[i]).cpu().numpy()
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
        torch.save({
                "epoch": epoch,
                "autoencoder_model_state_dict": autoencoder.state_dict(),
                "discriminator_model_state_dict": discriminator.state_dict(),
                "optimizer_generator_state_dict": optimizer_generator.state_dict(),
                "optimizer_discriminator_state_dict": optimizer_discriminator.state_dict(),
                "scanner_gain_regressor_state_dict": scanner_gain_regressor.state_dict(),
                "acq_param_encoder_model_state_dict": acq_param_encoder.state_dict(),
            },
            out_checkpoint
        )
        print(f"Saved checkpoint to {out_checkpoint}")


    #######################################################
    # Evaluation
    #######################################################
    acq_param_encoder.eval()
    autoencoder.eval()
    scanner_gain_regressor.eval()
    val_loss_sum = 0

    mse_func = nn.MSELoss()
    val_mse_sum = 0
    mae_func = nn.L1Loss()
    val_mae_sum = 0
    ms_ssim_func = MultiScaleSSIMMetric(spatial_dims=2, data_range=1.0, kernel_size=4)
    val_ms_ssim_scores = []
    val_psnr_sum = 0

    for index, val_batch in enumerate(tqdm(val_loader, desc="Evaluating model")):
        with torch.no_grad():
            val_images = val_batch["images"].to(device)
            N, C, H, W = val_images.shape
            val_image_mask = val_batch["image_mask"].to(device)
            val_acq_params = val_batch["acq_params"].to(device)
            val_modality_id = val_batch["modality_id"].to(device)

            val_inputs = scale_transform(val_images)
            val_targets = val_images

            if args.adagn:
                val_acq_params_norm = val_batch["acq_params_norm"].to(device).float()
                adagn_input = acq_param_encoder(val_acq_params_norm.view(N * C, 3))
                adagn_scale = adagn_input[:, 0]
                adagn_shift = adagn_input[:, 1]
                for module in autoencoder.encoder.modules():
                    if isinstance(module, AdaptiveGroupNorm):
                        module.assign_adaptive_paramters(scale=adagn_scale, shift=adagn_shift)

            mask = val_image_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            z_mu, z_sigma = autoencoder.encode(val_inputs.view(N * C, 1, H, W))
            z_mu = z_mu.view(N, C, latent_channels, 20, 28)
            z_sigma = z_sigma.view(N, C, latent_channels, 20, 28)

            # Product of experts (PoE); equation 8 in https://arxiv.org/pdf/2202.03242.pdf
            # sigma^(-2) = sum(sigma^(-2) * mask)
            # mu = sum((mu * mask) / sigma^(-2)) * sigma^2
            z_sigma_squared = z_sigma * z_sigma
            z_sigma_squared_rcp = 1.0 / z_sigma_squared
            z_sigma_squared_fused_rcp = (z_sigma_squared_rcp * mask).sum(1)
            z_sigma_squared_fused = 1.0 / z_sigma_squared_fused_rcp
            z_sigma_fused = torch.sqrt(z_sigma_squared_fused)
            z_mu_fused = ((z_mu * mask) * z_sigma_squared_rcp).sum(1) * z_sigma_squared_fused

            z = autoencoder.sampling(z_mu_fused, z_sigma_fused)
            val_norm_log_q_map = autoencoder.decode(z)

            if args.use_exp_offset:
                val_log_q_map = denormalize_log_q_map(
                    norm_log_q_map=val_norm_log_q_map,
                    log_pd_mean=log_pd_prior_mean,
                    log_t1_mean=log_t1_prior_mean,
                    log_t2_mean=log_t2_prior_mean
                )
            else:
                val_log_q_map = val_norm_log_q_map

            # predict the scanner gain directly from the input images
            # pass the input images through the shared log scanner gain regression model
            # masked the predictions based on the available modalities
            # and average the masked predictions
            if args.use_regressor:
                log_scanner_gain = (scanner_gain_regressor(val_inputs.view(N * C, 1, H, W)).view(N, C) * val_image_mask).sum(1) / val_image_mask.sum(1)
                log_scanner_gain = log_scanner_gain.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            else:
                log_scanner_gain = default_log_scanner_gain

            val_reconstruction = mr_signal_model(val_log_q_map, val_acq_params, val_modality_id, log_scanner_gain)
            val_loss_sum += train_reconstruction_loss_func(val_reconstruction.float(), val_targets.float())

            val_mse_sum += mse_func(val_reconstruction.float(), val_targets.float())
            val_mae_sum += mae_func(val_reconstruction.float(), val_targets.float())

            psnr_func = PeakSignalNoiseRatio()
            psnr_func.update(val_reconstruction.float(), val_targets.float())
            val_psnr_sum += psnr_func.compute()

            scaling_transform = ScaleIntensity()
            scaled_val_reconstruction = scaling_transform(val_reconstruction.float())
            scaled_val_targets = scaling_transform(val_targets.float())
            val_ms_ssim_scores.append(ms_ssim_func(scaled_val_reconstruction, scaled_val_targets))

        # validation logging
        # get a random image from the validation batch
        for log_index in range(2):
            image_index = index * 2 + log_index
            i = random.randrange(val_images.shape[0])
            tb_writer.add_image("Eval Input", img_tensor=tb_image_scaler(val_inputs[i:i+1].transpose(0, 1)), global_step=image_index, dataformats="NCHW")
            tb_writer.add_image("Eval Reconstruction", img_tensor=tb_image_scaler(val_reconstruction[i:i+1].transpose(0, 1)), global_step=image_index, dataformats="NCHW")

            q_map = torch.exp(val_log_q_map.detach()[i]).cpu().numpy()
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

            tb_writer.add_figure('Eval Q-Maps', figure=fig, global_step=image_index)
            plt.close(fig)

    val_loss = val_loss_sum / len(val_loader)
    val_mse = val_mse_sum / len(val_loader)
    val_mae = val_mae_sum / len(val_loader)
    ms_ssim_scores = torch.cat(val_ms_ssim_scores, dim=0)
    val_psnr = val_psnr_sum / len(val_loader)

    metric_dict = {
        "MSE": val_mse.item(),
        "MAE": val_mae.item(),
        "MS-SSIM": ms_ssim_scores.mean().item(),
        "PSNR": val_psnr.item()
    }
    print("Hyperparameters:")
    print(json.dumps(hparams, indent=4))
    print("Results:")
    print(json.dumps(metric_dict, indent=4))

    tb_writer.add_hparams(
        hparams,
        metric_dict=metric_dict
    )
    tb_writer.close()


if __name__ == "__main__":
    main()