from typing import Tuple

import torch
import torch.nn as nn
from generative.networks.nets import AutoencoderKL

from utils import (
    ECHO_TIME_INDEX,
    REPETITION_TIME_INDEX,
    INVERSION_TIME_INDEX,
    T1W_MODALITY_ID,
    T2W_MODALITY_ID,
    FLAIR_MODALITY_ID
)


def denormalize_log_q_map(norm_log_q_map, log_pd_mean, log_t1_mean, log_t2_mean):
    """
    Transform the log_q_map ~N(0, 1) from to N(log_prior_mean, log_prior_variance) 
    """

    log_pd_std = 1.0
    log_t1_std = 1.0
    log_t2_std = 1.0

    log_q_map = torch.zeros_like(norm_log_q_map)
    log_q_map[:, 0] = norm_log_q_map[:, 0] * log_pd_std + log_pd_mean
    log_q_map[:, 1] = norm_log_q_map[:, 1] * log_t1_std + log_t1_mean
    log_q_map[:, 2] = norm_log_q_map[:, 2] * log_t2_std + log_t2_mean

    return log_q_map


class MRSignalModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, log_q_map, acq_params, modality_ids, log_scanner_gain):

        log_pd = log_q_map[:, 0:1, :, :]
        log_t1_t2 = torch.exp(torch.clamp(log_q_map[:, 1:3, :, :], -20, 30))
        t1 = log_t1_t2[:, 0:1, :, :]
        t2 = log_t1_t2[:, 1:2, :, :]
        
        echo_time = acq_params[:, :, ECHO_TIME_INDEX].unsqueeze(-1).unsqueeze(-1)
        repetition_time = acq_params[:, :, REPETITION_TIME_INDEX].unsqueeze(-1).unsqueeze(-1)
        inversion_time = acq_params[:, :, INVERSION_TIME_INDEX].unsqueeze(-1).unsqueeze(-1)

        t1w_mask = (modality_ids == T1W_MODALITY_ID).unsqueeze(-1).unsqueeze(-1)
        t2w_mask = (modality_ids == T2W_MODALITY_ID).unsqueeze(-1).unsqueeze(-1)
        flair_mask = (modality_ids == FLAIR_MODALITY_ID).unsqueeze(-1).unsqueeze(-1)

        t1w_mprage = (1 - (2 * torch.exp(-inversion_time / t1)) / (1 + torch.exp(-repetition_time / t1)))
        t2w_spin_echo = (1 - torch.exp(-repetition_time / t1))
        flair = (1 - 2 * torch.exp(-inversion_time / t1) + torch.exp(-repetition_time / t1))
        
        return (
            torch.exp(log_scanner_gain + log_pd) 
            * (t1w_mprage * t1w_mask + t2w_spin_echo * t2w_mask + flair * flair_mask) 
            * torch.exp(-echo_time / t2)
        )
    

class PhysicalVAE(nn.Module):
    """
    Multimodal product of experts wrapper around MONAI's AutoencoderKL
    for compatibility with the LatentDiffusionInferer
    """

    def __init__(self, autoencoder: AutoencoderKL) -> None:
        super().__init__()
        self.autoencoder = autoencoder

    def encode_stage_2_inputs(self, x: torch.Tensor) -> torch.Tensor:
        images = x
        N, C, H, W = images.shape

        z_mu, z_sigma = self.autoencoder.encode(images.view(N * C, 1, H, W))
        z_mu = z_mu.view(N, C, 6, 20, 28)
        z_sigma = z_sigma.view(N, C, 6, 20, 28)

        # Product of experts (PoE); equation 8 in https://arxiv.org/pdf/2202.03242.pdf
        # sigma^(-2) = sum(sigma^(-2) * mask)
        # mu = sum((mu * mask) / sigma^(-2)) * sigma^2
        z_sigma_squared = z_sigma * z_sigma
        z_sigma_squared_rcp = 1.0 / z_sigma_squared
        z_sigma_squared_fused_rcp = (z_sigma_squared_rcp).sum(1)
        z_sigma_squared_fused = 1.0 / z_sigma_squared_fused_rcp
        z_sigma_fused = torch.sqrt(z_sigma_squared_fused)
        z_mu_fused = ((z_mu) * z_sigma_squared_rcp).sum(1) * z_sigma_squared_fused

        z = self.autoencoder.sampling(z_mu_fused, z_sigma_fused)

        return z

    def decode_stage_2_outputs(self, z: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode(z)
    

class AdaptiveGroupNorm(nn.Module):

    def __init__(self, group_norm: nn.GroupNorm):
        super().__init__()
        self.group_norm = group_norm
        self.scale = None
        self.shift = None

    def assign_adaptive_paramters(self, scale, shift):
        self.scale = scale
        self.shift = shift

    def forward(self, x):
        x = self.group_norm(x)
        scale = self.scale.view(-1, *((1,) * (x.dim() - 1)))
        shift = self.shift.view(-1, *((1,) * (x.dim() - 1)))
        return x * scale + shift
    

def replace_groupnorm_with_adaptive_groupnorm(model):
    for name, module in model.named_children():
        if isinstance(module, nn.GroupNorm):
            ada_gn_layer = AdaptiveGroupNorm(group_norm=module)
            setattr(model, name, ada_gn_layer)
        else:
            # Recursively apply the function to child modules
            replace_groupnorm_with_adaptive_groupnorm(module)