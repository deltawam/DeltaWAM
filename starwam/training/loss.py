"""Loss utilities for StarWAM training."""

import torch
from torch import Tensor
from starwam.modules.scheduler import FlowMatchScheduler


def flow_matching_loss(
    pred: Tensor,
    target: Tensor,
    timesteps: Tensor,
    scheduler: FlowMatchScheduler,
    is_pad_mask: Tensor | None = None,
    element_weight: Tensor | None = None,
) -> Tensor:
    """Compute weighted MSE loss with flow-matching importance sampling.

    Args:
        pred: model prediction [B, ...]
        target: training target [B, ...]
        timesteps: timestep indices [B]
        scheduler: FlowMatchScheduler for computing weights
        is_pad_mask: optional bool mask [B, ...] where True = padded (ignore)
        element_weight: optional non-negative weights broadcastable to ``pred``.
            Weighted reduction is normalized per sample, so changing temporal
            emphasis does not change that sample's total loss scale.
    Returns:
        scalar loss
    """
    # Per-element MSE
    mse = (pred - target) ** 2

    # Average over non-batch dims
    reduce_dims = list(range(1, mse.dim()))
    if is_pad_mask is not None or element_weight is not None:
        # Mask out padded positions. Broadcast the mask to the tensor layout.
        # Action tensors use [B, T, D], while video latents use [B, C, T, H, W].
        effective_weight = torch.ones_like(mse)
        if is_pad_mask is not None:
            valid_mask = ~is_pad_mask
            if valid_mask.dim() == 2 and mse.dim() == 5:
                valid_mask = valid_mask[:, None, :, None, None]
            while valid_mask.dim() < mse.dim():
                valid_mask = valid_mask.unsqueeze(-1)
            effective_weight = effective_weight * valid_mask.to(mse.dtype)
        if element_weight is not None:
            element_weight = element_weight.to(device=mse.device, dtype=mse.dtype)
            while element_weight.dim() < mse.dim():
                element_weight = element_weight.unsqueeze(-1)
            if torch.any(element_weight < 0):
                raise ValueError("element_weight must be non-negative")
            effective_weight = effective_weight * element_weight
        effective_weight = effective_weight.expand_as(mse)
        denom = effective_weight.sum(dim=reduce_dims).clamp(min=1.0)
        mse = (mse * effective_weight).sum(dim=reduce_dims) / denom
    else:
        mse = mse.mean(dim=reduce_dims)

    # Apply per-sample importance weight
    weight = scheduler.training_weight(timesteps).to(mse.device, mse.dtype)
    weighted_mse = mse * weight

    return weighted_mse.mean()
