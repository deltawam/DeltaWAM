"""Executed-action history tokens for chunk-boundary continuity."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ExecutedActionHistoryEncoder(nn.Module):
    """Encode strictly past, actually executed controls as condition tokens.

    Inputs are right-aligned: the last slot is relative time ``-1`` and the
    first slot is ``-max_history_steps``. Left padding remains masked and is
    never replaced by duplicated actions.
    """

    def __init__(
        self,
        action_dim: int,
        hidden_dim: int,
        max_history_steps: int,
        dropout_probability: float = 0.2,
    ) -> None:
        super().__init__()
        if action_dim <= 0 or hidden_dim <= 0 or max_history_steps <= 0:
            raise ValueError("action_dim, hidden_dim, and max_history_steps must be positive")
        if not 0.0 <= dropout_probability <= 1.0:
            raise ValueError("dropout_probability must be in [0, 1]")
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_history_steps = int(max_history_steps)
        self.dropout_probability = float(dropout_probability)
        self.input_norm = nn.LayerNorm(action_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.position_embedding = nn.Embedding(max_history_steps, hidden_dim)
        self.type_embedding = nn.Parameter(torch.zeros(hidden_dim))
        self.output_norm = nn.LayerNorm(hidden_dim)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.normal_(self.type_embedding, std=0.02)

    def forward(
        self, actions: Tensor, valid_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        if actions.ndim != 3:
            raise ValueError(f"action_history must be [B,H,A], got {tuple(actions.shape)}")
        batch, length, action_dim = actions.shape
        if action_dim != self.action_dim:
            raise ValueError(
                f"expected action history dim {self.action_dim}, got {action_dim}"
            )
        if length <= 0 or length > self.max_history_steps:
            raise ValueError(
                f"action history length must be in [1,{self.max_history_steps}], got {length}"
            )
        parameter = self.input_projection[0].weight
        actions = actions.to(device=parameter.device, dtype=parameter.dtype)
        if valid_mask is None:
            valid_mask = torch.ones(batch, length, dtype=torch.bool, device=actions.device)
        else:
            if valid_mask.shape != (batch, length):
                raise ValueError(
                    f"action_history_mask must be {(batch, length)}, got {tuple(valid_mask.shape)}"
                )
            valid_mask = valid_mask.to(device=actions.device, dtype=torch.bool)

        position_ids = torch.arange(
            self.max_history_steps - length, self.max_history_steps, device=actions.device
        )
        hidden = self.input_projection(self.input_norm(actions))
        hidden = hidden + self.position_embedding(position_ids).to(hidden.dtype)[None]
        hidden = hidden + self.type_embedding.to(hidden.dtype)
        hidden = self.output_norm(hidden)

        # Drop the complete history for a sample, not independent action tokens.
        if self.training and self.dropout_probability > 0.0:
            keep = torch.rand(batch, 1, device=actions.device) >= self.dropout_probability
            valid_mask = valid_mask & keep
        return hidden, valid_mask
