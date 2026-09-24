"""End-to-end DeltaWorld feature-conditioned action flow model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

from starwam.modules.action_dit import ActionDiT
from starwam.modules.scheduler import FlowMatchScheduler
from starwam.training.flow import add_flow_noise
from starwam.training.loss import flow_matching_loss

from .action_history import ExecutedActionHistoryEncoder
from .action_kv import ActionKVCache, CachedActionDiT, DeltaTokenKVAdapter
from .world_tokens import DeltaWorldTokenRollout, WorldRolloutOutput


@dataclass(frozen=True)
class ConditionBundle:
    """Inspectable condition tokens plus their projected layer-wise cache."""

    future_tokens: Tensor
    future_mask: Tensor
    current_vfm_features: Tensor
    selected_branch: Tensor
    kv_cache: ActionKVCache
    text_context: Optional[Tensor] = None
    text_mask: Optional[Tensor] = None
    action_history_mask: Optional[Tensor] = None


class DeltaWorldFeatureActionModel(nn.Module):
    """Action flow matching conditioned on predicted DeltaWorld tokens.

    The class uses dependency injection rather than constructing checkpoints:
    callers load a ``DeltaWorld`` and an ``ActionDiT`` exactly as they already
    do elsewhere in the repository, then pass both objects here.
    """

    def __init__(
        self,
        world_rollout: DeltaWorldTokenRollout,
        action_dit: ActionDiT,
        action_scheduler: Optional[FlowMatchScheduler] = None,
        context_frames: int = 1,
        world_prediction_steps: int = 8,
        world_time_delta: float = 0.1,
        world_condition_mode: str = "delta_tokens",
        include_text: bool = True,
        action_history_steps: int = 0,
        action_history_dropout: float = 0.2,
        action_history_attention_mode: str = "separate_gated",
        action_history_gate_init: float = 0.01,
    ) -> None:
        super().__init__()
        if world_condition_mode not in {"delta_tokens", "anchor_only", "decoded_vfm"}:
            raise ValueError(
                "world_condition_mode must be delta_tokens, anchor_only, or decoded_vfm"
            )
        if context_frames <= 0 or world_prediction_steps <= 0:
            raise ValueError("context_frames and world_prediction_steps must be positive")
        if world_time_delta <= 0:
            raise ValueError("world_time_delta must be positive")
        self.action_history_steps = int(action_history_steps)
        if self.action_history_steps < 0:
            raise ValueError("action_history_steps must be non-negative")
        effective_history_mode = (
            str(action_history_attention_mode)
            if self.action_history_steps > 0 else "none"
        )
        self.world_rollout = world_rollout
        self.action_model = CachedActionDiT(
            action_dit,
            action_history_attention_mode=effective_history_mode,
            action_history_gate_init=action_history_gate_init,
        )
        self.kv_adapter = DeltaTokenKVAdapter(
            world_dim=world_rollout.output_dim,
            action_hidden_dim=action_dit.hidden_dim,
        )
        action_param = action_dit.action_encoder.weight
        self.kv_adapter.to(device=action_param.device, dtype=action_param.dtype)
        self.action_scheduler = action_scheduler or FlowMatchScheduler()
        self.context_frames = int(context_frames)
        self.world_prediction_steps = int(world_prediction_steps)
        self.world_condition_mode = str(world_condition_mode)
        self.world_time_delta = float(world_time_delta)
        self.include_text = bool(include_text)
        self.action_history_attention_mode = effective_history_mode
        self.action_history_encoder = (
            ExecutedActionHistoryEncoder(
                action_dim=action_dit.action_dim,
                hidden_dim=action_dit.hidden_dim,
                max_history_steps=self.action_history_steps,
                dropout_probability=action_history_dropout,
            )
            if self.action_history_steps > 0 else None
        )

    @property
    def action_dit(self) -> ActionDiT:
        return self.action_model.action_dit

    @staticmethod
    def _to_btchw(video: Tensor, layout: str) -> Tensor:
        if layout == "BTCHW":
            if video.ndim not in (5, 6):
                raise ValueError(f"BTCHW video must be 5D or 6D multi-view, got {tuple(video.shape)}")
            return video
        if layout == "BCTHW":
            if video.ndim != 5:
                raise ValueError(f"BCTHW video must be 5D, got {tuple(video.shape)}")
            return video.permute(0, 2, 1, 3, 4).contiguous()
        if layout == "BVCTHW":
            if video.ndim != 6:
                raise ValueError(f"BVCTHW video must be 6D, got {tuple(video.shape)}")
            return video.permute(0, 3, 1, 2, 4, 5).contiguous()
        raise ValueError(f"video_layout must be 'BTCHW', 'BCTHW', or 'BVCTHW', got {layout!r}")

    def _default_times(self, frames: Tensor) -> tuple[Tensor, Tensor]:
        batch, context_len = frames.shape[:2]
        observed = torch.arange(context_len, device=frames.device, dtype=torch.float32)
        observed = observed * self.world_time_delta
        future = torch.arange(
            1,
            self.world_prediction_steps + 1,
            device=frames.device,
            dtype=torch.float32,
        )
        future = observed[-1] + future * self.world_time_delta
        return observed.expand(batch, -1), future.expand(batch, -1)

    def encode_condition(
        self,
        observed_frames: Tensor,
        observed_timestamps: Optional[Tensor] = None,
        future_timestamps: Optional[Tensor] = None,
        text_context: Optional[Tensor] = None,
        text_mask: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
        future_frames: Optional[Tensor] = None,
        instruction: Optional[object] = None,
        action_history: Optional[Tensor] = None,
        action_history_mask: Optional[Tensor] = None,
    ) -> ConditionBundle:
        world_output = self.rollout_world(
            observed_frames,
            observed_timestamps=observed_timestamps,
            future_timestamps=future_timestamps,
            generator=generator,
            future_frames=future_frames,
            instruction=instruction,
        )
        return self.build_action_condition(
            world_output,
            text_context=text_context,
            text_mask=text_mask,
            action_history=action_history,
            action_history_mask=action_history_mask,
        )

    def rollout_world(
        self,
        observed_frames: Tensor,
        observed_timestamps: Optional[Tensor] = None,
        future_timestamps: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
        future_frames: Optional[Tensor] = None,
        instruction: Optional[object] = None,
    ) -> WorldRolloutOutput:
        """Produce world features without constructing ActionDiT K/V.

        This explicit profiling boundary includes VFM/DeltaTok encoding and
        the DeltaWorld rollout. ActionDiT condition projections and layer-wise
        K/V construction are deliberately excluded.
        """
        if observed_frames.shape[1] > self.context_frames:
            observed_frames = observed_frames[:, : self.context_frames]
            if observed_timestamps is not None:
                observed_timestamps = observed_timestamps[:, : self.context_frames]
        if (
            observed_timestamps is not None
            and observed_timestamps.shape[1] != observed_frames.shape[1]
        ):
            observed_timestamps = observed_timestamps[:, : observed_frames.shape[1]]
        if observed_timestamps is None or future_timestamps is None:
            default_observed, default_future = self._default_times(observed_frames)
            observed_timestamps = (
                default_observed if observed_timestamps is None else observed_timestamps
            )
            future_timestamps = default_future if future_timestamps is None else future_timestamps
        if self.world_condition_mode == "anchor_only":
            world_output = self.world_rollout.observed_only(observed_frames)
        else:
            rollout = (
                self.world_rollout.forward_decoded_features
                if self.world_condition_mode == "decoded_vfm"
                else self.world_rollout
            )
            world_output = rollout(
                observed_frames,
                observed_timestamps,
                future_timestamps,
                future_frames=future_frames,
                generator=generator,
                instruction=instruction,
            )
        return world_output

    def build_action_condition(
        self,
        world_output: WorldRolloutOutput,
        text_context: Optional[Tensor] = None,
        text_mask: Optional[Tensor] = None,
        action_history: Optional[Tensor] = None,
        action_history_mask: Optional[Tensor] = None,
    ) -> ConditionBundle:
        """Build ActionDiT-native condition tokens and layer-wise K/V cache."""
        action_history_hidden = None
        encoded_history_mask = None
        if self.action_history_encoder is not None:
            if action_history is None:
                batch = world_output.future_tokens.shape[0]
                device = world_output.future_tokens.device
                action_history = torch.zeros(
                    batch, self.action_history_steps, self.action_dit.action_dim,
                    device=device,
                )
                action_history_mask = torch.zeros(
                    batch, self.action_history_steps, dtype=torch.bool,
                    device=device,
                )
            action_history_hidden, encoded_history_mask = self.action_history_encoder(
                action_history, action_history_mask
            )
        elif action_history is not None or action_history_mask is not None:
            # Datasets may keep the schema stable by returning empty tensors
            # with shape [B,0,A] / [B,0] when action_history_steps=0. Treat
            # those as an explicit no-history condition, but still reject any
            # non-empty history because it would indicate a config mismatch.
            empty_history = action_history is None or (action_history.ndim == 3 and action_history.shape[1] == 0)
            empty_mask = action_history_mask is None or (action_history_mask.ndim == 2 and action_history_mask.shape[1] == 0)
            if not (empty_history and empty_mask):
                raise ValueError("action history was provided but action_history_steps=0")

        cache = self.kv_adapter(
            self.action_dit,
            world_output.future_tokens,
            world_output.current_vfm_features,
            world_output.future_mask,
            text_context=text_context if self.include_text else None,
            text_mask=text_mask if self.include_text else None,
            action_history_hidden=action_history_hidden,
            action_history_mask=encoded_history_mask,
            action_history_mode=self.action_history_attention_mode,
            action_history_gate=(
                self.action_model.history_gate_values()[0]
                if self.action_history_attention_mode == "unified" else None
            ),
            future_representation=self.world_condition_mode,
        )
        return ConditionBundle(
            future_tokens=world_output.future_tokens,
            future_mask=world_output.future_mask,
            current_vfm_features=world_output.current_vfm_features,
            selected_branch=world_output.selected_branch,
            kv_cache=cache,
            text_context=text_context if self.include_text else None,
            text_mask=text_mask if self.include_text else None,
            action_history_mask=encoded_history_mask,
        )

    def forward(
        self,
        noisy_action: Tensor,
        action_timestep: Tensor,
        condition: ConditionBundle | ActionKVCache,
    ) -> Tensor:
        if isinstance(condition, ConditionBundle):
            return self.action_model(
                noisy_action,
                action_timestep,
                condition.kv_cache,
            )
        return self.action_model(noisy_action, action_timestep, condition)

    def training_step(self, sample: dict[str, Any]) -> tuple[Tensor, dict[str, float]]:
        """StarWAM-style training step.

        Required keys are ``video`` and ``action``. ``video_layout`` defaults
        to StarWAM's ``BCTHW``. Optional keys: ``timestamps``,
        ``future_timestamps``, ``context``, ``context_mask``, and
        ``action_is_pad``.
        """
        action = sample["action"]
        video = self._to_btchw(sample["video"], sample.get("video_layout", "BCTHW"))
        observed_frames = video[:, : self.context_frames]
        condition = self.encode_condition(
            observed_frames,
            observed_timestamps=sample.get("timestamps"),
            future_timestamps=sample.get("future_timestamps"),
            text_context=sample.get("context"),
            text_mask=sample.get("context_mask"),
            future_frames=sample.get("future_video"),
            instruction=sample.get("instruction"),
            action_history=sample.get("action_history"),
            action_history_mask=sample.get("action_history_mask"),
        )
        noisy_action, target, timestep = add_flow_noise(self.action_scheduler, action)
        prediction = self(noisy_action, timestep, condition)
        loss = flow_matching_loss(
            prediction,
            target,
            timestep,
            self.action_scheduler,
            is_pad_mask=sample.get("action_is_pad"),
        )
        return loss, {
            "loss_action": float(loss.detach().item()),
            "loss_total": float(loss.detach().item()),
            "condition_tokens": float(condition.kv_cache.total_sequence_length),
        }

    @torch.no_grad()
    def sample_actions(
        self,
        observed_frames: Tensor,
        action_horizon: int,
        num_inference_steps: int = 10,
        observed_timestamps: Optional[Tensor] = None,
        future_timestamps: Optional[Tensor] = None,
        text_context: Optional[Tensor] = None,
        text_mask: Optional[Tensor] = None,
        seed: Optional[int] = None,
        instruction: Optional[object] = None,
        action_history: Optional[Tensor] = None,
        action_history_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict actions while reusing one condition K/V cache at every step."""
        if action_horizon <= 0 or num_inference_steps <= 0:
            raise ValueError("action_horizon and num_inference_steps must be positive")
        parameter = self.action_dit.action_encoder.weight
        # Preserve uint8/float RGB dtype so the VFM AutoImageProcessor applies the
        # same normalization in training and inference. Only move the device here.
        observed_frames = observed_frames.to(device=parameter.device)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=observed_frames.device).manual_seed(seed)
        condition = self.encode_condition(
            observed_frames,
            observed_timestamps,
            future_timestamps,
            text_context,
            text_mask,
            generator,
            instruction=instruction,
            action_history=action_history,
            action_history_mask=action_history_mask,
        )
        actions = torch.randn(
            observed_frames.shape[0],
            action_horizon,
            self.action_dit.action_dim,
            device=parameter.device,
            dtype=parameter.dtype,
            generator=generator,
        )
        timesteps, deltas = self.action_scheduler.build_inference_schedule(
            num_inference_steps, actions.device, actions.dtype
        )
        for timestep, delta in zip(timesteps, deltas):
            velocity = self(
                actions,
                timestep.expand(actions.shape[0]),
                condition,
            )
            actions = self.action_scheduler.step(velocity, delta, actions)
        return actions
