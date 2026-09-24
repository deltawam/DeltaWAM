"""Autoregressive DeltaWorld predictor rollout without pixel decoding."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from torch import Tensor

if TYPE_CHECKING:
    from models.deltaworld import DeltaWorld


@dataclass(frozen=True)
class ObservationEncoding:
    """Intermediate outputs of the original VFM -> DeltaTok path.

    Single-view tensors keep the legacy shapes. Multi-view tensors preserve an
    explicit view axis so the raw VFM / DeltaTok path stays camera-isolated
    until camera embeddings are deliberately injected into predictor tokens.
    """

    vfm_features: Tensor  # [B,T,N,D] or [B,V,T,N,D]
    initial_vfm_feature: Tensor  # [B,N,D] or [B,V,N,D], black-frame reference
    delta_tokens: Tensor  # [B,T,D] or [B,V,T,D]
    rope: tuple[Tensor, Tensor]


@dataclass(frozen=True)
class WorldRolloutOutput:
    """One coherent future trajectory plus the current-frame spatial anchor."""

    future_tokens: Tensor  # delta, decoded VFM patch maps, or an empty sequence
    future_mask: Tensor  # [B,L] or [B,L,V]
    current_vfm_features: Tensor  # [B,N,D] or [B,V,N,D]
    selected_branch: Tensor  # [B], oracle/dropout branch in training, zero in single-sample inference


class DeltaWorldTokenRollout(nn.Module):
    """Use only ``DeltaWorld.predictor`` to produce future delta tokens.

    DeltaWorld's pixel decoder is deliberately skipped: the predicted tokens
    are already the representation consumed by the action model. Best-of-many
    branches remain separate: training selects one oracle trajectory against
    tokenized future RGB, while inference returns one sampled branch.

    Multi-view mode uses scheme A for the predictor cache: tokens are interleaved
    by time, i.e. ``[t0_cam0, t0_cam1, t1_cam0, t1_cam1, ...]``. The BoM oracle
    scores each branch jointly over all views and all predicted future steps.
    """

    def __init__(
        self,
        delta_world: DeltaWorld,
        num_samples: int = 1,
        freeze_world: bool = True,
        max_cameras: int = 4,
        use_camera_embedding: bool = True,
        oracle_branch_probability: float = 1.0,
    ) -> None:
        super().__init__()
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        if max_cameras <= 0:
            raise ValueError(f"max_cameras must be positive, got {max_cameras}")
        if not 0.0 <= oracle_branch_probability <= 1.0:
            raise ValueError(
                "oracle_branch_probability must be in [0, 1], got "
                f"{oracle_branch_probability}"
            )
        if not delta_world.use_bom and num_samples != 1:
            raise ValueError("num_samples > 1 is useful only for a best-of-many DeltaWorld")
        self.delta_world = delta_world
        self.num_samples = int(num_samples)
        self.freeze_world = bool(freeze_world)
        self.use_camera_embedding = bool(use_camera_embedding)
        self.oracle_branch_probability = float(oracle_branch_probability)
        if hasattr(delta_world, "camera_embedding"):
            self.camera_embedding = delta_world.camera_embedding
        else:
            self.camera_embedding = nn.Embedding(max_cameras, self.output_dim)
            nn.init.normal_(self.camera_embedding.weight, std=float(delta_world.initializer_range))
        if self.freeze_world:
            self.delta_world.requires_grad_(False)
            self.camera_embedding.requires_grad_(False)
        self.train(False)

    @property
    def output_dim(self) -> int:
        return int(self.delta_world.backbone.hidden_size)

    def train(self, mode: bool = True):
        del mode
        return super().train(False)

    @staticmethod
    def _validate_raw_rgb(name: str, frames: Tensor) -> None:
        if frames.dtype != torch.uint8:
            raise TypeError(
                f"{name} must be raw uint8 RGB before VFM normalization; "
                f"got dtype={frames.dtype}. Do not pass normalized tensors."
            )
        channel_dim = 2 if frames.ndim == 5 else 3 if frames.ndim == 6 else None
        if channel_dim is None:
            raise ValueError(
                f"{name} must be [B,T,C,H,W] or [B,T,V,C,H,W], got {tuple(frames.shape)}"
            )
        if frames.shape[channel_dim] != 3:
            raise ValueError(f"{name} must have three RGB channels, got {frames.shape[channel_dim]}")

    @staticmethod
    def _validate_time_grid(observed: Tensor, future: Tensor) -> None:
        if observed.shape[1] > 1 and not bool((observed[:, 1:] > observed[:, :-1]).all()):
            raise ValueError("observed_timestamps must be strictly increasing")
        if future.shape[1] > 1 and not bool((future[:, 1:] > future[:, :-1]).all()):
            raise ValueError("future_timestamps must be strictly increasing")
        if not bool((future[:, :1] > observed[:, -1:]).all()):
            raise ValueError("every future timestamp must start after the current frame")

    def _camera_ids(self, num_views: int, device: torch.device) -> Tensor:
        if num_views > self.camera_embedding.num_embeddings:
            raise ValueError(
                f"num_views={num_views} exceeds camera embedding capacity "
                f"{self.camera_embedding.num_embeddings}"
            )
        return torch.arange(num_views, device=device)

    def _add_camera_embedding_to_tokens(self, tokens: Tensor) -> Tensor:
        if not self.use_camera_embedding:
            return tokens
        num_views = tokens.shape[1]
        camera = self.camera_embedding(self._camera_ids(num_views, tokens.device)).to(
            device=tokens.device, dtype=tokens.dtype
        )
        if tokens.ndim == 4:
            return tokens + camera[None, :, None, :]
        if tokens.ndim == 3:
            return tokens + camera[None, :, :]
        raise ValueError(f"unexpected token rank for camera embedding: {tokens.ndim}")

    @torch.no_grad()
    def encode_observations(self, observed_frames: Tensor) -> ObservationEncoding:
        """Run VFM -> DeltaTok, keeping multi-view black references isolated."""
        self._validate_raw_rgb("observed_frames", observed_frames)
        tokenizer = self.delta_world.tokenizer
        if observed_frames.ndim == 5:
            vfm_features = tokenizer.backbone(observed_frames)
            black_frames = torch.zeros_like(observed_frames[:, :1])
            initial_vfm_feature = tokenizer.backbone(black_frames)[:, 0]
            rope = tokenizer._rope(observed_frames)
            delta_tokens = tokenizer.tokenize_offline(initial_vfm_feature, vfm_features, rope)
            batch, num_frames = observed_frames.shape[:2]
            if vfm_features.ndim != 4 or vfm_features.shape[:2] != (batch, num_frames):
                raise RuntimeError("VFM backbone must return [B,T,N,D], got " f"{tuple(vfm_features.shape)}")
            expected = (batch, num_frames, 1, vfm_features.shape[-1])
            if delta_tokens.shape != expected:
                raise RuntimeError(f"DeltaTok.tokenize_offline must return {expected}, got {tuple(delta_tokens.shape)}")
            return ObservationEncoding(vfm_features, initial_vfm_feature, delta_tokens.squeeze(2), rope)

        batch, num_frames, num_views = observed_frames.shape[:3]
        flat = observed_frames.permute(0, 2, 1, 3, 4, 5).reshape(
            batch * num_views, num_frames, *observed_frames.shape[3:]
        )
        vfm_flat = tokenizer.backbone(flat)
        # Physical raw black images are created per flattened view before VFM normalization.
        black_flat = torch.zeros_like(flat[:, :1])
        initial_flat = tokenizer.backbone(black_flat)[:, 0]
        rope = tokenizer._rope(flat)
        delta_flat = tokenizer.tokenize_offline(initial_flat, vfm_flat, rope)
        if vfm_flat.ndim != 4 or vfm_flat.shape[:2] != (batch * num_views, num_frames):
            raise RuntimeError("VFM backbone must return [B*V,T,N,D] in multi-view mode, got " f"{tuple(vfm_flat.shape)}")
        expected = (batch * num_views, num_frames, 1, vfm_flat.shape[-1])
        if delta_flat.shape != expected:
            raise RuntimeError(f"DeltaTok.tokenize_offline must return {expected}, got {tuple(delta_flat.shape)}")
        num_patches, dim = vfm_flat.shape[2], vfm_flat.shape[3]
        return ObservationEncoding(
            vfm_features=vfm_flat.reshape(batch, num_views, num_frames, num_patches, dim),
            initial_vfm_feature=initial_flat.reshape(batch, num_views, num_patches, dim),
            delta_tokens=delta_flat.squeeze(2).reshape(batch, num_views, num_frames, dim),
            rope=rope,
        )

    @torch.no_grad()
    def observed_only(self, observed_frames: Tensor) -> WorldRolloutOutput:
        """Encode the current DINO feature map without running the predictor."""
        encoding = self.encode_observations(observed_frames)
        current_vfm = (
            encoding.vfm_features[:, -1]
            if observed_frames.ndim == 5
            else encoding.vfm_features[:, :, -1]
        )
        batch = observed_frames.shape[0]
        empty = current_vfm.new_empty((batch, 0, current_vfm.shape[-1]))
        empty_mask = torch.empty((batch, 0), dtype=torch.bool, device=current_vfm.device)
        selected = torch.zeros(batch, dtype=torch.long, device=current_vfm.device)
        return WorldRolloutOutput(empty, empty_mask, current_vfm, selected)

    @torch.no_grad()
    def decode_future_vfm_features(
        self,
        future_tokens: Tensor,
        current_vfm_features: Tensor,
        rope: tuple[Tensor, Tensor],
    ) -> Tensor:
        """Autoregressively reconstruct future DINO patch maps per camera."""
        tokenizer = self.delta_world.tokenizer
        if future_tokens.ndim == 3:
            reference = current_vfm_features
            decoded = []
            for step in range(future_tokens.shape[1]):
                reference = tokenizer.decode(
                    future_tokens[:, step : step + 1], reference, rope
                )
                decoded.append(reference)
            return torch.stack(decoded, dim=1)

        if future_tokens.ndim != 4 or current_vfm_features.ndim != 4:
            raise ValueError(
                "multi-view decode expects future [B,L,V,D] and current [B,V,N,D], "
                f"got {tuple(future_tokens.shape)} and {tuple(current_vfm_features.shape)}"
            )
        batch, future_len, num_views, dim = future_tokens.shape
        if current_vfm_features.shape[:2] != (batch, num_views):
            raise ValueError("future/current camera dimensions do not match")
        reference = current_vfm_features.reshape(
            batch * num_views, current_vfm_features.shape[2], dim
        )
        decoded = []
        for step in range(future_len):
            token = future_tokens[:, step].reshape(batch * num_views, 1, dim)
            reference = tokenizer.decode(token, reference, rope)
            decoded.append(
                reference.reshape(batch, num_views, reference.shape[1], reference.shape[2])
            )
        return torch.stack(decoded, dim=1)

    @torch.no_grad()
    def encode_future_targets(self, future_frames: Tensor, current_vfm_features: Tensor) -> Tensor:
        """Tokenize ground-truth future RGB relative to the current VFM frame."""
        self._validate_raw_rgb("future_frames", future_frames)
        tokenizer = self.delta_world.tokenizer
        if future_frames.ndim == 5:
            future_vfm = tokenizer.backbone(future_frames)
            rope = tokenizer._rope(future_frames)
            targets = tokenizer.tokenize_offline(current_vfm_features, future_vfm, rope)
            expected = (future_frames.shape[0], future_frames.shape[1], 1, future_vfm.shape[-1])
            if targets.shape != expected:
                raise RuntimeError(f"future DeltaTok targets must have shape {expected}, got {tuple(targets.shape)}")
            return targets.squeeze(2)

        batch, future_len, num_views = future_frames.shape[:3]
        if current_vfm_features.ndim != 4 or current_vfm_features.shape[:2] != (batch, num_views):
            raise ValueError("multi-view current_vfm_features must be [B,V,N,D], got " f"{tuple(current_vfm_features.shape)}")
        flat = future_frames.permute(0, 2, 1, 3, 4, 5).reshape(
            batch * num_views, future_len, *future_frames.shape[3:]
        )
        current_flat = current_vfm_features.reshape(batch * num_views, current_vfm_features.shape[2], current_vfm_features.shape[3])
        future_vfm = tokenizer.backbone(flat)
        rope = tokenizer._rope(flat)
        targets = tokenizer.tokenize_offline(current_flat, future_vfm, rope)
        expected = (batch * num_views, future_len, 1, future_vfm.shape[-1])
        if targets.shape != expected:
            raise RuntimeError(f"future DeltaTok targets must have shape {expected}, got {tuple(targets.shape)}")
        return targets.squeeze(2).reshape(batch, num_views, future_len, future_vfm.shape[-1]).permute(0, 2, 1, 3)

    def _make_query(self, batch_size: int, device: torch.device, dtype: torch.dtype, generator: Optional[torch.Generator]) -> Tensor:
        world = self.delta_world
        if world.use_bom:
            query = torch.randn(batch_size, 1, world.predictor_hidden_size, device=device, dtype=torch.float32, generator=generator) * float(world.initializer_range)
            return query.to(dtype=dtype)
        return world.q_embed.weight.to(device=device, dtype=dtype).expand(batch_size, 1, -1)

    def _make_multiview_query(self, batch_samples: int, num_views: int, device: torch.device, dtype: torch.dtype, generator: Optional[torch.Generator]) -> Tensor:
        query = self._make_query(batch_samples * num_views, device, dtype, generator).reshape(batch_samples, num_views, -1)
        if self.use_camera_embedding:
            camera = self.camera_embedding(self._camera_ids(num_views, device)).to(device=device, dtype=dtype)
            query = query + camera[None, :, :]
        return query

    def _select_training_branch(
        self,
        oracle_branch: Tensor,
        num_samples: int,
        generator: Optional[torch.Generator],
    ) -> Tensor:
        """Drop oracle labels to blind random branches for ActionDiT conditioning.

        ``oracle_branch_probability=1`` preserves the original BoM oracle
        behavior. Smaller values keep the K-rollout oracle computation intact
        but sometimes feed ActionDiT a randomly selected coherent branch, which
        narrows the train/inference gap caused by single random-query rollout.
        """
        if num_samples <= 1 or self.oracle_branch_probability >= 1.0:
            return oracle_branch
        batch = oracle_branch.shape[0]
        device = oracle_branch.device
        random_branch = torch.randint(
            num_samples, (batch,), device=device, generator=generator
        )
        if self.oracle_branch_probability <= 0.0:
            return random_branch
        keep_oracle = torch.rand(
            batch, device=device, generator=generator
        ) < self.oracle_branch_probability
        return torch.where(keep_oracle, oracle_branch, random_branch)

    def forward(
        self,
        observed_frames: Tensor,
        observed_timestamps: Tensor,
        future_timestamps: Tensor,
        future_frames: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
        instruction: Optional[object] = None,
    ) -> WorldRolloutOutput:
        if observed_frames.ndim not in (5, 6):
            raise ValueError("observed_frames must be [B,T_context,C,H,W] or [B,T_context,V,C,H,W], " f"got {tuple(observed_frames.shape)}")
        if observed_timestamps.ndim != 2 or future_timestamps.ndim != 2:
            raise ValueError("observed_timestamps and future_timestamps must be [B,T]")
        batch, context_len = observed_frames.shape[:2]
        future_len = future_timestamps.shape[1]
        is_multiview = observed_frames.ndim == 6
        if context_len == 0 or future_len == 0:
            raise ValueError("at least one context frame and future timestamp are required")
        if observed_timestamps.shape != (batch, context_len):
            raise ValueError("observed frame/timestamp shape mismatch")
        if future_timestamps.shape[0] != batch:
            raise ValueError("future timestamp batch mismatch")
        self._validate_raw_rgb("observed_frames", observed_frames)
        self._validate_time_grid(observed_timestamps, future_timestamps)
        if future_frames is not None:
            if future_frames.ndim != observed_frames.ndim or future_frames.shape[:2] != (batch, future_len):
                rank = "[B,L,V,C,H,W]" if is_multiview else "[B,L,C,H,W]"
                raise ValueError(f"future_frames must align with future_timestamps as {rank}, got {tuple(future_frames.shape)}")
            if future_frames.shape[2:] != observed_frames.shape[2:]:
                raise ValueError("observed and future RGB frame shapes must match")
            self._validate_raw_rgb("future_frames", future_frames)

        grad_context = torch.no_grad() if self.freeze_world else nullcontext()
        with grad_context:
            encoding = self.encode_observations(observed_frames)
            samples = self.num_samples if future_frames is not None else 1
            film_condition = (
                self.delta_world._encode_task_film_condition(instruction)
                if hasattr(self.delta_world, "_encode_task_film_condition")
                else None
            )
            film_condition_samples = (
                None
                if film_condition is None
                else film_condition.repeat_interleave(samples, dim=0)
            )

            if not is_multiview:
                observed_tokens = encoding.delta_tokens
                current_vfm = encoding.vfm_features[:, -1]
                kv = observed_tokens[:, None].expand(-1, samples, -1, -1).reshape(batch * samples, context_len, observed_tokens.shape[-1])
                key_times = observed_timestamps[:, None].expand(-1, samples, -1).reshape(batch * samples, context_len).to(kv.device)
                predicted_steps = []
                for step in range(future_len):
                    query = self._make_query(batch * samples, kv.device, kv.dtype, generator)
                    query_time = future_timestamps[:, step : step + 1][:, None].expand(-1, samples, -1).reshape(batch * samples, 1).to(kv.device)
                    predicted = self.delta_world.predictor(
                        query, kv, None, (query_time,), (key_times,), film_condition_samples
                    )
                    predicted_steps.append(predicted)
                    kv = torch.cat([kv, predicted], dim=1)
                    key_times = torch.cat([key_times, query_time], dim=1)
                candidates = torch.cat(predicted_steps, dim=1).reshape(batch, samples, future_len, -1)
                if future_frames is not None:
                    target = self.encode_future_targets(future_frames, current_vfm)
                    trajectory_loss = (candidates.float() - target[:, None].float()).square().mean(dim=(-2, -1))
                    selected_branch = self._select_training_branch(
                        trajectory_loss.argmin(dim=1), samples, generator
                    )
                else:
                    selected_branch = torch.zeros(batch, dtype=torch.long, device=kv.device)
                batch_index = torch.arange(batch, device=kv.device)
                future_tokens = candidates[batch_index, selected_branch]
                future_mask = torch.ones(future_tokens.shape[:2], dtype=torch.bool, device=future_tokens.device)
                return WorldRolloutOutput(future_tokens, future_mask, current_vfm, selected_branch)

            observed_tokens = self._add_camera_embedding_to_tokens(encoding.delta_tokens)
            current_vfm = encoding.vfm_features[:, :, -1]
            num_views = observed_tokens.shape[1]
            kv_base = observed_tokens.permute(0, 2, 1, 3).reshape(batch, context_len * num_views, -1)
            times_base = observed_timestamps[:, :, None].expand(batch, context_len, num_views).reshape(batch, context_len * num_views).to(kv_base.device)
            kv = kv_base[:, None].expand(-1, samples, -1, -1).reshape(batch * samples, context_len * num_views, kv_base.shape[-1])
            key_times = times_base[:, None].expand(-1, samples, -1).reshape(batch * samples, context_len * num_views)
            predicted_steps = []
            for step in range(future_len):
                query = self._make_multiview_query(batch * samples, num_views, kv.device, kv.dtype, generator)
                query_time = future_timestamps[:, step : step + 1][:, None, :].expand(-1, samples, num_views).reshape(batch * samples, num_views).to(kv.device)
                predicted = self.delta_world.predictor(
                    query, kv, None, (query_time,), (key_times,), film_condition_samples
                )
                predicted_view = predicted.reshape(batch * samples, num_views, -1)
                predicted_steps.append(predicted_view.reshape(batch, samples, num_views, -1))
                kv_predicted = self._add_camera_embedding_to_tokens(predicted_view)
                kv = torch.cat([kv, kv_predicted], dim=1)
                key_times = torch.cat([key_times, query_time], dim=1)
            candidates = torch.stack(predicted_steps, dim=2)  # [B,S,L,V,D]
            if future_frames is not None:
                target = self.encode_future_targets(future_frames, current_vfm)
                trajectory_loss = (candidates.float() - target[:, None].float()).square().mean(dim=(-3, -2, -1))
                selected_branch = self._select_training_branch(
                    trajectory_loss.argmin(dim=1), samples, generator
                )
            else:
                selected_branch = torch.zeros(batch, dtype=torch.long, device=kv.device)
            batch_index = torch.arange(batch, device=kv.device)
            future_tokens = candidates[batch_index, selected_branch]
            future_mask = torch.ones(future_tokens.shape[:3], dtype=torch.bool, device=future_tokens.device)
            return WorldRolloutOutput(future_tokens, future_mask, current_vfm, selected_branch)

    def forward_decoded_features(
        self,
        observed_frames: Tensor,
        observed_timestamps: Tensor,
        future_timestamps: Tensor,
        future_frames: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
        instruction: Optional[object] = None,
    ) -> WorldRolloutOutput:
        """Roll out DeltaWorld, then decode the selected branch to DINO maps."""
        output = self.forward(
            observed_frames,
            observed_timestamps,
            future_timestamps,
            future_frames=future_frames,
            generator=generator,
            instruction=instruction,
        )
        if observed_frames.ndim == 5:
            rope = self.delta_world.tokenizer._rope(observed_frames)
        else:
            batch, time, views = observed_frames.shape[:3]
            flat = observed_frames.permute(0, 2, 1, 3, 4, 5).reshape(
                batch * views, time, *observed_frames.shape[3:]
            )
            rope = self.delta_world.tokenizer._rope(flat)
        decoded = self.decode_future_vfm_features(
            output.future_tokens, output.current_vfm_features, rope
        )
        return WorldRolloutOutput(
            decoded,
            output.future_mask,
            output.current_vfm_features,
            output.selected_branch,
        )
