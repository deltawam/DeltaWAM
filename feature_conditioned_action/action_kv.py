"""ActionDiT cross-attention backed by a reusable DeltaWorld K/V cache."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from starwam.modules.action_dit import ActionDiT
from starwam.modules.wan_block import sinusoidal_embedding_1d


@dataclass(frozen=True)
class LayerKV:
    """Projected keys and values for one ActionDiT cross-attention layer."""

    key: Tensor  # [B, H, L, Dh]
    value: Tensor  # [B, H, L, Dh]


@dataclass(frozen=True)
class ActionKVCache:
    """Layer-wise K/V tensors shared by all action denoising steps."""

    layers: tuple[LayerKV, ...]
    key_mask: Optional[Tensor]  # [B, L], True means the token is visible
    num_anchor_tokens: int
    num_future_tokens: int
    num_native_tokens: int = 0
    num_action_history_tokens: int = 0
    attention_bias: Optional[Tensor] = None  # unified-mode [B,L] logit bias
    history_layers: Optional[tuple[LayerKV, ...]] = None
    history_key_mask: Optional[Tensor] = None  # separate-mode [B,H]
    # Exact ActionDiT-space condition tokens before the per-layer K/V
    # projections. Keeping this reference makes representation probes faithful
    # to the training path (camera/type/temporal embeddings included).
    condition_hidden: Optional[Tensor] = None  # [B,L,D]

    @property
    def batch_size(self) -> int:
        return self.layers[0].key.shape[0]

    @property
    def sequence_length(self) -> int:
        """Primary memory length (or full concatenated length in unified mode)."""
        return self.layers[0].key.shape[2]

    @property
    def total_sequence_length(self) -> int:
        history_length = (
            self.history_layers[0].key.shape[2]
            if self.history_layers is not None else 0
        )
        return self.sequence_length + history_length


class DeltaTokenKVAdapter(nn.Module):
    """Map DeltaWorld tokens into ActionDiT hidden space and precompute K/V.

    Single-view tensors use the legacy shapes. Multi-view tensors preserve the
    view axis until this adapter adds action-side camera embeddings and flattens
    the condition sequence. Future temporal positions are tied by physical time:
    both camera tokens for ``t+1`` get temporal position 1, both camera tokens
    for ``t+2`` get temporal position 2, and so on.
    """

    def __init__(
        self,
        world_dim: int,
        action_hidden_dim: int,
        max_temporal_positions: int = 256,
        max_cameras: int = 4,
        use_camera_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.world_norm = nn.LayerNorm(world_dim)
        self.world_projection = nn.Linear(world_dim, action_hidden_dim)
        if max_temporal_positions < 2:
            raise ValueError("max_temporal_positions must be at least 2")
        if max_cameras <= 0:
            raise ValueError("max_cameras must be positive")
        self.delta_type_embedding = nn.Parameter(torch.zeros(action_hidden_dim))
        self.anchor_type_embedding = nn.Parameter(torch.zeros(action_hidden_dim))
        self.temporal_position_embedding = nn.Embedding(max_temporal_positions, action_hidden_dim)
        nn.init.normal_(self.temporal_position_embedding.weight, std=0.02)
        self.use_camera_embedding = bool(use_camera_embedding)
        self.camera_embedding = nn.Embedding(max_cameras, world_dim)
        nn.init.normal_(self.camera_embedding.weight, std=0.02)
        # Diagnostics opt in; normal training keeps no extra hidden reference.
        self.capture_condition_hidden = False

    def _camera_ids(self, num_views: int, device: torch.device) -> Tensor:
        if num_views > self.camera_embedding.num_embeddings:
            raise ValueError(
                f"num_views={num_views} exceeds camera embedding capacity "
                f"{self.camera_embedding.num_embeddings}"
            )
        return torch.arange(num_views, device=device)

    def _prepare_delta_tokens(
        self, delta_tokens: Tensor, delta_mask: Optional[Tensor]
    ) -> tuple[Tensor, Tensor, int, int]:
        """Return flattened future tokens, flattened mask, L, V."""
        if delta_tokens.ndim == 3:
            batch, future_len, _ = delta_tokens.shape
            flat_tokens = delta_tokens
            if delta_mask is None:
                flat_mask = torch.ones(batch, future_len, dtype=torch.bool, device=delta_tokens.device)
            else:
                if delta_mask.shape != (batch, future_len):
                    raise ValueError(f"delta_mask must be [B,L], got {tuple(delta_mask.shape)}")
                flat_mask = delta_mask.to(device=delta_tokens.device, dtype=torch.bool)
            return flat_tokens, flat_mask, future_len, 1
        if delta_tokens.ndim != 4:
            raise ValueError(
                f"delta_tokens must be [B,L,D] or [B,L,V,D], got {tuple(delta_tokens.shape)}"
            )
        batch, future_len, num_views, _ = delta_tokens.shape
        tokens = delta_tokens
        if self.use_camera_embedding:
            camera = self.camera_embedding(self._camera_ids(num_views, tokens.device)).to(
                device=tokens.device, dtype=tokens.dtype
            )
            tokens = tokens + camera[None, None, :, :]
        flat_tokens = tokens.reshape(batch, future_len * num_views, -1)
        if delta_mask is None:
            flat_mask = torch.ones(batch, future_len, num_views, dtype=torch.bool, device=tokens.device)
        else:
            delta_mask = delta_mask.to(device=tokens.device, dtype=torch.bool)
            if delta_mask.shape == (batch, future_len):
                flat_mask = delta_mask[:, :, None].expand(batch, future_len, num_views)
            elif delta_mask.shape == (batch, future_len, num_views):
                flat_mask = delta_mask
            else:
                raise ValueError(
                    f"delta_mask must be [B,L] or [B,L,V], got {tuple(delta_mask.shape)}"
                )
        return flat_tokens, flat_mask.reshape(batch, future_len * num_views), future_len, num_views

    def _prepare_decoded_features(
        self, features: Tensor, feature_mask: Optional[Tensor]
    ) -> tuple[Tensor, Tensor, int, int]:
        """Flatten decoded DINO maps in time -> view -> patch order."""
        if features.ndim == 4:
            batch, future_len, num_patches, _ = features.shape
            flat_tokens = features.reshape(batch, future_len * num_patches, -1)
            if feature_mask is None:
                mask = torch.ones(batch, future_len, dtype=torch.bool, device=features.device)
            else:
                if feature_mask.shape != (batch, future_len):
                    raise ValueError("single-view decoded feature mask must be [B,L]")
                mask = feature_mask.to(device=features.device, dtype=torch.bool)
            flat_mask = mask[:, :, None].expand(batch, future_len, num_patches)
            return flat_tokens, flat_mask.reshape(batch, -1), future_len, num_patches

        if features.ndim != 5:
            raise ValueError(
                "decoded features must be [B,L,N,D] or [B,L,V,N,D], got "
                f"{tuple(features.shape)}"
            )
        batch, future_len, num_views, num_patches, _ = features.shape
        tokens = features
        if self.use_camera_embedding:
            camera = self.camera_embedding(self._camera_ids(num_views, tokens.device)).to(
                device=tokens.device, dtype=tokens.dtype
            )
            tokens = tokens + camera[None, None, :, None, :]
        flat_tokens = tokens.reshape(batch, future_len * num_views * num_patches, -1)
        if feature_mask is None:
            mask = torch.ones(
                batch, future_len, num_views, dtype=torch.bool, device=tokens.device
            )
        else:
            feature_mask = feature_mask.to(device=tokens.device, dtype=torch.bool)
            if feature_mask.shape == (batch, future_len):
                mask = feature_mask[:, :, None].expand(batch, future_len, num_views)
            elif feature_mask.shape == (batch, future_len, num_views):
                mask = feature_mask
            else:
                raise ValueError("multi-view decoded feature mask must be [B,L] or [B,L,V]")
        flat_mask = mask[:, :, :, None].expand(
            batch, future_len, num_views, num_patches
        )
        return (
            flat_tokens,
            flat_mask.reshape(batch, -1),
            future_len,
            num_views * num_patches,
        )

    def _prepare_anchor_tokens(self, current_vfm_features: Tensor) -> tuple[Tensor, int]:
        if current_vfm_features.ndim == 3:
            return current_vfm_features, current_vfm_features.shape[1]
        if current_vfm_features.ndim != 4:
            raise ValueError(
                "current_vfm_features must be [B,N,D] or [B,V,N,D], got "
                f"{tuple(current_vfm_features.shape)}"
            )
        batch, num_views, num_patches, _ = current_vfm_features.shape
        tokens = current_vfm_features
        if self.use_camera_embedding:
            camera = self.camera_embedding(self._camera_ids(num_views, tokens.device)).to(
                device=tokens.device, dtype=tokens.dtype
            )
            tokens = tokens + camera[None, :, None, :]
        return tokens.reshape(batch, num_views * num_patches, -1), num_views * num_patches

    def forward(
        self,
        action_dit: ActionDiT,
        delta_tokens: Tensor,
        current_vfm_features: Tensor,
        delta_mask: Optional[Tensor] = None,
        text_context: Optional[Tensor] = None,
        text_mask: Optional[Tensor] = None,
        action_history_hidden: Optional[Tensor] = None,
        action_history_mask: Optional[Tensor] = None,
        action_history_mode: str = "separate_gated",
        action_history_gate: Optional[Tensor] = None,
        future_representation: str = "delta_tokens",
    ) -> ActionKVCache:
        if future_representation not in {"delta_tokens", "anchor_only", "decoded_vfm"}:
            raise ValueError(
                "future_representation must be delta_tokens, anchor_only, or decoded_vfm, "
                f"got {future_representation!r}"
            )
        if delta_tokens.shape[0] != current_vfm_features.shape[0]:
            raise ValueError("current VFM anchor and future delta token batch mismatch")
        if delta_tokens.shape[-1] != current_vfm_features.shape[-1]:
            raise ValueError("current VFM anchor and future delta token feature dim mismatch")
        if not action_dit.blocks:
            raise ValueError("ActionDiT must contain at least one block")

        projection_param = self.world_projection.weight
        delta_tokens = delta_tokens.to(device=projection_param.device, dtype=projection_param.dtype)
        current_vfm_features = current_vfm_features.to(device=projection_param.device, dtype=projection_param.dtype)

        if future_representation == "decoded_vfm":
            flat_future, delta_mask, future_len, temporal_repeats = (
                self._prepare_decoded_features(delta_tokens, delta_mask)
            )
        else:
            if future_representation == "anchor_only" and delta_tokens.shape[1] != 0:
                raise ValueError("anchor_only requires an empty future sequence")
            flat_future, delta_mask, future_len, temporal_repeats = (
                self._prepare_delta_tokens(delta_tokens, delta_mask)
            )
        flat_anchor, num_anchor_tokens = self._prepare_anchor_tokens(current_vfm_features)

        future_hidden = self.world_projection(self.world_norm(flat_future))
        if future_len + 1 > self.temporal_position_embedding.num_embeddings:
            raise ValueError(
                f"future length {future_len} exceeds temporal embedding capacity "
                f"{self.temporal_position_embedding.num_embeddings - 1}"
            )
        future_positions = torch.arange(1, future_len + 1, device=future_hidden.device)
        if temporal_repeats > 1:
            future_positions = future_positions[:, None].expand(
                future_len, temporal_repeats
            ).reshape(-1)
        future_hidden = (
            future_hidden
            + self.delta_type_embedding.to(future_hidden.dtype)
            + self.temporal_position_embedding(future_positions).to(future_hidden.dtype)[None]
        )

        anchor_hidden = self.world_projection(self.world_norm(flat_anchor))
        anchor_hidden = (
            anchor_hidden
            + self.anchor_type_embedding.to(anchor_hidden.dtype)
            + self.temporal_position_embedding.weight[0].to(anchor_hidden.dtype)
        )
        hidden_parts = [anchor_hidden, future_hidden]

        masks: list[Tensor] = [
            torch.ones(anchor_hidden.shape[:2], dtype=torch.bool, device=anchor_hidden.device),
            delta_mask.to(device=anchor_hidden.device, dtype=torch.bool),
        ]

        num_native_tokens = 0
        if text_context is not None:
            if text_context.ndim != 3:
                raise ValueError(
                    f"text_context must be [B, L, D], got {tuple(text_context.shape)}"
                )
            if text_context.shape[0] != delta_tokens.shape[0]:
                raise ValueError(
                    "world/text batch mismatch: "
                    f"{delta_tokens.shape[0]} vs {text_context.shape[0]}"
                )
            text_param = action_dit.text_embedding[0].weight
            text_context = text_context.to(device=text_param.device, dtype=text_param.dtype)
            text_hidden = action_dit.text_embedding(text_context)
            num_native_tokens = text_hidden.shape[1]
            if text_mask is None:
                text_mask = torch.ones(
                    text_hidden.shape[:2], dtype=torch.bool, device=text_hidden.device
                )
            else:
                if text_mask.shape != text_hidden.shape[:2]:
                    raise ValueError(
                        f"text_mask must be {text_hidden.shape[:2]}, got {tuple(text_mask.shape)}"
                    )
                text_mask = text_mask.to(device=text_hidden.device, dtype=torch.bool)
            if not bool(text_mask.any(dim=1).all().item()):
                raise ValueError("every sample needs at least one visible text/proprio token")
            hidden_parts.append(text_hidden)
            masks.append(text_mask)
        elif text_mask is not None:
            raise ValueError("text_mask was provided without text_context")

        if action_history_mode not in {"none", "unified", "separate_gated"}:
            raise ValueError(
                "action_history_mode must be 'none', 'unified', or 'separate_gated', "
                f"got {action_history_mode!r}"
            )
        num_action_history_tokens = 0
        separate_history_hidden = None
        separate_history_mask = None
        if action_history_hidden is not None:
            if action_history_mode == "none":
                raise ValueError("action history tokens require a non-'none' history mode")
            if action_history_hidden.ndim != 3:
                raise ValueError(
                    "action_history_hidden must be [B,H,D], got "
                    f"{tuple(action_history_hidden.shape)}"
                )
            expected = (delta_tokens.shape[0], action_dit.hidden_dim)
            if (action_history_hidden.shape[0], action_history_hidden.shape[2]) != expected:
                raise ValueError(
                    "action history batch/hidden mismatch: expected "
                    f"B,D={expected}, got {tuple(action_history_hidden.shape)}"
                )
            action_history_hidden = action_history_hidden.to(
                device=anchor_hidden.device, dtype=anchor_hidden.dtype
            )
            num_action_history_tokens = action_history_hidden.shape[1]
            if action_history_mask is None:
                action_history_mask = torch.ones(
                    action_history_hidden.shape[:2], dtype=torch.bool, device=anchor_hidden.device
                )
            else:
                if action_history_mask.shape != action_history_hidden.shape[:2]:
                    raise ValueError(
                        "action_history_mask must be "
                        f"{action_history_hidden.shape[:2]}, got {tuple(action_history_mask.shape)}"
                    )
                action_history_mask = action_history_mask.to(
                    device=anchor_hidden.device, dtype=torch.bool
                )
            if action_history_mode == "unified":
                hidden_parts.append(action_history_hidden)
                masks.append(action_history_mask)
            else:
                separate_history_hidden = action_history_hidden
                separate_history_mask = action_history_mask
        elif action_history_mask is not None:
            raise ValueError("action_history_mask was provided without action_history_hidden")

        # Primary memory never includes history in separate_gated mode.
        hidden = torch.cat(hidden_parts, dim=1)
        key_mask = torch.cat(masks, dim=1)
        if not bool(key_mask.any(dim=1).all().item()):
            raise ValueError("every sample needs at least one visible conditioning token")

        attention_bias = None
        if action_history_gate is not None:
            if action_history_mode != "unified":
                raise ValueError("action_history_gate logit bias is only valid in unified mode")
            if num_action_history_tokens <= 0:
                raise ValueError("action_history_gate requires action history tokens")
            gate = action_history_gate.to(device=hidden.device, dtype=hidden.dtype)
            if gate.numel() != 1:
                raise ValueError("action_history_gate must be scalar")
            # Invalid tokens get -inf. Valid history receives log(gate), so a
            # 0.1 cold-start gate contributes ~10% of an ordinary token's
            # softmax mass without changing K/V ordering.
            attention_bias = torch.zeros(
                key_mask.shape, device=hidden.device, dtype=hidden.dtype
            )
            attention_bias = attention_bias.masked_fill(~key_mask, float("-inf"))
            history_start = hidden.shape[1] - num_action_history_tokens
            history_valid = key_mask[:, history_start:]
            history_bias = torch.log(gate.clamp_min(torch.finfo(hidden.dtype).tiny))
            attention_bias[:, history_start:] = torch.where(
                history_valid, history_bias, attention_bias[:, history_start:]
            )

        layers = []
        history_layers = [] if separate_history_hidden is not None else None
        for block in action_dit.blocks:
            cross_attn = block.cross_attn
            key = cross_attn.norm_k(cross_attn.k(hidden))
            value = cross_attn.v(hidden)
            batch, length, _ = key.shape
            key = key.view(batch, length, cross_attn.num_heads, cross_attn.attn_head_dim).transpose(1, 2)
            value = value.view(batch, length, cross_attn.num_heads, cross_attn.attn_head_dim).transpose(1, 2)
            layers.append(LayerKV(key=key, value=value))
            if history_layers is not None:
                history_key = cross_attn.norm_k(cross_attn.k(separate_history_hidden))
                history_value = cross_attn.v(separate_history_hidden)
                history_length = history_key.shape[1]
                history_key = history_key.view(
                    batch, history_length, cross_attn.num_heads, cross_attn.attn_head_dim
                ).transpose(1, 2)
                history_value = history_value.view(
                    batch, history_length, cross_attn.num_heads, cross_attn.attn_head_dim
                ).transpose(1, 2)
                history_layers.append(LayerKV(key=history_key, value=history_value))
        return ActionKVCache(
            layers=tuple(layers),
            key_mask=key_mask,
            num_anchor_tokens=num_anchor_tokens,
            num_future_tokens=future_hidden.shape[1],
            num_native_tokens=num_native_tokens,
            num_action_history_tokens=num_action_history_tokens,
            attention_bias=attention_bias,
            history_layers=(tuple(history_layers) if history_layers is not None else None),
            history_key_mask=separate_history_mask,
            condition_hidden=(hidden if self.capture_condition_hidden else None),
        )


class CachedActionDiT(nn.Module):
    """Run an existing ActionDiT with preprojected cross-attention K/V."""

    def __init__(
        self,
        action_dit: ActionDiT,
        action_history_attention_mode: str = "none",
        action_history_gate_init: float = 0.01,
    ) -> None:
        super().__init__()
        if action_history_attention_mode not in {"none", "unified", "separate_gated"}:
            raise ValueError(
                "action_history_attention_mode must be 'none', 'unified', or "
                f"'separate_gated', got {action_history_attention_mode!r}"
            )
        if action_history_attention_mode == "unified" and not 0.0 < action_history_gate_init < 1.0:
            raise ValueError("unified history gate init must be strictly between 0 and 1")
        self.action_dit = action_dit
        self.action_history_attention_mode = action_history_attention_mode
        if action_history_attention_mode == "separate_gated":
            self.history_gates = nn.Parameter(
                torch.full((len(action_dit.blocks),), float(action_history_gate_init))
            )
        elif action_history_attention_mode == "unified":
            initial = torch.logit(torch.tensor(float(action_history_gate_init)))
            self.history_gates = nn.Parameter(initial.reshape(1))
        else:
            self.register_parameter("history_gates", None)

    def history_gate_values(self) -> Tensor:
        if self.history_gates is None:
            return torch.empty(0, device=self.action_dit.action_encoder.weight.device)
        if self.action_history_attention_mode == "unified":
            return self.history_gates.sigmoid()
        return self.history_gates.tanh()

    @staticmethod
    def _attention_probabilities(
        query: Tensor,
        key: Tensor,
        mask: Optional[Tensor],
    ) -> Tensor:
        """Materialize SDPA probabilities for diagnostics only.

        The normal model path continues to use fused SDPA. This helper is
        called only when ``return_cross_attention=True`` and mirrors PyTorch's
        default scale and bool/additive mask semantics.
        """
        logits = torch.matmul(
            query.float(), key.float().transpose(-2, -1)
        ) / math.sqrt(query.shape[-1])
        if mask is not None:
            mask = mask.to(device=logits.device)
            if mask.dtype == torch.bool:
                logits = logits.masked_fill(~mask, float("-inf"))
            else:
                logits = logits + mask.float()
        return torch.softmax(logits, dim=-1).detach()

    @staticmethod
    def _sdpa(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        num_heads: int,
        head_dim: int,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch, query_len, _ = query.shape
        key_len = key.shape[1]
        query = query.view(batch, query_len, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch, key_len, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch, key_len, num_heads, head_dim).transpose(1, 2)
        output = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        return output.transpose(1, 2).reshape(batch, query_len, num_heads * head_dim)

    def forward(
        self,
        action_tokens: Tensor,
        timestep: Tensor,
        kv_cache: ActionKVCache,
        text_context: Optional[Tensor] = None,
        text_mask: Optional[Tensor] = None,
        self_attention_mask: Optional[Tensor] = None,
        return_cross_attention: bool = False,
        cross_attention_layers: Optional[tuple[int, ...]] = None,
    ) -> Tensor | tuple[Tensor, dict[int, Tensor]]:
        dit = self.action_dit
        if kv_cache.num_native_tokens and text_context is not None:
            raise ValueError(
                "text/proprio K/V are already present in kv_cache; do not pass text_context again"
            )
        if len(kv_cache.layers) != len(dit.blocks):
            raise ValueError(
                f"cache has {len(kv_cache.layers)} layers, ActionDiT has {len(dit.blocks)}"
            )
        if kv_cache.history_layers is not None:
            if self.action_history_attention_mode != "separate_gated":
                raise ValueError("separate history cache requires separate_gated mode")
            if len(kv_cache.history_layers) != len(dit.blocks):
                raise ValueError(
                    f"history cache has {len(kv_cache.history_layers)} layers, "
                    f"ActionDiT has {len(dit.blocks)}"
                )
        if action_tokens.ndim != 3:
            raise ValueError(
                f"action_tokens must be [B, T, action_dim], got {tuple(action_tokens.shape)}"
            )
        if action_tokens.shape[0] != kv_cache.batch_size:
            raise ValueError(
                f"action/cache batch mismatch: {action_tokens.shape[0]} vs {kv_cache.batch_size}"
            )

        action_param = dit.action_encoder.weight
        action_tokens = action_tokens.to(
            device=action_param.device, dtype=action_param.dtype
        )
        batch, action_len, _ = action_tokens.shape
        tokens = dit.action_encoder(action_tokens)

        time_input = timestep.reshape(-1).to(device=tokens.device, dtype=tokens.dtype)
        if time_input.numel() == 1 and batch > 1:
            time_input = time_input.expand(batch)
        if time_input.numel() != batch:
            raise ValueError(f"timestep must have 1 or B={batch} elements")
        time_embedding = sinusoidal_embedding_1d(dit.freq_dim, time_input)
        time_embedding = dit.time_embedding(time_embedding).reshape(batch, dit.hidden_dim)
        time_modulation = dit.time_projection(time_embedding).reshape(
            batch, 6, dit.hidden_dim
        )
        freqs = dit._get_freqs(tokens.device)[:action_len]

        text_hidden = None
        text_cross_mask = None
        if text_context is not None:
            if text_context.ndim != 3:
                raise ValueError(
                    f"text_context must be [B, L, D], got {tuple(text_context.shape)}"
                )
            if text_context.shape[0] != batch:
                raise ValueError(
                    f"action/text batch mismatch: {batch} vs {text_context.shape[0]}"
                )
            text_param = dit.text_embedding[0].weight
            text_context = text_context.to(device=text_param.device, dtype=text_param.dtype)
            text_hidden = dit.text_embedding(text_context)
            if text_mask is None:
                text_mask = torch.ones(
                    text_hidden.shape[:2], dtype=torch.bool, device=text_hidden.device
                )
            else:
                expected_text_mask = text_hidden.shape[:2]
                if text_mask.shape != expected_text_mask:
                    raise ValueError(
                        f"text_mask must be {expected_text_mask}, got {tuple(text_mask.shape)}"
                    )
                text_mask = text_mask.to(device=text_hidden.device, dtype=torch.bool)
            if not bool(text_mask.any(dim=1).all().item()):
                raise ValueError("every sample needs at least one visible text/proprio token")
            text_cross_mask = text_mask[:, None, None, :].to(tokens.device)
        elif text_mask is not None:
            raise ValueError("text_mask was provided without text_context")

        cross_mask = None
        if kv_cache.key_mask is not None:
            expected_mask_shape = (batch, kv_cache.sequence_length)
            if kv_cache.key_mask.shape != expected_mask_shape:
                raise ValueError(
                    f"condition key mask must be {expected_mask_shape}, got "
                    f"{tuple(kv_cache.key_mask.shape)}"
                )
            world_end = kv_cache.num_anchor_tokens + kv_cache.num_future_tokens
            if not bool(kv_cache.key_mask[:, :world_end].all().item()):
                raise ValueError(
                    "spatial anchor and future condition tokens cannot be masked"
                )
            # Query-independent [B,1,1,K] mask broadcasts identical full
            # condition visibility to every action query. It cannot encode a
            # triangular/causal relation over the condition sequence.
            if kv_cache.attention_bias is not None:
                if kv_cache.attention_bias.shape != expected_mask_shape:
                    raise ValueError(
                        f"condition attention bias must be {expected_mask_shape}, got "
                        f"{tuple(kv_cache.attention_bias.shape)}"
                    )
                cross_mask = kv_cache.attention_bias[:, None, None, :].to(
                    device=tokens.device, dtype=tokens.dtype
                )
            else:
                cross_mask = kv_cache.key_mask[:, None, None, :].to(tokens.device)

        history_cross_mask = None
        if kv_cache.history_layers is not None:
            history_length = kv_cache.history_layers[0].key.shape[2]
            expected_history_mask = (batch, history_length)
            if kv_cache.history_key_mask is None or kv_cache.history_key_mask.shape != expected_history_mask:
                actual = None if kv_cache.history_key_mask is None else tuple(kv_cache.history_key_mask.shape)
                raise ValueError(
                    f"history key mask must be {expected_history_mask}, got {actual}"
                )
            history_cross_mask = kv_cache.history_key_mask[:, None, None, :].to(tokens.device)

        captured_cross_attention: dict[int, Tensor] = {}
        if cross_attention_layers is None:
            capture_layers = set(range(len(dit.blocks)))
        else:
            capture_layers = {int(index) for index in cross_attention_layers}
            invalid = sorted(
                index for index in capture_layers
                if not 0 <= index < len(dit.blocks)
            )
            if invalid:
                raise ValueError(
                    f"cross_attention_layers contains invalid indices {invalid}; "
                    f"model has {len(dit.blocks)} layers"
                )

        for layer_index, block in enumerate(dit.blocks):
            # Native ActionDiT self-attention.
            query, key, value = block.get_qkv(tokens, time_modulation, freqs)
            self_output = self._sdpa(
                query,
                key,
                value,
                block.num_heads,
                block.attn_head_dim,
                self_attention_mask,
            )
            _, _, gate_msa, shift_mlp, scale_mlp, gate_mlp = block._split_modulation(
                time_modulation
            )
            tokens = block.gate(
                tokens, gate_msa, block.self_attn.o(self_output)
            )

            # Cross-attention query is recomputed for each diffusion step;
            # DeltaWorld-derived keys and values come directly from the cache.
            layer_cache = kv_cache.layers[layer_index]
            cross = block.cross_attn
            cross_query = cross.norm_q(cross.q(block.norm3(tokens)))
            cross_query = cross_query.view(
                batch, action_len, cross.num_heads, cross.attn_head_dim
            ).transpose(1, 2)
            cross_key = layer_cache.key
            cross_value = layer_cache.value
            layer_cross_mask = cross_mask
            if text_hidden is not None:
                text_key = cross.norm_k(cross.k(text_hidden))
                text_value = cross.v(text_hidden)
                text_key = text_key.view(
                    batch, text_hidden.shape[1], cross.num_heads, cross.attn_head_dim
                ).transpose(1, 2)
                text_value = text_value.view(
                    batch, text_hidden.shape[1], cross.num_heads, cross.attn_head_dim
                ).transpose(1, 2)
                cross_key = torch.cat([cross_key, text_key], dim=2)
                cross_value = torch.cat([cross_value, text_value], dim=2)
                if layer_cross_mask is not None and layer_cross_mask.dtype != torch.bool:
                    text_additive_mask = torch.zeros(
                        text_cross_mask.shape, device=tokens.device, dtype=tokens.dtype
                    ).masked_fill(~text_cross_mask, float("-inf"))
                else:
                    text_additive_mask = text_cross_mask
                layer_cross_mask = (
                    text_additive_mask
                    if layer_cross_mask is None
                    else torch.cat([layer_cross_mask, text_additive_mask], dim=-1)
                )
            cross_output = F.scaled_dot_product_attention(
                cross_query,
                cross_key,
                cross_value,
                attn_mask=layer_cross_mask,
            )
            if return_cross_attention and layer_index in capture_layers:
                captured_cross_attention[layer_index] = self._attention_probabilities(
                    cross_query, cross_key, layer_cross_mask
                ).cpu()
            cross_output = cross_output.transpose(1, 2).reshape(
                batch, action_len, cross.attn_hidden_dim
            )
            tokens = tokens + cross.o(cross_output)

            if kv_cache.history_layers is not None:
                # History is an auxiliary correction memory. It has its own
                # softmax and cannot steal probability mass from primary memory.
                history_query = cross.norm_q(cross.q(block.norm3(tokens)))
                history_query = history_query.view(
                    batch, action_len, cross.num_heads, cross.attn_head_dim
                ).transpose(1, 2)
                history_layer = kv_cache.history_layers[layer_index]
                history_output = F.scaled_dot_product_attention(
                    history_query,
                    history_layer.key,
                    history_layer.value,
                    attn_mask=history_cross_mask,
                )
                # All-masked dropout samples must remain a finite no-op.
                history_output = torch.nan_to_num(history_output)
                history_output = history_output.transpose(1, 2).reshape(
                    batch, action_len, cross.attn_hidden_dim
                )
                history_residual = cross.o(history_output)
                # ``cross.o`` has a bias. An all-masked history sample must be
                # an exact no-op (including sample-level history dropout), so
                # suppress the complete projected residual rather than relying
                # only on SDPA returning zeros for an empty memory.
                history_visible = kv_cache.history_key_mask.any(dim=1).to(
                    device=tokens.device, dtype=tokens.dtype
                ).reshape(batch, 1, 1)
                history_gate = self.history_gates[layer_index].tanh().to(tokens.dtype)
                tokens = tokens + history_gate * history_visible * history_residual

            ffn_input = block.norm2(tokens) * (1 + scale_mlp) + shift_mlp
            tokens = block.gate(tokens, gate_mlp, block.ffn(ffn_input))

        output = dit.post_dit(tokens)
        if return_cross_attention:
            return output, captured_cross_attention
        return output
