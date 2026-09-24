import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig

from models.dinov3 import DINOv3_KAGGLE_MODEL, _resolve_backbone_name
from transformers.models.dinov3_vit.modeling_dinov3_vit import (
    DINOv3ViTLayerScale,
    DINOv3ViTMLP,
)

DINOV3_TEMPLATE = DINOv3_KAGGLE_MODEL


def _apply_rope_axis(x: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    half_size = x.shape[-1] // 2
    inv_freq = 1.0 / (10.0 ** torch.linspace(-2, 2, half_size, device=x.device))
    x_even, x_odd = x[..., :half_size], x[..., half_size : 2 * half_size]
    angle = (2 * torch.pi) * position * inv_freq.view(1, 1, 1, half_size)
    sin_vals, cos_vals = angle.sin(), angle.cos()
    return torch.cat(
        [x_even * cos_vals - x_odd * sin_vals, x_even * sin_vals + x_odd * cos_vals], -1
    )


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_q: list[torch.Tensor],
    pos_k: list[torch.Tensor],
    rope_axis_sizes: list[int],
    rope_unrotated_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_rotated, k_rotated, offset = [], [], 0
    for i, axis_size in enumerate(rope_axis_sizes):
        q_rotated.append(
            _apply_rope_axis(q[..., offset : offset + axis_size], pos_q[i])
        )
        k_rotated.append(
            _apply_rope_axis(k[..., offset : offset + axis_size], pos_k[i])
        )
        offset += axis_size
    q_rotated.append(q[..., offset : offset + rope_unrotated_size])
    k_rotated.append(k[..., offset : offset + rope_unrotated_size])
    return torch.cat(q_rotated, -1), torch.cat(k_rotated, -1)


def _reshape_heads(
    x: torch.Tensor, seq_len: int, batch_size: int, num_heads: int, head_size: int
) -> torch.Tensor:
    return x.reshape(batch_size, seq_len, num_heads, head_size).permute(0, 2, 1, 3)



class _FeatureWiseLinearModulation(nn.Module):
    """Apply feature-wise gamma/beta to token features.

    The generator emits gamma deltas around zero; the actual scale is
    ``1 + gamma_delta`` so zero-initialized generators preserve the original
    pre-norm Transformer exactly at initialization.
    """

    def forward(
        self,
        x: torch.Tensor,
        gamma_delta: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        while gamma_delta.ndim < x.ndim:
            gamma_delta = gamma_delta.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return x * (1.0 + gamma_delta.to(dtype=x.dtype)) + beta.to(dtype=x.dtype)


class _PredictorFiLMGenerator(nn.Module):
    """Map one task embedding to per-layer pre-norm FiLM/AdaLN parameters."""

    def __init__(
        self,
        condition_dim: int,
        hidden_size: int,
        num_layers: int,
        film_scale: float = 0.1,
        gated_residual: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.film_scale = float(film_scale)
        self.gated_residual = bool(gated_residual)
        self.params_per_layer = 6 if self.gated_residual else 4
        self.last_stats: dict[str, torch.Tensor] = {}
        self.net = nn.Sequential(
            nn.LayerNorm(condition_dim),
            nn.Linear(condition_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, num_layers * self.params_per_layer * hidden_size),
        )
        # Identity init:
        # - gamma_delta=0, beta=0 preserves both pre-norm sites.
        # - gated_residual uses residual multiplier ``1 + gate_delta`` below, so
        #   gate_delta=0 also preserves warm-started predictor block dynamics.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, condition: torch.Tensor | None) -> torch.Tensor | None:
        if condition is None:
            return None
        params = self.net(condition) * self.film_scale
        shaped = params.view(
            condition.shape[0],
            self.num_layers,
            self.params_per_layer,
            self.hidden_size,
        )
        with torch.no_grad():
            if self.gated_residual:
                gamma = shaped[:, :, (0, 3)]
                beta = shaped[:, :, (1, 4)]
                gate = shaped[:, :, (2, 5)]
            else:
                gamma = shaped[:, :, (0, 2)]
                beta = shaped[:, :, (1, 3)]
                gate = None
            self.last_stats = {
                "gamma_abs_mean": gamma.detach().float().abs().mean(),
                "gamma_abs_max": gamma.detach().float().abs().amax(),
                "beta_abs_mean": beta.detach().float().abs().mean(),
                "beta_abs_max": beta.detach().float().abs().amax(),
                "condition_abs_mean": condition.detach().float().abs().mean(),
                "condition_std": condition.detach().float().std(),
            }
            if gate is not None:
                self.last_stats.update(
                    {
                        "gate_delta_abs_mean": gate.detach().float().abs().mean(),
                        "gate_delta_abs_max": gate.detach().float().abs().amax(),
                    }
                )
        return shaped

class _QueryWisePredictorFiLMGenerator(_PredictorFiLMGenerator):
    """Map one aligned action embedding per query to per-layer AdaLN params."""

    def forward(self, condition: torch.Tensor | None) -> torch.Tensor | None:
        if condition is None:
            return None
        if condition.ndim != 3:
            raise ValueError(
                "query-wise action condition must be [B,Q,D], "
                f"got {tuple(condition.shape)}"
            )
        batch, queries = condition.shape[:2]
        params = self.net(condition) * self.film_scale
        # [B,Q,L,P,D] -> [B,L,Q,P,D], keeping query/action alignment explicit.
        shaped = params.view(
            batch,
            queries,
            self.num_layers,
            self.params_per_layer,
            self.hidden_size,
        ).permute(0, 2, 1, 3, 4)
        with torch.no_grad():
            if self.gated_residual:
                gamma = shaped[..., (0, 3), :]
                beta = shaped[..., (1, 4), :]
                gate = shaped[..., (2, 5), :]
            else:
                gamma = shaped[..., (0, 2), :]
                beta = shaped[..., (1, 3), :]
                gate = None
            self.last_stats = {
                "gamma_abs_mean": gamma.detach().float().abs().mean(),
                "gamma_abs_max": gamma.detach().float().abs().amax(),
                "beta_abs_mean": beta.detach().float().abs().mean(),
                "beta_abs_max": beta.detach().float().abs().amax(),
                "condition_abs_mean": condition.detach().float().abs().mean(),
                "condition_std": condition.detach().float().std(),
            }
            if gate is not None:
                self.last_stats.update(
                    {
                        "gate_delta_abs_mean": gate.detach().float().abs().mean(),
                        "gate_delta_abs_max": gate.detach().float().abs().amax(),
                    }
                )
        return shaped



class Predictor(nn.Module):

    class _TaskCrossAttn(nn.Module):
        """Cross-attend query tokens to always-visible language/task memory.

        Task memory is not a frame token: it does not receive temporal RoPE and
        does not participate in the visual-history causal mask.
        """

        def __init__(
            self,
            hidden_size: int,
            num_attention_heads: int,
        ) -> None:
            super().__init__()
            self.hidden_size = int(hidden_size)
            self.num_heads = int(num_attention_heads)
            self.head_size = self.hidden_size // self.num_heads
            self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
            self.kv_proj = nn.Linear(self.hidden_size, 2 * self.hidden_size)
            self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)

        def forward(
            self,
            q_input: torch.Tensor,
            task_memory: torch.Tensor,
            task_memory_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            batch_size, query_len = q_input.shape[:2]
            key_len = task_memory.shape[1]
            q = _reshape_heads(
                self.q_proj(q_input),
                query_len,
                batch_size,
                self.num_heads,
                self.head_size,
            )
            k, v = self.kv_proj(task_memory).chunk(2, dim=-1)
            k = _reshape_heads(k, key_len, batch_size, self.num_heads, self.head_size)
            v = _reshape_heads(v, key_len, batch_size, self.num_heads, self.head_size)

            attn_mask = None
            if task_memory_mask is not None:
                if task_memory_mask.shape != (batch_size, key_len):
                    raise ValueError(
                        "task_memory_mask must be [B,K] matching task_memory, "
                        f"got {tuple(task_memory_mask.shape)} for memory {tuple(task_memory.shape)}"
                    )
                visible = task_memory_mask.to(device=q.device, dtype=torch.bool)
                attn_mask = torch.zeros(
                    batch_size,
                    1,
                    1,
                    key_len,
                    device=q.device,
                    dtype=q.dtype,
                )
                attn_mask = attn_mask.masked_fill(
                    ~visible[:, None, None, :], torch.finfo(q.dtype).min
                )

            out = F.scaled_dot_product_attention(q, k, v, attn_mask)
            return self.out_proj(
                out.permute(0, 2, 1, 3).reshape(
                    batch_size, query_len, self.hidden_size
                )
            )

    class _CrossAttn(nn.Module):
        def __init__(
            self,
            backbone_hidden_size: int,
            rope_axis_sizes: list[int],
            rope_unrotated_size: int,
            predictor_hidden_size: int,
            num_attention_heads: int,
        ) -> None:
            super().__init__()
            self.rope_axis_sizes = rope_axis_sizes
            self.rope_unrotated_size = rope_unrotated_size
            self.predictor_hidden_size = predictor_hidden_size
            self.num_heads = num_attention_heads
            self.head_size = predictor_hidden_size // num_attention_heads
            self.q_proj = nn.Linear(predictor_hidden_size, predictor_hidden_size)
            self.kv_proj = nn.Linear(backbone_hidden_size, 2 * predictor_hidden_size)
            self.out_proj = nn.Linear(predictor_hidden_size, predictor_hidden_size)

        def forward(
            self,
            q_input: torch.Tensor,
            kv_input: torch.Tensor,
            attn_mask: torch.Tensor | None,
            pos_q: tuple[torch.Tensor, ...],
            pos_k: tuple[torch.Tensor, ...],
        ) -> torch.Tensor:
            batch_size, query_len, key_len = (
                q_input.shape[0],
                q_input.shape[1],
                kv_input.shape[1],
            )

            q = _reshape_heads(
                self.q_proj(q_input),
                query_len,
                batch_size,
                self.num_heads,
                self.head_size,
            )
            k, v = self.kv_proj(kv_input).chunk(2, dim=-1)
            k = _reshape_heads(k, key_len, batch_size, self.num_heads, self.head_size)
            v = _reshape_heads(v, key_len, batch_size, self.num_heads, self.head_size)

            def _expand_pos(pos_vals, seq_len):
                return (
                    pos_vals[:, None, :]
                    .unsqueeze(-1)
                    .expand(batch_size, self.num_heads, seq_len, 1)
                )

            q, k = _apply_rope(
                q,
                k,
                [_expand_pos(pos_vals, query_len) for pos_vals in pos_q],
                [_expand_pos(pos_vals, key_len) for pos_vals in pos_k],
                self.rope_axis_sizes,
                self.rope_unrotated_size,
            )
            out = F.scaled_dot_product_attention(q, k, v, attn_mask)
            return self.out_proj(
                out.permute(0, 2, 1, 3).reshape(
                    batch_size, query_len, self.predictor_hidden_size
                )
            )

    class _Block(nn.Module):
        def __init__(
            self,
            cfg: AutoConfig,
            backbone_hidden_size: int,
            rope_axis_sizes: list[int],
            rope_unrotated_size: int,
            predictor_hidden_size: int,
            num_attention_heads: int,
            use_task_cross_attn: bool = False,
            task_cross_attn_gate_init: float = 0.0,
            task_cross_attn_scale: float = 1.0,
        ) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(predictor_hidden_size)
            self.attention = Predictor._CrossAttn(
                backbone_hidden_size,
                rope_axis_sizes,
                rope_unrotated_size,
                predictor_hidden_size,
                num_attention_heads,
            )
            self.layer_scale1 = DINOv3ViTLayerScale(cfg)
            self.task_norm = (
                nn.LayerNorm(predictor_hidden_size) if use_task_cross_attn else None
            )
            self.task_attention = (
                Predictor._TaskCrossAttn(predictor_hidden_size, num_attention_heads)
                if use_task_cross_attn
                else None
            )
            self.task_cross_attn_scale = float(task_cross_attn_scale)
            self.task_cross_attn_gate = (
                nn.Parameter(torch.tensor(float(task_cross_attn_gate_init)))
                if use_task_cross_attn
                else None
            )
            self.norm2 = nn.LayerNorm(predictor_hidden_size)
            self.mlp = DINOv3ViTMLP(cfg)
            self.layer_scale2 = DINOv3ViTLayerScale(cfg)
            self.film = _FeatureWiseLinearModulation()

        @staticmethod
        def _apply_residual_gate(
            residual: torch.Tensor, gate_delta: torch.Tensor | None
        ) -> torch.Tensor:
            if gate_delta is None:
                return residual
            while gate_delta.ndim < residual.ndim:
                gate_delta = gate_delta.unsqueeze(1)
            return residual * (1.0 + gate_delta.to(dtype=residual.dtype))

        def forward(
            self,
            q: torch.Tensor,
            kv: torch.Tensor,
            mask: torch.Tensor | None,
            pos_q: tuple[torch.Tensor, ...],
            pos_k: tuple[torch.Tensor, ...],
            film_params: torch.Tensor | None = None,
            task_memory: torch.Tensor | None = None,
            task_memory_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            attn_input = self.norm1(q)
            gate_attn = None
            gate_mlp = None
            if film_params is not None:
                if film_params.shape[-2] == 6:
                    (
                        gamma_attn,
                        beta_attn,
                        gate_attn,
                        gamma_mlp,
                        beta_mlp,
                        gate_mlp,
                    ) = film_params.unbind(dim=-2)
                elif film_params.shape[-2] == 4:
                    gamma_attn, beta_attn, gamma_mlp, beta_mlp = film_params.unbind(dim=-2)
                else:
                    raise ValueError(
                        "predictor film_params must have 4 FiLM params or 6 gated AdaLN params "
                        f"per layer, got {film_params.shape[-2]}"
                    )
                attn_input = self.film(attn_input, gamma_attn, beta_attn)
            attn_residual = self.layer_scale1(
                self.attention(attn_input, kv, mask, pos_q, pos_k)
            )
            q = q + self._apply_residual_gate(attn_residual, gate_attn)

            if self.task_attention is not None and task_memory is not None:
                task_residual = self.task_attention(
                    self.task_norm(q), task_memory, task_memory_mask
                )
                q = q + (
                    self.task_cross_attn_gate.to(dtype=q.dtype)
                    * self.task_cross_attn_scale
                    * task_residual
                )

            mlp_input = self.norm2(q)
            if film_params is not None:
                mlp_input = self.film(mlp_input, gamma_mlp, beta_mlp)
            mlp_residual = self.layer_scale2(self.mlp(mlp_input))
            q = q + self._apply_residual_gate(mlp_residual, gate_mlp)
            return q

    def __init__(
        self,
        backbone_hidden_size: int,
        initializer_range: float,
        rope_axis_sizes: tuple[int, ...],
        rope_unrotated_size: int,
        layer_scale_init: float,
        predictor_hidden_size: int,
        predictor_num_hidden_layers: int,
        predictor_num_heads: int,
        mlp_ratio: int,
        film_condition_dim: int | None = None,
        film_scale: float = 0.1,
        film_gated_residual: bool = False,
        use_task_cross_attn: bool = False,
        task_cross_attn_gate_init: float = 0.0,
        task_cross_attn_scale: float = 1.0,
        action_film_condition_dim: int | None = None,
        action_film_scale: float = 1.0,
        action_film_gated_residual: bool = True,
    ) -> None:
        super().__init__()
        cfg = AutoConfig.from_pretrained(_resolve_backbone_name(DINOV3_TEMPLATE))
        cfg.hidden_size = predictor_hidden_size
        cfg.intermediate_size = predictor_hidden_size * mlp_ratio

        self.blocks = nn.ModuleList()
        for _ in range(predictor_num_hidden_layers):
            blk = self._Block(
                cfg,
                backbone_hidden_size,
                list(rope_axis_sizes),
                rope_unrotated_size,
                predictor_hidden_size,
                predictor_num_heads,
                use_task_cross_attn=use_task_cross_attn,
                task_cross_attn_gate_init=task_cross_attn_gate_init,
                task_cross_attn_scale=task_cross_attn_scale,
            )
            for module in blk.modules():
                if isinstance(module, nn.Linear):
                    nn.init.trunc_normal_(module.weight, std=initializer_range)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            blk.layer_scale1.lambda1.data.fill_(layer_scale_init)
            blk.layer_scale2.lambda1.data.fill_(layer_scale_init)
            self.blocks.append(blk)

        self.film_generator = (
            _PredictorFiLMGenerator(
                film_condition_dim,
                predictor_hidden_size,
                predictor_num_hidden_layers,
                film_scale,
                gated_residual=film_gated_residual,
            )
            if film_condition_dim is not None
            else None
        )
        self.action_film_generator = (
            _QueryWisePredictorFiLMGenerator(
                action_film_condition_dim,
                predictor_hidden_size,
                predictor_num_hidden_layers,
                action_film_scale,
                gated_residual=action_film_gated_residual,
            )
            if action_film_condition_dim is not None
            else None
        )
        self.norm = nn.LayerNorm(predictor_hidden_size)
        self.head = nn.Linear(predictor_hidden_size, backbone_hidden_size)
        nn.init.trunc_normal_(self.head.weight, std=initializer_range)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        mask: torch.Tensor | None,
        pos_q: tuple[torch.Tensor, ...],
        pos_k: tuple[torch.Tensor, ...],
        film_condition: torch.Tensor | None = None,
        task_memory: torch.Tensor | None = None,
        task_memory_mask: torch.Tensor | None = None,
        action_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        film = (
            self.film_generator(film_condition)
            if self.film_generator is not None
            else None
        )
        action_film = (
            self.action_film_generator(action_condition)
            if self.action_film_generator is not None
            else None
        )
        for layer_idx, blk in enumerate(self.blocks):
            layer_film = None if film is None else film[:, layer_idx]
            layer_action_film = (
                None if action_film is None else action_film[:, layer_idx]
            )
            if layer_action_film is not None:
                if layer_film is None:
                    layer_film = layer_action_film
                else:
                    if layer_film.shape[-2] != layer_action_film.shape[-2]:
                        raise ValueError(
                            "task and action FiLM parameter counts must match when combined"
                        )
                    layer_film = layer_film.unsqueeze(1) + layer_action_film
            q = blk(
                q,
                kv,
                mask,
                pos_q,
                pos_k,
                layer_film,
                task_memory=task_memory,
                task_memory_mask=task_memory_mask,
            )
        return self.head(self.norm(q))
