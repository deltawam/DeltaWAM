from itertools import chain

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.dinov3_vit.modeling_dinov3_vit import (
    DINOv3ViTLayer,
    DINOv3ViTRopePositionEmbedding,
)

from models.gated_attn import enable_gated_attn
from models.dinov3 import _resolve_backbone_name
from models.predictor import DINOV3_TEMPLATE
from models.qk_norm import enable_dinov3_qk_norm
from training.auto_prepare import ensure_kinetics_ckpt


def _with_qk_norm_aliases(sd: dict, module_sd: dict) -> dict:
    """Normalize historical DINOv3 QK-norm key names.

    Some released DeltaTok checkpoints store pre-RoPE QK norm parameters as
    ``attention.q_norm_prerope`` / ``attention.k_norm_prerope`` while the
    current patched DINOv3 attention modules expose them as
    ``attention.q_norm`` / ``attention.k_norm``.  They are the same per-head
    LayerNorm tensors in this code path, so alias the keys at load time instead
    of forcing every training config to toggle ``use_qk_norm`` by checkpoint
    vintage.
    """
    aliases = (
        (".attention.q_norm_prerope.", ".attention.q_norm."),
        (".attention.k_norm_prerope.", ".attention.k_norm."),
        (".attention.q_norm.", ".attention.q_norm_prerope."),
        (".attention.k_norm.", ".attention.k_norm_prerope."),
    )
    out = dict(sd)
    for src, dst in aliases:
        for key, tensor in sd.items():
            if src not in key:
                continue
            alias = key.replace(src, dst)
            if alias in out or alias not in module_sd:
                continue
            if getattr(tensor, "shape", None) != module_sd[alias].shape:
                continue
            out[alias] = tensor
    return out


def _load_sd(module: nn.Module, sd: dict):
    """Load same-name, same-shape tensors without importing training utilities."""
    sd = sd.get("state_dict", sd)
    module = getattr(module, "_orig_mod", module)
    module_sd = module.state_dict()
    sd = _with_qk_norm_aliases(sd, module_sd)

    used = {}
    unmapped = []
    for ckpt_key, tensor in sd.items():
        if ckpt_key not in module_sd:
            continue
        if tensor.shape != module_sd[ckpt_key].shape:
            unmapped.append(ckpt_key)
            continue
        used[ckpt_key] = tensor

    missing = [
        name for name, parameter in module.named_parameters(remove_duplicate=False)
        if parameter.requires_grad and name not in used
    ]
    if missing or unmapped:
        raise RuntimeError(f"missing={missing}, unmapped={unmapped}")
    return module.load_state_dict(used, strict=False)


class DeltaTok(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_hidden_layers=12,
        use_delta=True,
        layer_scale_init=1e-5,
        use_qk_norm=True,
        use_gated_attn=True,
        use_swiglu=True,
        use_rope_aug=True,
        ckpt_path: str | None = None,
    ):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.use_delta = use_delta
        self.backbone = backbone.requires_grad_(False).eval()

        cfg = AutoConfig.from_pretrained(_resolve_backbone_name(DINOV3_TEMPLATE))
        cfg._attn_implementation = "sdpa"

        cfg.hidden_size = self.backbone.hidden_size
        cfg.num_attention_heads = self.backbone.num_heads
        cfg.patch_size = self.backbone.patch_size

        if use_swiglu:
            cfg.use_gated_mlp = True
            cfg.intermediate_size = max(1, (2 * int(cfg.intermediate_size)) // 3)
            cfg.hidden_act = "silu"

        self.rope_embeddings = DINOv3ViTRopePositionEmbedding(cfg)
        if not use_rope_aug:
            self.rope_embeddings.eval()

        self.z_embed = nn.Embedding(1, self.backbone.hidden_size)
        nn.init.trunc_normal_(self.z_embed.weight, std=cfg.initializer_range)
        self.xy_embed = nn.Embedding(2, self.backbone.hidden_size)
        nn.init.trunc_normal_(self.xy_embed.weight, std=cfg.initializer_range)

        self.encoder_blocks = nn.ModuleList(
            [DINOv3ViTLayer(cfg) for _ in range(num_hidden_layers)]
        )
        self.decoder_blocks = nn.ModuleList(
            [DINOv3ViTLayer(cfg) for _ in range(num_hidden_layers)]
        )
        for blk in chain(self.encoder_blocks, self.decoder_blocks):
            for m in blk.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=cfg.initializer_range)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            blk.layer_scale1.lambda1.data.fill_(layer_scale_init)
            blk.layer_scale2.lambda1.data.fill_(layer_scale_init)
            if use_qk_norm:
                enable_dinov3_qk_norm(blk)
            if use_gated_attn:
                enable_gated_attn(blk)

        self.norm = nn.LayerNorm(self.backbone.hidden_size, cfg.layer_norm_eps)

        if ckpt_path:
            _load_sd(self, torch.load(ensure_kinetics_ckpt(ckpt_path)))

    def forward(
        self, frames: torch.Tensor, *_, horizon: int | None = None
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        if self.training:
            return self._forward_train(frames)
        return self._forward_eval(frames, horizon or frames.shape[1])

    def encode(
        self, x: torch.Tensor, y: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        z = self.z_embed.weight[None].repeat(y.shape[0], 1, 1)
        if self.use_delta:
            hidden = torch.cat(
                (
                    z,
                    x + self.xy_embed.weight[0],
                    y + self.xy_embed.weight[1],
                ),
                1,
            )
            rope = (
                torch.cat((rope[0], rope[0]), -2),
                torch.cat((rope[1], rope[1]), -2),
            )
        else:
            hidden = torch.cat((z, y + self.xy_embed.weight[1]), 1)
        for blk in self.encoder_blocks:
            hidden = blk(hidden, position_embeddings=rope)
        z = hidden[:, :1]
        return self.norm(z)

    def decode(
        self, z: torch.Tensor, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        if not self.use_delta:
            x = self.xy_embed.weight[0][None, None].expand_as(x)
        hidden = torch.cat((z, x), 1)
        for blk in self.decoder_blocks:
            hidden = blk(hidden, position_embeddings=rope)
        return hidden[:, 1:]

    def tokenize_offline(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        x = torch.cat((x.unsqueeze(1), y[:, :-1]), 1)
        z = self.encode(x.flatten(0, 1), y.flatten(0, 1), rope)
        return z.unflatten(0, (y.shape[0], -1))

    def _rope(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rope_embeddings(torch.zeros_like(frames[:1, 0], dtype=torch.float))

    def _forward_train(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rope = self._rope(frames)
        y = self.backbone(frames)
        z = self.encode(y[:, -2], y[:, -1], rope)
        y_hat = self.decode(z, y[:, -2], rope)
        return y_hat, y[:, -1]

    def rollout_init(self, frames: torch.Tensor, horizon: int) -> dict:
        y = self.backbone(frames)
        rope = self._rope(frames)
        if horizon == y.shape[1]:  # no context: start from black frame
            x = self.backbone(torch.zeros_like(frames[:, :1]))[:, 0]
        else:
            x = y[:, -horizon - 1]
        return {"y": y, "rope": rope, "x": x}

    def rollout_step(
        self, state: dict, tgt_frame_idx: int, _: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        y = state["y"][:, tgt_frame_idx]
        z = self.encode(state["x"], y, state["rope"])
        y_hat = self.decode(z, state["x"], state["rope"])
        state = {**state, "x": y_hat.detach()}
        return y_hat, y, state

    @torch.compiler.disable
    def _forward_eval(
        self, frames: torch.Tensor, horizon: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self.rollout_init(frames, horizon)
        preds, tgts = [], []
        for t in range(horizon):
            y_hat, y, state = self.rollout_step(
                state, frames.shape[1] - horizon + t, None
            )
            preds.append(y_hat)
            tgts.append(y)
        preds = torch.stack(preds, 1)
        tgts = torch.stack(tgts, 1)
        return preds, tgts, state["y"][:, : frames.shape[1] - horizon]
