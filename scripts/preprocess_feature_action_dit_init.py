#!/usr/bin/env python3
"""Preprocess Wan2.2 video-DiT weights for the DeltaWorld-conditioned ActionDiT.

This is the current-project analogue of StarWAM ActionDiT init:
video-DiT backbone/text/time/block weights are copied or linearly interpolated
into the action-side ActionDiT, while action-specific input/output modules stay
random (or the output head can be zeroed/payload-initialized at load time).

Examples:
    # From a local Wan2.2-TI2V-5B model directory:
    python scripts/preprocess_feature_action_dit_init.py \
      --config configs/deltaworld_feature_action_libero_spatial.yaml \
      --pretrained-model-id /path/to/Wan2.2-TI2V-5B \
      --output ckpt/action_dit_init/deltaworld_actiondit_wan22_linear_interp_768.pt

    # From a pre-extracted Wan2.2 DiT state_dict:
    python scripts/preprocess_feature_action_dit_init.py \
      --config configs/deltaworld_feature_action_libero_spatial.yaml \
      --video-state-dict /path/to/wan22_dit_state_dict.pt \
      --output ckpt/action_dit_init/deltaworld_actiondit_wan22_linear_interp_768.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT,):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from starwam.modules.action_dit import ActionDiT


def parse_dtype(name: str) -> torch.dtype:
    value = str(name).strip().lower()
    if value == "float32":
        return torch.float32
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype={name!r}")


def interpolate_last_dim(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    if tensor.shape[-1] == new_size:
        return tensor
    flat = tensor.reshape(-1, 1, tensor.shape[-1]).to(torch.float32)
    flat = F.interpolate(flat, size=new_size, mode="linear", align_corners=True)
    return flat.reshape(*tensor.shape[:-1], new_size)


def resize_to_shape(src: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    if tuple(src.shape) == target_shape:
        return src
    out = src.to(torch.float32)
    while out.ndim < len(target_shape):
        out = out.unsqueeze(0)
    while out.ndim > len(target_shape):
        if out.shape[0] != 1:
            raise ValueError(f"cannot reduce rank: src={tuple(src.shape)}, target={target_shape}")
        out = out.squeeze(0)
    for dim, new_size in enumerate(target_shape):
        if out.shape[dim] == new_size:
            continue
        perm = [i for i in range(out.ndim) if i != dim] + [dim]
        inv = [0] * out.ndim
        for i, p in enumerate(perm):
            inv[p] = i
        moved = out.permute(*perm).contiguous()
        prefix = moved.shape[:-1]
        moved = interpolate_last_dim(moved, new_size).reshape(*prefix, new_size)
        out = moved.permute(*inv).contiguous()
    if tuple(out.shape) != target_shape:
        raise ValueError(
            f"resize produced wrong shape: src={tuple(src.shape)}, target={target_shape}, got={tuple(out.shape)}"
        )
    return out.to(dtype=src.dtype)


def convert_tensor(src: torch.Tensor, target: torch.Tensor, alpha_scaling: bool) -> tuple[torch.Tensor, bool]:
    target_shape = tuple(target.shape)
    if tuple(src.shape) == target_shape:
        value = src
        resized = False
    else:
        value = resize_to_shape(src, target_shape)
        if alpha_scaling and src.ndim >= 2 and src.shape[-1] != target_shape[-1]:
            value = value.to(torch.float32) * (float(src.shape[-1]) / float(target_shape[-1])) ** 0.5
        resized = True
    return value.detach().to(dtype=target.dtype, device="cpu").contiguous(), resized


def load_video_state_from_file(path: str) -> dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and isinstance(raw.get("state_dict"), dict):
        return raw["state_dict"]
    if isinstance(raw, dict) and isinstance(raw.get("model_state_dict"), dict):
        return raw["model_state_dict"]
    if isinstance(raw, dict):
        return raw
    raise ValueError(f"unsupported video state format: {type(raw)}")


def load_wan22_video_state(model_dir: str, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    from starwam.backbone.wan22 import Wan22Dit
    from starwam.utils.checkpoint import infer_backbone_info

    info = infer_backbone_info(model_dir)
    dit = Wan22Dit(info)
    try:
        dit.load_pretrained(model_dir, dtype=dtype)
        return dit.state_dict()
    except FileNotFoundError:
        # Some local Wan2.2 mirrors contain shards named
        # diffusion_pytorch_model-00001-of-00003.safetensors but omit the
        # diffusion_pytorch_model.safetensors.index.json file expected by
        # Wan22Dit.load_pretrained().  Load those shards directly.
        from safetensors.torch import load_file

        model_path = Path(model_dir)
        shards = sorted(model_path.glob("diffusion_pytorch_model-*-of-*.safetensors"))
        if not shards:
            raise
        state: dict[str, torch.Tensor] = {}
        for shard in shards:
            print(f"[INFO] Loading Wan2.2 shard {shard}")
            state.update(load_file(str(shard), device="cpu"))
        if dtype is not None:
            state = {key: value.to(dtype) for key, value in state.items()}
        result = dit.load_state_dict(state, strict=False)
        print(
            f"[INFO] Loaded sharded Wan2.2 DiT directly "
            f"(missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)})"
        )
        return dit.state_dict()


def build_action_dit_from_current_yaml(config_path: Path) -> tuple[ActionDiT, dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    args = dict(config["model"]["init_args"])
    hidden = int(args.get("action_hidden_dim", 768))
    action_dim = int(args.get("action_dim", 7))
    num_layers = int(args.get("action_num_layers", 6))
    num_heads = int(args.get("action_num_heads", 12))
    head_dim = int(args.get("action_head_dim", hidden // num_heads))
    ffn_ratio = int(args.get("action_ffn_ratio", 4))
    text_dim = int(args.get("action_text_dim", 256))
    freq_dim = int(args.get("action_freq_dim", 256))
    eps = float(args.get("action_eps", 1e-6))
    horizon = int(args.get("action_horizon", 16))
    action_dit = ActionDiT(
        hidden_dim=hidden,
        action_dim=action_dim,
        ffn_dim=hidden * ffn_ratio,
        text_dim=text_dim,
        freq_dim=freq_dim,
        eps=eps,
        num_heads=num_heads,
        attn_head_dim=head_dim,
        num_layers=num_layers,
        max_seq_len=max(256, horizon * 2),
        use_gradient_checkpointing=False,
    )
    meta = {
        "hidden_dim": hidden,
        "ffn_dim": hidden * ffn_ratio,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "attn_head_dim": head_dim,
        "text_dim": text_dim,
        "freq_dim": freq_dim,
        "eps": eps,
        "action_dim": action_dim,
        "action_horizon": horizon,
        "source_config": str(config_path),
    }
    return action_dit, meta


def summary_path_for(output: Path) -> Path:
    return output.parent / f"{output.name}.json"


def maybe_write_json_summary(output: Path, payload: dict[str, Any], stats: dict[str, Any]) -> None:
    summary = {
        "output": str(output),
        "policy": payload.get("policy", {}),
        "meta": payload.get("meta", {}),
        "stats": stats,
    }
    summary_path_for(output).write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-state-dict", default=None)
    parser.add_argument("--pretrained-model-id", default=None, help="Local Wan2.2 model directory")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--head-init", default="random", choices=["random", "zero", "payload"])
    parser.add_argument("--no-alpha-scaling", dest="alpha_scaling", action="store_false")
    parser.add_argument("--strict-missing", action="store_true", help="Fail if any ActionDiT backbone key is absent in Wan state")
    parser.set_defaults(alpha_scaling=True)
    args = parser.parse_args()

    if bool(args.video_state_dict) == bool(args.pretrained_model_id):
        raise ValueError("pass exactly one of --video-state-dict or --pretrained-model-id")

    dtype = parse_dtype(args.dtype)
    action_dit, meta = build_action_dit_from_current_yaml(args.config)
    action_state = action_dit.state_dict()
    backbone_keys = sorted(action_dit.backbone_key_set(action_state.keys()))

    if args.video_state_dict:
        print(f"[INFO] Loading Wan2.2 video state_dict from {args.video_state_dict}")
        video_state = load_video_state_from_file(args.video_state_dict)
        source = "video_state_dict"
    else:
        print(f"[INFO] Loading Wan2.2 video DiT from {args.pretrained_model_id}")
        video_state = load_wan22_video_state(args.pretrained_model_id, dtype=dtype)
        source = "wan22_model_dir"

    mapped: dict[str, torch.Tensor] = {}
    missing: list[str] = []
    copied = interpolated = 0
    shape_changes: dict[str, dict[str, list[int]]] = {}
    for key in backbone_keys:
        if key not in video_state:
            missing.append(key)
            continue
        value, resized = convert_tensor(video_state[key], action_state[key], args.alpha_scaling)
        mapped[key] = value
        if resized:
            interpolated += 1
            shape_changes[key] = {
                "source": list(video_state[key].shape),
                "target": list(action_state[key].shape),
            }
        else:
            copied += 1

    if missing and args.strict_missing:
        raise ValueError(f"{len(missing)} ActionDiT backbone keys missing in Wan state: {missing[:20]}")
    if not mapped:
        raise ValueError("no ActionDiT weights were mapped from the source state")

    payload: dict[str, Any] = {
        "policy": {
            "source": source,
            "source_backbone": "wan22",
            "alpha_scaling": bool(args.alpha_scaling),
            "interpolation": "sequential_1d_linear_align_corners_true",
            "action_backbone_skip_prefixes": list(action_dit.ACTION_BACKBONE_SKIP_PREFIXES),
            "head_init": args.head_init,
            "missing_allowed": not bool(args.strict_missing),
        },
        "backbone_state_dict": mapped,
        "meta": {
            **meta,
            "mapped_keys": len(mapped),
            "missing_keys": len(missing),
        },
    }
    if args.head_init == "zero":
        payload["head_state_dict"] = {
            "head.weight": torch.zeros_like(action_dit.head.weight, device="cpu"),
            "head.bias": torch.zeros_like(action_dit.head.bias, device="cpu"),
        }
    elif args.head_init == "payload":
        payload["head_state_dict"] = {
            "head.weight": action_dit.head.weight.detach().cpu().contiguous(),
            "head.bias": action_dit.head.bias.detach().cpu().contiguous(),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    stats = {
        "total_action_backbone_keys": len(backbone_keys),
        "mapped": len(mapped),
        "copied": copied,
        "interpolated": interpolated,
        "missing": missing,
        "shape_changes": shape_changes,
    }
    maybe_write_json_summary(args.output, payload, stats)
    print(
        f"[INFO] Saved ActionDiT init payload to {args.output} "
        f"(mapped={len(mapped)}/{len(backbone_keys)}, copied={copied}, "
        f"interpolated={interpolated}, missing={len(missing)}, head_init={args.head_init})."
    )
    if missing:
        print(f"[WARN] Missing keys (first 20): {missing[:20]}")
    print(f"[INFO] JSON summary: {summary_path_for(args.output)}")
    print(f"[INFO] Set model.init_args.action_init_path: {args.output}")
    print(f"[INFO] Set model.init_args.action_head_init: {args.head_init}")


if __name__ == "__main__":
    main()
