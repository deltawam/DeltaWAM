from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel


def _kaggle_handle_from_url(backbone_name: str) -> str:
    prefix = "https://www.kaggle.com/models/"
    if backbone_name.startswith("kaggle://"):
        handle = backbone_name.removeprefix("kaggle://")
    elif backbone_name.startswith(prefix):
        handle = backbone_name.removeprefix(prefix).strip("/")
    else:
        return backbone_name

    parts = [part for part in handle.split("/") if part]
    if len(parts) == 2:
        parts.extend(["transformers", "default"])
    return "/".join(parts)


def _resolve_backbone_name(backbone_name: str) -> str:
    if not (
        backbone_name.startswith("kaggle://")
        or backbone_name.startswith("https://www.kaggle.com/models/")
    ):
        return backbone_name

    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "Loading DINOv3 from Kaggle requires kagglehub. "
            "Install it with `python -m pip install kagglehub`."
        ) from exc

    handle = _kaggle_handle_from_url(backbone_name)
    download_dir = Path(kagglehub.model_download(handle))
    candidates = [download_dir, *download_dir.rglob("*")]
    for path in candidates:
        if not path.is_dir():
            continue
        has_config = (path / "config.json").exists()
        has_processor = any(
            (path / name).exists()
            for name in (
                "preprocessor_config.json",
                "processor_config.json",
                "image_processor_config.json",
            )
        )
        if has_config and has_processor:
            return str(path)

    raise FileNotFoundError(
        f"Downloaded Kaggle model {handle!r} to {download_dir}, but could not find "
        "a Transformers model directory containing config.json and processor config."
    )


DINOv3_KAGGLE_MODEL = (
    "https://www.kaggle.com/models/x1an9l1/facebookdinov3-vitb16-pretrain-lvd1689m"
)


class DINOv3(nn.Module):
    def __init__(self, backbone_name: str = DINOv3_KAGGLE_MODEL):
        super().__init__()
        backbone_name = _resolve_backbone_name(backbone_name)
        self.processor = AutoImageProcessor.from_pretrained(
            backbone_name, do_resize=False, do_center_crop=False
        )
        self.backbone = AutoModel.from_pretrained(backbone_name)
        cfg = self.backbone.config
        self.patch_size = int(cfg.patch_size)
        self.hidden_size = int(cfg.hidden_size)
        self.num_heads = int(cfg.num_attention_heads)
        self.initializer_range = float(cfg.initializer_range)
        self.num_prefix_tokens = int(cfg.num_register_tokens) + 1

    @torch.no_grad
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, _, h, w = frames.shape
        assert (
            h % self.patch_size == 0 and w % self.patch_size == 0
        ), f"Frame size ({h}, {w}) must be a multiple of patch size ({self.patch_size})"
        x = self.processor(
            frames.reshape(batch_size * num_frames, *frames.shape[2:]),
            return_tensors="pt",
        )["pixel_values"].to(frames.device)
        y = self.backbone(x).last_hidden_state[:, self.num_prefix_tokens :]
        return y.reshape(batch_size, num_frames, -1, self.hidden_size)
