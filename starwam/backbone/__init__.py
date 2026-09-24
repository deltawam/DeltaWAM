"""Backbone builders for StarWAM."""

from __future__ import annotations

import torch

from starwam.backbone.base import BaseBackbone, BackboneInfo
from starwam.config import BackboneConfig


WAN_BACKBONES = {"wan22_5b", "wan22_14b"}
COSMOS_PREDICT2_BACKBONES = {"cosmos_predict2", "cosmos_predict2_2b"}


def build_backbone(
    config: BackboneConfig,
    device: str = "cpu",
    dtype=None,
    *,
    load_dit: bool = True,
) -> BaseBackbone:
    """Build a StarWAM backbone."""

    if dtype is None:
        dtype = torch.bfloat16
    if config.type in WAN_BACKBONES:
        from starwam.backbone.wan22 import Wan22Backbone

        return Wan22Backbone(
            config,
            device=device,
            dtype=dtype,
            load_dit=load_dit,
            load_text_encoder=getattr(config, "load_text_encoder", False),
        )
    if config.type in COSMOS_PREDICT2_BACKBONES:
        from starwam.backbone.cosmos_predict2 import CosmosPredict2Backbone

        return CosmosPredict2Backbone(
            config,
            device=device,
            dtype=dtype,
            load_dit=load_dit,
            load_text_encoder=getattr(config, "load_text_encoder", False),
        )
    raise ValueError(
        f"Unknown StarWAM backbone type: {config.type}. "
        f"Available: {sorted(WAN_BACKBONES | COSMOS_PREDICT2_BACKBONES)}"
    )


__all__ = ["BaseBackbone", "BackboneInfo", "Wan22Backbone", "build_backbone"]


def __getattr__(name):
    if name == "Wan22Backbone":
        from starwam.backbone.wan22 import Wan22Backbone

        return Wan22Backbone
    raise AttributeError(name)
