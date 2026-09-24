"""Utility namespace for StarWAM.

Keep this package initializer lazy: importing ``starwam.utils.checkpoint`` during
ActionDiT init must not eagerly import Wan backbones, otherwise Wan22's own
checkpoint helpers create a circular import.
"""

__all__ = ["infer_backbone_info", "save_checkpoint", "load_checkpoint"]


def __getattr__(name):
    if name in __all__:
        from starwam.utils import checkpoint

        return getattr(checkpoint, name)
    raise AttributeError(name)
