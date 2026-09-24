"""Lightning training wrapper for DeltaWorld-conditioned ActionDiT on LIBERO."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import lightning
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from feature_conditioned_action import (
    DeltaWorldFeatureActionModel, DeltaWorldTokenRollout, T5InstructionEncoder,
)
from starwam.action_model import load_action_dit_init
from starwam.modules.action_dit import ActionDiT
from starwam.modules.scheduler import FlowMatchScheduler
from starwam.training.flow import add_flow_noise
from starwam.training.loss import flow_matching_loss
from training.auto_prepare import ensure_action_dit_init_payload
from training.base import Base, load_sd


LOGGER = logging.getLogger(__name__)


class FeatureConditionedAction(Base):
    """Train ActionDiT with a pretrained DeltaWorld feature predictor."""

    def __init__(
        self,
        delta_world: nn.Module,
        world_ckpt_path: str,
        action_dim: int = 7,
        action_horizon: int = 16,
        action_hidden_dim: int = 768,
        action_num_layers: int = 6,
        action_num_heads: int = 12,
        action_head_dim: int = 64,
        action_ffn_ratio: int = 4,
        action_text_dim: int = 256,
        action_freq_dim: int = 256,
        action_eps: float = 1e-6,
        action_init_path: str | None = None,
        action_head_init: str = "random",
        num_world_samples: int = 1,
        freeze_world: bool = True,
        oracle_branch_probability: float = 1.0,
        context_frames: int = 4,
        world_prediction_steps: int = 3,
        world_time_delta: float = 0.1,
        world_condition_mode: str = "delta_tokens",
        text_encoder_name: str = "google-t5/t5-base",
        text_encoder_max_length: int = 64,
        freeze_text_encoder: bool = True,
        proprio_dim: int = 9,
        action_history_steps: int = 0,
        action_history_dropout: float = 0.2,
        action_history_attention_mode: str = "separate_gated",
        action_history_gate_init: float = 0.01,
        action_warmstart_path: str | None = None,
        action_warmstart_exclude_prefixes: tuple[str, ...] = (),
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        warmup_steps: int = 1000,
        min_lr_ratio: float = 0.01,
        lr_scheduler_type: str = "cosine",
        resume_lr_warmup_steps: int = 0,
        scheduler_train_shift: float = 5.0,
        scheduler_infer_shift: float = 5.0,
        scheduler_num_train_timesteps: int = 1000,
        use_compile: bool = False,
    ) -> None:
        # Base init is world-loss-specific; direct Lightning init preserves CLI compatibility.
        lightning.LightningModule.__init__(self)
        if action_hidden_dim != action_num_heads * action_head_dim:
            raise ValueError(
                "action_hidden_dim must equal action_num_heads * action_head_dim, "
                f"got {action_hidden_dim} vs {action_num_heads}*{action_head_dim}"
            )
        if proprio_dim < 0:
            raise ValueError("proprio_dim must be non-negative")
        if not 0.0 <= oracle_branch_probability <= 1.0:
            raise ValueError(
                "oracle_branch_probability must be in [0, 1], got "
                f"{oracle_branch_probability}"
            )

        self.save_hyperparameters(ignore=["delta_world"])
        self.ckpt_path = str(world_ckpt_path)
        self.lr = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.lr_warmup_steps = int(warmup_steps)
        self.min_lr_ratio = float(min_lr_ratio)
        self.lr_scheduler_type = str(lr_scheduler_type).strip().lower()
        if self.lr_scheduler_type not in {"cosine", "constant"}:
            raise ValueError(
                "lr_scheduler_type must be 'cosine' or 'constant', got "
                f"{lr_scheduler_type!r}"
            )
        self.resume_lr_warmup_steps = int(resume_lr_warmup_steps)
        if self.resume_lr_warmup_steps < 0:
            raise ValueError("resume_lr_warmup_steps must be non-negative")
        self._resume_lr_ramp: dict[str, float | int] | None = None
        # Base.setup() reads this flag. Dynamic BoM selection and reusable KV caches
        # stay in eager mode by default for predictable first-stage training.
        self.use_compile = bool(use_compile)
        self.action_horizon = int(action_horizon)
        self.proprio_dim = int(proprio_dim)

        world_path = Path(world_ckpt_path)
        if not world_path.is_file():
            raise FileNotFoundError(f"DeltaWorld checkpoint not found: {world_path}")
        checkpoint = torch.load(world_path, map_location="cpu", weights_only=False, mmap=True)
        incompatible = load_sd(delta_world, checkpoint)
        LOGGER.info(
            "Loaded DeltaWorld checkpoint %s (missing=%d, unexpected=%d)",
            world_path, len(incompatible.missing_keys), len(incompatible.unexpected_keys),
        )

        action_dit = ActionDiT(
            hidden_dim=action_hidden_dim,
            action_dim=action_dim,
            ffn_dim=action_hidden_dim * action_ffn_ratio,
            text_dim=action_text_dim,
            freq_dim=action_freq_dim,
            eps=action_eps,
            num_heads=action_num_heads,
            attn_head_dim=action_head_dim,
            num_layers=action_num_layers,
            max_seq_len=max(256, action_horizon * 2),
            use_gradient_checkpointing=False,
        )
        action_init_path = ensure_action_dit_init_payload(
            action_init_path,
            action_dit,
            head_init=action_head_init,
            dtype=torch.bfloat16,
        )
        load_action_dit_init(action_dit, action_init_path, head_init=action_head_init)
        world_rollout = DeltaWorldTokenRollout(
            delta_world,
            num_samples=num_world_samples,
            freeze_world=freeze_world,
            oracle_branch_probability=oracle_branch_probability,
        )
        self.network = DeltaWorldFeatureActionModel(
            world_rollout,
            action_dit,
            action_scheduler=FlowMatchScheduler(
                num_train_timesteps=scheduler_num_train_timesteps,
                shift=scheduler_train_shift,
            ),
            context_frames=context_frames,
            world_prediction_steps=world_prediction_steps,
            world_time_delta=world_time_delta,
            world_condition_mode=world_condition_mode,
            include_text=True,
            action_history_steps=action_history_steps,
            action_history_dropout=action_history_dropout,
            action_history_attention_mode=action_history_attention_mode,
            action_history_gate_init=action_history_gate_init,
        )
        self.inference_scheduler = FlowMatchScheduler(
            num_train_timesteps=scheduler_num_train_timesteps,
            shift=scheduler_infer_shift,
        )
        self.instruction_encoder = T5InstructionEncoder(
            model_name_or_path=text_encoder_name,
            output_dim=action_text_dim,
            max_length=text_encoder_max_length,
            freeze_encoder=freeze_text_encoder,
        )
        self.proprio_encoder = (
            nn.Sequential(
                nn.LayerNorm(proprio_dim),
                nn.Linear(proprio_dim, action_text_dim),
                nn.SiLU(),
                nn.Linear(action_text_dim, action_text_dim),
            )
            if proprio_dim > 0 else None
        )
        self.action_warmstart_path = (
            str(action_warmstart_path) if action_warmstart_path else None
        )
        self.action_warmstart_exclude_prefixes = tuple(
            str(prefix) for prefix in action_warmstart_exclude_prefixes
        )
        if self.action_warmstart_path is not None:
            self._load_action_warmstart(self.action_warmstart_path)

    def _load_action_warmstart(self, checkpoint_path: str) -> None:
        """Load model tensors only; intentionally start a fresh optimizer/scheduler."""
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Action warm-start checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        source = checkpoint.get("state_dict", checkpoint)
        current = self.state_dict()
        compatible = {
            key: value for key, value in source.items()
            if key in current and current[key].shape == value.shape
            and not any(
                key.startswith(prefix)
                for prefix in self.action_warmstart_exclude_prefixes
            )
        }
        excluded = [
            key for key in source
            if any(
                key.startswith(prefix)
                for prefix in self.action_warmstart_exclude_prefixes
            )
        ]
        incompatible = lightning.LightningModule.load_state_dict(
            self, compatible, strict=False
        )
        history_missing = [
            key for key in incompatible.missing_keys
            if key.startswith("network.action_history_encoder.")
            or key == "network.action_model.history_gates"
        ]
        LOGGER.info(
            "Warm-started action model from %s global_step=%s loaded=%d "
            "excluded=%d new_history_parameters=%d (optimizer/scheduler reset)",
            path, checkpoint.get("global_step"), len(compatible), len(excluded),
            len(history_missing),
        )

    def _conditioning_tokens(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
        context, mask = self.instruction_encoder(batch["instruction"])
        if self.proprio_encoder is not None:
            proprio = batch["proprio"]
            if proprio.shape[-1] != self.proprio_dim:
                raise ValueError(
                    f"expected proprio_dim={self.proprio_dim}, got {proprio.shape[-1]}"
                )
            proprio_token = self.proprio_encoder(proprio).unsqueeze(1)
            if proprio_token.shape[0] != context.shape[0]:
                raise ValueError("instruction/proprio batch size mismatch")
            context = torch.cat([context, proprio_token], dim=1)
            proprio_mask = torch.ones(
                proprio_token.shape[:2], dtype=torch.bool, device=context.device
            )
            mask = torch.cat([mask, proprio_mask], dim=1)
        return context, mask

    def _shared_step(self, batch: dict[str, Any], stage: str) -> Tensor:
        action = batch["action"].float()
        if action.shape[1] != self.action_horizon:
            raise ValueError(
                f"expected action_horizon={self.action_horizon}, got {action.shape[1]}"
            )
        context, context_mask = self._conditioning_tokens(batch)
        condition = self.network.encode_condition(
            batch["video"],
            observed_timestamps=batch["timestamps"],
            future_timestamps=batch["future_timestamps"],
            text_context=context,
            text_mask=context_mask,
            future_frames=batch["future_video"],
            instruction=batch.get("instruction"),
            action_history=batch.get("action_history"),
            action_history_mask=batch.get("action_history_mask"),
        )
        noisy_action, target, timestep = add_flow_noise(
            self.network.action_scheduler, action
        )
        prediction = self.network(noisy_action, timestep, condition)
        loss = flow_matching_loss(
            prediction, target, timestep, self.network.action_scheduler,
            is_pad_mask=batch.get("action_is_pad"),
        )
        self.log(
            f"losses/{stage}", loss, prog_bar=True, sync_dist=True,
            on_step=stage == "train", on_epoch=stage != "train",
            batch_size=action.shape[0],
        )
        if stage == "train":
            if "sampling_hard_branch" in batch:
                phase_id = batch["sampling_phase_id"].detach()
                sampling_metrics = {
                    "monitor/data/base_branch_fraction": batch[
                        "sampling_base_branch"
                    ].detach().float().mean(),
                    "monitor/data/hard_branch_fraction": batch[
                        "sampling_hard_branch"
                    ].detach().float().mean(),
                    "monitor/data/hard_task_fraction": batch[
                        "sampling_task_is_hard"
                    ].detach().float().mean(),
                    "monitor/data/phase_fraction": batch[
                        "sampling_is_phase"
                    ].detach().float().mean(),
                    "monitor/data/phase_grasp_fraction": (phase_id == 1).float().mean(),
                    "monitor/data/phase_release_fraction": (phase_id == 2).float().mean(),
                    "monitor/data/phase_precision_fraction": (phase_id == 3).float().mean(),
                    "monitor/data/phase_terminal_fraction": (phase_id == 4).float().mean(),
                }
                self.log_dict(
                    sampling_metrics,
                    sync_dist=True,
                    on_step=True,
                    on_epoch=False,
                    batch_size=action.shape[0],
                )
            if condition.action_history_mask is not None:
                history_gates = self.network.action_model.history_gate_values().detach()
                self.log_dict(
                    {
                        "monitor/action_history_gate_mean": history_gates.mean(),
                        "monitor/action_history_gate_abs_mean": history_gates.abs().mean(),
                        "monitor/action_history_gate_min": history_gates.min(),
                        "monitor/action_history_gate_max": history_gates.max(),
                    },
                    sync_dist=True, on_step=True, on_epoch=False,
                    batch_size=action.shape[0],
                )
                self.log(
                    "monitor/action_history_valid",
                    condition.action_history_mask.detach().float().sum(dim=1).mean(),
                    sync_dist=True, on_step=True, on_epoch=False,
                    batch_size=action.shape[0],
                )
            self.log(
                "monitor/action_velocity_l1", prediction.detach().float().abs().mean(),
                sync_dist=True, on_step=True, on_epoch=False,
                batch_size=action.shape[0],
            )
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        del batch_idx
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        del batch_idx
        return self._shared_step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        """Do not run Base image-prediction visualization for action validation."""
        # losses/val is already reduced by Lightning via self.log(on_epoch=True).
        return None

    def _lr_multiplier(self, step: int) -> float:
        ramp = self._resume_lr_ramp
        schedule_start = self.lr_warmup_steps
        if ramp is not None:
            start_step = int(ramp["start_step"])
            warmup_steps = int(ramp["warmup_steps"])
            start_lr = float(ramp["start_lr"])
            target_lr = float(ramp["target_lr"])
            schedule_start = start_step + warmup_steps
            if step <= schedule_start:
                progress = min(
                    max((step - start_step) / max(1, warmup_steps), 0.0), 1.0
                )
                desired_lr = start_lr + (target_lr - start_lr) * progress
                return desired_lr / target_lr
        elif self.lr_warmup_steps > 0 and step < self.lr_warmup_steps:
            return max(step, 1) / self.lr_warmup_steps

        if self.lr_scheduler_type == "constant":
            return 1.0

        max_steps = int(self.trainer.max_steps)
        if max_steps <= 0:
            # Epoch-based training sets max_steps=-1. Lightning accounts for
            # dataset length, devices and gradient accumulation here.
            max_steps = int(self.trainer.estimated_stepping_batches)
        max_steps = max(max_steps, schedule_start + 1)
        progress = (step - schedule_start) / max(
            1, max_steps - schedule_start
        )
        cosine = 0.5 * (
            1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0))
        )
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def configure_optimizers(self) -> dict[str, Any]:
        decay, no_decay = [], []
        for parameter in self.parameters():
            if parameter.requires_grad:
                (decay if parameter.ndim >= 2 else no_decay).append(parameter)
        if not decay and not no_decay:
            raise RuntimeError("No trainable action-model parameters")
        optimizer = AdamW(
            [
                {"params": decay, "weight_decay": self.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.lr,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": LambdaLR(optimizer, self._lr_multiplier),
                "interval": "step",
            },
        }

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # Frozen DeltaWorld is restored from world_ckpt_path, not duplicated per action ckpt.
        trainable = {name for name, p in self.named_parameters() if p.requires_grad}
        checkpoint["state_dict"] = {
            key: value for key, value in checkpoint["state_dict"].items()
            if key in trainable
        }
        checkpoint["external_world_checkpoint"] = self.ckpt_path
        if self._resume_lr_ramp is not None:
            checkpoint["action_resume_lr_ramp"] = dict(self._resume_lr_ramp)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Preserve optimizer state and optionally install a resume LR ramp.

        ``FeatureConditionedAction`` inherits from the DeltaWorld ``Base`` wrapper
        for CLI compatibility, but ``Base.on_load_checkpoint`` intentionally
        clears LR schedulers so world-model finetunes can restart with a new
        schedule. ActionDiT instead retains optimizer moments and scheduler
        position. A configured ramp changes only the saved LR curve.
        """
        if self.resume_lr_warmup_steps == 0:
            return

        optimizer_states = checkpoint.get("optimizer_states") or []
        scheduler_states = checkpoint.get("lr_schedulers") or []
        if not optimizer_states or not scheduler_states:
            LOGGER.warning(
                "resume_lr_warmup_steps=%d but checkpoint has no optimizer/scheduler "
                "state; using the fresh-run schedule",
                self.resume_lr_warmup_steps,
            )
            return

        param_groups = optimizer_states[0].get("param_groups") or []
        current_lrs = [float(group["lr"]) for group in param_groups]
        if not current_lrs:
            raise RuntimeError("resume checkpoint optimizer has no parameter groups")
        if max(current_lrs) - min(current_lrs) > 1e-12:
            raise RuntimeError(
                "resume LR ramp requires equal LR across optimizer parameter groups, "
                f"got {current_lrs}"
            )

        saved_ramp = checkpoint.get("action_resume_lr_ramp")
        if (
            isinstance(saved_ramp, dict)
            and math.isclose(
                float(saved_ramp.get("target_lr", -1.0)),
                self.lr,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and int(saved_ramp.get("warmup_steps", -1))
            == self.resume_lr_warmup_steps
        ):
            ramp = dict(saved_ramp)
        else:
            ramp = {
                "start_step": int(checkpoint.get("global_step", 0)),
                "warmup_steps": self.resume_lr_warmup_steps,
                "start_lr": current_lrs[0],
                "target_lr": self.lr,
            }
        self._resume_lr_ramp = ramp

        for optimizer_state in optimizer_states:
            for group in optimizer_state.get("param_groups", []):
                group["initial_lr"] = self.lr
        for scheduler_state in scheduler_states:
            count = len(scheduler_state.get("base_lrs", current_lrs))
            scheduler_state["base_lrs"] = [self.lr] * count
            scheduler_state["_last_lr"] = current_lrs[:count]
        checkpoint["action_resume_lr_ramp"] = dict(ramp)

        LOGGER.info(
            "Resume LR ramp: step=%d lr=%.8g -> %.8g over %d steps; "
            "scheduler_after_ramp=%s",
            int(checkpoint.get("global_step", 0)),
            current_lrs[0],
            self.lr,
            self.resume_lr_warmup_steps,
            self.lr_scheduler_type,
        )

    def on_train_start(self) -> None:
        """Do not apply DeltaWorld Base LR reset logic to ActionDiT training."""
        return None

    def load_state_dict(
        self, state_dict: dict[str, Tensor], strict: bool = False
    ) -> torch.nn.modules.module._IncompatibleKeys:
        return lightning.LightningModule.load_state_dict(self, state_dict, strict=False)
