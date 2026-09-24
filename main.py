import json
import logging
import os
import platform
import subprocess
import sys
import warnings
from datetime import datetime
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from lightning import Trainer
from lightning.pytorch import cli
from lightning.pytorch.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)

from training.base import Base
from training.slurm_checkpoint import SlurmPreemptionCheckpoint


def _run_cmd(cmd: list[str]) -> str:
    try:
        return subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parent,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except Exception as exc:
        return f"<failed: {type(exc).__name__}: {exc}>"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().item() if value.numel() == 1 else str(tuple(value.shape))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class LocalExperimentLog(Callback):
    """Write a local manifest for every run, independent of external loggers."""

    def __init__(self, root: str = "experiment_logs") -> None:
        self.root = Path(root)
        self.run_dir: Path | None = None
        self.start_time: str | None = None
        self._file_handler: logging.Handler | None = None

    def setup(self, trainer: Trainer, pl_module: Base, stage: str) -> None:
        if not trainer.is_global_zero or self.run_dir is not None:
            return

        repo_root = Path(__file__).resolve().parent
        self.start_time = datetime.now().astimezone().isoformat()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = repo_root / self.root / f"{stamp}_{stage or 'run'}_{os.getpid()}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._file_handler = logging.FileHandler(self.run_dir / "runtime.log")
        self._file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logging.getLogger().addHandler(self._file_handler)

        manifest = {
            "start_time": self.start_time,
            "stage": stage,
            "command": sys.argv,
            "command_string": " ".join(sys.argv),
            "cwd": str(Path.cwd()),
            "python": sys.executable,
            "platform": platform.platform(),
            "git_commit": _run_cmd(["git", "rev-parse", "HEAD"]),
            "git_branch": _run_cmd(["git", "branch", "--show-current"]),
            "env": {
                key: os.environ.get(key)
                for key in sorted(os.environ)
                if key.endswith(("_ROOT", "_PATH")) or key in {"CUDA_VISIBLE_DEVICES", "WANDB_MODE"}
            },
            "trainer": {
                "devices": _jsonable(trainer.num_devices),
                "num_nodes": _jsonable(trainer.num_nodes),
                "precision": _jsonable(trainer.precision),
                "max_steps": _jsonable(trainer.max_steps),
                "max_epochs": _jsonable(trainer.max_epochs),
                "val_check_interval": _jsonable(trainer.val_check_interval),
            },
            "model": {
                "class": type(pl_module).__module__ + "." + type(pl_module).__name__,
                "ckpt_path": getattr(pl_module, "ckpt_path", None),
                "lr": getattr(pl_module, "lr", None),
                "weight_decay": getattr(pl_module, "weight_decay", None),
                "lr_scheduler_type": getattr(pl_module, "lr_scheduler_type", None),
                "lr_warmup_steps": getattr(pl_module, "lr_warmup_steps", None),
                "resume_lr_warmup_steps": getattr(
                    pl_module, "resume_lr_warmup_steps", None
                ),
                "min_lr_ratio": getattr(pl_module, "min_lr_ratio", None),
            },
            "data": _jsonable(getattr(trainer, "datamodule", None).__dict__ if getattr(trainer, "datamodule", None) else None),
            "logger_dir": str(getattr(getattr(trainer, "logger", None), "save_dir", "")),
        }
        (self.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        (self.run_dir / "command.txt").write_text(manifest["command_string"] + "\n")
        (self.run_dir / "git_status.txt").write_text(_run_cmd(["git", "status", "--short"]) + "\n")
        (self.run_dir / "git_diff.patch").write_text(_run_cmd(["git", "diff", "--", ".", ":(exclude)data", ":(exclude)ckpt"]) + "\n")
        pkgs = sorted(f"{d.name}=={d.version}" for d in distributions())
        (self.run_dir / "pip-freeze.txt").write_text("\n".join(pkgs) + "\n")
        logging.info("Local experiment log: %s", self.run_dir)

    def _write_final(self, trainer: Trainer, stage: str) -> None:
        if not trainer.is_global_zero or self.run_dir is None:
            return
        metrics = {key: _jsonable(value) for key, value in trainer.callback_metrics.items()}
        final = {
            "start_time": self.start_time,
            "end_time": datetime.now().astimezone().isoformat(),
            "stage": stage,
            "global_step": trainer.global_step,
            "metrics": metrics,
            "checkpoint_dirs": [
                str(getattr(callback, "dirpath", ""))
                for callback in trainer.callbacks
                if isinstance(callback, ModelCheckpoint)
            ],
        }
        (self.run_dir / "final.json").write_text(json.dumps(final, indent=2, sort_keys=True))
        if self._file_handler is not None:
            self._file_handler.flush()

    def on_fit_end(self, trainer: Trainer, pl_module: Base) -> None:
        self._write_final(trainer, "fit")

    def on_validation_end(self, trainer: Trainer, pl_module: Base) -> None:
        if not trainer.sanity_checking:
            self._write_final(trainer, "validate")

    def on_test_end(self, trainer: Trainer, pl_module: Base) -> None:
        self._write_final(trainer, "test")


class LogRun(Callback):
    """Capture everything needed to reproduce the run."""

    def setup(self, trainer: Trainer, pl_module: Base, stage: str) -> None:
        log_code = os.environ.get("DELTATOK_WANDB_LOG_CODE", "1").strip().lower()
        if (
            trainer.is_global_zero
            and trainer.logger is not None
            and log_code not in {"0", "false", "no", "off"}
        ):
            trainer.logger.experiment.log_code(
                ".", include_fn=lambda filename: filename.endswith((".py", ".yaml", ".env"))
            )

    def on_train_start(self, trainer: Trainer, pl_module: Base) -> None:
        if not trainer.is_global_zero or trainer.logger is None:
            return
        experiment = trainer.logger.experiment
        experiment_dir = getattr(experiment, "dir", None)
        if callable(experiment_dir):
            experiment_dir = experiment_dir()

        if experiment_dir is not None and Path(experiment_dir).exists():
            pkgs = sorted(f"{d.name}=={d.version}" for d in distributions())
            pkg_path = Path(experiment_dir) / f"pip-freeze-step{trainer.global_step}.txt"
            pkg_path.write_text("\n".join(pkgs))
            experiment.save(str(pkg_path), policy="now")

        entry = f"[step {trainer.global_step}] {datetime.now().isoformat()}\n$ {' '.join(sys.argv)}"
        experiment.notes = f"{experiment.notes or ''}\n{entry}".strip()


class LightningCLI(cli.LightningCLI):
    def _parse_ckpt_path(self) -> None:
        """Keep the current YAML/CLI model configuration on full-state resume.

        LightningCLI normally merges ``hyper_parameters`` stored in
        ``--ckpt_path`` back into the parsed configuration before constructing
        the model. That makes an old checkpoint silently override intentional
        changes in the current YAML (for example, ``constant`` replacing a new
        ``cosine`` LR schedule). Trainer still receives ``ckpt_path`` and
        restores weights, optimizer state, scheduler state, and global step;
        only this pre-instantiation hyperparameter override is disabled.
        """

        return

    def __init__(self, *args: Any, **kw: Any) -> None:
        load_dotenv()
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.suppress_errors = True
        torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
        logging.getLogger().setLevel(logging.INFO)
        for msg in [
            r".*isinstance.*LeafSpec.*is deprecated.*",
            r".*functools.partial will be a method descriptor in future Python versions*",
            r".*No device id is provided via `init_process_group` or `barrier.*",
            r".*Precision bf16-mixed is not supported by the model summary.*",
            r".*Dynamo detected a call to a `functools.lru_cache`-wrapped function.*",
            r".*Trying to infer the `batch_size` from an ambiguous collection.*",
            r".*Found .* module\(s\) in eval mode at the start of training.*",
        ]:
            warnings.filterwarnings("ignore", message=msg)

        super().__init__(*args, **kw)


def cli_main() -> None:
    LightningCLI(
        Base,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=None,
        seed_everything_default=0,
        trainer_defaults={
            "max_epochs": -1,
            "devices": 8,
            "enable_model_summary": False,
            "check_val_every_n_epoch": None,
            "val_check_interval": 5000,
            "callbacks": [
                ModelSummary(max_depth=3),
                LearningRateMonitor(logging_interval="step"),
                LogRun(),
                LocalExperimentLog(),
                SlurmPreemptionCheckpoint(),
                ModelCheckpoint(
                    every_n_train_steps=50000,
                    save_top_k=-1,
                ),
                ModelCheckpoint(
                    monitor="losses/val",
                    mode="min",
                    save_top_k=1,
                ),
                ModelCheckpoint(
                    every_n_train_steps=500,
                    save_last=True,
                    save_top_k=0,
                    enable_version_counter=False,
                ),
            ],
            "log_every_n_steps": 1,
            "logger": {
                "class_path": "lightning.pytorch.loggers.wandb.WandbLogger",
                "init_args": {
                    "project": "deltawam",
                    "save_dir": str(Path(__file__).resolve().parent / "runs"),
                },
            },
            "precision": "bf16-mixed",
        },
    )


if __name__ == "__main__":
    cli_main()
