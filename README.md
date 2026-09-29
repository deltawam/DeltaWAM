# DeltaWAM

Official implementation of **DeltaWAM: What Changes Is What Matters:
Transition Tokens for Efficient World-Action Models**.

DeltaWAM predicts compact future transition tokens with DeltaWorld and
conditions a flow-matching ActionDiT on two complementary visual memories:
dense DINOv3 patches from the current observation and one predicted delta token
per future step and camera view.

## Repository contents

```text
models/                       DeltaTok and DeltaWorld
feature_conditioned_action/  DeltaWorld rollout, K/V adapter, and policy
training/                     Lightning training modules
datasets/                     LIBERO data loading and temporal sampling
starwam/                      Minimal vendored ActionDiT dependency
configs/release/              Paper training and ablation recipes
scripts/                      Training, evaluation, audit, and init tools
splits/                       Replay-verified LIBERO demonstration lists
checkpoints/                  Expected external weight layout (no weights)
```

FastWAM and raw datasets are not bundled.

## Installation

Create a clean Python environment and install the package:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
cp .env.example .env
```

Closed-loop evaluation additionally requires LIBERO and its MuJoCo/robosuite
dependencies. Install LIBERO from its official repository in the same
environment and set:

```bash
export LIBERO_HOME=/path/to/LIBERO
export LIBERO_ROOT=/path/to/libero_hdf5
```

DINOv3 access may require accepting its model license and authenticating with
the hosting service. T5-Base is downloaded through Transformers.

## Checkpoints

Checkpoint files are intentionally excluded from this anonymous supplementary
archive. Place separately obtained checkpoints in the expected local layout
below before running training or evaluation commands that require them.

The complete expected directory layout is:

```text
checkpoints/
├── actiondit/
│   ├── actiondit_libero90_pretrained.ckpt
│   ├── actiondit_libero_spatial.ckpt
│   ├── actiondit_libero_object.ckpt
│   ├── actiondit_libero_goal.ckpt
│   └── actiondit_libero_10.ckpt
├── deltaworld/
│   ├── deltaworld_finetuned_1.ckpt
│   └── deltaworld_finetuned_2.ckpt
├── deltatok_kinetics.bin          # prepare separately
├── deltaworld_kinetics.bin        # prepare separately
└── actiondit_wan22_init.pt        # prepare separately
```

The checkpoint files under `actiondit/` and `deltaworld/` are not bundled. The
three initialization assets at the root of `checkpoints/` must also be prepared
separately when needed.

`deltaworld_finetuned_1.ckpt` is used by the released Spatial policy;
`deltaworld_finetuned_2.ckpt` is used by the other released policies.
Spatial, Object, and Goal use full-state specialization from the LIBERO-90
checkpoint. LIBERO-10 uses the same checkpoint as a weights-only warm start.

## LIBERO data

Expected layout:

```text
${LIBERO_ROOT}/
├── libero_spatial/*.hdf5
├── libero_object/*.hdf5
├── libero_goal/*.hdf5
├── libero_90/*.hdf5
└── libero_10/*.hdf5
```

Each standard task starts with 50 demonstrations. Suite specialization uses
the replay-verified demonstration lists under `splits/`. The lists contain
only relative HDF5 names and demonstration identifiers and do not redistribute
LIBERO data.

To regenerate a list from a local LIBERO installation, use:

```bash
python scripts/audit_libero_expert_demos.py --help
```

## Training

### 1. Adapt DeltaWorld to LIBERO

```bash
bash scripts/train_deltaworld.sh
```

This recipe uses dual views, suite-balanced sampling, a frozen DINOv3
ViT-B/16 and DeltaTok, and a 12-layer language-conditioned DeltaWorld
predictor.

### 2. Train the unified LIBERO-90 action policy

```bash
bash scripts/train_actiondit_libero90.sh
```

The policy uses a 12-layer, width-768 ActionDiT initialized from the
preprocessed Wan2.2 payload and trained with eight executed-action history
tokens.

If the initialization payload is not released, regenerate it from a local
Wan2.2 model or state dictionary:

```bash
python scripts/preprocess_feature_action_dit_init.py \
  --config configs/release/actiondit_libero90_pretrain.yaml \
  --pretrained-model-id /path/to/Wan2.2-TI2V-5B \
  --output checkpoints/actiondit_wan22_init.pt \
  --dtype bfloat16
```

### 3. Specialize to a standard suite

```bash
bash scripts/finetune_suite.sh spatial
bash scripts/finetune_suite.sh object
bash scripts/finetune_suite.sh goal
bash scripts/finetune_suite.sh libero_10
```

Spatial, Object, and Goal resume the full LIBERO-90 state at step 28,737 for
5,000 more optimizer steps. LIBERO-10 uses a weights-only warm start and a new
5,000-step optimizer schedule.

## Configuration reference

| Configuration | Purpose |
|---|---|
| `deltaworld_libero_all.yaml` | DeltaWorld adaptation on all LIBERO suites |
| `actiondit_libero90_pretrain.yaml` | Unified LIBERO-90 ActionDiT training |
| `actiondit_libero_spatial.yaml` | Spatial specialization |
| `actiondit_libero_object.yaml` | Object specialization |
| `actiondit_libero_goal.yaml` | Goal specialization |
| `actiondit_libero_10.yaml` | LIBERO-10 weights-only specialization |

All files are under `configs/release/` and use repository-relative paths.
The `configs/release/ablations/` directory contains current-DINO-only and
decoded-future-DINO recipes for every standard suite.

Spatial, Object, and Goal use full-state resume:

```bash
python main.py fit -c configs/release/actiondit_libero_object.yaml \
  --ckpt_path checkpoints/actiondit/actiondit_libero90_pretrained.ckpt
```

LIBERO-10 loads `actiondit/actiondit_libero90_pretrained.ckpt` through
`action_warmstart_path` and starts a new optimizer schedule.

## Inference and evaluation

Run closed-loop inference for one complete LIBERO suite:

```bash
bash scripts/evaluate_libero.sh spatial
bash scripts/evaluate_libero.sh object
bash scripts/evaluate_libero.sh goal
bash scripts/evaluate_libero.sh libero_10
```

The wrapper uses 50 fixed initial-state trials per task, ten flow-matching
steps, eight executed actions per policy query, and temporal ensembling with
decay 0.1. Extra evaluator arguments may be appended to the command.

The evaluator writes per-task successes, trial counts, and execution metadata
under `outputs/feature_action_libero/`.

## License and attribution

DeltaWAM is released under Apache License 2.0. A minimal Apache-2.0 StarWAM
subset is vendored under `starwam/`; its license is preserved under
`third_party/starwam/`. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)
and [`NOTICE`](NOTICE) for attribution.
