# Fourier Neural Operator for Magnetized Plasmas

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

This project implements a 3D Fourier Neural Operator (FNO) surrogate for
2D magnetohydrodynamic (MHD) turbulence. The current supported workflow is
based on **Idefix** simulations of the **Orszag-Tang vortex**, spanning an
ensemble of viscosities and magnetic diffusivities. The model learns to map
a short initial window of simulation frames and physical parameters to
future states for quantities such as density, velocity, and magnetic-field
components.

---
**CURRENT STATUS**
This repository contains working scripts for training, inference, dense
reference preparation, and scalar/vector visualization. The supported data
workflow is Idefix-based. Older FARGO3D assets remain only as historical
context and are not a supported path. A Dockerized environment is not
currently provided.

---

![Demo](https://github.com/rsnemmen/rsnemmen.github.io/blob/ea3fb56c0b4aff19ea168753a924d4d59114f2b4/assets/video/magfield.webp)
**Figure 1:** Comparison between ground truth simulated data of MHD vortex (*left panel*) and FNO prediction (*center*) using this codebase including residuals (*right panel*). Colors indicate the magnetic field strength. Time in units of the Alfven time for the computational box.


## Overview

The Fourier Neural Operator (FNO) is a deep learning architecture that learns mappings between infinite-dimensional function spaces. Unlike traditional neural networks that learn point-wise mappings, FNOs learn entire operator mappings, making them particularly effective for solving PDEs.

This implementation:
- Operates on 2D spatial grids (128×128) with temporal evolution
- Learns in the frequency domain using Fast Fourier Transforms
- Predicts future MHD simulation states from initial conditions and physical parameters (viscosity ν, diffusivity μ)
- Supports multiple physical quantities (gas density, velocity components, magnetic field components)

## Installation

### Requirements

- Python 3.8 or higher
- CUDA-capable GPU (required for training)
- 8GB+ GPU memory recommended
- Numpy, matplotlib, scipy, PyTorch


## Model Architecture

### FNO3d Overview

The model consists of:
- **Input layer**: Linear projection from 10 channels (7 data channels + 3 spatial/temporal coordinates added internally) to hidden width
- **5 Fourier layers**: Each combines spectral convolution with a 1×1×1 skip convolution and GELU activation
- **Output layers**: Two fully-connected layers projecting back to physical space

### Architecture Details

```
Input: (batch, 128, 128, 20, 7)
       └─ 128×128 spatial grid, 20 temporal positions, 7 channels
          (5 input snapshots + viscosity ν + diffusivity μ)
  ↓ [Append x/y/t grid coordinates → 10 channels total]
  ↓ [Linear projection (10 → width=30) + permute]
  ↓ [5× Fourier Layer with GELU activation]
  ↓ [Unpad + permute]
  ↓ [Linear projection (30 → 128 → 1)]
Output: (batch, 128, 128, 20, 1) → next 20 timesteps
```

**Key hyperparameters:**
- Fourier modes: 64 (x-direction), 64 (y-direction), 5 (time)
- Hidden width: 30
- Temporal padding: 6 (for non-periodic temporal boundaries)
- Activation: GELU

## Usage

### Training a Model

```bash
cd src
python train.py --param <parameter_name> \
  [--experiments-dir <path>] [--fast] [--batch-size N] [--patience N]
```

**Available parameters (Idefix dataset):**
- `density` - Gas density
- `vx` - Gas velocity (x-component)
- `vy` - Gas velocity (y-component)
- `bx` - Magnetic field (x-component)
- `by` - Magnetic field (y-component)

Short leaf parameters such as `density` or `by` resolve automatically to the
preferred dataset snapshot under `data/idefix/numpy/t20/`. If you need a
specific dataset path, pass it explicitly, for example:

```bash
python train.py --param idefix/numpy/t10/density
```

The same short-param resolution applies to inference and visualization
scripts.

**Example:**
```bash
cd src
python train.py --param density
```

To create a cleaner run-scoped experiment directory, point
`--experiments-dir` at a dedicated run root:

```bash
cd src
python train.py \
  --experiments-dir ../experiments/64_30_autoreg--2026-31-03 \
  --param by
```

When `--experiments-dir` targets a dedicated run directory, new artifacts are
stored in a flatter layout such as:

```text
experiments/<run>/
  manifest.json
  params/
    by/
      model_best.pt
      model_64_30.pt
      training_state.pt
      loss_64_30.npy
      train.log
      predictions/
        pred_sim_000.npy
      renders/
        ...
  references/
    test/
      bx/
        ref_sim_000.npy
      by/
        ref_sim_000.npy
  vector/
    magnetic/
    velocity/
```

Existing legacy experiment trees remain supported.

Use `--fast` for a short smoke test that trains on a tiny subset of the data and saves a validation image every epoch.

**Training configuration:**
- Epochs: 5,000
- Batch size: 16 (per gradient step)
- Optimizer: Custom Adam with `weight_decay=1e-4`
- Learning rate: 0.001 with StepLR scheduler (`step_size=500`, `gamma=0.5`)
- Early stopping: `--patience 500` (default; monitors validation MAE, 0 = disabled)
- Loss: MAE (L1) on normalized targets

**Output** (paths are relative to project root):
- Legacy default root: `experiments/<param>/checkpoints/` and `experiments/<param>/visualizations/`
- Dedicated run root: `experiments/<run>/params/<param>/`
- Run-scoped prediction outputs: `experiments/<run>/params/<param>/predictions/`
- Optional run-scoped dense references: `experiments/<run>/references/test/<param>/`
- Validation/render images for run-scoped layouts: `experiments/<run>/params/<param>/renders/`

Checkpoints created before the MAE-only refactor are not compatible with `--resume`; start a fresh run instead of resuming an older `training_state.pt`.

### Inference

```bash
cd src
python inference.py --param <parameter_name> \
  [--experiments-dir <path>] [--checkpoint <path>] [--rollout-steps N]
```

Processes all discovered test samples and saves denormalized predictions as
`.npy` arrays.

**Example:**
```bash
cd src
python inference.py --param density
```

For a run-scoped experiment, you can use a short leaf parameter:

```bash
cd src
python inference.py \
  --experiments-dir ../experiments/64_30_autoreg--2026-31-03 \
  --param by
```

Autoregressive predictions are written under
`params/<param>/predictions/pred_sim_<id>.npy`. `--rollout-steps N` keeps
rolling the model forward for `N` chained 20-frame steps, so each saved array
has shape `(1, 128, 128, 20*N)`.

To prepare dense rollout references inside the same run root instead of writing
them back into `data/<param>/test/`, use:

```bash
cd src
python prepare_reference.py \
  --experiments-dir ../experiments/64_30_magfield \
  --param-prefix idefix/numpy/t20
```

### Visualization

After inference, render scalar or vector diagnostics from the same run root:

```bash
cd src
python viz_scalar.py \
  --experiments-dir ../experiments/64_30_autoreg--2026-31-03 \
  --param by

python viz_vector.py \
  --experiments-dir ../experiments/64_30_autoreg--2026-31-03 \
  --param bx
```

`viz_vector.py` requires paired predictions for both vector components under the
same run root. For example, magnetic plots need matching `bx` and `by` files
for the same `sim_id`.

This is the recommended way to organize a unified magnetic-field evaluation
bundle: keep the raw `x_sim_*.npy` / `y_sim_*.npy` inputs in `data/`, but point
both `bx` and `by` training, inference, and reference preparation at the same
`experiments/<run>/` root.

Legacy nested experiment layouts are still supported. For those runs you can
continue passing the full nested parameter path when needed.

---
**NOTE ABOUT FORECASTING**  
Inference is now rollout-only: after the initial prediction window, the model
feeds its own last 5 predicted frames back in as the next input state. This is
closer to real deployment, but prediction errors accumulate over time, so
longer rollouts are harder than one-step forecasts.

**The `test` split is reserved for final inference only.** Training validates against the separate `val` split so that `test` remains an unbiased final evaluation set.

---


## Training Details

### Loss Functions

The model optimizes a single loss:

1. **MAE Loss (L1)**: Mean absolute error on normalized predictions

Validation MAE is also used for early stopping and best-checkpoint selection.

### Normalization

- **Input snapshots** (first 5 of 7 channels): min-max scaled to [-1, 1] per batch
- **Physical parameters** ν and μ (last 2 channels): passed through as-is, not normalized
- **Target (y)**: min-max scaled to [-1, 1] during training; denormalized for saved predictions and diagnostic metrics

### Data Loading

- All data cached in memory at startup to avoid repeated disk I/O
- File counts are discovered dynamically from the resolved dataset path
- Current preferred snapshot `data/idefix/numpy/t20/<param>/` contains
  37 train, 6 val, and 2 test files per field
- Training validates against the **val** split; the **test** split is held out for final inference only

## Data Format

### Directory Structure

Short params like `density` resolve to the preferred nested path
`data/idefix/numpy/t20/density/`.

```
data/
├── idefix/
│   └── numpy/
│       ├── t20/
│       │   ├── density/
│       │   │   ├── train/
│       │   │   │   ├── x_sim_002.npy   # Input: 5 snapshots + ν + μ
│       │   │   │   ├── y_sim_002.npy   # Target: next 20 frames
│       │   │   │   └── ...
│       │   │   ├── val/
│       │   │   └── test/
│       │   ├── vx/
│       │   ├── vy/
│       │   ├── bx/
│       │   └── by/
│       └── t10/                        # Older legacy snapshot
└── fargo3d/                            # Historical only; not supported
```

### Array Shapes

- **Input files (`x_*.npy`)** for the preferred `t20` snapshot:
  `(20, 128, 128, 20, 7)`
  - 20 samples per file, 128×128 spatial grid, 20 temporal positions
  - 7 channels: 5 input snapshots + viscosity ν + diffusivity μ

- **Target files (`y_*.npy`)** for the preferred `t20` snapshot:
  `(20, 128, 128, 20)`
  - 20 samples per file, 128×128 spatial grid, 20 predicted frames

The legacy `t10` snapshot uses `10` instead of `20` in the temporal
dimension, but short params default to `t20`.

## Configuration

### Model Hyperparameters

To modify the model architecture, edit the instantiation in `src/train.py` and `src/inference.py`:

```python
model = FNO3d(modes1=64, modes2=64, modes3=5, width=30).cuda()
```

- `modes1`, `modes2`: Fourier modes in x, y directions
- `modes3`: Fourier modes in the time dimension
- `width`: hidden layer width

## Project Structure

```
fno/
├── paper/
│   └── main.tex             # Associated journal paper
├── src/
│   ├── architecture.py      # FNO3d and SpectralConv3d definitions (used by inference.py)
│   ├── train.py             # Main training script (also defines FNO3d inline)
│   ├── inference.py         # Inference script
│   ├── prepare_reference.py # Dense rollout reference generation from Idefix runs
│   ├── viz_scalar.py        # Scalar prediction rendering
│   ├── viz_vector.py        # Vector prediction rendering
│   ├── unify_legacy_magnetic_run.py # Merge legacy bx/by runs into one run root
│   ├── utilities.py         # Loss functions: LpLoss, HsLoss, FrequencyLoss
│   ├── Adam.py              # Custom Adam optimizer
│   └── experiment_layout.py # Shared legacy/run-scoped path resolver
└── experiments/             # Legacy and run-scoped experiment artifacts
```

## References

- Duarte, Nemmen & Lima-Santos (2025). Spectral Learning of Magnetized Plasma Dynamics: A Neural Operator Application. [*arXiv:2507.01388*](https://arxiv.org/abs/2507.01388)
- Li et al. (2020). Fourier Neural Operator for Parametric Partial Differential Equations. [*arXiv:2010.08895*](https://arxiv.org/abs/2010.08895)
