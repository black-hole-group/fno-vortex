# AGENTS.md

This file contains guidelines for agentic coding agents working in this repository.

## Overview

This is a PyTorch implementation of a 3D Fourier Neural Operator (FNO) for simulating 2D magnetohydrodynamic (MHD) turbulence. The specific problem is the **Orszag-Tang vortex**, a standard MHD benchmark. The model learns to predict future plasma dynamics (density, velocity, magnetic fields) from initial simulation frames and physical parameters (kinematic viscosity nu, Ohmic diffusivity eta).

The paper associated with this codebase is `paper/main.tex`. Detailed documentation is in `CLAUDE.md`.

## Training and Inference

### Setup

- Requires Python 3.8+ with PyTorch 1.10+
- Install dependencies: `pip install torch numpy matplotlib scipy`
- GPU required (CUDA, no CPU fallback)

### Training

```bash
python src/train.py --param <parameter_name>
```

- **Parameters:** `density`, `vy`, `by` (defaults to `density`)
- **Epochs:** 10,000
- **Learning rate:** 0.001 with StepLR scheduler (step=500, gamma=0.5)
- **Batch size:** 4
- **Loss:** MAE + Relative L2 (`LpLoss`)
- **Outputs:**
  - Model checkpoint: `experiments/<param>/checkpoints/model_64_30.pt`
  - Loss history: `experiments/<param>/checkpoints/loss_64_30.npy`
  - Validation images: `experiments/<param>/visualizations/`

### Inference

```bash
python src/inference.py --param <parameter_name>
```

- Loads model from `experiments/<param>/checkpoints/model_64_30.pt`
- Runs over 21 test files, saves predictions to `experiments/<param>/visualizations/pred_<j>.npy`

### Linting

- No explicit linting specified; code follows PEP8 conventions
- Use `pylint` or `flake8` for code quality checks

## Data Pipeline

### Input Data Structure

- **Training:** `input_data/<param>/train/[x|y]_<idx>.npy`, indices 0-89 (90 files)
- **Test:** `input_data/<param>/test/[x|y]_<idx>.npy`, indices 0-20 (21 files)
- **Input shape:** `(20, 128, 128, 10, 7)` -- 20 samples, 128x128 grid, 10 frames, 7 channels
- **Output shape:** `(20, 128, 128, 10)` -- 10 predicted frames

### Data Generation

Raw data comes from FARGO3D simulations of the Orszag-Tang vortex:

- 50 simulations, each with 1,000 timesteps
- Parameters: nu = mu sampled in [1e-5, 5e-2]
- Test cases (held out): nu = mu = 5e-5 and nu = mu = 3e-4
- FARGO3D outputs binary `.dat` files (16,384 values = 128x128 per field)
- A preprocessing script (**location TBD**) converts `.dat` to `.npy` format, assembling sliding-window input/output blocks with nu and mu appended as the last 2 channels

### Normalization

- Input frames (first 5 channels): min-max scaled to [-1, 1]
- Physical parameters nu and mu: **not normalized**
- Target `y`: min-max scaled to [-1, 1] during training

## Architecture

**Model:** `FNO3d(64, 64, 5, 30)` -- 64 spatial Fourier modes, 5 temporal modes, width 30

- **Input:** `(batch, 128, 128, 10, 7)` -> 5 frames + 2 physical params
- **Internal:** `get_grid()` appends 3 coordinate channels -> 10 channels, lifted to width 30
- **Output:** `(batch, 128, 128, 10)` -- 10 predicted timesteps
- **Layers:** 5 Fourier layers (SpectralConv3d + 1x1x1 Conv), GELU activation
- **Time padding:** 6 (non-periodic temporal boundaries)

**SpectralConv3d:** 3D real FFT -> complex multiplication (4 weight tensors for x/y octants) -> inverse FFT

**File locations:**
- Model definitions: `src/architecture.py` (used by `inference.py`)
- Training copy: `src/train.py` (contains duplicate definitions -- must stay in sync)

## Code Style Guidelines

### Imports

- All imports at top of file
- Standard library -> third-party -> local imports
- Use `import torch` and `import torch.nn as nn` for PyTorch

### Formatting

- Follow PEP8 conventions
- 4 spaces for indentation (no tabs)
- Maximum line length of 79 characters
- Use descriptive variable names (e.g., `batch_size` not `bs`)

### Naming Conventions

- Class names: PascalCase (e.g., `FNO3d`, `SpectralConv3d`)
- Function names: snake_case (e.g., `get_grid`, `unormalize`)
- Constants: UPPER_CASE (e.g., `LEARNING_RATE = 0.001`)

### Code Structure

- `src/architecture.py` -- model definitions
- `src/train.py` -- training logic
- `src/inference.py` -- inference logic
- `src/utilities.py` -- loss functions and normalizers
- `src/Adam.py` -- custom Adam optimizer

## Known Issues

1. **Duplicate model definitions:** `FNO3d` and `SpectralConv3d` are defined in both `architecture.py` and `train.py`. Changes must be applied to both files.

2. **Import mismatch:** `train.py` imports `from utilities3 import *` but the file is `utilities.py`. Either rename the file or fix the import.

3. **CUDA required:** Code uses `.cuda()` throughout with no CPU fallback. Training will fail without a GPU.

4. **Unused BatchNorm layers:** `bn0`-`bn3` are defined in `FNO3d.__init__` but never called in `forward()`.

5. **Preprocessing script missing:** The script that converts FARGO3D `.dat` outputs to `.npy` format is not in this repository (location TBD).

## Path Conventions

All paths are relative to the project root:

- **Data:** `input_data/<param>/[train|test]/`
- **Checkpoints:** `experiments/<param>/checkpoints/`
- **Visualizations:** `experiments/<param>/visualizations/`
