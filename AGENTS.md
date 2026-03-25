# AGENTS.md

This file contains guidelines for agentic coding agents working in this repository.

## Overview

This is a PyTorch implementation of a 3D Fourier Neural Operator (FNO) for simulating 2D magnetohydrodynamic (MHD) turbulence. The specific problem is the **Orszag-Tang vortex**, a standard MHD benchmark. The model learns to predict future plasma dynamics (density, velocity, magnetic fields) from initial simulation frames and physical parameters (kinematic viscosity nu, Ohmic diffusivity eta).

The paper associated with this codebase is `paper/main.tex`. Detailed documentation is in `CLAUDE.md`. Purpose of current branch can be found in `THIS_BRANCH.md`.

## Training and Inference

### Setup

- Requires Python 3.8+ with PyTorch 1.10+
- Install dependencies: `pip install torch numpy matplotlib scipy`
- GPU required (CUDA, no CPU fallback)

### Training

```bash
python src/train.py --param <parameter_name> [--experiments-dir <path>]
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
python src/inference.py --param <parameter_name> [--experiments-dir <path>]
```

- Loads model from `experiments/<param>/checkpoints/model_64_30.pt`
- Runs over all available test files (counted dynamically), saves predictions to `experiments/<param>/visualizations/pred_<j>.npy`
- **Not autoregressive (teacher-forced):** each test file contains pre-assembled ground-truth windows. The model runs one forward pass per window; predictions are never fed back as inputs. Reported metrics reflect teacher-forced performance, not free-running rollout.

### Visualization

```bash
python src/visualize_results.py --param <parameter_name> [--experiments-dir <path>]
```

- Loads `pred_<j>.npy` files saved by `inference.py` (does **not** re-run the model)
- For each test file `j`, using sample index 0:
  - 10 PNGs: `sample_{j:02d}_time_{t:02d}.png` — 3-panel (target | prediction | error)
  - 1 GIF: `sample_{j:02d}_evolution.gif` — animation across all 10 timesteps
- Outputs saved to `experiments/<param>/visualizations/`

### Linting

- No explicit linting specified; code follows PEP8 conventions
- Use `pylint` or `flake8` for code quality checks

## Data Pipeline

### Input Data Structure

- **Training:** `data/<param>/train/[x|y]_<idx>.npy` (count determined dynamically)
- **Test:** `data/<param>/test/[x|y]_<idx>.npy` (count determined dynamically)
- For FARGO3D data, `<param>` includes the solver prefix, e.g. `fargo3d/density`
- **Input shape:** `(20, 128, 128, 10, 7)` -- 20 samples, 128x128 grid, 10 frames, 7 channels
- **Output shape:** `(20, 128, 128, 10)` -- 10 predicted frames

### Data Generation

Two numerical solvers have been used:

**Idefix pipeline (`data/idefix/`)** — current:

1. `python generate_params.py [--seed 42] [--nsims 25]` → `params.csv` (sim_id, nu, mu, split)
   - 2 hardcoded test cases (nu=mu=5e-5 and nu=mu=3e-4), rest random log-uniform in [1e-5, 5e-2]
2. `bash build.sh` — builds Idefix binary (requires `$IDEFIX_DIR`, CUDA, MHD enabled)
3. `python run_simulations.py [--params params.csv] [--gpus 0,1]` — runs sims in parallel across GPUs
   - Each sim produces ~1001 VTK files in `runs/sim_XXX/`, dt=0.05, tstop=50
4. `python convert_to_npy.py [--runs-dir runs] [--output-dir ../../data]` → `data/<param>/[train|test]/[x|y]_<idx>.npy`
   - Requires `pip install idefix-pytools`
   - Fields: RHO→density, VX1→vx, VX2→vy, BX1→bx, BX2→by
   - 20 sliding windows per sim; input: 5 frames spaced 20 apart; output: 10 frames spaced 80 apart from frame 160
   - nu and mu appended as channels 5–6
   - **File count note:** 25 sims → 23 train + 2 test files per field; `train.py` and `inference.py` count files dynamically, so no code changes needed

**FARGO3D** — original dataset:
- 50 simulations, 1,000 timesteps each, domain [0, 2π]², nu=mu sampled in [1e-5, 5e-2]
- Binary `.dat` output (16,384 values = 128×128 per field), same sliding-window conversion to `.npy`
- Preprocessed data at `data/fargo3d/<param>/[train|test]/`; use e.g. `--param fargo3d/density`

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
- `src/utilities.py` -- loss functions, normalizers, and `DenseNet`
- `src/Adam.py` -- custom Adam optimizer
- `src/visualize_results.py` -- visualization of inference results
- `src/architecture_diagram.py` -- architecture diagram generation

## Known Issues

1. **Duplicate model definitions:** `FNO3d` and `SpectralConv3d` are defined in both `architecture.py` and `train.py`. Changes must be applied to both files.

2. **Import mismatch:** `train.py` imports `from utilities3 import *` but the file is `utilities.py`. Either rename the file or fix the import.

3. **CUDA required:** Code uses `.cuda()` throughout with no CPU fallback. Training will fail without a GPU.

4. **Unused BatchNorm layers:** `bn0`-`bn3` are defined in `FNO3d.__init__` but never called in `forward()`.

5. **Preprocessing script:** `data/idefix/convert_to_npy.py` handles VTK → `.npy` conversion for Idefix output. For FARGO3D `.dat` files, an equivalent conversion script is not in this repository.

6. **No autoregressive rollout:** inference is teacher-forced — the model always receives ground-truth frames as input, never its own predictions. Implementing free-running rollout would require a loop in `inference.py` that slides the input window forward using predicted frames.

## Path Conventions

All paths are relative to the project root:

- **Data:** `data/<param>/[train|test]/`
- **Checkpoints:** `experiments/<param>/checkpoints/`
- **Visualizations:** `experiments/<param>/visualizations/`
