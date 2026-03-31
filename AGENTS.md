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
python src/train.py --param <parameter_name> [--experiments-dir <path>] [--fast]
```

- **Parameters:** `density`, `vy`, `by` (defaults to `density`)
- **Epochs:** 5,000
- **Learning rate:** 0.001 with StepLR scheduler (step=500, gamma=0.5)
- **Batch size:** 16
- **Loss:** MAE (`F.l1_loss`) on normalized targets
- **Validation:** uses `data/<param>/val/` for model selection and early stopping; `test/` is never read during training
- `--fast` runs a tiny smoke test with fewer files, fewer samples, and more frequent visualizations
- **Outputs:**
  - **Legacy layout:** checkpoints in `experiments/<param>/checkpoints/`, validation images in `experiments/<param>/visualizations/`
  - **Run-scoped layout:** artifacts in `experiments/<run>/params/<param>/`, validation images in `experiments/<run>/params/<param>/renders/`
  - New run-scoped experiments also write `experiments/<run>/manifest.json`
- Checkpoints created before the MAE-only refactor are intentionally incompatible with `--resume`

### Inference

```bash
python src/inference.py --param <parameter_name> [--experiments-dir <path>] [--rollout-steps N]
```

- Resolves model artifacts from `--experiments-dir` and `--param`
  - **Legacy layout:** loads from `experiments/<param>/checkpoints/`
  - **Run-scoped layout:** loads from `experiments/<run>/params/<param>/`
- Runs over all available test files (counted dynamically), saves predictions to the resolved artifact directory as `pred_sim_<id>.npy`
- `--rollout-steps N` (default 1): autoregressive rollout feeds the last 5 of 20 predicted frames back as input and repeats N chained times, producing 20×N frames; saved as `pred_sim_<id>.npy` with shape `(1, 128, 128, 20*N)`.
- Under a run-scoped root, short leaf params like `by` and `bx` are supported; legacy nested experiment trees also remain supported in place

### Visualization

```bash
python src/viz_scalar.py --param <parameter_name> [--experiments-dir <path>]
python src/viz_vector.py --param <parameter_name> [--experiments-dir <path>]
```

- `viz_scalar.py` loads `pred_sim_*.npy` files saved by `inference.py` and renders scalar diagnostics without re-running the model
- `viz_vector.py` loads paired component predictions (e.g. `bx` + `by` or `vx` + `vy`) and renders vector diagnostics
- Scalar renders are written to the resolved render directory
  - **Legacy layout:** `experiments/<param>/visualizations/`
  - **Run-scoped layout:** `experiments/<run>/params/<param>/renders/`
- Vector renders are written to a shared family directory
  - **Legacy layout:** nested family directory such as `.../magnetic_vector_visualizations/`
  - **Run-scoped layout:** `experiments/<run>/vector/<family>/`

### Linting

- No explicit linting specified; code follows PEP8 conventions
- Use `pylint` or `flake8` for code quality checks

## Data Pipeline

### Input Data Structure

- **Training:** `data/<param>/train/[x|y]_<idx>.npy` (count determined dynamically)
- **Validation:** `data/<param>/val/[x|y]_<idx>.npy` (count determined dynamically)
- **Test:** `data/<param>/test/[x|y]_<idx>.npy` (count determined dynamically; used only for final evaluation)
- For FARGO3D data, `<param>` includes the solver prefix, e.g. `fargo3d/density`
- **Input shape:** `(20, 128, 128, 20, 7)` -- 20 samples, 128x128 grid, 20 frames, 7 channels
- **Output shape:** `(20, 128, 128, 20)` -- 20 predicted frames

### Data Generation

Two numerical solvers have been used:

**Idefix pipeline (`data/idefix/`)** — current:

1. `python generate_params.py [--seed 42] [--nsims 50] [--nval 6]` → `params.csv` (sim_id, nu, mu, split)
   - 2 hardcoded test cases (nu=mu=5e-5 and nu=mu=3e-4), 6 val (default), rest train
   - Current dataset: 50 simulations → 42 train + 6 val + 2 test
2. `bash build.sh` — builds Idefix binary (requires `$IDEFIX_DIR`, CUDA, MHD enabled)
3. `python run_simulations.py [--params params.csv] [--gpus 0,1]` — runs sims in parallel across GPUs
   - Each sim produces ~1001 VTK files in `runs/sim_XXX/`, dt=0.05, tstop=50
4. `python convert_to_npy.py [--runs-dir runs] [--output-dir ../../data]` → `data/<param>/[train|test]/[x|y]_<idx>.npy`
   - Requires `pip install idefix-pytools`
   - Fields: RHO→density, VX1→vx, VX2→vy, BX1→bx, BX2→by
   - 20 sliding windows per sim; input: 5 consecutive frames; output: 20 consecutive frames immediately after input
   - nu and mu appended as channels 5–6
   - **File count note:** Current Idefix setup has 50 sims → 42 train + 6 val + 2 test files per field; `train.py` and `inference.py` count files dynamically, so no code changes needed when the number of simulations changes

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

- **Input:** `(batch, 128, 128, 20, 7)` -> 5 frames + 2 physical params, T_in=20
- **Internal:** `get_grid()` appends 3 coordinate channels -> 10 channels, lifted to width 30
- **Output:** `(batch, 128, 128, 20)` -- 20 predicted timesteps
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
- `src/viz_scalar.py` -- scalar prediction visualization
- `src/viz_vector.py` -- vector prediction visualization
- `src/experiment_layout.py` -- shared legacy/run-scoped experiment path resolver
- `src/utilities.py` -- loss functions, normalizers, and `DenseNet`
- `src/Adam.py` -- custom Adam optimizer
- `src/architecture_diagram.py` -- architecture diagram generation

## Known Issues

1. **Duplicate model definitions:** `FNO3d` and `SpectralConv3d` are defined in both `architecture.py` and `train.py`. Changes must be applied to both files.

2. **Unused BatchNorm layers:** `bn0`-`bn3` are defined in `FNO3d.__init__` but never called in `forward()`.

3. **Preprocessing script:** `data/idefix/convert_to_npy.py` handles VTK → `.npy` conversion for Idefix output. For FARGO3D `.dat` files, an equivalent conversion script is not in this repository.

4. **Autoregressive error accumulation:** the `--rollout-steps N` flag in `inference.py` enables free-running rollout, and prediction errors compound at each chained step.

## Path Conventions

All paths are relative to the project root:

- **Data:** `data/<param>/[train|val|test]/`
- **Legacy experiments:** `experiments/<param>/checkpoints/` and `experiments/<param>/visualizations/`
- **Run-scoped experiments:** `experiments/<run>/manifest.json`, `experiments/<run>/params/<param>/`, and `experiments/<run>/vector/<family>/`
