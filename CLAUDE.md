# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a PyTorch implementation of a 3D Fourier Neural Operator (FNO) for simulating 2D magnetohydrodynamic (MHD) turbulence. The specific problem is the **Orszag–Tang vortex**, a standard MHD benchmark solved with the FARGO3D finite-difference code. The model learns to map a short initial window of simulation frames (plus physical parameters) to future states for physical quantities such as density, velocity components, and magnetic field components.

The paper associated with this codebase is `paper/main.tex`.

## Training and Inference

**Training a model:**
```bash
cd src
python train.py --param <parameter_name>
```
- `--param` selects which physical field to train on (e.g. `density`, `vy`, `by`; defaults to `density`)
- Trains for 10,000 epochs with learning rate 0.001 and StepLR scheduling (step=500, γ=0.5)
- Saves model checkpoint to `experiments/<param>/checkpoints/model_64_30.pt`
- Saves loss history to `experiments/<param>/checkpoints/loss_64_30.npy`
- Saves one validation prediction image per epoch to `experiments/<param>/visualizations/`

**Running inference:**
```bash
cd src
python inference.py --param <parameter_name>
```
- Loads model from `experiments/<param>/checkpoints/model_64_30.pt`
- Runs over 21 test files and saves denormalized predictions as `.npy` arrays to
  `experiments/<param>/visualizations/pred_<j>.npy`

**Note on paths:** Data is read from `input_data/<param>/[train|test]/` and results are written to `experiments/<param>/`.

## Architecture Details

**FNO3d Model (`src/architecture.py` and duplicated in `src/train.py`):**
- Instantiated as `FNO3d(64, 64, 5, 30)` — modes_x=64, modes_y=64, modes_t=5, width=30
- **Input to `forward()`:** `(batch, 128, 128, T_in, 7)` where T_in=10 and the 7 channels are:
  - 5 input simulation frames (temporal snapshots of one field)
  - 2 physical parameters: kinematic viscosity ν and Ohmic diffusivity μ/η
- Internally, `get_grid()` appends 3 coordinate channels (x, y, t grid) → 10 channels total, then lifted by `fc0 = Linear(10, width)`
- **Output:** `(batch, 128, 128, 10)` — 10 predicted future timesteps
- 5 Fourier layers (conv0–conv4) each running in parallel with a 1×1×1 convolution (w0–w4), summed and passed through GELU activation
- Padding of 6 applied in the time dimension to handle non-periodic temporal boundaries
- Projection: `fc1 = Linear(width, 128)` → GELU → `fc2 = Linear(128, 1)`

**SpectralConv3d:**
- Performs 3D real FFT → complex multiplication in frequency domain (4 weight tensors for the four x/y octants) → inverse FFT
- Complex multiplication implemented via `torch.einsum`

**Note:** `FNO3d` and `SpectralConv3d` are defined twice — once in `src/architecture.py` (used by `inference.py`) and once inline in `src/train.py`. These should be kept in sync.

**Custom Adam Optimizer (`src/Adam.py`):**
- Modified PyTorch Adam implementation with explicit functional API
- Applied with `weight_decay=1e-4` in training

## Data Pipeline

**Input data structure:**
- Training data: `input_data/<param>/train/[x|y]_<idx>.npy`, indices 0–89 (90 files)
- Test data: `input_data/<param>/test/[x|y]_<idx>.npy`, indices 0–20 (21 files)
- Each `x` file has shape `(20, 128, 128, 10, 7)` — 20 samples, 128×128 spatial grid, 10 temporal frames, 7 channels
- Each `y` file has shape `(20, 128, 128, 10)` — same samples, 10 output frames

**Batch construction:**
- Training iterates over 90 files; within each file, samples are loaded in slices of 4 (`l:l+4` for l in 0, 4, 8, 12), giving 4 gradient steps per file
- Effective batch size per gradient update: **4 samples**

**Normalization (`train.py` and `inference.py`):**
- Input frames (first 5 channels, i.e. `x[:,:,:,:,:-2]`): min-max scaled to [-1, 1] per batch
- Physical parameters ν and μ (last 2 channels): **not normalized**, passed through as-is
- Target `y`: min-max scaled to [-1, 1] during training
- Denormalization in `unormalize()` inverts the y scaling using the original y min/max from training data

## Loss Functions

Training combines two losses:
1. **MAE** (`F.l1_loss`) on normalized predictions
2. **Relative L2** (`LpLoss`) on denormalized predictions: `‖pred − target‖₂ / ‖target‖₂`
- Combined as: `loss = mae + l2`

**`src/utilities.py` provides:**
- `LpLoss`: Relative/absolute Lp norm loss (used in training)
- `HsLoss`: Sobolev norm loss comparing derivatives in frequency domain
- `FrequencyLoss`: Compares log power spectra in Fourier space
- `UnitGaussianNormalizer`, `GaussianNormalizer`, `RangeNormalizer`: alternative normalizers (not used in current training loop)
- `MatReader`: utility for loading `.mat` files (legacy, not used for current `.npy` data)

**Known naming issue:** `train.py` imports via `from utilities3 import *` but the file is `src/utilities.py`. This must match for training to run.

## Model I/O

**Saving/loading:**
- Saved with `torch.save(model.state_dict(), path)`
- Loaded with `model.load_state_dict(torch.load(path))`
- Must instantiate `FNO3d(64, 64, 5, 30)` before loading
- Loss history is a 1D array alternating MAE and combined loss per epoch

## Data Generation

**Raw FARGO3D output:** each simulation produces binary `.dat` files, one per field component per timestep. Each file contains a flat vector of 16,384 values (= 128×128) that must be reshaped into a 2D array.

**FARGO3D setup:** the Orszag–Tang vortex is a built-in problem in FARGO3D. Viscosity ν and diffusivity μ are set via the parameter file. 50 simulations were run in total, each for 1,000 timesteps, sampling ν = μ across [10⁻⁵, 5×10⁻²]. 48 simulations are used for training/validation and 2 are held out for testing (ν = μ = 5×10⁻⁵ and ν = μ = 3×10⁻⁴).

**Conversion to `.npy`:** a preprocessing script (**location TBD**) reads the binary FARGO3D outputs, reshapes them to 128×128, assembles the sliding-window input/output blocks, appends ν and μ as the last 2 channels of `x`, and writes the paired `x_<idx>.npy` / `y_<idx>.npy` files consumed by `train.py`. The resulting shapes are `(20, 128, 128, 10, 7)` for `x` and `(20, 128, 128, 10)` for `y`.

## Physical Context

- **Problem:** 2D Orszag–Tang vortex on a 128×128 periodic spatial grid, domain [0, 2π] × [0, 2π]
- **Numerical solver:** FARGO3D (finite-difference, GPU-accelerated, constrained transport for ∇·B=0)
- **Key parameters:** kinematic viscosity ν and Ohmic diffusivity η (referred to as μ in some parts of the code), sampled in [10⁻⁵, 5×10⁻²]
- **Training regime:** model is conditioned on the first ~160 frames (≈ 0.73 Alfvén times) and predicts frames 160–1000 (≈ 0.73–4.39 Alfvén times)
- **Test cases (held out):** ν = μ = 5×10⁻⁵ and ν = μ = 3×10⁻⁴

## Key Implementation Notes

- The model operates on CUDA by default (`.cuda()` calls throughout); no CPU fallback in training
- **Inference is teacher-forced, not autoregressive:** at inference time every prediction window receives ground-truth FARGO3D frames as input — the model's own predictions are never fed back in. The 21 test files contain pre-assembled windows that already cover all temporal segments; the model runs one forward pass per window independently. Benchmarked performance therefore reflects teacher-forced evaluation and will degrade if predictions were fed back as inputs (autoregressive rollout).
- Batch normalization layers (`bn0`–`bn3`) are defined in `FNO3d.__init__` but never called in `forward()`
- The time dimension is padded by 6 before the Fourier layers and unpadded after (`x[..., :-self.padding]`)
- Spatial dimensions are not padded (the Orszag–Tang problem has periodic spatial boundaries)
- `architecture.py` and `train.py` contain duplicate definitions of `FNO3d` and `SpectralConv3d`; `inference.py` imports from `architecture.py`
