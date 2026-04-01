# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a PyTorch implementation of a 3D Fourier Neural Operator (FNO) for simulating 2D magnetohydrodynamic (MHD) turbulence. The specific problem is the **Orszag–Tang vortex**, a standard MHD benchmark. The model learns to map a short initial window of simulation frames (plus physical parameters) to future states for physical quantities such as density, velocity components, and magnetic field components.

Two numerical solvers have been used to generate training data:
- **FARGO3D** (finite-difference, GPU-accelerated) — original dataset, domain [0, 2π]², 128×128 grid
- **Idefix** (Godunov finite-volume, GPU-accelerated) — newer dataset, domain [0, 1]², 128×128 grid

The paper associated with this codebase is `paper/main.tex`.

Purpose of current branch can be found in `THIS_BRANCH.md`.

## Training and Inference

**Training a model:**
```bash
cd src
python train.py --param <parameter_name> [--experiments-dir <path>] [--fast] [--batch-size N] [--patience N] [--resume]
```
- `--param` selects which physical field to train on (e.g. `density`, `vy`, `by`; defaults to `density`)
- Trains for 5,000 epochs with learning rate 0.001 and StepLR scheduling (step=500, γ=0.5)
- `--batch-size N` (default 16): samples per gradient update
- `--patience N` (default 500): early stopping — stops if val MAE does not improve by >1e-4 for N consecutive epochs; 0 disables early stopping
- `--resume`: resumes from `training_state.pt` checkpoint; requires loss definition version to match `'normalized-mae-only-v1'`
- Legacy layout writes checkpoints to `experiments/<param>/checkpoints/` and validation images to `experiments/<param>/visualizations/`
- Run-scoped layout writes artifacts to `experiments/<run>/params/<param>/`, validation images to `experiments/<run>/params/<param>/renders/`, and metadata to `experiments/<run>/manifest.json`
- `--fast` runs a short smoke test (3 epochs, batch_size=4, 1 train/val file, 4 samples) for quick training/display checks
- Checkpoints created before the MAE-only refactor are intentionally incompatible with `--resume`
- Uses automatic mixed precision (bfloat16 if supported, float16 otherwise; float16 disabled for complex-parameter layers)

**Running inference:**
```bash
cd src
python inference.py --param <parameter_name> [--experiments-dir <path>] [--rollout-steps N] [--checkpoint <path>]
```
- Resolves checkpoints from the parameter artifact directory under `--experiments-dir`; prefers `model_best.pt` if present, falls back to `model_64_30.pt`
- `--checkpoint <path>`: override checkpoint path explicitly
- Saves denormalized predictions as `pred_sim_<id>.npy` in that same resolved artifact directory
- Under a run-scoped root, short leaf params like `by` and `bx` are supported; legacy nested experiment trees remain supported in place
- `--rollout-steps N` (default 1): runs autoregressive rollout by predicting 20 frames, feeding the last 5 predicted frames back as input, and repeating N times total, producing 20×N frames. Saves as `pred_sim_<id>.npy` with shape `(1, 128, 128, 20*N)`

**Visualizing scalar results** (run after inference):
```bash
cd src
python viz_scalar.py --param <parameter_name> [--experiments-dir <path>] [--force] [--max-frames N] [--workers N]
```
- Loads `pred_sim_*.npy` files saved by `inference.py` (does **not** re-run the model); also loads `ref_sim_<id>.npy` dense reference trajectories if present (produced by `prepare_reference.py`), falling back to `y_sim_<id>.npy`
- `--force`: regenerate existing PNGs and movies
- `--max-frames N`: limit frames rendered per simulation
- `--workers N`: parallel PNG rendering via `ProcessPoolExecutor`
- Writes scalar renders to the resolved render directory
  - legacy layout: `experiments/<param>/visualizations/`
  - run-scoped layout: `experiments/<run>/params/<param>/renders/`
- Generates MP4 movies via `ffmpeg` concat after rendering frames

**Visualizing vector results** (run after inference):
```bash
cd src
python viz_vector.py --param <parameter_name> [--experiments-dir <path>] [--force] [--max-frames N] [--workers N] [--quiver-stride N]
```
- Loads paired component predictions such as `bx` + `by` or `vx` + `vy`
- `--quiver-stride N` (default 8): subsampling stride for quiver arrows
- `--force`, `--max-frames`, `--workers`: same as `viz_scalar.py`
- Writes vector renders (magnitude colormaps with quiver arrows) to:
  - legacy layout: nested family directories such as `.../magnetic_vector_visualizations/`
  - run-scoped layout: `experiments/<run>/vector/<family>/`
- Generates MP4 movies via `ffmpeg` concat after rendering frames

**Note on paths:** Data is read from `data/<param>/[train|val|test]/`. Experiment artifacts support two layouts: the original legacy layout under `experiments/<param>/...` and a newer run-scoped layout under `experiments/<run>/...`. Path resolution is centralized in `src/experiment_layout.py`, and scripts still resolve repository-relative defaults via `__file__`, so they can be run from any working directory.

## Architecture Details

**FNO3d Model (`src/architecture.py`):**
- Instantiated as `FNO3d(64, 64, 5, 30)` — modes_x=64, modes_y=64, modes_t=5, width=30
- Defined once in `src/architecture.py`; imported by both `train.py` and `inference.py`
- **Input to `forward()`:** `(batch, 128, 128, T_in, 7)` where T_in=20 and the 7 channels are:
  - 5 input simulation frames (temporal snapshots of one field)
  - 2 physical parameters: kinematic viscosity ν and Ohmic diffusivity μ/η
- Internally, `get_grid()` appends 3 coordinate channels (x, y, t grid) → 10 channels total, then lifted by `fc0 = Linear(10, width)`; grid is cached via `register_buffer`
- **Output:** `(batch, 128, 128, 20)` — 20 predicted future timesteps
- 5 Fourier layers (conv0–conv4) each running in parallel with a 1×1×1 convolution (w0–w4), summed and passed through GELU activation
- Padding of 6 applied in the time dimension to handle non-periodic temporal boundaries
- Projection: `fc1 = Linear(width, 128)` → GELU → `fc2 = Linear(128, 1)`

**SpectralConv3d:**
- Performs 3D real FFT → complex multiplication in frequency domain (4 weight tensors for the four x/y octants) → inverse FFT
- Complex multiplication implemented via `torch.einsum`
- `forward()` decorated with `@torch.amp.autocast('cuda', enabled=False)` — always casts to float32 before FFT

**Custom Adam Optimizer (`src/Adam.py`):**
- Modified PyTorch Adam implementation with explicit functional API
- **Note:** `train.py` currently uses `torch.optim.Adam` (standard PyTorch) with `weight_decay=1e-4`; `src/Adam.py` is not used by the training pipeline

## Data Pipeline

**Input data structure:**
- Training data: `data/<param>/train/x_sim_<id>.npy` / `y_sim_<id>.npy` (count determined dynamically)
- Validation data: `data/<param>/val/x_sim_<id>.npy` / `y_sim_<id>.npy` (count determined dynamically)
- Test data: `data/<param>/test/x_sim_<id>.npy` / `y_sim_<id>.npy` (count determined dynamically; used only for final evaluation, never during training)
- For the FARGO3D dataset, `<param>` includes the solver prefix, e.g. `fargo3d/density`
- Each `x` file has shape `(20, 128, 128, 20, 7)` — 20 samples, 128×128 spatial grid, 20 temporal frames, 7 channels
- Each `y` file has shape `(20, 128, 128, 20)` — same samples, 20 output frames

**Batch construction:**
- All training and validation files are pre-loaded into memory by `load_dataset()` at startup
- Training iterates over all available files; within each file, samples are sliced in batches of `--batch-size` (default 16), giving `ceil(20/batch_size)` gradient steps per file
- Effective default batch size per gradient update: **16 samples**

**Normalization (`train.py` and `inference.py`):**
- Input frames (first 5 channels, i.e. `x[:,:,:,:,:-2]`): min-max scaled to [-1, 1] per file at load time
- Physical parameters ν and μ (last 2 channels): **not normalized**, passed through as-is
- Target `y`: min-max scaled to [-1, 1] per file at load time
- Denormalization inverts the y scaling using the stored per-file min/max

## Loss Functions

Training uses one loss:
1. **MAE** (`F.l1_loss`) on normalized predictions

Validation MAE is also used for early stopping and best-model selection.

**`src/utilities.py` provides:**
- `LpLoss`: Relative/absolute Lp norm loss (available but not used in current training loop)

## Model I/O

**Saving/loading:**
- `model_64_30.pt`: periodic model state dict (saved every 100 epochs and at end of training)
- `model_best.pt`: best model state dict by validation MAE (used preferentially by `inference.py`)
- `training_state.pt`: full checkpoint (epoch, model/optimizer/scheduler/scaler state dicts, loss history, early stopping state) — used by `--resume`
- `loss_64_30.npy`: loss history array with columns `(train_mae, val_mae)` per epoch
- Must instantiate `FNO3d(64, 64, 5, 30)` before loading a state dict

## Data Generation

### Idefix pipeline (`data/idefix/`)

The current data generation pipeline uses Idefix. All scripts live in `data/idefix/`.

**Step 1 — generate parameter table:**
```bash
python generate_params.py [--seed 42] [--nsims 50] [--nval 6]
```
- Writes `params.csv` with columns `sim_id, nu, mu, split`
- 2 hardcoded test cases (ν=μ=5×10⁻⁵ and ν=μ=3×10⁻⁴), `--nval` (default 6) randomly chosen for val, remainder train
- Current dataset: 50 simulations → 42 train + 6 val + 2 test

**Step 2 — build Idefix binary** (once):
```bash
# Requires $IDEFIX_DIR set to Idefix source tree
bash build.sh
```
Builds with MHD + CUDA enabled (Pascal sm_60 arch).

**Step 3 — run simulations:**
```bash
python run_simulations.py [--params params.csv] [--start N] [--end N] [--gpus 0,1]
```
- Creates `runs/sim_XXX/` per simulation, patches `idefix.ini` with ν and μ, runs Idefix
- Dispatches simulations in parallel across GPUs (round-robin)
- Each simulation produces ~1001 VTK files (`data.0000.vtk` … `data.1000.vtk`), dt=0.05, tstop=50

**Step 4 — convert VTK to `.npy`:**
```bash
python convert_to_npy.py [--runs-dir runs] [--params params.csv] [--output-dir ../../data]
```
- Reads VTK files via `idefix-pytools` (`pip install idefix-pytools`)
- Fields extracted: `RHO→density`, `VX1→vx`, `VX2→vy`, `BX1→bx`, `BX2→by`
- Builds 20 evenly-spaced sliding-window samples per simulation (stride 51 frames between window starts)
  - Input: 5 consecutive frames starting at `start`
  - Output: 20 consecutive frames immediately after input (frames start+5 .. start+24)
- Appends ν and μ as channels 5–6 of `x`, broadcast over (128, 128, 20)
- Writes `data/<param>/[train|val|test]/x_sim_<id>.npy` and `y_sim_<id>.npy`

**Note on file counts:** with 50 simulations (42 train + 6 val + 2 test), `convert_to_npy.py` produces the corresponding files per field. Both `train.py` and `inference.py` dynamically count available files, so no code changes are needed when the number of simulations changes.

### FARGO3D (original dataset)

**Raw output:** binary `.dat` files, one per field per timestep; flat vector of 16,384 values (128×128). 50 simulations total, each 1,000 timesteps, ν=μ sampled in [10⁻⁵, 5×10⁻²]. 48 train, 2 test.

**Conversion to `.npy`:** same sliding-window logic as the Idefix pipeline above. Some of the files are located in `data/fargo3d`, with notes about their preprocessed structure in `data/fargo3d/README.md`.

**FARGO3D data path:** preprocessed `.npy` files live under `data/fargo3d/<param>/[train|test]/`. To train on FARGO3D data pass e.g. `--param fargo3d/density`.

## Physical Context

- **Problem:** 2D Orszag–Tang vortex on a 128×128 periodic spatial grid
- **Numerical solvers:** FARGO3D (domain [0, 2π]²) and Idefix (domain [0, 1]², HLLD solver, UCT contact EMF, γ=5/3)
- **Key parameters:** kinematic viscosity ν and Ohmic diffusivity η (referred to as μ in some parts of the code), sampled in [10⁻⁵, 5×10⁻²]
- **Training regime:** model is conditioned on 5 consecutive input frames and predicts the next 20 consecutive frames (short-horizon, fine-grained temporal prediction)
- **Test cases (held out):** ν = μ = 5×10⁻⁵ and ν = μ = 3×10⁻⁴

## Key Implementation Notes

- The model operates on CUDA by default (`.cuda()` calls throughout).
- **Inference is autoregressive-only:** `--rollout-steps N` controls how many chained 20-frame rollout segments are generated. The last 5 of each step's 20 predicted frames are fed back as input for the next step, so autoregressive error compounds across steps.
- Batch normalization layers (`bn0`–`bn3`) are defined in `FNO3d.__init__` but never called in `forward()`
- The time dimension is padded by 6 before the Fourier layers and unpadded after (`x[..., :-self.padding]`)
- Spatial dimensions are not padded (the Orszag–Tang problem has periodic spatial boundaries)
- `architecture.py` is the single canonical definition of `FNO3d` and `SpectralConv3d`; both `train.py` and `inference.py` import from it
- Mixed precision: `train.py` uses `torch.amp.autocast` (bf16 preferred, fp16 fallback) with `GradScaler` for fp16; `SpectralConv3d.forward` opts out of autocast via decorator
- `src/architecture_diagram.py`: standalone matplotlib script to generate an architecture diagram (not part of training/inference pipeline)
- `src/prepare_reference.py`: converts raw Idefix VTK output to dense `ref_sim_<id>.npy` reference trajectories for visualization; used by `viz_scalar.py` when present
- `src/unify_legacy_magnetic_run.py`: one-off migration utility that copies legacy `bx`/`by` experiment artifacts into a unified run-scoped root
- `data/idefix/render_vtk.py`: renders 3-panel (density, |B|, |v|) images from VTK snapshots and assembles an MP4; independent of the training pipeline
- Idefix simulation source files (`setup.cpp`, `definitions.hpp`, `idefix.ini`, `Makefile`) live in `data/idefix/` alongside the pipeline scripts

## Dependencies

- **Python packages:** `torch`, `numpy`, `matplotlib`, `scipy`, `tqdm`, `asciichartpy`
- **Optional:** `idefix-pytools` (for `data/idefix/convert_to_npy.py` and `prepare_reference.py`)
- **System:** `ffmpeg` (for MP4 movie generation in `viz_scalar.py` and `viz_vector.py`)
