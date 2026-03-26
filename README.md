# Fourier Neural Operator for Magnetized Plasmas

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

This project implements a 3D Fourier Neural Operator (FNO) surrogate for 2D magnetohydrodynamic (MHD) turbulence. The model is trained on the **Orszag–Tang vortex**, a standard MHD benchmark, simulated with the [FARGO3D](https://fargo3d.bitbucket.io) code across an ensemble of viscosities and magnetic diffusivities. It learns to map a short initial window of simulation frames to future states for physical quantities such as density, velocity, and magnetic field components.

---
**REPOSITORY UNDER CONSTRUCTION**
We are working to make this repository useful and inference-ready, including a Docker image and train/test data. For the time being, check out our paper: *Spectral Learning of Magnetized Plasma Dynamics: A Neural Operator Application. [arXiv:2507.01388](https://arxiv.org/abs/2507.01388)*.

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
Input: (batch, 128, 128, 10, 7)
       └─ 128×128 spatial grid, 10 temporal frames, 7 channels
          (5 input snapshots + viscosity ν + diffusivity μ)
  ↓ [Append x/y/t grid coordinates → 10 channels total]
  ↓ [Linear projection (10 → width=30) + permute]
  ↓ [5× Fourier Layer with GELU activation]
  ↓ [Unpad + permute]
  ↓ [Linear projection (30 → 128 → 1)]
Output: (batch, 128, 128, 10, 1) → next 10 timesteps
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
python train.py --param <parameter_name> [--fast]
```

**Available parameters (Idefix dataset):**
- `density` - Gas density
- `vx` - Gas velocity (x-component)
- `vy` - Gas velocity (y-component)
- `bx` - Magnetic field (x-component)
- `by` - Magnetic field (y-component)

*FARGO3D dataset additionally includes `vz`, `bz`, `br`; pass e.g. `--param fargo3d/density`.*

**Example:**
```bash
cd src
python train.py --param density
```

Use `--fast` for a short smoke test that trains on a tiny subset of the data and saves a validation image every epoch.

**Training configuration:**
- Epochs: 10,000
- Batch size: 4 (per gradient step)
- Optimizer: Custom Adam with `weight_decay=1e-4`
- Learning rate: 0.001 with StepLR scheduler (`step_size=500`, `gamma=0.5`)
- Loss: Combined MAE (L1) + Relative L2 loss

**Output** (paths are relative to project root):
- Model checkpoint: `experiments/<param>/checkpoints/model_64_30.pt`
- Loss history: `experiments/<param>/checkpoints/loss_64_30.npy`
- Validation images: `experiments/<param>/visualizations/` (one per epoch)

### Inference

```bash
cd src
python inference.py --param <parameter_name>
```

Processes 21 test samples and saves denormalized predictions as `.npy` arrays.

**Example:**
```bash
cd src
python inference.py --param density
```

---
**NOTE ABOUT FORECASTING**  
Each of the 21 test files contains pre-assembled ground-truth windows covering a different temporal segment of the FARGO3D simulation. At inference time the model receives real simulation frames as input for every window; *its own predictions are never fed back in*. This means the reported metrics reflect teacher-forced performance, which is typically much better than free-running (autoregressive) inference where prediction errors would accumulate over time. 

---


## Training Details

### Loss Functions

The model uses a composite loss (defined in `src/utilities.py`):

1. **MAE Loss (L1)**: Mean absolute error on normalized predictions
2. **Relative L2 Loss (LpLoss)**: `‖prediction − target‖₂ / ‖target‖₂`
3. **Combined**: `loss = mae + l2`

### Normalization

- **Input snapshots** (first 5 of 7 channels): min-max scaled to [-1, 1] per batch
- **Physical parameters** ν and μ (last 2 channels): passed through as-is, not normalized
- **Target (y)**: min-max scaled to [-1, 1] during training; denormalized for loss computation and saved predictions

### Data Loading

- Online loading: one batch file loaded at a time to manage GPU memory
- Training files: 90 | Test files: 21
- Batch size per gradient update: 4 samples

## Data Format

### Directory Structure

```
data/
├── density/
│   ├── train/
│   │   ├── x_0.npy   # Input: 5 frames + ν + μ
│   │   ├── y_0.npy   # Target: next 10 frames
│   │   └── ...       # x_1.npy, y_1.npy, ..., x_89.npy, y_89.npy
│   └── test/
│       ├── x_0.npy
│       ├── y_0.npy
│       └── ...       # up to x_20.npy, y_20.npy
├── vy/
├── by/
└── ...
```

### Array Shapes

- **Input files (`x_*.npy`)**: `(20, 128, 128, 10, 7)`
  - 20 samples per file, 128×128 spatial grid, 10 temporal frames
  - 7 channels: 5 input snapshots + viscosity ν + diffusivity μ

- **Target files (`y_*.npy`)**: `(20, 128, 128, 10)`
  - 20 samples per file, 128×128 spatial grid, 10 predicted frames

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
│   ├── utilities.py         # Loss functions: LpLoss, HsLoss, FrequencyLoss
│   ├── Adam.py              # Custom Adam optimizer
│   └── visualize_results.py # Result visualization utilities
└── experiments/             # Model checkpoints and visualizations
```

## References

- Duarte, Nemmen & Lima-Santos (2025). Spectral Learning of Magnetized Plasma Dynamics: A Neural Operator Application. [*arXiv:2507.01388*](https://arxiv.org/abs/2507.01388)
- Li et al. (2020). Fourier Neural Operator for Parametric Partial Differential Equations. [*arXiv:2010.08895*](https://arxiv.org/abs/2010.08895)

