# Fourier Neural Operator for Magnetized Plasmas

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

This project implements a 3D Fourier Neural Operator (FNO) surrogate for 2D magnetohydrodynamic (MHD) turbulence. The model is trained on the **Orszag–Tang vortex**, a standard MHD benchmark, simulated with the [FARGO3D](https://fargo3d.bitbucket.io) code across an ensemble of viscosities and magnetic diffusivities. It learns to map a short initial window of simulation frames to future states for physical quantities such as density, velocity, and magnetic field components.

---
**REPOSITORY UNDER CONSTRUCTION**
We are working to make this repository useful and inference-ready, including a Docker image and train/test data. For the time being, check out our paper: *Spectral Learning of Magnetized Plasma Dynamics: A Neural Operator Application. [arXiv:2507.01388](https://arxiv.org/abs/2507.01388)*.

---

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

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd fno

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib scipy

# Verify PyTorch CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

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

### SpectralConv3d

Performs operations in Fourier space:
1. **FFT**: Transform to frequency domain using `torch.fft.rfftn`
2. **Complex multiplication**: 4 learned weight tensors multiply Fourier coefficients across x/y octants
3. **Inverse FFT**: Transform back to physical space using `torch.fft.irfftn`
4. **Parallel path**: 1×1×1 convolution acts as skip connection

## Usage

### Training a Model

```bash
cd src
python train.py --param <parameter_name>
```

**Available parameters:**
- `density` - Gas density
- `vy` - Gas velocity (y-component)
- `vz` - Gas velocity (z-component)
- `by` - Magnetic field (y-component)
- `bz` - Magnetic field (z-component)
- `br` - Magnetic field (radial component)

**Example:**
```bash
cd src
python train.py --param density
```

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

### Running Inference

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

## Physical Parameters

| Parameter | Description | Physical Quantity |
|-----------|-------------|-------------------|
| `density` | Gas density | ρ |
| `vy` | Gas velocity Y | v_y component |
| `vz` | Gas velocity Z | v_z component |
| `by` | Magnetic field Y | B_y component |
| `bz` | Magnetic field Z | B_z component |
| `br` | Magnetic field radial | B_r component |

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
input_data/
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

### Updating Data Paths

Data and results paths are currently hardcoded in both scripts. To adapt to your environment, update the `os.path.join(home_dir, ...)` calls in:

- `src/train.py`: `data()` function (input) and `unormalize()` function (for denormalization)
- `src/inference.py`: input data path and results output path

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

## Troubleshooting

### Common Issues

**CUDA out of memory:**
- Reduce the slice size in the inner training loop in `src/train.py`
- Use a smaller model width or fewer Fourier modes

**File not found errors:**
- Update the hardcoded paths in `src/train.py` and `src/inference.py` to match your data location
- Ensure data files follow the expected naming convention: `x_<idx>.npy`, `y_<idx>.npy`

**Import errors on `utilities3`:**
- `src/train.py` imports `from utilities3 import *` but the file is `src/utilities.py` — rename or update the import before running training

**Model not converging:**
- Check that normalization is applied correctly (physical parameter channels must not be normalized)
- Verify input data quality and range

## References

- Duarte, Nemmen & Lima-Santos (2025). Spectral Learning of Magnetized Plasma Dynamics: A Neural Operator Application. [*arXiv:2507.01388*](https://arxiv.org/abs/2507.01388)
- Li et al. (2020). Fourier Neural Operator for Parametric Partial Differential Equations. [*arXiv:2010.08895*](https://arxiv.org/abs/2010.08895)

---

## TODO

- [x] make repo public
- [ ] figures, movie and dataset
- [ ] inference guide w/ test dataset
- [ ] include link to original sim. data and conversion script
- [ ] reproducibility: include Docker image
