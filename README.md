# Fourier Neural Operator for Magnetized Plasmas

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

This project implements a 3D Fourier Neural Operator (FNO) for solving time-dependent partial differential equations in astrophysical magnetohydrodynamic simulations. The model learns operator mappings from initial conditions to future states (next 10 timesteps) for various physical quantities in MHD simulations.

---
**REPOSITORY UNDER CONSTRUCTION**  
We are working to make this repository useful and inference-ready, including a Docker image. For the time being, check out our paper: *Spectral Learning of Magnetized Plasma Dynamics: A Neural Operator Application. [arXiv:2507.01388](https://arxiv.org/abs/2507.01388)*.

---

## Overview

The Fourier Neural Operator (FNO) is a deep learning architecture that learns mappings between infinite-dimensional function spaces. Unlike traditional neural networks that learn point-wise mappings, FNOs learn entire operator mappings, making them particularly effective for solving PDEs.

This implementation:
- Operates on 3D spatial grids (128×128) with temporal evolution (10 timesteps)
- Learns in the frequency domain using Fast Fourier Transforms
- Predicts future states of astrophysical gas dynamics from initial conditions
- Supports multiple physical quantities (gas density, velocity components, magnetic fields)

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
- **Input layer**: Linear projection from 10 channels (7 physical + 3 spatial coordinates) to hidden width
- **5 Fourier layers**: Each combines spectral convolution with skip connection
- **Output layers**: Two fully-connected layers projecting back to physical space

### Architecture Details

```
Input: (batch, 128, 128, 10, 7)  → 128×128 spatial grid, 10 timesteps, 7 channels
  ↓ [Add grid coordinates]
  ↓ (batch, 128, 128, 10, 10)
  ↓ [Linear projection + permute]
  ↓ [Fourier Layer 0-4 with GELU activation]
Output: (batch, 128, 128, 10, 1) → Next 10 timesteps prediction
```

**Key hyperparameters:**
- Fourier modes: 64 (x-direction), 64 (y-direction), 5 (time)
- Hidden width: 30
- Temporal padding: 6 (for non-periodic boundary conditions)
- Activation: GELU

### SpectralConv3d

Performs operations in Fourier space:
1. **FFT**: Transform to frequency domain using `torch.fft.rfftn`
2. **Complex multiplication**: Learned weights multiply Fourier coefficients
3. **Inverse FFT**: Transform back to physical space using `torch.fft.irfftn`
4. **Parallel path**: 1×1×1 convolution acts as skip connection

## Usage

### Training a Model

```bash
python teste2.py --param <parameter_name>
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
python teste2.py --param density
```

**Training configuration:**
- Epochs: 10,000
- Batch size: 4
- Optimizer: Custom Adam with weight_decay=1e-4
- Learning rate: 0.001 with StepLR scheduler (step_size=100, gamma=0.5)
- Loss: Combined MAE (L1) + Relative L2 loss

**Output:**
- Model checkpoints: `experiments/<param>/checkpoints/model_64_30.pt`
- Loss history: `experiments/<param>/checkpoints/loss_64_30.npy`
- Validation images: `experiments/<param>/visualizations/` (generated once per epoch)

### Running Inference

```bash
python inference.py --param <parameter_name>
```

Processes 21 test samples and generates prediction visualizations comparing ground truth vs. predictions.

**Example:**
```bash
python inference.py --param density
```

## Physical Parameters

This model supports various physical quantities from MHD simulations:

| Parameter | Description | Physical Quantity |
|-----------|-------------|-------------------|
| `density` | Gas density | ρ (mass per unit volume) |
| `vy` | Gas velocity Y | v_y component of velocity field |
| `vz` | Gas velocity Z | v_z component of velocity field |
| `by` | Magnetic field Y | B_y component of magnetic field |
| `bz` | Magnetic field Z | B_z component of magnetic field |
| `br` | Magnetic field radial | B_r radial component |

## Training Details

### Loss Functions

The model uses a composite loss (defined in `utilities3.py`):

1. **MAE Loss (L1)**: Mean absolute error on normalized predictions
   ```python
   mae = |prediction - target|
   ```

2. **Relative L2 Loss (LpLoss)**: Normalized error in L2 norm
   ```python
   l2 = ||prediction - target||_2 / ||target||_2
   ```

3. **Combined Loss**:
   ```python
   loss = mae + l2
   ```

### Normalization

**Critical**: Input features and targets are normalized separately:

- **Input (x)**: Min-max normalized to [-1, 1] (except last 2 spatial coordinate channels)
- **Target (y)**: Min-max normalized to [-1, 1] during training
- **Predictions**: Denormalized using original target statistics for loss computation

### Data Loading

- **Online loading**: Loads one batch file at a time to manage memory
- **Training samples**: 90 files
- **Test samples**: 21 files
- **Batch size**: 4

## Data Format

### Directory Structure

```
input_data/
├── density/
│   ├── train/
│   │   ├── x_0.npy  # Input: first 10 timesteps
│   │   ├── y_0.npy  # Target: next 10 timesteps
│   │   ├── x_1.npy
│   │   ├── y_1.npy
│   │   └── ...
│   └── test/
│       ├── x_0.npy
│       ├── y_0.npy
│       └── ...
├── vy/
├── vz/
└── ...

experiments/
└── <param>/
    ├── checkpoints/              # Model checkpoints and loss history
    └── visualizations/  # Validation/test visualizations
```
/home/roberta/DL_new/FNO/Data/
├── gasdens/
│   ├── train/
│   │   ├── x_0.npy  # Input: first 10 timesteps
│   │   ├── y_0.npy  # Target: next 10 timesteps
│   │   ├── x_1.npy
│   │   ├── y_1.npy
│   │   └── ...
│   └── test/
│       ├── x_0.npy
│       ├── y_0.npy
│       └── ...
├── gasvy/
├── gasvz/
└── ...
```

### Array Shapes

- **Input files (x_*.npy)**: `(20, 128, 128, 10, 7)`
  - 20 samples per file
  - 128×128 spatial resolution
  - 10 timesteps
  - 7 physical channels

- **Target files (y_*.npy)**: `(20, 128, 128, 10, 1)`
  - 20 samples per file
  - 128×128 spatial resolution
  - 10 future timesteps
  - 1 predicted channel

## Configuration

### Updating Data Paths

All data paths are currently hardcoded. To adapt to your environment, update the following locations:

1. **Training script** (`teste2.py`):
   ```python
   # Line ~184: data() function
   dir = f'/your/path/Data/{param}/train/'

   # Line ~216: unormalize() function
   dir = f'/your/path/Data/{param}/test/'
   ```

2. **Inference script** (`inference.py`):
   ```python
   # Lines ~27, 36, 39: Update base paths
   dir = f'/your/path/Data/{param}/test/'
   ```

### Model Hyperparameters

To modify model architecture, edit the instantiation in `teste2.py`:

```python
model = FNO3d(modes1=64, modes2=64, modes3=5, width=30).cuda()
```

- `modes1`, `modes2`: Number of Fourier modes in x, y directions
- `modes3`: Number of Fourier modes in time dimension
- `width`: Hidden layer width

## Project Structure

```
fno/
├── architecture.py      # FNO3d and SpectralConv3d model definitions
├── teste2.py           # Main training script
├── inference.py        # Inference and visualization script
├── utilities3.py       # Loss functions (LpLoss, HsLoss, FrequencyLoss)
├── Adam.py            # Custom Adam optimizer implementation
├── input_data/        # Input data directory (created automatically)
│   └── <param>/
│       ├── train/              # Training data  
│       └── test/               # Test data
└── experiments/       # Output directory (created automatically)
    └── <param>/
        ├── checkpoints/        # Model checkpoints and loss history
        └── visualizations/     # Validation/test visualizations
```

## Results

### Output Files

After training, results are organized as:

```
experiments/
└── <param>/
    ├── checkpoints/
    │   ├── model_64_30.pt      # Trained model state dict
    │   └── loss_64_30.npy      # Training history (alternating MAE and combined loss)
    └── visualizations/
        ├── prediction_0.png     # Visualization of test sample 0
        ├── prediction_1.png
        └── ...
```

### Visualization

Prediction images show:
- Side-by-side comparison of ground truth vs. predicted timesteps
- Generated for validation samples during training (1 per epoch)
- Generated for all 21 test samples during inference

## Troubleshooting

### Common Issues

**CUDA out of memory:**
- Reduce batch size in `teste2.py` (line ~228: `data(idx, param, batch=4)`)
- Use a smaller model width or fewer Fourier modes

**File not found errors:**
- Verify data paths are correctly updated (see [Configuration](#configuration))
- Ensure data files follow the expected naming convention: `x_<idx>.npy`, `y_<idx>.npy`

**Model not converging:**
- Check data normalization is applied correctly
- Verify input data quality and range
- Try adjusting learning rate or scheduler parameters

**Import errors:**
- Ensure PyTorch is installed with CUDA support: `torch.cuda.is_available()` should return `True`
- Install missing dependencies: `pip install numpy matplotlib scipy`

### Debugging Tips

- Monitor loss values in `experiments/<param>/checkpoints/loss_64_30.npy`
- Check validation images during training to verify model is learning
- Use smaller epoch counts for initial testing (modify line ~227 in `teste2.py`)

## References

- Paper reporting the MHD application: Duarte, Nemmen & Lima (2025). Spectral Learning of Magnetized Plasma Dynamics: A Neural Operator Application. [*arXiv:2507.01388*](https://arxiv.org/abs/2507.01388)
- For more information on FNOs and their application to PDEs: Li et al. (2020). "Fourier Neural Operator for Parametric Partial Differential Equations." [*arXiv:2010.08895*](https://arxiv.org/abs/2010.08895)

---

## TODO

- [x] make repo public
- [ ] figures and movie! 
- [ ] inference guide w/ test dataset
- [ ] serve the model w/ API
- [ ] reproducibility: include Docker image