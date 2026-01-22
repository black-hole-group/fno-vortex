# Fourier Neural Operator for Astrophysical Gas Dynamics

This project implements a 3D Fourier Neural Operator (FNO) for solving time-dependent partial differential equations in astrophysical gas dynamics simulations. The model learns operator mappings from initial conditions to future states for various physical quantities.

## Overview

The Fourier Neural Operator (FNO) is a deep learning approach that learns the mapping between input and output functions of PDEs directly in the frequency domain. This implementation handles 3D gas dynamics simulations with time evolution.

## Model Architecture

- **FNO3d Model**: A 3D Fourier neural operator with 5 spectral layers
- **SpectralConv3d**: Performs FFT → complex multiplication in frequency domain → inverse FFT
- **Input**: First 10 timesteps of solution + 3 spatial coordinates (u(1, x, y), ..., u(10, x, y), x, y, t)
- **Output**: Next 40 timesteps prediction
- **Features**: Uses 64 modes in x/y directions, 5 modes in time, width=30

## Key Components

- `architecture.py`: FNO3d model definition
- `teste2.py`: Training script with 10,000 epochs 
- `inference.py`: Inference script for testing
- `utilities3.py`: Loss functions and utilities (LpLoss, HsLoss, etc.)
- `Adam.py`: Custom Adam optimizer

## Usage

### Training
```bash
python teste2.py --param <parameter_name>
```
Available parameters: `gasdens`, `gasvy`, `gasvz`, `by`, `bz`, `br`

### Inference
```bash
python inference.py --param <parameter_name>
```

## Requirements

- Python 3.8+
- PyTorch 1.10+
- numpy, matplotlib, scipy

## Data Format

- Training data: `/home/roberta/DL_new/FNO/Data/<param>/train/[x|y]_<idx>.npy`
- Test data: `/home/roberta/DL_new/FNO/Data/<param>/test/[x|y]_<idx>.npy`
- Each file contains numpy arrays with shape `(20, 128, 128, 10, [7|1])`

## Results

- Models saved to: `Results/<param>/model/`
- Predictions saved to: `Results/<param>/predictions_image/`

## Note

All paths are hardcoded. Update path constants in `teste2.py` and `inference.py` when adapting to a different environment.

## Acknowledgements

Based on research in Fourier Neural Operators for PDE solving in astrophysical simulations.