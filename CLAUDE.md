# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a PyTorch implementation of a 3D Fourier Neural Operator (FNO) for solving time-dependent partial differential equations in astrophysical gas dynamics simulations. The model learns operator mappings from initial conditions (first 10 timesteps) to future states (next 40 timesteps) for various physical quantities like gas density and velocity components.

## Training and Inference

**Training a model:**
```bash
python src/teste2.py --param <parameter_name>
```
- Available parameters: `density`, `vy`, `vz`, `by`, `bz`, `br`
- Trains for 10,000 epochs with learning rate 0.001 and StepLR scheduling
- Saves model checkpoints and loss history to `experiments/<param>/checkpoints/`
- Generates validation visualizations in `experiments/<param>/visualizations/`
- Uses batch size of 4 with data loaded from relative paths in `input_data/`

**Running inference:**
```bash
python src/inference.py --param <parameter_name>
```
- Loads trained model from `experiments/<param>/checkpoints/model_64_30.pt`
- Processes 21 test samples and saves predictions to `experiments/<param>/visualizations/`
- Data paths are relative to the current directory

## Architecture Details

**FNO3d Model (src/architecture.py, src/teste2.py):**
- Input: `(batch, 128, 128, 10, 7)` - first 10 timesteps of solution + 3 spatial coordinates
- Output: `(batch, 128, 128, 10, 1)` - next 10 timesteps prediction
- The model uses 5 Fourier layers (conv0-conv4) in parallel with 1x1x1 convolutions (w0-w4)
- Key hyperparameters: 64 modes in x/y directions, 5 modes in time, width=30
- Padding of 6 applied in time dimension for non-periodic boundary conditions
- Grid coordinates are automatically generated and concatenated to input via `get_grid()`

**SpectralConv3d:**
- Performs FFT → complex multiplication in frequency domain → inverse FFT
- Uses 4 weight tensors to handle different octants of the frequency space
- Complex multiplication implemented via einsum for efficiency

**Custom Adam Optimizer (Adam.py):**
- Modified PyTorch Adam implementation with explicit functional API
- Used instead of torch.optim.Adam, likely for research/debugging purposes
- Applied with weight_decay=1e-4 in training

## Data Pipeline

**Input data structure:**
- Training data: `input_data/<param>/train/[x|y]_<idx>.npy`
- Test data: `input_data/<param>/test/[x|y]_<idx>.npy`
- Each file contains numpy arrays with shape `(20, 128, 128, 10, [7|1])`
- Training uses 90 files, test uses 21 files

**Normalization (teste2.py:184-214):**
- Input features (x): Min-max normalized to [-1, 1], except last 2 channels (spatial coords)
- Target (y): Min-max normalized to [-1, 1] during training
- Predictions are denormalized using original y statistics in `unormalize()` function
- This dual normalization is critical - inputs and targets are normalized separately per batch

## Loss Functions

**Training combines two losses (teste2.py:228, 257-258):**
1. MAE (L1 loss) on normalized predictions
2. Relative L2 loss (LpLoss) on denormalized predictions
- `LpLoss` is from utilities3.py and computes relative error: `||pred - target||_p / ||target||_p`
- Combined as: `loss = mae + l2`

**utilities3.py provides:**
- `LpLoss`: Relative/absolute Lp norm loss
- `HsLoss`: Sobolev (HS) norm loss comparing derivatives in frequency domain
- `FrequencyLoss`: Compares log power spectra in Fourier space

## Model I/O

**Loading models:**
- Models are saved/loaded with `torch.load()` and `model.load_state_dict()`
- Expected path format: `experiments/<param>/checkpoints/model_64_30.pt`
- Model architecture must be instantiated before loading: `FNO3d(64, 64, 5, 30)`

**Data format:**
- All data is `.npy` format (NumPy arrays)
- Loss history saved as 1D array alternating MAE and combined loss per epoch

## Important Path Conventions

- Results directory structure: `experiments/<param>/[checkpoints|visualizations]/`
- All training/test data paths are now relative to the project directory
- When adapting code, update paths in `data()` function (teste2.py:184), `unormalize()` (teste2.py:216), and inference loading (inference.py:27, 36, 39)

## Key Implementation Notes

- The model operates on CUDA by default (`.cuda()` calls throughout)
- Training uses online data loading - loads one batch file at a time to manage memory
- Validation visualization is generated once per epoch on a random test sample
- Batch normalization layers (bn0-bn3) are defined but never used in forward pass
- Time dimension undergoes padding/unpadding (teste2.py:90, 121) but spatial dims don't
