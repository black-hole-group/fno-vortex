# AGENTS.md

This file contains guidelines for agentic coding agents working in this repository.

## Build/Lint/Test Commands

### Setup
- Requires Python 3.8+ with PyTorch 1.10+
- Install dependencies: `pip install torch numpy matplotlib scipy`

### Build Commands
- Training: `python teste2.py --param <parameter_name>`
  - Available parameters: `gasdens`, `gasvy`, `gasvz`, `by`, `bz`, `br`
- Inference: `python inference.py --param <parameter_name>`

### Lint Commands
- No explicit linting specified, but code generally follows PEP8 conventions
- Use `pylint` or `flake8` for code quality checks

### Test Commands
- No specific test framework found
- Training and inference files can be run directly
- To run a single test: `python inference.py --param gasdens`
- For training a specific parameter: `python teste2.py --param gasdens`

## Code Style Guidelines

### Imports
- All imports at top of file
- Standard library imports first, then third-party, then local imports
- Use `import torch` and `import torch.nn as nn` for PyTorch modules
- Import specific utilities like `from utilities3 import *`

### Formatting
- Follow PEP8 conventions
- 4 spaces for indentation (no tabs)
- Maximum line length of 79 characters
- Use descriptive variable names (e.g., `batch_size` not `bs`)

### Naming Conventions
- Class names: PascalCase (e.g., `FNO3d`, `SpectralConv3d`)
- Function names: snake_case (e.g., `get_grid`, `unormalize`)
- Constants: UPPER_CASE (e.g., `LEARNING_RATE = 0.001`)
- Variables: snake_case (e.g., `batch_size`, `train_loss`)

### Types
- Use type hints where possible
- PyTorch tensors, numpy arrays, and standard Python types used
- No explicit type checking enforced in code

### Error Handling
- Uses standard Python exception handling
- Custom exception handling is kept minimal
- Errors are logged via print statements or standard Python exception flow
- No explicit try/catch blocks for expected errors in training/inference

### Code Structure
- Model definitions in `architecture.py`
- Training logic in `teste2.py`
- Inference logic in `inference.py`
- Utility functions in `utilities3.py`
- Custom optimizer in `Adam.py`

### Documentation
- Docstrings for all classes and functions
- Inline comments for complex operations
- Code is documented in CLAUDE.md file for better understanding

### Path Conventions
- All hardcoded paths in `teste2.py` and `inference.py`
- Data paths in `/home/roberta/DL_new/FNO/Data/`
- Results paths in `/home/roberta/DL_new/FNO/Results/`
- Paths should be updated when adapting code for different environments

### Additional Notes
- GPU usage is enabled by default with `.cuda()` calls
- Data normalization is handled internally in training
- Model checkpoints saved in `Results/<param>/model/`
- Validation visualizations saved in `Results/<param>/predictions_image/`