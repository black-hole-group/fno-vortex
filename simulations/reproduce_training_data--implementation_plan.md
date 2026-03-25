# Plan: Set up Idefix Orszag-Tang vortex simulations

## Context

The FNO model in this repo was trained on FARGO3D simulation data of the Orszag-Tang vortex. The goal is to reproduce the simulation dataset using **Idefix** instead, running on a Linux server with an RTX 3090 Ti GPU. We need Idefix setup files, a batch runner for 25 simulations, and a conversion script to produce the `.npy` files consumed by `train.py`.

## Files to create

All new files go under `simulations/idefix/`.

### 1. `simulations/idefix/setup.cpp` — Initial conditions

C++ setup file defining the Orszag-Tang vortex initial conditions from the paper:

```
ρ   = 25 / (36π)
P   = 5 / (12π)
vx  = -sin(2πy)
vy  = sin(2πx)
Bx  = -sin(2πy) / √(4π)
By  = sin(4πx) / √(4π)
```

- Use Idefix's `Setup::InitFlow()` to fill the grid arrays (`Vc(RHO,...)`, `Vc(VX1,...)`, etc.)
- Magnetic field initialized via vector potential `Az = cos(4πx)/(4π√(4π)) + cos(2πy)/(2π√(4π))` to ensure ∇·B = 0 from the start (Idefix convention for CT)

### 2. `simulations/idefix/definitions.hpp` — Compile-time options

```cpp
#define COMPONENTS 2
#define DIMENSIONS 2
#define GEOMETRY CARTESIAN
```

MHD enabled via cmake flag (`-DIdefix_MHD=ON`).

### 3. `simulations/idefix/idefix.ini` — Runtime parameters

```ini
[Grid]
  X1-grid    1  0.0  128  6.283185307  u
  X2-grid    1  0.0  128  6.283185307  u

[TimeIntegrator]
  CFL         0.3
  tstop       50.0
  first_dt    1e-4

[Hydro]
  gamma       1.666666667
  solver      hlld
  emf         uct_contact
  viscosity   explicit
  resistivity explicit

[Viscosity]
  nu          1e-3          # placeholder, overwritten by batch script

[Resistivity]
  eta         1e-3          # placeholder, overwritten by batch script

[Boundary]
  X1-beg      periodic
  X1-end      periodic
  X2-beg      periodic
  X2-end      periodic

[Output]
  vtk         0.05          # output every Δt = 0.05 → 1000 snapshots
  log         100
```

### 4. `simulations/idefix/generate_params.py` — Parameter generation

Generates 25 (ν, μ) pairs sampled independently in log₁₀ space, uniform in [log₁₀(1e-5), log₁₀(5e-2)] = [-5, -1.301]. Saves to `params.csv`. Uses a fixed random seed for reproducibility.

Two simulations are reserved as test set:
- ν = μ = 5×10⁻⁵
- ν = μ = 3×10⁻⁴

These are hardcoded (not random). The remaining 23 are randomly generated.

### 5. `simulations/idefix/run_simulations.py` — Batch runner (sequential)

1. Reads `params.csv`
2. For each of the 25 simulations:
   - Creates run directory: `simulations/idefix/runs/sim_XXX/`
   - Copies `idefix.ini` and patches `nu` and `eta` using `inifix`
   - Symlinks the compiled `idefix` binary
   - Runs simulation via subprocess, waits for completion
3. Supports `--start`/`--end` flags for partial runs
4. Logs progress

Dependencies: `inifix` (`pip install inifix`)

### 6. `simulations/idefix/convert_to_npy.py` — VTK → npy conversion

Reads Idefix VTK output and produces `.npy` files for `train.py`.

**Field mapping:** `RHO` → density, `VX1` → vx, `VX2` → vy, `BX1` → bx, `BX2` → by

**Output shapes:**
- `x_<idx>.npy`: `(20, 128, 128, 10, 7)` — 20 samples, 128² grid, 10 time frames, 7 channels
- `y_<idx>.npy`: `(20, 128, 128, 10)` — 10 output frames

**Sliding window construction:**
- **x channels 0–4:** 5 frames from first 160 snapshots, spaced 20 frames apart (Δt=1.0). Slide starting frame to get 20 samples.
- **y:** 10 frames from snapshots 160–1000, spaced 80 frames apart (Δt=4.0)
- **x channels 5–6:** ν and μ broadcast to (128, 128, 10)

**Split:** 23 simulations → `data/<param>/train/`, 2 test sims → `data/<param>/test/`

Dependencies: `idefix-pytools` (`pip install idefix-pytools`), `numpy`

### 7. `simulations/idefix/build.sh` — Build helper

```bash
#!/bin/bash
cmake $IDEFIX_DIR -DIdefix_MHD=ON -DKokkos_ENABLE_CUDA=ON \
      -DKokkos_ARCH_AMPERE86=ON
make -j$(nproc)
```

## Workflow

```bash
# 1. Generate parameter table
cd simulations/idefix
python generate_params.py

# 2. Build Idefix (once)
bash build.sh

# 3. Run all 25 simulations sequentially
python run_simulations.py

# 4. Convert VTK output to .npy training data
python convert_to_npy.py --output-dir ../../data
```

## Verification

- After build: run a short test (tstop=1.0), verify VTK output is readable
- After full runs: verify each simulation produced 1001 VTK files
- After conversion: check `.npy` shapes `(20, 128, 128, 10, 7)` and `(20, 128, 128, 10)`
- Sanity check: visualize a density snapshot to confirm Orszag-Tang vortex structure
