# Idefix Simulation Setup — Usage

## Dependencies (Python)

```bash
pip install inifix idefix-pytools tqdm
```

## Step 1 — Generate parameter table

Run once to create `params.csv` with 25 (ν, μ) pairs (2 hardcoded test cases + 23 random training cases sampled log-uniformly in [1e-5, 5e-2]):

```bash
cd simulations/idefix
python generate_params.py
```

## Step 2 — Build Idefix

From the `simulations/idefix/` directory:

```bash
export IDEFIX_DIR=/path/to/idefix
bash build.sh
```

Targets the RTX 3090 Ti (Ampere sm_86). Requires `$IDEFIX_DIR` to point to the Idefix source tree.

## Step 3 — Run all 25 simulations sequentially

```bash
python run_simulations.py
```

To run a subset (e.g., simulations 5 through 10):

```bash
python run_simulations.py --start 5 --end 10
```

Each simulation runs in its own directory under `runs/sim_XXX/` and produces 1001 VTK snapshots (one per timestep at Δt = 0.05).

- Added --gpus argument (default "0,1")
- run_simulation() now accepts a gpu_id and sets CUDA_VISIBLE_DEVICES in the subprocess
  environment
- Main loop replaced with ProcessPoolExecutor, assigning GPUs round-robin — so sim 0 → GPU 0, sim 1 → GPU 1, sim 2 → GPU 0, etc., with at most one simulation running per GPU at a time

To run with only one GPU (e.g., for testing): python run_simulations.py --gpus 0


## Step 4 — Convert VTK output to `.npy`

```bash
python convert_to_npy.py --output-dir ../../data
```

Produces files for each physical field (`density`, `vx`, `vy`, `bx`, `by`):

```
data/<param>/train/x_<idx>.npy   shape (20, 128, 128, 10, 7)
data/<param>/train/y_<idx>.npy   shape (20, 128, 128, 10)
data/<param>/test/x_<idx>.npy
data/<param>/test/y_<idx>.npy
```

## Notes

- **Test cases**: `ν = μ = 5×10⁻⁵` and `ν = μ = 3×10⁻⁴` are held out and written to `test/`.
- **VTK field names**: Idefix outputs `RHO`, `VX1`, `VX2`, `BX1`, `BX2`, mapped to `density`, `vx`, `vy`, `bx`, `by`.
- **Reproducibility**: the 23 random (ν, μ) pairs are generated with a fixed seed (42) in `generate_params.py`.
