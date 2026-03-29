# Idefix Simulation Setup — Usage

## Step 1 — Generate parameter table

Run once to create `params.csv` with (ν, μ) pairs (2 hardcoded test cases + random val/train cases sampled log-uniformly in [1e-5, 5e-2]):

```bash
cd data/idefix
python generate_params.py [--nsims 50] [--nval 6] [--seed 42]
```

Default: 50 simulations with 6 val + 42 train + 2 fixed test cases. Use `--nsims` and `--nval` to change counts.

## Step 2 — Build Idefix

From the `data/idefix/` directory:

```bash
export IDEFIX_DIR=/path/to/idefix
bash build.sh
```

Targets Pascal GPUs (sm_60, e.g. Quadro GP100/P6000). Requires `$IDEFIX_DIR` to point to the Idefix source tree.

## Step 3 — Run simulations in parallel

Simulations are dispatched in parallel across GPUs (one per GPU at a time, round-robin):

```bash
python run_simulations.py [--gpus 0,1]
```

To run a subset (e.g., simulations 5 through 10):

```bash
python run_simulations.py --start 5 --end 10
```

To run with only one GPU (e.g., for testing):

```bash
python run_simulations.py --gpus 0
```

Each simulation runs in its own directory under `runs/sim_XXX/` and produces 1001 VTK snapshots (one per timestep at Δt = 0.05). GPUs are assigned round-robin: sim 0 → GPU 0, sim 1 → GPU 1, sim 2 → GPU 0, etc.


## Step 4 — Convert VTK output to `.npy`

```bash
python convert_to_npy.py --output-dir ../../data
```

Produces files for each physical field (`density`, `vx`, `vy`, `bx`, `by`):

```
data/<param>/train/x_sim_<id>.npy   shape (20, 128, 128, 20, 7)
data/<param>/train/y_sim_<id>.npy   shape (20, 128, 128, 20)
data/<param>/val/x_sim_<id>.npy
data/<param>/val/y_sim_<id>.npy
data/<param>/test/x_sim_<id>.npy
data/<param>/test/y_sim_<id>.npy
```

Prints per-split conversion counts at the end for quick verification.

## Notes

- **Split roles**: `train/` is used during gradient updates; `val/` is used for model selection and early stopping; `test/` is held out for final inference only and is **never read during training**.
- **Test cases**: `ν = μ = 5×10⁻⁵` and `ν = μ = 3×10⁻⁴` are always assigned to `test/`.
- **VTK field names**: Idefix outputs `RHO`, `VX1`, `VX2`, `BX1`, `BX2`, mapped to `density`, `vx`, `vy`, `bx`, `by`.
- **Reproducibility**: the random (ν, μ) pairs and val/train assignment are generated with a fixed seed (42) in `generate_params.py`.
