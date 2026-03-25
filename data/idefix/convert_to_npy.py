"""
Convert Idefix VTK output to .npy training data for the FNO model.

For each simulation in params.csv, reads all VTK snapshots and constructs
sliding-window input/output blocks matching the format expected by train.py.

Output per physical field (density, vx, vy, bx, by):
  data/<param>/train/x_<idx>.npy  shape (20, 128, 128, 10, 7)
  data/<param>/train/y_<idx>.npy  shape (20, 128, 128, 10)
  data/<param>/test/x_<idx>.npy   shape (20, 128, 128, 10, 7)
  data/<param>/test/y_<idx>.npy   shape (20, 128, 128, 10)

Sliding window details (from paper):
  - 1000 snapshots total (frames 0-999, dt=0.05)
  - Input  : 5 frames from frames 0-159, spaced 20 apart (dt=1.0 code units)
  - Output : 10 frames from frames 160-999, spaced 80 apart (dt=4.0 code units)
  - 20 sliding windows per simulation (start frames 0..19)
  - x channels 5-6: nu and mu broadcast to fill (128, 128, 10)

Usage:
  python convert_to_npy.py [--runs-dir runs] [--params params.csv]
                           [--output-dir ../../data]

Dependencies:
  pip install numpy tqdm
  $IDEFIX_DIR must point to the Idefix source tree (for pytools/vtk.py)
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ── locate Idefix pytools/vtk.py ──────────────────────────────────────────────
_idefix_dir = os.environ.get("IDEFIX_DIR")
if _idefix_dir:
    sys.path.insert(0, _idefix_dir)
else:
    _guess = Path(__file__).resolve().parents[4] / "idefix"
    if _guess.exists():
        sys.path.insert(0, str(_guess))
    else:
        print("Error: $IDEFIX_DIR is not set and idefix/ was not found automatically.")
        print("Set $IDEFIX_DIR to the Idefix source directory and re-run.")
        sys.exit(1)

import pytools.vtk as pvtk  # noqa: E402

def read_vtk(path):
    reader = pvtk.VtkReader(str(path))
    return reader.data


SCRIPT_DIR = Path(__file__).parent.resolve()

# Idefix field name → output param name
FIELD_MAP = {
    "RHO": "density",
    "VX1": "vx",
    "VX2": "vy",
    "BX1": "bx",
    "BX2": "by",
}

N_SNAPSHOTS = 1000      # frames 0-999
N_INPUT_FRAMES = 160    # first 160 frames are the input window
INPUT_SPACING = 20      # input frames spaced 20 apart → dt=1.0
N_INPUT_CHANNELS = 5    # 5 input frames per sample
OUTPUT_SPACING = 80     # output frames spaced 80 apart → dt=4.0
N_OUTPUT_FRAMES = 10    # 10 output frames per sample
N_SAMPLES = 20          # 20 sliding windows per simulation
T_IN = 10               # temporal dimension in x array (= N_SAMPLES // 2, matches model)


def load_params(params_file):
    with open(params_file) as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_all_snapshots(run_dir, field_key):
    """Load all VTK snapshots for a simulation and extract one field.

    Returns array of shape (N_SNAPSHOTS, 128, 128).
    """
    run_dir = Path(run_dir)
    vtk_files = sorted(run_dir.glob("data.????.vtk"))

    if len(vtk_files) == 0:
        raise FileNotFoundError(f"No VTK files found in {run_dir}")
    if len(vtk_files) < N_SNAPSHOTS:
        print(f"  WARNING: expected {N_SNAPSHOTS} VTK files, found {len(vtk_files)}")

    frames = []
    iter_files = tqdm(vtk_files[:N_SNAPSHOTS], desc="  Reading VTK", leave=False) \
        if HAS_TQDM else vtk_files[:N_SNAPSHOTS]

    for vtk_path in iter_files:
        data = read_vtk(vtk_path)
        field = np.array(data[field_key]).squeeze()  # (128, 128)
        frames.append(field)

    return np.stack(frames, axis=0)  # (N_SNAPSHOTS, 128, 128)


def build_windows(snapshots, nu, mu):
    """Build sliding-window (x, y) blocks from a simulation's snapshots.

    snapshots: (N_SNAPSHOTS, 128, 128)

    Returns:
      x: (N_SAMPLES, 128, 128, T_IN, 7)
      y: (N_SAMPLES, 128, 128, N_OUTPUT_FRAMES)
    """
    H, W = snapshots.shape[1], snapshots.shape[2]

    x_all = np.zeros((N_SAMPLES, H, W, T_IN, 7), dtype=np.float32)
    y_all = np.zeros((N_SAMPLES, H, W, N_OUTPUT_FRAMES), dtype=np.float32)

    for sample_idx in range(N_SAMPLES):
        start = sample_idx  # slide starting frame by 1 each sample

        # Input: 5 frames spaced INPUT_SPACING apart, starting at `start`
        input_frames = [snapshots[start + k * INPUT_SPACING] for k in range(N_INPUT_CHANNELS)]
        for ch, frame in enumerate(input_frames):
            x_all[sample_idx, :, :, :, ch] = frame[:, :, np.newaxis]  # broadcast to T_IN

        # Physical parameters: nu and mu (channels 5 and 6)
        x_all[sample_idx, :, :, :, 5] = nu
        x_all[sample_idx, :, :, :, 6] = mu

        # Output: 10 frames spaced OUTPUT_SPACING apart, starting at frame 160
        output_frames = [snapshots[N_INPUT_FRAMES + k * OUTPUT_SPACING] for k in range(N_OUTPUT_FRAMES)]
        for t, frame in enumerate(output_frames):
            y_all[sample_idx, :, :, t] = frame

    return x_all, y_all


def save_npy(output_dir, param_name, split, file_idx, x, y):
    out = Path(output_dir) / param_name / split
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"x_{file_idx}.npy", x)
    np.save(out / f"y_{file_idx}.npy", y)


def main():
    parser = argparse.ArgumentParser(description="Convert Idefix VTK output to .npy")
    parser.add_argument("--runs-dir", default=str(SCRIPT_DIR / "runs"),
                        help="Directory containing sim_XXX run subdirectories")
    parser.add_argument("--params", default=str(SCRIPT_DIR / "params.csv"),
                        help="Path to params.csv")
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "../../data"),
                        help="Root output directory (data/ in repo)")
    args = parser.parse_args()

    params = load_params(args.params)
    runs_dir = Path(args.runs_dir)

    # Track per-split file counters
    train_counters = {p: 0 for p in FIELD_MAP.values()}
    test_counters  = {p: 0 for p in FIELD_MAP.values()}

    outer = tqdm(params, desc="Simulations") if HAS_TQDM else params

    for row in outer:
        sim_id = int(row["sim_id"])
        nu = float(row["nu"])
        mu = float(row["mu"])
        split = row["split"]
        run_dir = runs_dir / f"sim_{sim_id:03d}"

        if not run_dir.exists():
            print(f"WARNING: {run_dir} not found, skipping sim {sim_id}")
            continue

        print(f"\n[sim_{sim_id:03d}] nu={nu:.3e}  mu={mu:.3e}  split={split}")

        for field_key, param_name in FIELD_MAP.items():
            snapshots = load_all_snapshots(run_dir, field_key)  # (1000, 128, 128)
            x, y = build_windows(snapshots, nu, mu)             # (20, 128, 128, 10, 7)

            if split == "train":
                idx = train_counters[param_name]
                train_counters[param_name] += 1
            else:
                idx = test_counters[param_name]
                test_counters[param_name] += 1

            save_npy(args.output_dir, param_name, split, idx, x, y)
            print(f"  {param_name:8s}: {split}/x_{idx}.npy  x={x.shape}  y={y.shape}")

    print("\nConversion complete.")
    print("Train file counts:", train_counters)
    print("Test  file counts:", test_counters)


if __name__ == "__main__":
    main()
