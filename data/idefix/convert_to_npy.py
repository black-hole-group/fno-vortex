"""
Convert Idefix VTK output to .npy training data for the FNO model.

For each simulation in params.csv, reads all VTK snapshots and constructs
sliding-window input/output blocks matching the format expected by train.py.

Output per physical field (density, vx, vy, bx, by):
  data/<param>/train/x_<idx>.npy  shape (20, 128, 128, 20, 7)
  data/<param>/train/y_<idx>.npy  shape (20, 128, 128, 20)
  data/<param>/test/x_<idx>.npy   shape (20, 128, 128, 20, 7)
  data/<param>/test/y_<idx>.npy   shape (20, 128, 128, 20)

Sliding window details:
  - 1000 snapshots total (frames 0-999, dt=0.05)
  - Input  : 5 consecutive frames starting at `start` (dt=0.05 code units each)
  - Output : 20 consecutive frames immediately after input (frames start+5 .. start+24)
  - 20 sliding windows per simulation, evenly strided (stride=51, start frames 0,51,102,...,969)
  - x channels 5-6: nu and mu broadcast to fill (128, 128, 20)

Usage:
  python convert_to_npy.py [--runs-dir runs] [--params params.csv]
                           [--output-dir ../../data]

Dependencies:
  pip install numpy tqdm
  $IDEFIX_DIR must point to the Idefix source tree (for pytools/vtk_io.py)
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

# ── locate Idefix pytools/vtk_io.py ───────────────────────────────────────────
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

from pytools.vtk_io import readVTK  # noqa: E402

def read_vtk(path):
    V = readVTK(str(path))
    return V.data


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
N_INPUT_FRAMES = 5      # number of consecutive input frames per sample
INPUT_SPACING = 1       # input frames consecutive → dt=0.05
N_INPUT_CHANNELS = 5    # 5 input frames per sample
OUTPUT_SPACING = 1      # output frames consecutive → dt=0.05
N_OUTPUT_FRAMES = 20    # 20 output frames per sample
N_SAMPLES = 20          # 20 sliding windows per simulation
T_IN = 20               # temporal dimension in x array (= N_OUTPUT_FRAMES, matches model)
# Stride between window start frames: evenly spaces N_SAMPLES windows across the full
# timeline so the last window starts at frame N_SNAPSHOTS - N_INPUT_FRAMES - N_OUTPUT_FRAMES.
# With defaults: (1000 - 5 - 20) // (20 - 1) = 51, covering frames 0..969.
WINDOW_STRIDE = (N_SNAPSHOTS - N_INPUT_FRAMES - N_OUTPUT_FRAMES) // (N_SAMPLES - 1)
MIN_SNAPSHOTS = N_INPUT_FRAMES + N_OUTPUT_FRAMES + (N_SAMPLES - 1) * WINDOW_STRIDE


def load_params(params_file):
    with open(params_file) as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_all_snapshots(run_dir):
    """Load all VTK snapshots for a simulation and extract all fields in one pass.

    Returns dict mapping field_key -> array of shape (N_SNAPSHOTS, 128, 128),
    or None if the simulation is incomplete or corrupted.
    """
    run_dir = Path(run_dir)
    vtk_files = sorted(run_dir.glob("data.????.vtk"))

    if len(vtk_files) == 0:
        raise FileNotFoundError(f"No VTK files found in {run_dir}")
    if len(vtk_files) < MIN_SNAPSHOTS:
        return None

    frames = {key: [] for key in FIELD_MAP}
    for vtk_path in vtk_files[:N_SNAPSHOTS]:
        try:
            data = read_vtk(vtk_path)
        except RuntimeError as e:
            print(f"\n  WARNING: corrupted VTK file {vtk_path}: {e}")
            return None
        for key in FIELD_MAP:
            frames[key].append(np.array(data[key]).squeeze())  # (128, 128)

    return {key: np.stack(frames[key], axis=0) for key in FIELD_MAP}  # each (N_SNAPSHOTS, 128, 128)


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
        start = sample_idx * WINDOW_STRIDE  # evenly spaced across full timeline

        # Input: 5 frames spaced INPUT_SPACING apart, starting at `start`
        input_frames = [snapshots[start + k * INPUT_SPACING] for k in range(N_INPUT_CHANNELS)]
        for ch, frame in enumerate(input_frames):
            x_all[sample_idx, :, :, :, ch] = frame[:, :, np.newaxis]  # broadcast to T_IN

        # Physical parameters: nu and mu (channels 5 and 6)
        x_all[sample_idx, :, :, :, 5] = nu
        x_all[sample_idx, :, :, :, 6] = mu

        # Output: 20 consecutive frames immediately after the input window
        output_frames = [snapshots[start + N_INPUT_FRAMES + k * OUTPUT_SPACING] for k in range(N_OUTPUT_FRAMES)]
        for t, frame in enumerate(output_frames):
            y_all[sample_idx, :, :, t] = frame

    return x_all, y_all


def save_npy(output_dir, param_name, split, sim_id, x, y):
    out = Path(output_dir) / param_name / split
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"x_sim_{sim_id:03d}.npy", x)
    np.save(out / f"y_sim_{sim_id:03d}.npy", y)


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

    bar    = tqdm(params, desc="Converting", unit="sim", position=1) if HAS_TQDM else None
    status = tqdm(total=0, bar_format="{desc}", position=0, leave=False) if HAS_TQDM else None
    iterable = bar if bar else params

    n_converted = 0
    n_skipped = 0

    for row in iterable:
        sim_id = int(row["sim_id"])
        nu = float(row["nu"])
        mu = float(row["mu"])
        split = row["split"]
        run_dir = runs_dir / f"sim_{sim_id:03d}"

        if status:
            status.set_description_str(f"sim_{sim_id:03d} nu={nu:.1e} mu={mu:.1e} checking")

        if not run_dir.exists():
            if status:
                status.set_description_str(f"sim_{sim_id:03d} MISSING, skipped")
            if bar:
                bar.update(1)
            n_skipped += 1
            continue

        if status:
            status.set_description_str(f"sim_{sim_id:03d} nu={nu:.1e} mu={mu:.1e} loading VTK")

        all_snapshots = load_all_snapshots(run_dir)
        if all_snapshots is None:
            if status:
                status.set_description_str(f"sim_{sim_id:03d} incomplete, skipped")
            if bar:
                bar.update(1)
            n_skipped += 1
            continue

        for field_key, param_name in FIELD_MAP.items():
            if status:
                status.set_description_str(f"sim_{sim_id:03d} nu={nu:.1e} mu={mu:.1e} {param_name}")
            x, y = build_windows(all_snapshots[field_key], nu, mu)
            save_npy(args.output_dir, param_name, split, sim_id, x, y)

        if status:
            status.set_description_str(f"sim_{sim_id:03d} nu={nu:.1e} mu={mu:.1e} done ({split})")
        if bar:
            bar.update(1)
        n_converted += 1

    if status:
        status.close()
    if bar:
        bar.close()

    print("\nConversion complete.")
    print(f"Converted: {n_converted}  Skipped: {n_skipped}")


if __name__ == "__main__":
    main()
