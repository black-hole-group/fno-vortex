"""
Prepare dense reference trajectories for rollout visualization.

Reads raw Idefix VTK simulation outputs and saves a compact reference file
per simulation per field:

    data/<param>/test/ref_sim_<id>.npy   shape (N_frames, 128, 128)

This file is read by visualize_results.py to show the true simulation
evolution alongside autoregressive rollout predictions.

By default (when --param is omitted) all supported fields are prepared in a
single pass, reading each VTK snapshot only once per simulation. Providing
--param restricts output to that one field.

Usage — prepare all fields at once (recommended):

    python src/prepare_reference.py \\
        --param-prefix idefix/numpy/t20 \\
        [--experiments-dir experiments/<run>] \\
        [--runs-dir data/idefix/runs] \\
        [--params-csv data/idefix/params.csv] \\
        [--data-dir data] \\
        [--split test]

Usage — single field (backward compatible):

    python src/prepare_reference.py \\
        --param idefix/numpy/t20/by \\
        [--experiments-dir experiments/<run>] \\
        [--runs-dir data/idefix/runs] \\
        [--params-csv data/idefix/params.csv] \\
        [--data-dir data] \\
        [--split test]

Requires $IDEFIX_DIR to be set (or idefix/ to exist at the repo root)
so that pytools/vtk_io.py can be found.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

from experiment_layout import (
    ensure_param_layout,
    reference_dir,
    resolve_experiment_param,
    write_manifest,
)

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent

# ── locate Idefix pytools/vtk_io.py ──────────────────────────────────────────
_idefix_dir = os.environ.get("IDEFIX_DIR")
if _idefix_dir:
    sys.path.insert(0, _idefix_dir)
else:
    _guess = ROOT_DIR / "idefix"
    if _guess.exists():
        sys.path.insert(0, str(_guess))

try:
    from pytools.vtk_io import readVTK  # noqa: E402
    _HAS_VTK = True
except ImportError:
    _HAS_VTK = False

# Reverse of convert_to_npy.py FIELD_MAP: leaf param name → VTK field key
_PARAM_TO_VTK = {
    "density": "RHO",
    "vx":      "VX1",
    "vy":      "VX2",
    "bx":      "BX1",
    "by":      "BX2",
}


def _field_from_param(param):
    """Extract the leaf field name from a --param path.

    E.g. 'idefix/numpy/t20/by' → 'by'
         'density'               → 'density'
    """
    return Path(param).name


def load_sim_fields(run_dir, vtk_fields, n_frames=1000):
    """Load n_frames snapshots of multiple fields from run_dir in one pass.

    Parameters
    ----------
    run_dir   : path to the simulation directory containing data.????.vtk files
    vtk_fields : dict mapping field name to VTK key, e.g. {'bx': 'BX1', ...}
    n_frames  : how many VTK frames to read

    Returns
    -------
    dict mapping field name → ndarray of shape (n_frames, H, W), float32
    """
    run_dir = Path(run_dir)
    vtk_files = sorted(run_dir.glob("data.????.vtk"))[:n_frames]

    if len(vtk_files) == 0:
        raise FileNotFoundError(f"No VTK files found in {run_dir}")
    if len(vtk_files) < n_frames:
        print(f"  WARNING: only {len(vtk_files)} frames available in {run_dir}")

    accum = {field: [] for field in vtk_fields}
    for vtk_path in vtk_files:
        vtk_data = readVTK(str(vtk_path)).data
        for field, vtk_key in vtk_fields.items():
            accum[field].append(
                np.array(vtk_data[vtk_key]).squeeze().astype(np.float32)
            )

    return {field: np.stack(frames, axis=0) for field, frames in accum.items()}


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Save dense reference trajectories for rollout visualisation. "
            "When --param is omitted all supported fields are prepared in a "
            "single VTK-reading pass."
        )
    )
    param_group = parser.add_mutually_exclusive_group()
    param_group.add_argument(
        "--param", type=str,
        help=(
            "Single-field parameter path, e.g. idefix/numpy/t20/by. "
            "When given, only that field is prepared."
        ),
    )
    param_group.add_argument(
        "--param-prefix", type=str,
        help=(
            "Path prefix shared by all fields, e.g. idefix/numpy/t20. "
            "All supported fields are appended automatically. "
            "Required when --param is omitted."
        ),
    )
    parser.add_argument("--runs-dir",
                        default=str(ROOT_DIR / "data" / "idefix" / "runs"),
                        help="Directory containing sim_NNN/ subdirectories")
    parser.add_argument("--params-csv",
                        default=str(ROOT_DIR / "data" / "idefix" / "params.csv"),
                        help="Path to params.csv")
    parser.add_argument("--data-dir",
                        default=str(ROOT_DIR / "data"),
                        help="Root data directory")
    parser.add_argument(
        "--experiments-dir",
        default=None,
        help=(
            "Optional run root for writing prepared ref_sim_*.npy files under "
            "experiments/<run>/references/<split>/<param>. When omitted, "
            "references are written back into data/<param>/<split>/."
        ),
    )
    parser.add_argument("--split", type=str, default="test",
                        help="Which split to prepare reference files for")
    parser.add_argument("--n-frames", type=int, default=400,
                        help="Number of frames to extract per simulation")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing ref_sim_*.npy files")
    args = parser.parse_args()

    if not _HAS_VTK:
        print(
            "ERROR: pytools.vtk_io not found.\n"
            "Set $IDEFIX_DIR to your Idefix source tree, then re-run."
        )
        sys.exit(1)

    # Build the mapping: {leaf_field: (vtk_key, output_param_path)}
    if args.param is not None:
        # Single-field mode — backward compatible
        leaf = _field_from_param(args.param)
        if leaf not in _PARAM_TO_VTK:
            print(
                f"ERROR: unknown field '{leaf}'.\n"
                f"Supported fields: {', '.join(_PARAM_TO_VTK)}"
            )
            sys.exit(1)
        field_map = {leaf: (
            _PARAM_TO_VTK[leaf],
            args.param,
        )}
    else:
        # All-fields mode — require a prefix
        if args.param_prefix is None:
            parser.error(
                "Either --param or --param-prefix must be provided. "
                "Example: --param-prefix idefix/numpy/t20"
            )
        prefix = args.param_prefix
        field_map = {
            leaf: (vtk_key, f"{prefix}/{leaf}")
            for leaf, vtk_key in _PARAM_TO_VTK.items()
        }

    runs_dir = Path(args.runs_dir)
    data_dir = Path(args.data_dir)

    # Load params.csv to find simulations in the requested split
    with open(args.params_csv) as f:
        rows = list(csv.DictReader(f))
    split_rows = [r for r in rows if r["split"] == args.split]

    if not split_rows:
        print(f"No simulations with split='{args.split}' found in {args.params_csv}")
        sys.exit(1)

    all_fields = sorted(field_map)
    print(f"Fields : {', '.join(all_fields)}")
    print(f"Split  : {args.split}  ({len(split_rows)} sims)")
    print(f"Runs   : {runs_dir}")
    if args.experiments_dir:
        print(f"Output : run-local references under {args.experiments_dir}")
    else:
        print(f"Output : dataset references under {data_dir}")

    # Pre-create output directories
    out_dirs = {}
    for leaf, (_, param_path) in field_map.items():
        if args.experiments_dir is not None:
            paths = resolve_experiment_param(
                args.experiments_dir, param_path, data_dir, create=True,
            )
            ensure_param_layout(paths)
            out_dir = reference_dir(paths, split=args.split, create=True)
            if out_dir is None:
                raise ValueError(
                    "Run-local references require a run-scoped experiments "
                    f"directory, got {args.experiments_dir!r}."
                )
            write_manifest(
                paths,
                metadata={
                    'cli': {
                        'experiments_dir': str(paths.exp_dir),
                        'param': param_path,
                    },
                    'data': {
                        'param': paths.data_param,
                    },
                    'references': {
                        'split': args.split,
                        'source_runs_dir': str(runs_dir),
                        'n_frames': args.n_frames,
                    },
                    'params': {
                        paths.param_key: {
                            'reference_preparation': {
                                'split': args.split,
                                'output_dir': str(
                                    out_dir.relative_to(paths.exp_dir)
                                ),
                                'source_runs_dir': str(runs_dir),
                                'n_frames': args.n_frames,
                            },
                        },
                    },
                },
            )
        else:
            out_dir = data_dir / param_path / args.split
            out_dir.mkdir(parents=True, exist_ok=True)
        out_dirs[leaf] = out_dir

    for row in split_rows:
        sim_id = int(row["sim_id"])
        sim_id_str = f"{sim_id:03d}"

        run_dir = runs_dir / f"sim_{sim_id_str}"
        if not run_dir.exists():
            print(
                f"  sim_{sim_id_str}: run directory not found ({run_dir}), skipping"
            )
            continue

        # Determine which fields still need to be written
        fields_needed = {}
        for leaf, (vtk_key, _) in field_map.items():
            out_path = out_dirs[leaf] / f"ref_sim_{sim_id_str}.npy"
            if out_path.exists() and not args.force:
                print(
                    f"  sim_{sim_id_str}/{leaf}: already exists, "
                    "skipping (--force to overwrite)"
                )
            else:
                fields_needed[leaf] = vtk_key

        if not fields_needed:
            continue

        print(
            f"  sim_{sim_id_str}: loading {args.n_frames} frames "
            f"[{', '.join(sorted(fields_needed))}] ... ",
            end="",
            flush=True,
        )
        try:
            trajectories = load_sim_fields(run_dir, fields_needed, args.n_frames)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        for leaf, traj in trajectories.items():
            out_path = out_dirs[leaf] / f"ref_sim_{sim_id_str}.npy"
            np.save(out_path, traj)

        saved = [
            f"{leaf}→{out_dirs[leaf] / f'ref_sim_{sim_id_str}.npy'}"
            for leaf in sorted(trajectories)
        ]
        shapes = [f"{leaf}:{traj.shape}" for leaf, traj in trajectories.items()]
        print(f"saved  ({', '.join(shapes)})")

    print("Done.")


if __name__ == "__main__":
    main()
