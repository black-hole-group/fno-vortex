"""
Batch runner for Idefix Orszag-Tang vortex simulations.

For each row in params.csv:
  1. Creates a run directory: runs/sim_XXX/
  2. Copies idefix.ini and patches nu and eta using inifix
  3. Symlinks the compiled idefix binary
  4. Runs the simulation (sequential, one at a time)

Usage:
  python run_simulations.py [--start N] [--end N] [--params params.csv]

Dependencies:
  pip install inifix
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

import inifix


SCRIPT_DIR = Path(__file__).parent.resolve()
RUNS_DIR = SCRIPT_DIR / "runs"
INI_TEMPLATE = SCRIPT_DIR / "idefix.ini"
IDEFIX_BIN = SCRIPT_DIR / "idefix"


def load_params(params_file):
    with open(params_file) as f:
        reader = csv.DictReader(f)
        return list(reader)


def run_simulation(sim_id, nu, mu, split):
    run_dir = RUNS_DIR / f"sim_{sim_id:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy and patch idefix.ini
    ini_dst = run_dir / "idefix.ini"
    shutil.copy(INI_TEMPLATE, ini_dst)

    conf = inifix.load(str(ini_dst))
    conf["Viscosity"]["nu"] = float(nu)
    conf["Resistivity"]["eta"] = float(mu)
    inifix.dump(conf, str(ini_dst))

    # Symlink the compiled binary
    bin_link = run_dir / "idefix"
    if bin_link.exists() or bin_link.is_symlink():
        bin_link.unlink()
    bin_link.symlink_to(IDEFIX_BIN)

    print(f"\n[sim_{sim_id:03d}] nu={float(nu):.3e}  mu={float(mu):.3e}  split={split}")
    print(f"  Run dir: {run_dir}")

    log_path = run_dir / "idefix.log"
    with open(log_path, "w") as log_file:
        result = subprocess.run(
            ["./idefix"],
            cwd=run_dir,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    if result.returncode != 0:
        print(f"  ERROR: simulation failed (exit code {result.returncode}). "
              f"Check {log_path}")
        return False

    # Quick sanity check: expect ~1001 VTK files
    vtk_files = list(run_dir.glob("data.????.vtk"))
    print(f"  Done. VTK files produced: {len(vtk_files)}")
    if len(vtk_files) < 1000:
        print(f"  WARNING: expected ~1001 VTK files, got {len(vtk_files)}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Run Idefix Orszag-Tang simulations")
    parser.add_argument("--params", default=str(SCRIPT_DIR / "params.csv"),
                        help="Path to params.csv")
    parser.add_argument("--start", type=int, default=0,
                        help="First sim_id to run (inclusive)")
    parser.add_argument("--end", type=int, default=None,
                        help="Last sim_id to run (inclusive). Defaults to last.")
    args = parser.parse_args()

    if not IDEFIX_BIN.exists():
        print(f"Error: Idefix binary not found at {IDEFIX_BIN}")
        print("Run build.sh first.")
        sys.exit(1)

    params = load_params(args.params)

    end = args.end if args.end is not None else len(params) - 1
    subset = [p for p in params if args.start <= int(p["sim_id"]) <= end]

    print(f"Running {len(subset)} simulations (sim {args.start} to {end})")

    failed = []
    for p in subset:
        ok = run_simulation(int(p["sim_id"]), p["nu"], p["mu"], p["split"])
        if not ok:
            failed.append(p["sim_id"])

    print(f"\n{'='*50}")
    print(f"Completed {len(subset) - len(failed)}/{len(subset)} simulations successfully.")
    if failed:
        print(f"Failed sim_ids: {failed}")


if __name__ == "__main__":
    main()
