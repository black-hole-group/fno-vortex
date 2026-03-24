#!/usr/bin/env python3
"""
Render Idefix Orszag-Tang VTK snapshots as 3-panel images and assemble a movie.

Panels: density | magnetic field (|B| + vectors) | velocity field (|v| + vectors)

Usage:
  python render_vtk.py <sim_dir> [--fps 15] [--stride 1] [--output movie.mp4]

  <sim_dir>  path to a run directory, e.g. runs/sim_000
  --fps      frames per second in the output movie (default: 15)
  --stride   only render every Nth VTK file (default: 1)
  --output   output movie path (default: <sim_dir>/movie.mp4)

Requires:
  pip install matplotlib numpy tqdm
  ffmpeg must be available on $PATH
  $IDEFIX_DIR must point to the Idefix source tree (for pytools/vtk_io.py)
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
from tqdm import tqdm
import matplotlib.pyplot as plt

# ── locate Idefix pytools/vtk_io.py ───────────────────────────────────────────
_idefix_dir = os.environ.get("IDEFIX_DIR")
if _idefix_dir:
    sys.path.insert(0, _idefix_dir)
else:
    # Fall back: look for idefix/ four levels up from this script
    _guess = Path(__file__).resolve().parents[4] / "idefix"
    if _guess.exists():
        sys.path.insert(0, str(_guess))
    else:
        print("Error: $IDEFIX_DIR is not set and idefix/ was not found automatically.")
        print("Set $IDEFIX_DIR to the Idefix source directory and re-run.")
        sys.exit(1)

from pytools.vtk_io import readVTK  # noqa: E402

# Subsample vector arrows every N cells (128/8 = 16 arrows per axis)
QUIVER_STRIDE = 8


def compute_clims(vtk_files: list) -> dict:
    """First pass: compute global min/max for rho, |B|, |v| across all frames."""
    rho_min, rho_max = np.inf, -np.inf
    bmag_min, bmag_max = np.inf, -np.inf
    vmag_min, vmag_max = np.inf, -np.inf

    for vtk in tqdm(vtk_files, unit="file", desc="scanning"):
        V = readVTK(str(vtk))
        rho  = V.data["RHO"][:, :, 0]
        vx   = V.data["VX1"][:, :, 0]
        vy   = V.data["VX2"][:, :, 0]
        bx   = V.data["BX1"][:, :, 0]
        by   = V.data["BX2"][:, :, 0]
        bmag = np.sqrt(bx**2 + by**2)
        vmag = np.sqrt(vx**2 + vy**2)

        rho_min,  rho_max  = min(rho_min,  rho.min()),  max(rho_max,  rho.max())
        bmag_min, bmag_max = min(bmag_min, bmag.min()), max(bmag_max, bmag.max())
        vmag_min, vmag_max = min(vmag_min, vmag.min()), max(vmag_max, vmag.max())

    return {
        "rho":  (rho_min,  rho_max),
        "bmag": (bmag_min, bmag_max),
        "vmag": (vmag_min, vmag_max),
    }


def render_frame(vtk_path: Path, out_path: Path, clims: dict) -> None:
    V = readVTK(str(vtk_path))

    # 2D fields: (nx, ny, nz=1) → (nx, ny)
    rho = V.data["RHO"][:, :, 0]
    vx  = V.data["VX1"][:, :, 0]
    vy  = V.data["VX2"][:, :, 0]
    bx  = V.data["BX1"][:, :, 0]
    by  = V.data["BX2"][:, :, 0]

    x, y = V.x, V.y
    X, Y = np.meshgrid(x, y, indexing="ij")   # (nx, ny)

    s = QUIVER_STRIDE
    Xq, Yq   = X[::s, ::s],  Y[::s, ::s]
    vxq, vyq = vx[::s, ::s], vy[::s, ::s]
    bxq, byq = bx[::s, ::s], by[::s, ::s]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    fig.suptitle(f"t = {float(V.t):.3f}", fontsize=11)

    # Panel 1: density
    ax = axes[0]
    im = ax.pcolormesh(x, y, rho.T, cmap="inferno", shading="auto",
                       vmin=clims["rho"][0], vmax=clims["rho"][1])
    fig.colorbar(im, ax=ax)
    ax.set_title(r"Density  $\rho$")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Panel 2: magnetic field magnitude + vectors
    ax = axes[1]
    bmag = np.sqrt(bx**2 + by**2)
    im = ax.pcolormesh(x, y, bmag.T, cmap="cividis", shading="auto",
                       vmin=clims["bmag"][0], vmax=clims["bmag"][1])
    fig.colorbar(im, ax=ax, label="|B|")
    ax.quiver(Xq, Yq, bxq, byq, color="white", alpha=0.8, pivot="mid")
    ax.set_title(r"Magnetic field  $\mathbf{B}$")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Panel 3: velocity magnitude + vectors
    ax = axes[2]
    vmag = np.sqrt(vx**2 + vy**2)
    im = ax.pcolormesh(x, y, vmag.T, cmap="viridis", shading="auto",
                       vmin=clims["vmag"][0], vmax=clims["vmag"][1])
    fig.colorbar(im, ax=ax, label="|v|")
    ax.quiver(Xq, Yq, vxq, vyq, color="white", alpha=0.8, pivot="mid")
    ax.set_title(r"Velocity field  $\mathbf{v}$")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Render Idefix VTK snapshots and assemble a movie"
    )
    parser.add_argument("sim_dir",
                        help="Path to simulation run directory (e.g. runs/sim_000)")
    parser.add_argument("--fps", type=float, default=15.0,
                        help="Movie frame rate (default: 15)")
    parser.add_argument("--stride", type=int, default=1,
                        help="Render every Nth VTK file (default: 1)")
    parser.add_argument("--output", default=None,
                        help="Output movie path (default: <sim_dir>/movie.mp4)")
    args = parser.parse_args()

    sim_dir = Path(args.sim_dir).resolve()
    frames_dir = sim_dir / "frames"
    shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir()

    vtk_files = sorted(sim_dir.glob("data.????.vtk"))[::args.stride]
    if not vtk_files:
        print(f"No VTK files found in {sim_dir}")
        sys.exit(1)

    print(f"Found {len(vtk_files)} VTK files.")
    clims = compute_clims(vtk_files)

    print(f"Rendering frames to {frames_dir}/")
    frame_paths = []
    with tqdm(vtk_files, unit="frame", desc="rendering") as pbar:
        for i, vtk in enumerate(pbar):
            out = frames_dir / f"frame_{i:04d}.png"
            pbar.set_postfix_str(f"{vtk.name} → {out.name}")
            render_frame(vtk, out, clims)
            frame_paths.append(out)

    out_movie = Path(args.output) if args.output else sim_dir / "movie.mp4"
    print(f"\nAssembling {len(frame_paths)} frames → {out_movie}")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(args.fps),
        "-i", str(frames_dir / "frame_%04d.png"),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(out_movie),
    ]
    subprocess.run(cmd, check=True)
    print("Done.")


if __name__ == "__main__":
    main()
