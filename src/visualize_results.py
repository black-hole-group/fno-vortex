"""
FNO Result Visualization Script

Supports both teacher-forced and autoregressive rollout predictions produced
by inference.py.

Teacher-forced (pred_sim_<id>.npy):
  Shape (20, 128, 128, T_out) — 20 sliding-window samples.
  For each sample s and output step t, absolute frame = 5 + s + t.
  Renders: reference | prediction | error

Autoregressive rollout (pred_sim_<id>_rollout.npy):
  Shape (1, 128, 128, 20*N) — single long trajectory from sample 0.
  Absolute frame index i maps to frame 5 + i.

  Reference data is taken from (in order of preference):
    1. ref_sim_<id>.npy in the test directory — dense full-trajectory
       array (N_frames, 128, 128) produced by prepare_reference.py.
       When available, every rollout frame gets a reference|pred|error panel.
    2. y_sim_<id>.npy sample 0 — sparse supervised windows.
       Only the first T_out frames have reference; beyond that the panel
       shows prediction only.

Usage:
    python visualize_results.py --param <parameter_name> \\
        [--experiments-dir <path>] [--force]

Run inference.py first to produce pred_sim_*.npy files.
For full-horizon rollout reference, run prepare_reference.py first.
"""

import numpy as np
import argparse
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
DATA_DIR        = os.path.join(ROOT_DIR, 'data')
EXPERIMENTS_DIR = os.path.join(ROOT_DIR, 'experiments')

# Number of initial input frames fed to the model (matches inference.py)
N_INPUT_FRAMES = 5


def parse_pred_file(path):
    """Return (sim_id, is_rollout) from a pred_sim_*.npy filename.

    Examples:
        pred_sim_042.npy        → ('042', False)
        pred_sim_042_rollout.npy → ('042', True)
    """
    stem = path.stem  # strip .npy
    without_prefix = stem[len('pred_sim_'):]
    if without_prefix.endswith('_rollout'):
        return without_prefix[:-len('_rollout')], True
    return without_prefix, False


def _save_frame_with_gt(vis_dir, fname, param, sim_id, frame,
                        target, pred_t, vmin, vmax, eabs, suffix=''):
    """Render a 3-panel (target | prediction | error) PNG."""
    error = target - pred_t
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    title = f'{param}  |  sim {sim_id}  |  frame {frame}'
    if suffix:
        title += f'  |  {suffix}'
    fig.suptitle(title, fontsize=12)

    im0 = axes[0].pcolormesh(target, vmin=vmin, vmax=vmax, cmap='viridis')
    axes[0].set_title('Reference')
    axes[0].set_aspect('equal')

    im1 = axes[1].pcolormesh(pred_t, vmin=vmin, vmax=vmax, cmap='viridis')
    axes[1].set_title('Prediction')
    axes[1].set_aspect('equal')

    im2 = axes[2].pcolormesh(error, vmin=-eabs, vmax=eabs, cmap='RdBu_r')
    axes[2].set_title('Error (Target - Prediction)')
    axes[2].set_aspect('equal')

    for ax, im in zip(axes, [im0, im1, im2]):
        fig.colorbar(im, ax=ax, location='bottom', shrink=0.9, pad=0.08)

    plt.savefig(vis_dir / fname, dpi=100)
    plt.close(fig)


def _save_frame_pred_only(vis_dir, fname, param, sim_id, frame,
                          pred_t, vmin, vmax, suffix=''):
    """Render a single-panel (prediction only) PNG, same figure size as 3-panel."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    title = f'{param}  |  sim {sim_id}  |  frame {frame}  |  prediction only'
    if suffix:
        title += f'  |  {suffix}'
    fig.suptitle(title, fontsize=12)

    im1 = axes[1].pcolormesh(pred_t, vmin=vmin, vmax=vmax, cmap='viridis')
    axes[1].set_title('Prediction')
    axes[1].set_aspect('equal')
    fig.colorbar(im1, ax=axes[1], location='bottom', shrink=0.9, pad=0.08)

    # Blank flanking panels keep figure size consistent for movies
    for ax in (axes[0], axes[2]):
        ax.set_visible(False)

    plt.savefig(vis_dir / fname, dpi=100)
    plt.close(fig)


def _load_rollout_reference(test_dir, sim_id):
    """Load the reference trajectory for rollout visualization.

    Tries, in order:
      1. ref_sim_<id>.npy — dense full-trajectory array (N_frames, H, W)
         produced by prepare_reference.py.  Returns (ref_array, True).
      2. y_sim_<id>.npy sample 0 — sparse supervised windows (H, W, T_out).
         Returns (ref_sparse, False).
    """
    ref_path = test_dir / f"ref_sim_{sim_id}.npy"
    if ref_path.exists():
        ref = np.load(ref_path)  # (N_frames, H, W)
        return ref, True

    # Fallback: sparse reference from first supervised window only
    y = np.load(test_dir / f"y_sim_{sim_id}.npy")  # (n_samples, H, W, T_out)
    return y[0], False             # (H, W, T_out)


def visualize_teacher_forced(pred_path, sim_id, test_dir, vis_dir,
                             param, force):
    """Visualize a teacher-forced prediction file.

    pred shape: (n_samples, 128, 128, T_out)
    y    shape: (n_samples, 128, 128, T_out)
    """
    pred = np.load(pred_path).squeeze(-1)
    y    = np.load(test_dir / f"y_sim_{sim_id}.npy")

    vmin_global = min(pred.min(), y.min())
    vmax_global = max(pred.max(), y.max())
    eabs_global = float(np.max(np.abs(y - pred)))

    total = 0
    skipped = 0
    n_samples = pred.shape[0]

    for s in tqdm(range(n_samples), desc=f"  sim {sim_id}", unit="s",
                  leave=False):
        for t in range(pred.shape[3]):
            frame = N_INPUT_FRAMES + s + t
            fname = f'frame_{frame:04d}_sim_{sim_id}.png'
            if (vis_dir / fname).exists() and not force:
                skipped += 1
                continue
            _save_frame_with_gt(
                vis_dir, fname, param, sim_id, frame,
                y[s, :, :, t], pred[s, :, :, t],
                vmin_global, vmax_global, eabs_global,
            )
            total += 1

    return total, skipped


def visualize_rollout(pred_path, sim_id, test_dir, vis_dir, param, force):
    """Visualize an autoregressive rollout prediction file.

    pred shape: (1, 128, 128, 20*N)  — single trajectory, sample 0 only

    Reference data (in order of preference):
      - ref_sim_<id>.npy  (dense, shape (N_frames, H, W)) — every rollout
        frame is compared against the true simulation.
      - y_sim_<id>.npy sample 0 (sparse, covers indices 0..T_out-1 only);
        frames beyond T_out fall back to prediction-only.

    Absolute frame index i → absolute simulation frame N_INPUT_FRAMES + i.
    Dense reference: ref[N_INPUT_FRAMES + i].
    """
    pred = np.load(pred_path)          # (1, 128, 128, 20*N)
    pred = pred[0]                     # (128, 128, 20*N)
    n_frames = pred.shape[2]

    ref, dense = _load_rollout_reference(test_dir, sim_id)
    # dense=True:  ref shape (N_sim_frames, H, W)
    # dense=False: ref shape (H, W, T_out) — sparse window 0 only

    if dense:
        n_ref = ref.shape[0]
        ref_available = n_ref  # ref[abs_frame] valid for abs_frame < n_ref
        # Build arrays for color scale computation over available overlap
        max_abs = min(N_INPUT_FRAMES + n_frames, n_ref)
        pred_for_scale = pred[:, :, :max_abs - N_INPUT_FRAMES]
        ref_for_scale  = ref[N_INPUT_FRAMES:max_abs]
        vmin_global = min(float(pred.min()), float(ref_for_scale.min()))
        vmax_global = max(float(pred.max()), float(ref_for_scale.max()))
        # Transpose ref_for_scale to (H,W,T) for error calc
        eabs_global = float(
            np.max(np.abs(ref_for_scale - pred_for_scale.transpose(2, 0, 1)))
        )
        if n_ref < N_INPUT_FRAMES + n_frames:
            print(
                f"  sim {sim_id}: reference has {n_ref} frames, "
                f"rollout needs {N_INPUT_FRAMES + n_frames}; "
                f"last {N_INPUT_FRAMES + n_frames - n_ref} frames will be "
                "prediction-only"
            )
    else:
        # Sparse fallback: ref is y0 with shape (H, W, T_out)
        t_out = ref.shape[2]
        overlap = min(n_frames, t_out)
        vmin_global = min(float(pred.min()), float(ref.min()))
        vmax_global = max(float(pred.max()), float(ref.max()))
        eabs_global = float(np.max(np.abs(ref[:, :, :overlap] - pred[:, :, :overlap])))
        print(
            f"  sim {sim_id}: no ref_sim_{sim_id}.npy found; "
            f"using sparse y_sim reference (frames 0..{t_out - 1} only). "
            "Run prepare_reference.py for full-horizon reference."
        )

    total = 0
    skipped = 0

    for i in tqdm(range(n_frames), desc=f"  sim {sim_id} rollout", unit="frame",
                  leave=False):
        frame = N_INPUT_FRAMES + i
        fname = f'frame_{frame:04d}_sim_{sim_id}_rollout.png'

        if (vis_dir / fname).exists() and not force:
            skipped += 1
            continue

        pred_t = pred[:, :, i]

        # Determine if reference exists for this frame
        if dense:
            has_ref = frame < ref.shape[0]
        else:
            has_ref = i < ref.shape[2]

        if has_ref:
            ref_t = ref[frame] if dense else ref[:, :, i]
            _save_frame_with_gt(
                vis_dir, fname, param, sim_id, frame,
                ref_t, pred_t,
                vmin_global, vmax_global, eabs_global,
                suffix='rollout',
            )
        else:
            _save_frame_pred_only(
                vis_dir, fname, param, sim_id, frame,
                pred_t, vmin_global, vmax_global,
                suffix='rollout',
            )
        total += 1

    return total, skipped


def render_movie(vis_dir, png_glob, movie_name, force):
    """Build an mp4 from PNGs matching png_glob (sorted by name)."""
    movie_path = vis_dir / movie_name

    if movie_path.exists() and not force:
        print(f"  Skipping movie {movie_name} (already exists)")
        return

    png_list = sorted(vis_dir.glob(png_glob))
    if not png_list:
        print(f"  No PNGs found for {movie_name}, skipping")
        return

    concat_file = vis_dir / f'_concat_{movie_name}.txt'
    with open(concat_file, 'w') as f:
        for p in png_list:
            f.write(f"file '{p.name}'\nduration 0.0667\n")  # ~15 fps

    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat', '-safe', '0',
        '-i', concat_file.name,
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
        '-pix_fmt', 'yuv420p',
        '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
        movie_name,
    ]
    print(f"  Rendering {movie_name}...")
    subprocess.run(cmd, check=True, cwd=str(vis_dir))
    concat_file.unlink()
    print(f"  Saved: {movie_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--param", type=str, default='density')
    parser.add_argument("--experiments-dir", type=str, default=EXPERIMENTS_DIR)
    parser.add_argument("--force", action="store_true",
                        help="Regenerate PNGs/movies even if they already exist")
    opt = parser.parse_args()

    exp_dir = opt.experiments_dir
    vis_dir = Path(exp_dir) / opt.param / 'visualizations'
    test_dir = Path(DATA_DIR) / opt.param / 'test'

    all_pred_files = sorted(vis_dir.glob("pred_sim_*.npy"))
    if not all_pred_files:
        raise FileNotFoundError(
            f"No pred_sim_*.npy files found in {vis_dir}. "
            "Run inference.py first."
        )

    # Separate teacher-forced from rollout files
    tf_files = []
    rollout_files = []
    for p in all_pred_files:
        sim_id, is_rollout = parse_pred_file(p)
        if is_rollout:
            rollout_files.append((p, sim_id))
        else:
            tf_files.append((p, sim_id))

    print(f"Processing parameter: {opt.param}")
    print(f"Predictions directory: {vis_dir}")
    print(f"Ground truth directory: {test_dir}")
    print(f"Found {len(tf_files)} teacher-forced file(s), "
          f"{len(rollout_files)} rollout file(s)")

    total_pngs = 0
    skipped_pngs = 0

    # --- teacher-forced ---
    for pred_path, sim_id in tqdm(tf_files, desc="Teacher-forced", unit="file"):
        n, s = visualize_teacher_forced(
            pred_path, sim_id, test_dir, vis_dir, opt.param, opt.force
        )
        total_pngs += n
        skipped_pngs += s

    # --- rollout ---
    for pred_path, sim_id in tqdm(rollout_files, desc="Rollout", unit="file"):
        n, s = visualize_rollout(
            pred_path, sim_id, test_dir, vis_dir, opt.param, opt.force
        )
        total_pngs += n
        skipped_pngs += s

    print(f"\nDone. Generated {total_pngs} PNGs, "
          f"skipped {skipped_pngs} already existing, in {vis_dir}/")

    # --- movies ---
    print("\nRendering movies...")
    for _, sim_id in tf_files:
        render_movie(
            vis_dir,
            png_glob=f'frame_*_sim_{sim_id}.png',
            movie_name=f'movie_sim_{sim_id}.mp4',
            force=opt.force,
        )
    for _, sim_id in rollout_files:
        render_movie(
            vis_dir,
            png_glob=f'frame_*_sim_{sim_id}_rollout.png',
            movie_name=f'movie_sim_{sim_id}_rollout.mp4',
            force=opt.force,
        )


if __name__ == '__main__':
    main()
