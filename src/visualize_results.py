"""
FNO Result Visualization Script

Loads pre-computed predictions from inference.py (pred_sim_<id>.npy) and generates:
- 200 PNGs per test file: 3-panel (target | prediction | error) for each of
  20 samples × 10 timesteps, labeled with absolute simulation frame numbers

Frame numbering: sample s, output step t → absolute frame 160 + s + t*80

Usage:
    python visualize_results.py --param <parameter_name> [--experiments-dir <path>]

Run inference.py first to produce pred_sim_*.npy files.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--param", type=str, default='density')
    parser.add_argument("--experiments-dir", type=str, default=EXPERIMENTS_DIR)
    parser.add_argument("--force", action="store_true", help="Regenerate PNGs even if they already exist")
    opt = parser.parse_args()

    exp_dir = opt.experiments_dir
    vis_dir = Path(exp_dir) / opt.param / 'visualizations'
    test_dir = Path(DATA_DIR) / opt.param / 'test'

    pred_files = sorted(vis_dir.glob("pred_sim_*.npy"))
    if not pred_files:
        raise FileNotFoundError(f"No pred_sim_*.npy files found in {vis_dir}. Run inference.py first.")
    n_test = len(pred_files)

    print(f"Processing parameter: {opt.param}")
    print(f"Predictions directory: {vis_dir}")
    print(f"Ground truth directory: {test_dir}")
    print(f"Found {n_test} prediction files")

    total_pngs = 0
    skipped_pngs = 0

    for pred_path in tqdm(pred_files, desc="Test files", unit="file"):
        sim_id = pred_path.stem.split("_")[-1]
        pred = np.load(pred_path).squeeze(-1)               # (20, 128, 128, 10)
        y    = np.load(test_dir / f"y_sim_{sim_id}.npy")   # (20, 128, 128, 10)

        # Global color ranges for this simulation — fixed across all frames
        vmin_global = min(pred.min(), y.min())
        vmax_global = max(pred.max(), y.max())
        eabs_global = np.max(np.abs(y - pred))

        n_samples = pred.shape[0]  # 20

        for s in tqdm(range(n_samples), desc=f"Samples sim_{sim_id}", unit="s", leave=False):
            for t in range(pred.shape[3]):
                frame = 160 + s + t * 80
                fname = vis_dir / f'frame_{frame:04d}_sim_{sim_id}.png'

                if fname.exists() and not opt.force:
                    skipped_pngs += 1
                    continue

                target = y[s, :, :, t]
                pred_t = pred[s, :, :, t]
                error  = target - pred_t

                fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
                fig.suptitle(f'{opt.param}  |  sim {sim_id}  |  frame {frame}', fontsize=12)

                im0 = axes[0].pcolormesh(target, vmin=vmin_global, vmax=vmax_global, cmap='viridis')
                axes[0].set_title('Target')
                axes[0].set_aspect('equal')

                im1 = axes[1].pcolormesh(pred_t, vmin=vmin_global, vmax=vmax_global, cmap='viridis')
                axes[1].set_title('Prediction')
                axes[1].set_aspect('equal')

                im2 = axes[2].pcolormesh(error, vmin=-eabs_global, vmax=eabs_global, cmap='RdBu_r')
                axes[2].set_title('Error (Target - Prediction)')
                axes[2].set_aspect('equal')

                for ax, im in zip(axes, [im0, im1, im2]):
                    fig.colorbar(im, ax=ax, location='bottom', shrink=0.9, pad=0.08)

                plt.savefig(fname, dpi=100)
                plt.close(fig)
                total_pngs += 1

    print(f"\nDone. Generated {total_pngs} PNGs, skipped {skipped_pngs} already existing, in {vis_dir}/")

    # --- movies (one per simulation) ---
    for pred_path in pred_files:
        sim_id = pred_path.stem.split("_")[-1]
        movie_path = vis_dir / f'movie_sim_{sim_id}.mp4'

        if movie_path.exists():
            print(f"\nSkipping movie for sim {sim_id} (already exists)")
            continue

        # build a concat list of all PNGs for this sim, sorted by frame number
        png_list = sorted(vis_dir.glob(f'frame_*_sim_{sim_id}.png'))
        concat_file = vis_dir / f'_concat_sim_{sim_id}.txt'
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
            movie_path.name
        ]
        print(f"\nRendering movie for sim {sim_id}...")
        subprocess.run(cmd, check=True, cwd=str(vis_dir))
        concat_file.unlink()
        print(f"Saved: {movie_path}")


if __name__ == '__main__':
    main()
