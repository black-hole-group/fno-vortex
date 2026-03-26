"""
FNO Result Visualization Script

Loads pre-computed predictions from inference.py (pred_sim_<id>.npy) and generates:
- 10 PNGs per test file: 3-panel (target | prediction | error) for each timestep
- 1 GIF per test file: animation of all 10 timesteps for sample 0

Usage:
    python visualize_results.py --param <parameter_name> [--experiments-dir <path>]

Run inference.py first to produce pred_sim_*.npy files.
"""

import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
    opt = parser.parse_args()

    exp_dir = opt.experiments_dir
    vis_dir = Path(exp_dir) / opt.param / 'visualizations'
    test_dir = Path(DATA_DIR) / opt.param / 'test'

    # Infer test files from saved predictions
    pred_files = sorted(vis_dir.glob("pred_sim_*.npy"))
    if not pred_files:
        raise FileNotFoundError(f"No pred_sim_*.npy files found in {vis_dir}. Run inference.py first.")
    n_test = len(pred_files)

    print(f"Processing parameter: {opt.param}")
    print(f"Predictions directory: {vis_dir}")
    print(f"Ground truth directory: {test_dir}")
    print(f"Found {n_test} prediction files")
    print(f"Generating {10 * n_test} PNGs and {n_test} GIFs...\n")

    total_pngs = 0
    total_gifs = 0

    for pred_path in tqdm(pred_files, desc="Test files", unit="file"):
        sim_id = pred_path.stem.split("_")[-1]
        pred = np.load(pred_path).squeeze(-1)               # (20, 128, 128, 10)
        y    = np.load(test_dir / f"y_sim_{sim_id}.npy")   # (20, 128, 128, 10)

        s = 0  # representative sample index

        # --- 10 static PNGs ---
        for t in tqdm(range(10), desc=f"PNG sim_{sim_id}", unit="t", leave=False):
            target = y[s, :, :, t]
            pred_t = pred[s, :, :, t]
            error  = target - pred_t

            vmin = min(target.min(), pred_t.min())
            vmax = max(target.max(), pred_t.max())
            eabs = max(abs(error.min()), abs(error.max()))

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f'{opt.param}  |  sim {sim_id}  |  timestep {t}', fontsize=12)

            im0 = axes[0].pcolormesh(target, vmin=vmin, vmax=vmax, cmap='viridis')
            axes[0].set_title('Target')
            axes[0].set_aspect('equal')

            im1 = axes[1].pcolormesh(pred_t, vmin=vmin, vmax=vmax, cmap='viridis')
            axes[1].set_title('Prediction')
            axes[1].set_aspect('equal')

            im2 = axes[2].pcolormesh(error, vmin=-eabs, vmax=eabs, cmap='RdBu_r')
            axes[2].set_title('Error (Target - Prediction)')
            axes[2].set_aspect('equal')

            for ax, im in zip(axes, [im0, im1, im2]):
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('bottom', size='5%', pad=0.4)
                fig.colorbar(im, cax=cax, orientation='horizontal')

            plt.tight_layout()
            fname = vis_dir / f'sample_sim_{sim_id}_time_{t:02d}.png'
            plt.savefig(fname, dpi=100, bbox_inches='tight')
            plt.close(fig)

        total_pngs += 10

        # --- 1 GIF animation ---
        with tqdm(desc=f"GIF sim_{sim_id}", unit="gif", leave=False) as pbar:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f'{opt.param} | sim {sim_id} | sample 0', fontsize=12)

            vmin_all = min(y[s].min(), pred[s].min())
            vmax_all = max(y[s].max(), pred[s].max())
            eabs_all = max(abs((y[s] - pred[s]).min()), abs((y[s] - pred[s]).max()))

            target0 = y[s, :, :, 0]
            pred0   = pred[s, :, :, 0]
            error0  = target0 - pred0

            im0 = axes[0].pcolormesh(target0, vmin=vmin_all, vmax=vmax_all, cmap='viridis')
            axes[0].set_title('Target')
            axes[0].set_aspect('equal')
            im1 = axes[1].pcolormesh(pred0,   vmin=vmin_all, vmax=vmax_all, cmap='viridis')
            axes[1].set_title('Prediction')
            axes[1].set_aspect('equal')
            im2 = axes[2].pcolormesh(error0,  vmin=-eabs_all, vmax=eabs_all, cmap='RdBu_r')
            axes[2].set_title('Error (Target - Prediction)')
            axes[2].set_aspect('equal')

            for ax, im in zip(axes, [im0, im1, im2]):
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('bottom', size='5%', pad=0.4)
                fig.colorbar(im, cax=cax, orientation='horizontal')

            plt.tight_layout()
            time_text = fig.text(0.5, 1.01, '', ha='center', va='bottom',
                                 transform=axes[1].transAxes, fontsize=11)

            def update(t):
                target_t = y[s, :, :, t]
                pred_t   = pred[s, :, :, t]
                error_t  = target_t - pred_t
                im0.set_array(target_t.ravel())
                im1.set_array(pred_t.ravel())
                im2.set_array(error_t.ravel())
                time_text.set_text(f'timestep {t}')
                return im0, im1, im2, time_text

            ani = animation.FuncAnimation(fig, update, frames=10, interval=400, blit=False)
            gif_path = vis_dir / f'sample_sim_{sim_id}_evolution.gif'
            ani.save(gif_path, writer='pillow', dpi=80)
            plt.close(fig)
            pbar.update(1)
        total_gifs += 1

    print(f"\nDone. Generated {total_pngs} PNGs and {total_gifs} GIFs in {vis_dir}/")


if __name__ == '__main__':
    main()
