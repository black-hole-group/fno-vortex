"""
FNO Result Visualization Script

Loads pre-computed predictions from inference.py (pred_<j>.npy) and generates:
- 10 PNGs per test file: 3-panel (target | prediction | error) for each timestep
- 1 GIF per test file: animation of all 10 timesteps for sample 0

Usage:
    python visualize_results.py --param <parameter_name> [--experiments-dir <path>]

Run inference.py first to produce pred_*.npy files.
"""

import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
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
    vis_dir = os.path.join(exp_dir, opt.param, 'visualizations')
    test_dir = os.path.join(DATA_DIR, opt.param, 'test')

    # Infer number of test files from saved predictions
    n_test = sum(1 for f in os.listdir(vis_dir) if f.startswith('pred_') and f.endswith('.npy'))
    if n_test == 0:
        raise FileNotFoundError(f"No pred_*.npy files found in {vis_dir}. Run inference.py first.")

    print(f"Processing parameter: {opt.param}")
    print(f"Predictions directory: {vis_dir}")
    print(f"Ground truth directory: {test_dir}")
    print(f"Found {n_test} prediction files")
    print(f"Generating {10 * n_test} PNGs and {n_test} GIFs...\n")

    total_pngs = 0
    total_gifs = 0

    for j in tqdm(range(n_test), desc="Test files", unit="file"):
        pred = np.load(os.path.join(vis_dir, f'pred_{j}.npy')).squeeze(-1)   # (20, 128, 128, 10)
        y    = np.load(os.path.join(test_dir, 'y_'+str(j)+'.npy'))  # (20, 128, 128, 10)

        s = 0  # representative sample index

        # --- 10 static PNGs ---
        for t in tqdm(range(10), desc=f"PNG {j}", unit="t", leave=False):
            target = y[s, :, :, t]
            pred_t = pred[s, :, :, t]
            error  = target - pred_t

            vmin = min(target.min(), pred_t.min())
            vmax = max(target.max(), pred_t.max())
            eabs = max(abs(error.min()), abs(error.max()))

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f'{opt.param}  |  test file {j}  |  timestep {t}', fontsize=12)

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
            fname = os.path.join(vis_dir, f'sample_{j:02d}_time_{t:02d}.png')
            plt.savefig(fname, dpi=100, bbox_inches='tight')
            plt.close(fig)

        total_pngs += 10

        # --- 1 GIF animation ---
        with tqdm(desc=f"GIF {j}", unit="gif", leave=False) as pbar:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f'{opt.param} | test file {j} | sample 0', fontsize=12)

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
            gif_path = os.path.join(vis_dir, f'sample_{j:02d}_evolution.gif')
            ani.save(gif_path, writer='pillow', dpi=80)
            plt.close(fig)
            pbar.update(1)
        total_gifs += 1

    print(f"\nDone. Generated {total_pngs} PNGs and {total_gifs} GIFs in {vis_dir}/")


if __name__ == '__main__':
    main()
