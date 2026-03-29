import torch
import numpy as np
import argparse
from timeit import default_timer
from tqdm import tqdm
from pathlib import Path

from architecture import FNO3d

import os

torch.manual_seed(0)
np.random.seed(0)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
EXPERIMENTS_DIR = os.path.join(ROOT_DIR, 'experiments')

T_IN = 20  # temporal grid dimension (matches model's T_OUT)


def _normalize_frames(x):
    """Min-max normalize frame channels (0-4) of x to [-1, 1] in-place. Returns (xmin, xmax)."""
    frames = x[:, :, :, :, :-2]
    xmin = frames.min()
    xmax = frames.max()
    x[:, :, :, :, :-2] = 2 * ((frames - xmin) / (xmax - xmin)) - 1
    return float(xmin), float(xmax)


def run_rollout(model, x_init, y_path, rollout_steps):
    """Run autoregressive rollout.

    Each step predicts 20 frames from 5 input frames. The last 5 predicted
    frames become the input for the next step.

    Args:
        model: FNO3d model (on CUDA, in eval mode)
        x_init: numpy array (N_samples, 128, 128, T_IN, 7) — uses sample 0 only
        y_path: path to ground-truth y file (used for step-1 denormalization stats)
        rollout_steps: number of chained prediction steps

    Returns:
        numpy array (1, 128, 128, 20*rollout_steps) in physical units
    """
    # Step 1 — normalize and run; denormalize using ground-truth y statistics
    x = x_init[0:1].copy()  # (1, 128, 128, T_IN, 7)
    _normalize_frames(x)
    out = model(torch.from_numpy(x).float().cuda()).cpu().numpy().squeeze(-1)  # (1, 128, 128, 20)

    y = np.load(y_path)
    ymin, ymax = float(y.min()), float(y.max())
    out_denorm = ((out + 1) / 2) * (ymax - ymin) + ymin  # (1, 128, 128, 20)

    all_frames = [out_denorm]

    nu = float(x_init[0, 0, 0, 0, 5])
    mu = float(x_init[0, 0, 0, 0, 6])

    for _ in range(1, rollout_steps):
        # Use last 5 of the 20 predicted frames as the new input channels
        last5 = out_denorm[0, :, :, 15:]  # (128, 128, 5)

        x_new = np.zeros((1, 128, 128, T_IN, 7), dtype=np.float32)
        for ch in range(5):
            x_new[0, :, :, :, ch] = last5[:, :, ch:ch+1]  # broadcast frame to T_IN
        x_new[0, :, :, :, 5] = nu
        x_new[0, :, :, :, 6] = mu

        # Normalize frame channels; keep their stats for denormalization
        frame_min, frame_max = _normalize_frames(x_new)

        out = model(torch.from_numpy(x_new).float().cuda()).cpu().numpy().squeeze(-1)
        out_denorm = ((out + 1) / 2) * (frame_max - frame_min) + frame_min

        all_frames.append(out_denorm)

    return np.concatenate(all_frames, axis=3)  # (1, 128, 128, 20*rollout_steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--param", type=str, default='density')
    parser.add_argument("--experiments-dir", type=str, default=EXPERIMENTS_DIR)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file. Defaults to <experiments-dir>/<param>/checkpoints/model_64_30.pt")
    parser.add_argument("--rollout-steps", type=int, default=1,
                        help="Number of autoregressive 20-frame prediction steps (default: 1 = teacher-forced)")
    opt = parser.parse_args()

    exp_dir = opt.experiments_dir
    vis_dir = os.path.join(exp_dir, opt.param, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    checkpoint_path = opt.checkpoint if opt.checkpoint else \
        os.path.join(exp_dir, opt.param, 'checkpoints', 'model_64_30.pt')

    print(f"Processing parameter: {opt.param}")

    model = FNO3d(64, 64, 5, 30).cuda()
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()

    test_dir = Path(DATA_DIR) / opt.param / 'test'
    test_files = sorted(test_dir.glob("x_sim_*.npy"))
    n_test = len(test_files)

    if n_test == 0:
        print(f"ERROR: No test files (x_sim_*.npy) found in {test_dir}")
        return

    print(f"Test data directory: {test_dir}")
    print(f"Output directory: {vis_dir}")
    print(f"Loading model from: {checkpoint_path}")

    if opt.rollout_steps > 1:
        print(f"Autoregressive rollout: {opt.rollout_steps} steps ({opt.rollout_steps * 20} frames total per simulation)")

    t1 = default_timer()
    print(f"\nRunning inference on {n_test} test files...")

    with torch.no_grad():
        for x_path in tqdm(test_files, desc="Inference", unit="file"):
            sim_id = x_path.stem.split("_")[-1]
            y_path = test_dir / f"y_sim_{sim_id}.npy"

            x = np.load(x_path)

            if opt.rollout_steps == 1:
                # Teacher-forced inference (original behavior, unchanged)
                x[:,:,:,:,:-2] = 2*((x[:,:,:,:,:-2] - np.min(x[:,:,:,:,:-2])) / (np.max(x[:,:,:,:,:-2]) - np.min(x[:,:,:,:,:-2]))) - 1
                x = torch.from_numpy(x).float().cuda()

                out = model(x).cpu().numpy()

                del x

                y = np.load(y_path)
                out = ((out + 1)/2)*(np.max(y) - np.min(y)) + np.min(y)

                np.save(os.path.join(vis_dir, f'pred_sim_{sim_id}.npy'), out)
            else:
                # Autoregressive rollout
                out = run_rollout(model, x, y_path, opt.rollout_steps)
                np.save(os.path.join(vis_dir, f'pred_sim_{sim_id}_rollout.npy'), out)

    t2 = default_timer()

    print(f"\nInference completed in {t2-t1:.2f}s")
    print(f"Saved {n_test} predictions to {vis_dir}/")


if __name__ == '__main__':
    main()
