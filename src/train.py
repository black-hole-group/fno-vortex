import torch
import numpy as np
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from timeit import default_timer
from torch.optim import Adam
import os
from pathlib import Path
from datetime import datetime
import asciichartpy

from architecture import FNO3d
from utilities import LpLoss

torch.manual_seed(0)
np.random.seed(0)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
DATA_DIR        = os.path.join(ROOT_DIR, 'data')
EXPERIMENTS_DIR = os.path.join(ROOT_DIR, 'experiments')


def load_dataset(param, split):
    """Load and normalize all .npy files for a dataset split into memory.

    Returns a list of tuples (x_normalized, y_normalized, y_min, y_max).
    y_min/y_max are file-level scalars used for consistent denormalization.
    """
    cache = []
    data_dir = Path(DATA_DIR) / param / split
    for x_path in sorted(data_dir.glob("x_sim_*.npy")):
        sim_id = x_path.stem.split("_")[-1]
        y_path = data_dir / f"y_sim_{sim_id}.npy"

        x = np.load(x_path)
        x[:,:,:,:,:-2] = 2*((x[:,:,:,:,:-2] - np.min(x[:,:,:,:,:-2]))/(np.max(x[:,:,:,:,:-2]) - np.min(x[:,:,:,:,:-2]))) - 1
        x = torch.from_numpy(x).float()

        y_raw = np.load(y_path)
        y_min = float(np.min(y_raw))
        y_max = float(np.max(y_raw))
        y_norm = torch.from_numpy(2*((y_raw - y_min)/(y_max - y_min)) - 1).float()

        cache.append((x, y_norm, y_min, y_max))
    return cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--param", type=str, default='density')
    parser.add_argument("--experiments-dir", type=str, default=EXPERIMENTS_DIR)
    opt = parser.parse_args()

    exp_dir = opt.experiments_dir
    os.makedirs(os.path.join(exp_dir, opt.param, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, opt.param, 'visualizations'), exist_ok=True)

    n_train = len(list((Path(DATA_DIR) / opt.param / 'train').glob('x_sim_*.npy')))
    n_test  = len(list((Path(DATA_DIR) / opt.param / 'test').glob('x_sim_*.npy')))

    learning_rate = 0.001
    scheduler_step = 500
    scheduler_gamma = 0.5
    epochs = 10000
    batch_size = 16

    model = FNO3d(64, 64, 5, 30).cuda()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    # Load all data into memory once — avoids ~460K disk reads over 10k epochs
    print(f"Loading {n_train} train + {n_test} test files into memory...")
    train_cache = load_dataset(opt.param, 'train')
    test_cache  = load_dataset(opt.param, 'test')
    print("Data loaded.")

    n_params = sum(p.numel() for p in model.parameters())
    x0_shape = train_cache[0][0].shape
    y0_shape = train_cache[0][1].shape
    print(f"\n{'='*55}")
    print(f"  FNO3d(modes=64/64/5, width=30)")
    print(f"  Parameters : {n_params:,}")
    print(f"  Input shape: {tuple(x0_shape)}  (samples, x, y, t, channels)")
    print(f"  Output shape: {tuple(y0_shape)}  (samples, x, y, t)")
    print(f"  Train files: {n_train}  |  Test files: {n_test}")
    print(f"  Epochs: {epochs}  |  Batch size: {batch_size}  |  LR: {learning_rate}")
    print(f"  Scheduler: StepLR(step={scheduler_step}, gamma={scheduler_gamma})")
    print(f"  Param: {opt.param}")
    print(f"{'='*55}\n")

    log_path = os.path.join(exp_dir, opt.param, 'checkpoints', 'train.log')
    with open(log_path, 'a') as log:
        log.write(f"\n{'='*70}\n")
        log.write(f"  Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"  Param: {opt.param}\n")
        log.write(f"  FNO3d(modes=64/64/5, width=30)  |  Parameters: {n_params:,}\n")
        log.write(f"  Input shape: {tuple(x0_shape)}  |  Output shape: {tuple(y0_shape)}\n")
        log.write(f"  Train files: {n_train}  |  Test files: {n_test}\n")
        log.write(f"  Epochs: {epochs}  |  Batch size: {batch_size}  |  LR: {learning_rate}\n")
        log.write(f"  Scheduler: StepLR(step={scheduler_step}, gamma={scheduler_gamma})\n")
        log.write(f"{'='*70}\n")
        log.write(f"{'epoch':>6}  {'time_s':>7}  {'lr':>10}  {'mae':>12}  {'loss':>12}  {'log10_mae':>10}  {'eta':>12}\n")
        log.write(f"{'-'*84}\n")

    myloss = LpLoss(size_average=False)
    loss_function = []
    log10_mae_history = []
    t1_final = default_timer()
    vis_idx = 0
    # 20 samples per file, batch_size=4 → 5 gradient steps per file
    steps_per_file = (20 + batch_size - 1) // batch_size

    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_mae = 0
        train_loss = 0

        for bs in range(n_train):
            x_all, y_all, y_min, y_max = train_cache[bs]

            for l in range(0, 20, batch_size):
                x = x_all[l:l+batch_size].cuda()
                y = y_all[l:l+batch_size].cuda()
                actual_bs = len(x)

                optimizer.zero_grad()
                out = model(x).view(actual_bs, 128, 128, 10)

                mae = F.l1_loss(out, y, reduction='mean')

                # Denormalize using file-level min/max (consistent with normalization)
                y_denorm   = y_min + ((y   + 1) * (y_max - y_min) / 2)
                out_denorm = y_min + ((out + 1) * (y_max - y_min) / 2)
                l2 = myloss(out_denorm.view(actual_bs, -1), y_denorm.view(actual_bs, -1))

                loss = mae + l2
                loss.backward()

                optimizer.step()
                train_mae  += mae.item()
                train_loss += loss.item()

        scheduler.step()
        model.eval()

        with torch.no_grad():
            j = np.random.randint(n_test)
            xt = test_cache[j][0][4:4+batch_size].cuda()
            yt = test_cache[j][1][4:4+batch_size].cpu().numpy()

            out = model(xt).cpu().detach().numpy()

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))

            imt   = ax1.pcolormesh(yt[0, :, :, 0])
            im    = ax2.pcolormesh(out[0, :, :, 0, 0])
            error = ax3.pcolormesh(yt[0, :, :, 0] - out[0, :, :, 0, 0], cmap='hot')

            for ax, cm, label in [
                (ax3, error, 'Error'),
                (ax1, imt,   f'{opt.param} target'),
                (ax2, im,    f'{opt.param} prediction'),
            ]:
                divider = make_axes_locatable(ax)
                cax = divider.new_vertical(size='5%', pad=0.5, pack_start=True)
                fig.add_axes(cax)
                fig.colorbar(cm, cax=cax, orientation='horizontal')
                cax.set_xlabel(label)

            plt.savefig(os.path.join(exp_dir, opt.param, 'visualizations', str(vis_idx).zfill(4) + '.png'))
            plt.close(fig)
            vis_idx += 1

        total_steps = n_train * steps_per_file
        train_mae  /= total_steps
        train_loss /= total_steps

        t2 = default_timer()
        lr_now = scheduler.get_last_lr()[0]
        avg_epoch_time = (t2 - t1_final) / (ep + 1)
        eta_sec = int(avg_epoch_time * (epochs - ep - 1))
        eh, erem = divmod(eta_sec, 3600)
        em, es = divmod(erem, 60)
        eta_str = f"{eh}h {em}m {es}s"
        log10_mae_history.append(np.log10(train_mae))
        chart = asciichartpy.plot(log10_mae_history[-100:], {'height': 8, 'format': '{:8.3f}'})
        output = (
            f"Epoch {ep+1:>5}/{epochs}  |  time: {t2-t1:.1f}s  |  lr: {lr_now:.2e}  |  ETA: {eta_str}\n"
            f"  MAE: {train_mae:.6f}  |  Loss (MAE+L2): {train_loss:.6f}\n"
            f"{chart}\n"
            f"  log10(MAE): {log10_mae_history[-1]:.3f}"
        )
        n_lines = output.count('\n') + 1
        # Move cursor up to overwrite previous epoch's output
        if ep > 0:
            print(f"\033[{prev_n_lines}A\033[J", end='')
        print(output)
        prev_n_lines = n_lines

        loss_function.append(train_mae)
        loss_function.append(train_loss)

        with open(log_path, 'a') as log:
            log.write(f"{ep+1:>6}  {t2-t1:>7.1f}  {lr_now:>10.3e}  {train_mae:>12.6f}  {train_loss:>12.6f}  {log10_mae_history[-1]:>10.4f}  {eta_str:>12}\n")

        np.save(os.path.join(exp_dir, opt.param, 'checkpoints', 'loss_64_30.npy'), loss_function)
        torch.save(model.state_dict(), os.path.join(exp_dir, opt.param, 'checkpoints', 'model_64_30.pt'))

    torch.save(model.state_dict(), os.path.join(exp_dir, opt.param, 'checkpoints', 'model_64_30.pt'))
    t2_final = default_timer()
    elapsed = t2_final - t1_final
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    print(f"\nTraining complete. Total time: {h}h {m}m {s}s")
    with open(log_path, 'a') as log:
        log.write(f"{'-'*70}\n")
        log.write(f"  Run finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  Total time: {h}h {m}m {s}s\n")


if __name__ == '__main__':
    main()
