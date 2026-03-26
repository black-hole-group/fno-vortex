import torch
import numpy as np
import torch.nn.functional as F
import argparse
from contextlib import nullcontext
import shutil
import sys
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


def compress_series(values, max_points):
    if len(values) <= max_points:
        return values

    edges = np.linspace(0, len(values), max_points + 1, dtype=int)
    compressed = []
    for idx in range(max_points):
        start = edges[idx]
        end = edges[idx + 1]
        if start < end:
            compressed.append(float(np.mean(values[start:end])))
    return compressed


def visual_line_count(text, cols):
    total = 0
    for line in text.splitlines() or [""]:
        total += max(1, (len(line) + cols - 1) // cols)
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--param", type=str, default='density')
    parser.add_argument("--experiments-dir", type=str, default=EXPERIMENTS_DIR)
    parser.add_argument("--resume", action='store_true')
    opt = parser.parse_args()

    exp_dir = opt.experiments_dir
    os.makedirs(os.path.join(exp_dir, opt.param, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, opt.param, 'visualizations'), exist_ok=True)

    n_train = len(list((Path(DATA_DIR) / opt.param / 'train').glob('x_sim_*.npy')))
    n_test  = len(list((Path(DATA_DIR) / opt.param / 'test').glob('x_sim_*.npy')))

    learning_rate = 0.001
    scheduler_step = 500
    scheduler_gamma = 0.5
    epochs = 5000
    batch_size = 16

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    model = FNO3d(64, 64, 5, 30).cuda()
    # model = torch.compile(model)  # not supported on Python 3.14+
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    has_complex_params = any(torch.is_complex(param) for param in model.parameters())
    bf16_supported = torch.cuda.is_bf16_supported()
    if has_complex_params:
        autocast_enabled = bf16_supported
        autocast_dtype = torch.bfloat16 if bf16_supported else None
    else:
        autocast_enabled = True
        autocast_dtype = torch.bfloat16 if bf16_supported else torch.float16
    use_grad_scaler = autocast_enabled and autocast_dtype == torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=use_grad_scaler)

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
    if autocast_enabled:
        print(f"  Mixed precision: autocast {autocast_dtype} | grad scaler: {use_grad_scaler}")
    else:
        print("  Mixed precision: disabled (complex parameters require unsupported AMP unscale)")
    print(f"{'='*55}\n")

    log_path = os.path.join(exp_dir, opt.param, 'checkpoints', 'train.log')
    checkpoint_dir = os.path.join(exp_dir, opt.param, 'checkpoints')
    model_path = os.path.join(checkpoint_dir, 'model_64_30.pt')
    training_state_path = os.path.join(checkpoint_dir, 'training_state.pt')
    loss_path = os.path.join(checkpoint_dir, 'loss_64_30.npy')

    myloss = LpLoss(size_average=False)
    loss_function = []
    log10_mae_history = []

    def save_checkpoint(epoch_idx):
        checkpoint = {
            'epoch': epoch_idx,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'loss_function': loss_function,
            'log10_mae_history': log10_mae_history,
        }
        np.save(loss_path, loss_function)
        torch.save(model.state_dict(), model_path)
        torch.save(checkpoint, training_state_path)

    start_epoch = 0
    if opt.resume and os.path.exists(training_state_path):
        checkpoint = torch.load(training_state_path, map_location='cuda')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        loss_function = checkpoint.get('loss_function', loss_function)
        log10_mae_history = checkpoint.get('log10_mae_history', log10_mae_history)
        start_epoch = checkpoint.get('epoch', -1) + 1
        print(f"Resumed from epoch {start_epoch}")

    with open(log_path, 'a') as log:
        log.write(f"\n{'='*70}\n")
        log.write(f"  Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"  Param: {opt.param}\n")
        log.write(f"  Resume: {opt.resume}\n")
        log.write(f"  FNO3d(modes=64/64/5, width=30)  |  Parameters: {n_params:,}\n")
        log.write(f"  Input shape: {tuple(x0_shape)}  |  Output shape: {tuple(y0_shape)}\n")
        log.write(f"  Train files: {n_train}  |  Test files: {n_test}\n")
        log.write(f"  Epochs: {epochs}  |  Batch size: {batch_size}  |  LR: {learning_rate}\n")
        log.write(f"  Scheduler: StepLR(step={scheduler_step}, gamma={scheduler_gamma})\n")
        if autocast_enabled:
            log.write(
                f"  Mixed precision: autocast {autocast_dtype} | grad scaler: {use_grad_scaler}\n"
            )
        else:
            log.write(
                "  Mixed precision: disabled (complex parameters require unsupported AMP unscale)\n"
            )
        log.write(f"{'='*70}\n")
        log.write(f"{'epoch':>6}  {'time_s':>7}  {'lr':>10}  {'mae':>12}  {'loss':>12}  {'log10_mae':>10}  {'eta':>12}\n")
        log.write(f"{'-'*84}\n")

    t1_final = default_timer()
    vis_idx = 0
    # 20 samples per file, batch_size=4 → 5 gradient steps per file
    steps_per_file = (20 + batch_size - 1) // batch_size

    try:
        for ep in range(start_epoch, epochs):
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
                    if autocast_enabled:
                        autocast_context = torch.amp.autocast(
                            'cuda', dtype=autocast_dtype
                        )
                    else:
                        autocast_context = nullcontext()
                    with autocast_context:
                        out = model(x).view(actual_bs, 128, 128, 10)

                        mae = F.l1_loss(out, y, reduction='mean')

                        # Denormalize using file-level min/max (consistent with normalization)
                        y_denorm   = y_min + ((y   + 1) * (y_max - y_min) / 2)
                        out_denorm = y_min + ((out + 1) * (y_max - y_min) / 2)
                        l2 = myloss(out_denorm.view(actual_bs, -1), y_denorm.view(actual_bs, -1))

                        loss = mae + l2
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    train_mae  += mae.item()
                    train_loss += loss.item()

            scheduler.step()
            model.eval()

            if (ep + 1) % 50 == 0 or ep == 0:
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
            avg_epoch_time = (t2 - t1_final) / (ep - start_epoch + 1)
            eta_sec = int(avg_epoch_time * (epochs - ep - 1))
            eh, erem = divmod(eta_sec, 3600)
            em, es = divmod(erem, 60)
            eta_str = f"{eh}h {em}m {es}s"
            log10_mae_history.append(np.log10(train_mae))
            cols = shutil.get_terminal_size(fallback=(100, 24)).columns
            chart_cols = max(20, cols - 12)
            chart_history = compress_series(log10_mae_history[-100:], chart_cols)
            chart = asciichartpy.plot(
                chart_history,
                {'height': 8, 'format': '{:6.2f}', 'offset': 2},
            )
            eta_compact = f"{eh:02d}:{em:02d}:{es:02d}"
            output = (
                f"Ep {ep+1}/{epochs} | {t2-t1:.1f}s | lr {lr_now:.2e} | ETA {eta_compact}\n"
                f"MAE {train_mae:.4e} | Loss {train_loss:.4e}\n"
                f"{chart}\n"
                f"  log10(MAE): {log10_mae_history[-1]:.3f}"
            )
            n_lines = visual_line_count(output, cols)
            # Move cursor up to overwrite previous epoch's output
            if ep > start_epoch and sys.stdout.isatty():
                print(f"\033[{prev_n_lines}A\033[J", end='')
            print(output)
            prev_n_lines = n_lines

            loss_function.append(train_mae)
            loss_function.append(train_loss)

            with open(log_path, 'a') as log:
                log.write(f"{ep+1:>6}  {t2-t1:>7.1f}  {lr_now:>10.3e}  {train_mae:>12.6f}  {train_loss:>12.6f}  {log10_mae_history[-1]:>10.4f}  {eta_str:>12}\n")

            if (ep + 1) % 100 == 0 or ep == epochs - 1:
                save_checkpoint(ep)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving latest checkpoint...")
        if 'ep' in locals():
            save_checkpoint(ep)
        return

    save_checkpoint(epochs - 1)
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
