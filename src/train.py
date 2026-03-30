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
torch.manual_seed(0)
np.random.seed(0)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
DATA_DIR        = os.path.join(ROOT_DIR, 'data')
EXPERIMENTS_DIR = os.path.join(ROOT_DIR, 'experiments')
LOSS_DEFINITION_VERSION = 'normalized-mae-only-v1'
LOSS_HISTORY_COLUMNS = (
    'train_mae',
    'val_mae',
)


def load_dataset(param, split, max_files=None, max_samples=None):
    """Load and normalize all .npy files for a dataset split into memory.

    Returns a list of tuples (x_normalized, y_normalized, y_min, y_max).
    y_min/y_max are file-level scalars used for consistent denormalization.
    """
    cache = []
    data_dir = Path(DATA_DIR) / param / split
    for file_idx, x_path in enumerate(sorted(data_dir.glob("x_sim_*.npy"))):
        if max_files is not None and file_idx >= max_files:
            break
        sim_id = x_path.stem.split("_")[-1]
        y_path = data_dir / f"y_sim_{sim_id}.npy"

        x = np.load(x_path)
        if max_samples is not None:
            x = x[:max_samples]
        x_data = x[:, :, :, :, :-2]
        x_min = np.min(x_data)
        x_max = np.max(x_data)
        x_range = x_max - x_min
        if x_range > 0:
            x[:, :, :, :, :-2] = 2 * ((x_data - x_min) / x_range) - 1
        else:
            x[:, :, :, :, :-2] = 0
        x = torch.from_numpy(x).float()

        y_raw = np.load(y_path)
        if max_samples is not None:
            y_raw = y_raw[:max_samples]
        y_min = float(np.min(y_raw))
        y_max = float(np.max(y_raw))
        y_range = y_max - y_min
        if y_range > 0:
            y_norm = torch.from_numpy(2 * ((y_raw - y_min) / y_range) - 1).float()
        else:
            y_norm = torch.zeros_like(torch.from_numpy(y_raw).float())

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
    parser.add_argument(
        "--fast",
        action='store_true',
        help='Run a very small smoke-test training job',
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help='Training batch size',
    )
    parser.add_argument('--patience', type=int, default=500,
                        help='Early stopping patience in epochs (0 = disabled)')
    opt = parser.parse_args()
    if opt.batch_size <= 0:
        parser.error("--batch-size must be positive")

    exp_dir = opt.experiments_dir
    os.makedirs(os.path.join(exp_dir, opt.param, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, opt.param, 'visualizations'), exist_ok=True)

    n_train_available = len(list((Path(DATA_DIR) / opt.param / 'train').glob('x_sim_*.npy')))
    n_val_available = len(list((Path(DATA_DIR) / opt.param / 'val').glob('x_sim_*.npy')))

    learning_rate = 0.001
    scheduler_step = 500
    scheduler_gamma = 0.5
    epochs = 5000
    batch_size = opt.batch_size
    samples_per_file = 20
    train_files_limit = None
    val_files_limit = None
    viz_every = 50
    checkpoint_every = 100

    if opt.fast:
        epochs = 3
        batch_size = 4
        samples_per_file = 4
        train_files_limit = 1
        val_files_limit = 1
        viz_every = 1
        checkpoint_every = 1

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
    print(
        f"Loading {n_train_available} train + {n_val_available} val files into memory..."
    )
    train_cache = load_dataset(
        opt.param,
        'train',
        max_files=train_files_limit,
        max_samples=samples_per_file,
    )
    val_cache = load_dataset(
        opt.param,
        'val',
        max_files=val_files_limit,
        max_samples=samples_per_file,
    )
    print("Data loaded.")

    if len(train_cache) == 0 or len(val_cache) == 0:
        raise RuntimeError(
            "No training or validation files were loaded. "
            "Check that data/<param>/train/ and data/<param>/val/ exist."
        )

    n_train = len(train_cache)
    n_val = len(val_cache)

    n_params = sum(p.numel() for p in model.parameters())
    x0_shape = train_cache[0][0].shape
    y0_shape = train_cache[0][1].shape
    print(f"\n{'='*55}")
    print(f"  FNO3d(modes=64/64/5, width=30)")
    print(f"  Parameters : {n_params:,}")
    print(f"  Input shape: {tuple(x0_shape)}  (samples, x, y, t, channels)")
    print(f"  Output shape: {tuple(y0_shape)}  (samples, x, y, t)")
    print(f"  Train files: {n_train}  |  Val files: {n_val}")
    print(f"  Epochs: {epochs}  |  Batch size: {batch_size}  |  LR: {learning_rate}")
    print(f"  Scheduler: StepLR(step={scheduler_step}, gamma={scheduler_gamma})")
    print(f"  Param: {opt.param}")
    if opt.fast:
        print(
            "  Fast mode: enabled "
            f"(train files={train_files_limit}, val files={val_files_limit}, "
            f"samples/file={samples_per_file}, viz every epoch)"
        )
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

    loss_history = {name: [] for name in LOSS_HISTORY_COLUMNS}
    log10_mae_history = []
    log10_val_mae_history = []
    best_val_mae = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    best_model_path = os.path.join(checkpoint_dir, 'model_best.pt')

    def save_loss_history():
        if not loss_history['train_mae']:
            history_array = np.empty((0, len(LOSS_HISTORY_COLUMNS)), dtype=np.float32)
        else:
            history_array = np.column_stack(
                [np.asarray(loss_history[name], dtype=np.float32)
                 for name in LOSS_HISTORY_COLUMNS]
            )
        np.save(loss_path, history_array)

    def save_checkpoint(epoch_idx):
        checkpoint = {
            'epoch': epoch_idx,
            'loss_definition_version': LOSS_DEFINITION_VERSION,
            'loss_history_columns': LOSS_HISTORY_COLUMNS,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'loss_history': loss_history,
            'log10_mae_history': log10_mae_history,
            'log10_val_mae_history': log10_val_mae_history,
            'best_val_mae': best_val_mae,
            'best_epoch': best_epoch,
            'epochs_no_improve': epochs_no_improve,
        }
        save_loss_history()
        torch.save(model.state_dict(), model_path)
        torch.save(checkpoint, training_state_path)

    start_epoch = 0
    if opt.resume and os.path.exists(training_state_path):
        checkpoint = torch.load(training_state_path, map_location='cuda')
        checkpoint_loss_version = checkpoint.get('loss_definition_version')
        if checkpoint_loss_version != LOSS_DEFINITION_VERSION:
            raise RuntimeError(
                "Checkpoint resume is incompatible with the current loss "
                "definition. Start a fresh run without --resume after the "
                "loss refactor."
            )
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        loss_history = checkpoint.get('loss_history', loss_history)
        log10_mae_history = checkpoint.get('log10_mae_history', log10_mae_history)
        log10_val_mae_history = checkpoint.get('log10_val_mae_history', log10_val_mae_history)
        best_val_mae = checkpoint.get(
            'best_val_mae',
            best_val_mae,
        )
        best_epoch = checkpoint.get('best_epoch', best_epoch)
        epochs_no_improve = checkpoint.get('epochs_no_improve', epochs_no_improve)
        start_epoch = checkpoint.get('epoch', -1) + 1
        print(f"Resumed from epoch {start_epoch}")

    with open(log_path, 'a') as log:
        log.write(f"\n{'='*70}\n")
        log.write(f"  Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"  Param: {opt.param}\n")
        log.write(f"  Resume: {opt.resume}\n")
        log.write(f"  FNO3d(modes=64/64/5, width=30)  |  Parameters: {n_params:,}\n")
        log.write(f"  Input shape: {tuple(x0_shape)}  |  Output shape: {tuple(y0_shape)}\n")
        log.write(f"  Train files: {n_train}  |  Val files: {n_val}\n")
        if opt.fast:
            log.write(
                f"  Fast mode file/sample caps already applied before logging\n"
            )
        log.write(f"  Epochs: {epochs}  |  Batch size: {batch_size}  |  LR: {learning_rate}\n")
        log.write(f"  Scheduler: StepLR(step={scheduler_step}, gamma={scheduler_gamma})\n")
        if opt.fast:
            log.write(
                "  Fast mode: enabled "
                f"(train files={train_files_limit}, val files={val_files_limit}, "
                f"samples/file={samples_per_file}, viz every epoch)\n"
            )
        if autocast_enabled:
            log.write(
                f"  Mixed precision: autocast {autocast_dtype} | grad scaler: {use_grad_scaler}\n"
            )
        else:
            log.write(
                "  Mixed precision: disabled (complex parameters require unsupported AMP unscale)\n"
            )
        log.write(f"{'='*70}\n")
        log.write(
            f"{'epoch':>6}  {'time_s':>7}  {'lr':>10}  {'mae':>12}  "
            f"{'val_mae':>12}  {'log10_mae':>10}  {'eta':>12}\n"
        )
        log.write(f"{'-'*90}\n")

    t1_final = default_timer()
    vis_idx = 0

    try:
        for ep in range(start_epoch, epochs):
            model.train()
            t1 = default_timer()
            train_mae = 0
            train_steps = 0

            for bs in range(n_train):
                x_all, y_all, y_min, y_max = train_cache[bs]

                for l in range(0, len(x_all), batch_size):
                    x = x_all[l:l+batch_size].cuda()
                    y = y_all[l:l+batch_size].cuda()
                    actual_bs = len(x)

                    if actual_bs == 0:
                        continue

                    optimizer.zero_grad()
                    if autocast_enabled:
                        autocast_context = torch.amp.autocast(
                            'cuda', dtype=autocast_dtype
                        )
                    else:
                        autocast_context = nullcontext()
                    with autocast_context:
                        out = model(x).view(actual_bs, 128, 128, 20)

                        mae = F.l1_loss(out, y, reduction='mean')
                    scaler.scale(mae).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    train_mae += mae.item()
                    train_steps += 1

            scheduler.step()
            model.eval()

            # --- Validation pass (uses val split, never test) ---
            val_steps = 0
            val_mae_sum = 0.0
            with torch.no_grad():
                for j in range(n_val):
                    x_t, y_t, _, _ = val_cache[j]
                    for l in range(0, len(x_t), batch_size):
                        xb = x_t[l:l+batch_size].cuda()
                        yb = y_t[l:l+batch_size].cuda()
                        ab = len(xb)
                        if ab == 0:
                            continue
                        if autocast_enabled:
                            val_autocast = torch.amp.autocast('cuda', dtype=autocast_dtype)
                        else:
                            val_autocast = nullcontext()
                        with val_autocast:
                            ob = model(xb).view(ab, 128, 128, 20)
                            val_mae_b = F.l1_loss(ob, yb, reduction='mean')
                        val_mae_sum += val_mae_b.item()
                        val_steps += 1
            val_mae = val_mae_sum / max(1, val_steps)

            # --- Best checkpoint + early stopping ---
            # Save best model whenever validation MAE improves.
            if val_mae < best_val_mae - 1e-4:
                best_val_mae = val_mae
                best_epoch = ep + 1
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                epochs_no_improve += 1
                if opt.patience > 0 and epochs_no_improve >= opt.patience:
                    print(f"\nEarly stopping at epoch {ep+1} "
                           f"(no improvement for {opt.patience} epochs, "
                           f"best val MAE {best_val_mae:.4e} at epoch "
                           f"{best_epoch}).")
                    save_checkpoint(ep)
                    break

            if (ep + 1) % viz_every == 0 or ep == 0:
                with torch.no_grad():
                    j = np.random.randint(n_val)
                    xt = val_cache[j][0][:batch_size].cuda()
                    yt = val_cache[j][1][:batch_size].cpu().numpy()

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

            train_mae /= max(1, train_steps)

            t2 = default_timer()
            lr_now = scheduler.get_last_lr()[0]
            avg_epoch_time = (t2 - t1_final) / (ep - start_epoch + 1)
            eta_sec = int(avg_epoch_time * (epochs - ep - 1))
            eh, erem = divmod(eta_sec, 3600)
            em, es = divmod(erem, 60)
            eta_str = f"{eh}h {em}m {es}s"
            log10_mae_history.append(np.log10(train_mae))
            log10_val_mae_history.append(np.log10(val_mae))
            cols = shutil.get_terminal_size(fallback=(100, 24)).columns
            chart_cols = max(20, cols - 12)
            train_chart_data = compress_series(log10_mae_history[-100:], chart_cols)
            val_chart_data   = compress_series(log10_val_mae_history[-100:], chart_cols)
            chart = asciichartpy.plot(
                [train_chart_data, val_chart_data],
                {'height': 8, 'format': '{:6.2f}', 'offset': 2,
                 'colors': [None, asciichartpy.green]},
            )
            eta_compact = f"{eh:02d}:{em:02d}:{es:02d}"
            patience_str = f" | no-improve {epochs_no_improve}/{opt.patience}" if opt.patience > 0 else ""
            output = (
                f"Ep {ep+1}/{epochs} | {t2-t1:.1f}s | lr {lr_now:.2e} | ETA {eta_compact}\n"
                f"Train — MAE {train_mae:.4e}\n"
                f"\033[32mVal\033[0m   — MAE {val_mae:.4e}{patience_str}\n"
                f"{chart}\n"
                f"  log10(MAE): train {log10_mae_history[-1]:.3f}  \033[32mval {log10_val_mae_history[-1]:.3f}\033[0m"
            )
            n_lines = visual_line_count(output, cols)
            # Move cursor up to overwrite previous epoch's output
            if ep > start_epoch and sys.stdout.isatty():
                print(f"\033[{prev_n_lines}A\033[J", end='')
            print(output)
            prev_n_lines = n_lines

            loss_history['train_mae'].append(train_mae)
            loss_history['val_mae'].append(val_mae)

            with open(log_path, 'a') as log:
                log.write(
                    f"{ep+1:>6}  {t2-t1:>7.1f}  {lr_now:>10.3e}  "
                    f"{train_mae:>12.6f}  {val_mae:>12.6f}  "
                    f"{log10_mae_history[-1]:>10.4f}  {eta_str:>12}\n"
                )

            if (ep + 1) % checkpoint_every == 0 or ep == epochs - 1:
                save_checkpoint(ep)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving latest checkpoint...")
        if 'ep' in locals():
            save_checkpoint(ep)
        return

    t2_final = default_timer()
    elapsed = t2_final - t1_final
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    print(f"\nTraining complete. Total time: {h}h {m}m {s}s")
    print(f"Best val MAE: {best_val_mae:.4e}  (epoch {best_epoch})")
    with open(log_path, 'a') as log:
        log.write(f"{'-'*70}\n")
        log.write(
            f"  Run finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
            f"Total time: {h}h {m}m {s}s  |  "
            f"Best val MAE: {best_val_mae:.4e} at epoch {best_epoch}\n"
        )


if __name__ == '__main__':
    main()
