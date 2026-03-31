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
        [--experiments-dir <path>] [--force] [--max-frames N]
        [--workers N]

Run inference.py first to produce pred_sim_*.npy files.
For full-horizon rollout reference, run prepare_reference.py first.

For Idefix-derived data, frame titles also show time in Alfvén units. The
conversion is derived from the fixed Idefix box size, initial density,
magnetic-field amplitude, and stored timestep defined in data/idefix/.

Each rendered frame also includes a magnetic-dissipation diagnostic panel that
plots the evolution of epsilon_M over time using companion bx/by data. When
only one magnetic prediction component is available, the panel falls back to
the reference curve and prints a notice to the terminal.
"""

import numpy as np
import argparse
import multiprocessing
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
DATA_DIR        = os.path.join(ROOT_DIR, 'data')
EXPERIMENTS_DIR = os.path.join(ROOT_DIR, 'experiments')

# Number of initial input frames fed to the model (matches inference.py)
N_INPUT_FRAMES = 5

# Fixed Idefix setup parameters used to derive t_A = L / v_A. The magnetic
# field in setup.cpp already includes the 1/sqrt(4*pi) normalization factor, so
# the code-unit Alfvén speed is B0 / sqrt(rho0).
IDEFIX_BOX_LENGTH = 1.0
IDEFIX_TIMESTEP = 0.05
IDEFIX_INITIAL_DENSITY = 25.0 / (36.0 * np.pi)
IDEFIX_INITIAL_B_FIELD = 1.0 / np.sqrt(4.0 * np.pi)
IDEFIX_ALFVEN_SPEED = IDEFIX_INITIAL_B_FIELD / np.sqrt(IDEFIX_INITIAL_DENSITY)
IDEFIX_ALFVEN_TIME = IDEFIX_BOX_LENGTH / IDEFIX_ALFVEN_SPEED
IDEFIX_T_OVER_TA_PER_FRAME = IDEFIX_TIMESTEP / IDEFIX_ALFVEN_TIME
FARGO_BOX_LENGTH = 2.0 * np.pi


def _frame_to_alfven_time(param, frame):
    """Convert an absolute frame index to t/t_A when configured."""
    if param.startswith("fargo3d/"):
        return None

    return frame * IDEFIX_T_OVER_TA_PER_FRAME


def _format_alfven_time_label(param, frame):
    """Return a paper-style Alfvén-time label for figure titles."""
    t_over_ta = _frame_to_alfven_time(param, frame)
    if t_over_ta is None:
        return None

    return f"t = {t_over_ta:.2f} t_A"


def _replace_param_leaf(param, leaf):
    """Return a sibling parameter path with the final path segment replaced."""
    param_path = Path(param)
    parent = param_path.parent
    if str(parent) == '.':
        return leaf
    return str(parent / leaf)


def _frames_to_plot_x(param, frames):
    """Convert absolute frames to plotted x coordinates and an axis label."""
    times = [_frame_to_alfven_time(param, int(frame)) for frame in frames]
    if all(time is not None for time in times):
        return np.asarray(times, dtype=np.float64), r'$t / t_A$'
    return np.asarray(frames, dtype=np.float64), 'Frame'


def _curve_ylim(curves, pad_frac=0.08):
    """Return a stable linear y-limit for a collection of 1D curves."""
    valid = [np.asarray(curve) for curve in curves if curve is not None]
    if not valid:
        return None

    vals = np.concatenate(valid)
    if vals.size == 0:
        return None

    vmax = float(vals.max())
    vmin = float(vals.min())
    if np.isclose(vmin, vmax):
        pad = max(abs(vmax) * pad_frac, 1e-8)
        return vmin - pad, vmax + pad

    pad = (vmax - vmin) * pad_frac
    lower = 0.0 if vmin >= 0.0 else vmin - pad
    upper = vmax + pad
    return lower, upper


def _collapse_windowed_trajectory(windowed):
    """Collapse overlapping teacher-forced windows to one absolute timeline.

    The chosen policy keeps the shortest-horizon prediction for each absolute
    frame, which means lower output-step offsets take priority.
    """
    selected = {}
    n_samples = windowed.shape[0]
    t_out = windowed.shape[3]

    for lead in range(t_out):
        for sample in range(n_samples):
            frame = N_INPUT_FRAMES + sample + lead
            if frame not in selected:
                selected[frame] = windowed[sample, :, :, lead]

    frames = np.array(sorted(selected), dtype=np.int64)
    traj = np.stack([selected[frame] for frame in frames], axis=0)
    return frames, traj


def _rollout_prediction_trajectory(pred):
    """Return absolute frames and trajectory from a rollout prediction array."""
    frames = np.arange(
        N_INPUT_FRAMES, N_INPUT_FRAMES + pred.shape[2], dtype=np.int64
    )
    return frames, pred.transpose(2, 0, 1)


def _rollout_reference_trajectory(ref, dense, n_pred_frames):
    """Return reference frames/trajectory aligned to the rollout horizon."""
    if dense:
        start = N_INPUT_FRAMES
        stop = min(ref.shape[0], N_INPUT_FRAMES + n_pred_frames)
        frames = np.arange(start, stop, dtype=np.int64)
        return frames, ref[start:stop]

    stop = min(ref.shape[2], n_pred_frames)
    frames = np.arange(N_INPUT_FRAMES, N_INPUT_FRAMES + stop, dtype=np.int64)
    return frames, ref[:, :, :stop].transpose(2, 0, 1)


def _align_trajectories(frames_a, traj_a, frames_b, traj_b):
    """Intersect two trajectories onto the same absolute-frame timeline."""
    common = sorted(set(frames_a.tolist()) & set(frames_b.tolist()))
    if not common:
        raise ValueError("No overlapping frames available for epsilon_M.")

    index_a = {int(frame): idx for idx, frame in enumerate(frames_a)}
    index_b = {int(frame): idx for idx, frame in enumerate(frames_b)}
    common_frames = np.asarray(common, dtype=np.int64)
    aligned_a = np.stack([traj_a[index_a[int(frame)]] for frame in common_frames])
    aligned_b = np.stack([traj_b[index_b[int(frame)]] for frame in common_frames])
    return common_frames, aligned_a, aligned_b


def _magnetic_box_length(param):
    """Return the physical box length for the current dataset family."""
    if param.startswith("fargo3d/"):
        return FARGO_BOX_LENGTH
    return IDEFIX_BOX_LENGTH


def compute_magnetic_dissipation(bx_traj, by_traj, mu, param):
    """Compute epsilon_M(t) from aligned Bx/By trajectories."""
    bx_traj = np.asarray(bx_traj, dtype=np.float64)
    by_traj = np.asarray(by_traj, dtype=np.float64)
    if bx_traj.shape != by_traj.shape:
        raise ValueError(
            "Bx and By trajectories must have the same shape for epsilon_M."
        )

    _, height, width = bx_traj.shape
    box_length = _magnetic_box_length(param)
    dx = box_length / width
    dy = box_length / height

    dby_dx = (np.roll(by_traj, -1, axis=2) - np.roll(by_traj, 1, axis=2))
    dby_dx /= (2.0 * dx)
    dbx_dy = (np.roll(bx_traj, -1, axis=1) - np.roll(bx_traj, 1, axis=1))
    dbx_dy /= (2.0 * dy)
    current_density = dby_dx - dbx_dy

    cell_area = dx * dy
    return (mu / box_length) * np.sum(current_density ** 2, axis=(1, 2)) * cell_area


def _load_mu(test_dir, sim_id):
    """Load the magnetic diffusivity mu for a simulation from x_sim data."""
    x = np.load(test_dir / f"x_sim_{sim_id}.npy", mmap_mode='r')
    return float(x[0, 0, 0, 0, -1])


def _load_optional_prediction_trajectory(pred_path, is_rollout):
    """Load a magnetic prediction trajectory when the file exists."""
    if not pred_path.exists():
        return None

    pred = np.load(pred_path)
    if is_rollout:
        return _rollout_prediction_trajectory(pred[0])
    return _collapse_windowed_trajectory(pred.squeeze(-1))


def _build_magnetic_dissipation_data(
    exp_dir, param, sim_id, test_dir, is_rollout, rollout_n_frames=None
):
    """Build precomputed epsilon_M data for one simulation."""
    mu = _load_mu(test_dir, sim_id)
    magnetic_params = {
        field: _replace_param_leaf(param, field) for field in ('bx', 'by')
    }
    magnetic_test_dirs = {
        field: Path(DATA_DIR) / magnetic_params[field] / 'test'
        for field in magnetic_params
    }

    if is_rollout and rollout_n_frames is None:
        raise ValueError("rollout_n_frames is required for rollout epsilon_M.")

    ref_components = {}
    for field, field_test_dir in magnetic_test_dirs.items():
        ref, dense = _load_rollout_reference(field_test_dir, sim_id)
        if is_rollout:
            ref_components[field] = _rollout_reference_trajectory(
                ref, dense, rollout_n_frames
            )
        else:
            if dense:
                # Dense ref_sim covers the full simulation timeline — use it
                # so epsilon_M spans all available frames, not just y windows.
                n_ref = ref.shape[0]
                frames = np.arange(n_ref, dtype=np.int64)
                traj = ref          # (N_frames, H, W)
                ref_components[field] = (frames, traj)
            else:
                # Fall back to supervised windows when no dense file exists.
                y = np.load(field_test_dir / f"y_sim_{sim_id}.npy")
                ref_components[field] = _collapse_windowed_trajectory(y)

    ref_frames, ref_bx, ref_by = _align_trajectories(
        ref_components['bx'][0],
        ref_components['bx'][1],
        ref_components['by'][0],
        ref_components['by'][1],
    )
    ref_curve = compute_magnetic_dissipation(ref_bx, ref_by, mu, param)
    ref_x, xlabel = _frames_to_plot_x(param, ref_frames)

    pred_suffix = '_rollout' if is_rollout else ''
    pred_components = {}
    missing_pred = []
    for field, magnetic_param in magnetic_params.items():
        pred_path = (
            Path(exp_dir) / magnetic_param / 'visualizations' /
            f"pred_sim_{sim_id}{pred_suffix}.npy"
        )
        pred_traj = _load_optional_prediction_trajectory(pred_path, is_rollout)
        if pred_traj is None:
            missing_pred.append(field)
        else:
            pred_components[field] = pred_traj

    pred_frames = None
    pred_x = None
    pred_curve = None
    if missing_pred:
        missing = ', '.join(sorted(missing_pred))
        print(
            f"  sim {sim_id}: magnetic prediction curve unavailable; "
            f"missing {missing}. Showing reference epsilon_M only."
        )
    else:
        pred_frames, pred_bx, pred_by = _align_trajectories(
            pred_components['bx'][0],
            pred_components['bx'][1],
            pred_components['by'][0],
            pred_components['by'][1],
        )
        pred_curve = compute_magnetic_dissipation(pred_bx, pred_by, mu, param)
        pred_x, _ = _frames_to_plot_x(param, pred_frames)

    return {
        'xlabel': xlabel,
        'ref_frames': ref_frames,
        'ref_x': ref_x,
        'ref_curve': ref_curve,
        'pred_frames': pred_frames,
        'pred_x': pred_x,
        'pred_curve': pred_curve,
        'ylim': _curve_ylim([ref_curve, pred_curve]),
    }


def compute_power_spectrum(field):
    """Compute the 1D isotropic power spectrum of a 2D field.

    P(k) = sum_{|k|=k} |f_hat(k)|^2   (paper §Spectral Analysis)

    Returns
    -------
    k_bins : ndarray, shape (k_max,)   integer wavenumbers 1..k_max
    P_k    : ndarray, shape (k_max,)   summed power per wavenumber bin
    """
    H, W = field.shape
    f_hat = np.fft.fft2(field)
    psd2d = np.abs(f_hat) ** 2

    kx = np.fft.fftfreq(W) * W   # integer wavenumbers along x
    ky = np.fft.fftfreq(H) * H   # integer wavenumbers along y
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)

    k_int = np.round(K).astype(int)
    k_max = min(H, W) // 2
    k_bins = np.arange(1, k_max + 1)
    P_k = np.array(
        [psd2d[k_int == k].sum() for k in k_bins], dtype=np.float64
    )
    return k_bins, P_k


def _spectrum_ylim(P_arrays, pad=0.5):
    """Derive stable log-scale y-limits from a collection of power spectra.

    Parameters
    ----------
    P_arrays : iterable of ndarray
        1D power spectra P(k) collected across all frames to be shown in a
        single movie/simulation.
    pad : float
        Extra decades of padding added to each side of the dynamic range.

    Returns
    -------
    (ymin, ymax) : tuple of float, or None when no positive values exist.
    """
    pos_vals = np.concatenate(
        [P[P > 0] for P in P_arrays if np.any(P > 0)]
    )
    if pos_vals.size == 0:
        return None
    log_min = np.log10(pos_vals.min())
    log_max = np.log10(pos_vals.max())
    return 10 ** (log_min - pad), 10 ** (log_max + pad)


def _plot_spectra(ax, k_bins, P_target, P_pred, with_target=True,
                  ylim=None):
    """Draw 1D power spectra on a log-log axis.

    Parameters
    ----------
    with_target : bool
        When True, draw both reference and prediction spectra.
        When False, draw only the prediction spectrum (rollout-only frames).
    ylim : tuple of (float, float) or None
        Fixed (ymin, ymax) for the y-axis. When None, Matplotlib auto-scales.
        Pass a simulation-level value to keep the axis stable across frames.
    """
    if with_target and P_target is not None:
        ax.loglog(k_bins, P_target, color='steelblue', label='Reference')
    ax.loglog(k_bins, P_pred, color='darkorange',
              linestyle='--', label='Prediction')
    ax.set_xlabel('Wavenumber k')
    ax.set_ylabel('P(k)')
    ax.set_title('Power spectrum')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    if ylim is not None:
        ax.set_ylim(ylim)


def _plot_magnetic_dissipation(ax, frame, dissipation):
    """Draw the epsilon_M panel and highlight the current frame."""
    if dissipation is None:
        ax.set_visible(False)
        return

    ax.plot(
        dissipation['ref_x'],
        dissipation['ref_curve'],
        color='steelblue',
        linewidth=2,
        label='Reference',
    )
    if dissipation['pred_curve'] is not None:
        ax.plot(
            dissipation['pred_x'],
            dissipation['pred_curve'],
            color='darkorange',
            linestyle='--',
            linewidth=2,
            label='Prediction',
        )

    marker_sources = [
        (
            dissipation['pred_frames'],
            dissipation['pred_x'],
            dissipation['pred_curve'],
        ),
        (
            dissipation['ref_frames'],
            dissipation['ref_x'],
            dissipation['ref_curve'],
        ),
    ]
    for marker_frames, marker_x, marker_curve in marker_sources:
        if marker_curve is None:
            continue

        matches = np.where(marker_frames == frame)[0]
        if not matches.size:
            continue

        idx = int(matches[0])
        ax.plot(
            marker_x[idx],
            marker_curve[idx],
            marker='o',
            markersize=12,
            color='red',
            markeredgecolor='white',
            markeredgewidth=1.5,
            linestyle='None',
            label='Current frame',
            zorder=5,
        )
        break

    ax.set_yscale('log')
    ax.set_title(r'Magnetic dissipation $\epsilon_M$')
    ax.set_xlabel(dissipation['xlabel'])
    ax.set_ylabel(r'$\epsilon_M$')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=9)


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
                        target, pred_t, vmin, vmax, eabs, suffix='',
                        spec_ylim=None, dissipation=None):
    """Render a two-row PNG.

    Row 0: reference | prediction | error  (field panels)
    Row 1: power spectrum (target vs prediction) spanning first two columns
    """
    error = target - pred_t
    k_bins, P_target = compute_power_spectrum(target)
    _, P_pred = compute_power_spectrum(pred_t)

    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1.2])

    title = f'{param}  |  sim {sim_id}  |  frame {frame}'
    time_label = _format_alfven_time_label(param, frame)
    if time_label:
        title += f'  |  {time_label}'
    if suffix:
        title += f'  |  {suffix}'
    fig.suptitle(title, fontsize=12)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax_spec = fig.add_subplot(gs[1, :2])
    ax_eps = fig.add_subplot(gs[1, 2])

    im0 = ax0.pcolormesh(target, vmin=vmin, vmax=vmax, cmap='viridis')
    ax0.set_title('Reference')
    ax0.set_aspect('equal')

    im1 = ax1.pcolormesh(pred_t, vmin=vmin, vmax=vmax, cmap='viridis')
    ax1.set_title('Prediction')
    ax1.set_aspect('equal')

    im2 = ax2.pcolormesh(error, vmin=-eabs, vmax=eabs, cmap='RdBu_r')
    ax2.set_title('Error (Target - Prediction)')
    ax2.set_aspect('equal')

    for ax, im in zip([ax0, ax1, ax2], [im0, im1, im2]):
        fig.colorbar(im, ax=ax, location='bottom', shrink=0.9, pad=0.08)

    _plot_spectra(ax_spec, k_bins, P_target, P_pred, with_target=True,
                  ylim=spec_ylim)
    _plot_magnetic_dissipation(ax_eps, frame, dissipation)

    plt.savefig(vis_dir / fname, dpi=100)
    plt.close(fig)


def _save_frame_pred_only(vis_dir, fname, param, sim_id, frame,
                          pred_t, vmin, vmax, suffix='', spec_ylim=None,
                          dissipation=None):
    """Render a two-row PNG for prediction-only rollout frames.

    Row 0: blank | prediction | blank  (same column widths as _save_frame_with_gt)
    Row 1: prediction power spectrum spanning first two columns
    """
    k_bins, P_pred = compute_power_spectrum(pred_t)

    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1.2])

    title = f'{param}  |  sim {sim_id}  |  frame {frame}  |  prediction only'
    time_label = _format_alfven_time_label(param, frame)
    if time_label:
        title += f'  |  {time_label}'
    if suffix:
        title += f'  |  {suffix}'
    fig.suptitle(title, fontsize=12)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax_spec = fig.add_subplot(gs[1, :2])
    ax_eps = fig.add_subplot(gs[1, 2])

    im1 = ax1.pcolormesh(pred_t, vmin=vmin, vmax=vmax, cmap='viridis')
    ax1.set_title('Prediction')
    ax1.set_aspect('equal')
    fig.colorbar(im1, ax=ax1, location='bottom', shrink=0.9, pad=0.08)

    # Blank flanking panels keep figure size consistent for movies
    for ax in (ax0, ax2):
        ax.set_visible(False)

    _plot_spectra(ax_spec, k_bins, None, P_pred, with_target=False,
                  ylim=spec_ylim)
    _plot_magnetic_dissipation(ax_eps, frame, dissipation)

    plt.savefig(vis_dir / fname, dpi=100)
    plt.close(fig)


def _render_frame_job(job):
    """Top-level picklable worker for parallel PNG rendering.

    ``job`` is a plain dict produced by the main process. Calls
    ``_save_frame_with_gt`` when a reference field is present, or
    ``_save_frame_pred_only`` for prediction-only frames.
    Returns the output filename (used to count completions).
    """
    vis_dir = Path(job['vis_dir'])
    spec_ylim = job.get('spec_ylim')
    dissipation = job.get('dissipation')
    if job['target'] is not None:
        _save_frame_with_gt(
            vis_dir, job['fname'], job['param'], job['sim_id'],
            job['frame'], job['target'], job['pred_t'],
            job['vmin'], job['vmax'], job['eabs'],
            suffix=job.get('suffix', ''),
            spec_ylim=spec_ylim,
            dissipation=dissipation,
        )
    else:
        _save_frame_pred_only(
            vis_dir, job['fname'], job['param'], job['sim_id'],
            job['frame'], job['pred_t'],
            job['vmin'], job['vmax'],
            suffix=job.get('suffix', ''),
            spec_ylim=spec_ylim,
            dissipation=dissipation,
        )
    return job['fname']


def _run_jobs(jobs, desc, workers=None):
    """Render a list of frame jobs serially or with a process pool.

    Parameters
    ----------
    jobs    : list of dicts as produced by the visualize_* helpers
    desc    : tqdm progress-bar label
    workers : int or None — None / 1 → serial; >1 → process pool

    Returns the number of jobs executed.
    """
    if not jobs:
        return 0
    if workers and workers > 1:
        ctx = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=ctx
        ) as pool:
            futures = [pool.submit(_render_frame_job, j) for j in jobs]
            for _ in tqdm(
                as_completed(futures), total=len(futures),
                desc=desc, unit='frame', leave=False,
            ):
                pass
    else:
        for job in tqdm(jobs, desc=desc, unit='frame', leave=False):
            _render_frame_job(job)
    return len(jobs)


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


def visualize_teacher_forced(pred_path, sim_id, exp_dir, test_dir, vis_dir,
                             param, force, max_frames=None,
                             workers=None):
    """Visualize a teacher-forced prediction file.

    pred shape: (n_samples, 128, 128, T_out)
    y    shape: (n_samples, 128, 128, T_out)
    """
    pred = np.load(pred_path).squeeze(-1)
    y    = np.load(test_dir / f"y_sim_{sim_id}.npy")

    vmin_global = float(min(pred.min(), y.min()))
    vmax_global = float(max(pred.max(), y.max()))
    eabs_global = float(np.max(np.abs(y - pred)))

    # Precompute power spectra across all frames to get a stable y-axis range
    # for the spectrum subplot — avoids jumpy limits in the rendered movie.
    all_spectra = []
    for s in range(pred.shape[0]):
        for t in range(pred.shape[3]):
            _, P_y = compute_power_spectrum(y[s, :, :, t])
            _, P_p = compute_power_spectrum(pred[s, :, :, t])
            all_spectra.extend([P_y, P_p])
    spec_ylim_global = _spectrum_ylim(all_spectra)
    dissipation = _build_magnetic_dissipation_data(
        exp_dir, param, sim_id, test_dir, is_rollout=False,
    )

    jobs = []
    skipped = 0
    n_samples = pred.shape[0]

    for s in range(n_samples):
        if max_frames is not None and len(jobs) + skipped >= max_frames:
            break
        for t in range(pred.shape[3]):
            if max_frames is not None and len(jobs) + skipped >= max_frames:
                break
            frame = N_INPUT_FRAMES + s + t
            fname = f'frame_{frame:04d}_sim_{sim_id}.png'
            if (vis_dir / fname).exists() and not force:
                skipped += 1
                continue
            jobs.append({
                'vis_dir': str(vis_dir),
                'fname': fname,
                'param': param,
                'sim_id': sim_id,
                'frame': frame,
                'target': y[s, :, :, t].copy(),
                'pred_t': pred[s, :, :, t].copy(),
                'vmin': vmin_global,
                'vmax': vmax_global,
                'eabs': eabs_global,
                'suffix': '',
                'spec_ylim': spec_ylim_global,
                'dissipation': dissipation,
            })

    total = _run_jobs(
        jobs, desc=f"  sim {sim_id}", workers=workers,
    )
    return total, skipped


def visualize_rollout(pred_path, sim_id, exp_dir, test_dir, vis_dir, param,
                      force, max_frames=None, workers=None):
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
    dissipation = _build_magnetic_dissipation_data(
        exp_dir, param, sim_id, test_dir, is_rollout=True,
        rollout_n_frames=n_frames,
    )

    ref, dense = _load_rollout_reference(test_dir, sim_id)
    # dense=True:  ref shape (N_sim_frames, H, W)
    # dense=False: ref shape (H, W, T_out) — sparse window 0 only

    if dense:
        n_ref = ref.shape[0]
        # Build arrays for color scale computation over available overlap
        max_abs = min(N_INPUT_FRAMES + n_frames, n_ref)
        pred_for_scale = pred[:, :, :max_abs - N_INPUT_FRAMES]
        ref_for_scale  = ref[N_INPUT_FRAMES:max_abs]
        vmin_global = min(float(pred.min()), float(ref_for_scale.min()))
        vmax_global = max(float(pred.max()), float(ref_for_scale.max()))
        eabs_global = float(
            np.max(np.abs(
                ref_for_scale - pred_for_scale.transpose(2, 0, 1)
            ))
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
        eabs_global = float(
            np.max(np.abs(ref[:, :, :overlap] - pred[:, :, :overlap]))
        )
        print(
            f"  sim {sim_id}: no ref_sim_{sim_id}.npy found; "
            f"using sparse y_sim reference (frames 0..{t_out - 1} only). "
            "Run prepare_reference.py for full-horizon reference."
        )

    frame_limit = (
        n_frames if max_frames is None else min(n_frames, max_frames)
    )

    # Precompute power spectra across all frames for a stable spectrum y-axis.
    all_spectra = []
    for i in range(frame_limit):
        _, P_p = compute_power_spectrum(pred[:, :, i])
        all_spectra.append(P_p)
        if dense:
            frame_abs = N_INPUT_FRAMES + i
            if frame_abs < ref.shape[0]:
                _, P_r = compute_power_spectrum(ref[frame_abs])
                all_spectra.append(P_r)
        else:
            if i < ref.shape[2]:
                _, P_r = compute_power_spectrum(ref[:, :, i])
                all_spectra.append(P_r)
    spec_ylim_global = _spectrum_ylim(all_spectra)

    jobs = []
    skipped = 0

    for i in range(frame_limit):
        frame = N_INPUT_FRAMES + i
        fname = f'frame_{frame:04d}_sim_{sim_id}_rollout.png'

        if (vis_dir / fname).exists() and not force:
            skipped += 1
            continue

        pred_t = pred[:, :, i].copy()

        # Determine if reference exists for this frame
        if dense:
            has_ref = frame < ref.shape[0]
        else:
            has_ref = i < ref.shape[2]

        if has_ref:
            ref_t = (ref[frame] if dense else ref[:, :, i]).copy()
            jobs.append({
                'vis_dir': str(vis_dir),
                'fname': fname,
                'param': param,
                'sim_id': sim_id,
                'frame': frame,
                'target': ref_t,
                'pred_t': pred_t,
                'vmin': vmin_global,
                'vmax': vmax_global,
                'eabs': eabs_global,
                'suffix': 'rollout',
                'spec_ylim': spec_ylim_global,
                'dissipation': dissipation,
            })
        else:
            jobs.append({
                'vis_dir': str(vis_dir),
                'fname': fname,
                'param': param,
                'sim_id': sim_id,
                'frame': frame,
                'target': None,
                'pred_t': pred_t,
                'vmin': vmin_global,
                'vmax': vmax_global,
                'eabs': 0.0,
                'suffix': 'rollout',
                'spec_ylim': spec_ylim_global,
                'dissipation': dissipation,
            })

    total = _run_jobs(
        jobs, desc=f"  sim {sim_id} rollout", workers=workers,
    )
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
    parser.add_argument("--experiments-dir", type=str,
                        default=EXPERIMENTS_DIR)
    parser.add_argument("--force", action="store_true",
                        help="Regenerate PNGs/movies even if they exist")
    parser.add_argument("--max-frames", type=int, default=None, metavar="N",
                        help="Render only the first N frames per simulation "
                             "(default: render all frames)")
    parser.add_argument(
        "--workers", type=int, default=None, metavar="N",
        help="Number of worker processes for parallel PNG rendering. "
             "Defaults to serial (1 process). Use e.g. --workers 8 to "
             "saturate CPU cores when rendering hundreds of frames.",
    )
    opt = parser.parse_args()

    exp_dir = Path(opt.experiments_dir)
    vis_dir = exp_dir / opt.param / 'visualizations'
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
    if opt.workers and opt.workers > 1:
        print(f"Parallel rendering: {opt.workers} worker processes")
    else:
        print("Rendering: serial (use --workers N for parallelism)")

    total_pngs = 0
    skipped_pngs = 0

    # --- teacher-forced ---
    for pred_path, sim_id in tqdm(tf_files, desc="Teacher-forced",
                                  unit="file"):
        n, s = visualize_teacher_forced(
            pred_path, sim_id, exp_dir, test_dir, vis_dir, opt.param,
            opt.force,
            max_frames=opt.max_frames, workers=opt.workers,
        )
        total_pngs += n
        skipped_pngs += s

    # --- rollout ---
    for pred_path, sim_id in tqdm(rollout_files, desc="Rollout", unit="file"):
        n, s = visualize_rollout(
            pred_path, sim_id, exp_dir, test_dir, vis_dir, opt.param,
            opt.force,
            max_frames=opt.max_frames, workers=opt.workers,
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
