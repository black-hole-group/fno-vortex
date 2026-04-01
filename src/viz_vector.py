"""
Vector-field comparison visualization for paired autoregressive predictions.

This script renders per-frame comparisons for vector fields such as:

- magnetic field: B = (bx, by), colored by |B|
- velocity field: v = (vx, vy), colored by |v|

The figure layout reuses the existing two-row, three-column grid from
visualize_results.py:

- Row 0, col 0: reference vector field, colored by magnitude
- Row 0, col 1: prediction vector field, colored by predicted magnitude
- Row 0, col 2: magnitude residuals (reference - prediction)
- Row 1, col 0-1: magnitude power spectrum
- Row 1, col 2: family-specific time-series diagnostic

Magnetic mode uses epsilon_M(t) in the final panel. Velocity mode currently
falls back to a generic mean-squared-speed curve so the renderer remains
reusable for later velocity-specific diagnostics.
"""

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from experiment_layout import (
    prediction_files,
    reference_file,
    resolve_experiment_param,
    resolve_vector_output_dir,
)
from viz_scalar import (
    DATA_DIR,
    EXPERIMENTS_DIR,
    _align_trajectories,
    _curve_ylim,
    _format_alfven_time_label,
    _frames_to_plot_x,
    _load_mu,
    _load_optional_prediction_trajectory,
    _load_rollout_reference,
    _replace_param_leaf,
    _rollout_reference_trajectory,
    _spectrum_ylim,
    compute_magnetic_dissipation,
    compute_power_spectrum,
    parse_pred_file,
    render_movie,
)

VECTOR_FAMILIES = {
    'magnetic': {
        'components': ('bx', 'by'),
        'vector_symbol': 'B',
        'magnitude_label': r'$|B|$',
        'output_dirname': 'magnetic_vector_visualizations',
        'file_prefix': 'magnetic',
        'fallback_diag_title': 'Mean squared magnetic magnitude',
        'fallback_diag_ylabel': r'$\langle |B|^2 \rangle$',
    },
    'velocity': {
        'components': ('vx', 'vy'),
        'vector_symbol': 'v',
        'magnitude_label': r'$|v|$',
        'output_dirname': 'velocity_vector_visualizations',
        'file_prefix': 'velocity',
        'fallback_diag_title': 'Mean squared speed',
        'fallback_diag_ylabel': r'$\langle |v|^2 \rangle$',
    },
}

LEAF_TO_FAMILY = {
    component: family
    for family, spec in VECTOR_FAMILIES.items()
    for component in spec['components']
}


def _resolve_vector_spec(exp_dir, param):
    """Resolve the vector family and companion component paths."""
    param_paths = resolve_experiment_param(exp_dir, param, DATA_DIR)
    leaf = param_paths.param_key
    family = LEAF_TO_FAMILY.get(leaf)
    if family is None:
        supported = ', '.join(sorted(LEAF_TO_FAMILY))
        raise ValueError(
            f"Unsupported vector component '{leaf}'. Supported leaves: "
            f"{supported}."
        )

    spec = dict(VECTOR_FAMILIES[family])
    spec['family'] = family
    spec['param'] = param_paths.data_param
    spec['param_key'] = param_paths.param_key
    spec['paths'] = param_paths
    spec['component_params'] = {}
    spec['component_paths'] = {}
    for component in spec['components']:
        component_paths = resolve_experiment_param(
            exp_dir,
            _replace_param_leaf(param_paths.data_param, component),
            DATA_DIR,
        )
        spec['component_params'][component] = component_paths.data_param
        spec['component_paths'][component] = component_paths
    return spec


def _vector_output_dir(exp_dir, spec):
    """Return the shared visualization directory for this vector family."""
    return resolve_vector_output_dir(
        exp_dir, spec['family'], data_param=spec['param'],
    )


def _prediction_map(paths):
    """Map simulation ids to prediction file paths."""
    mapping = {}
    pred_files = prediction_files(paths)
    if not pred_files:
        return mapping

    for pred_path in pred_files:
        mapping[parse_pred_file(pred_path)] = pred_path
    return mapping


def _format_sim_list(sim_ids):
    """Return a compact simulation-id list for error messages."""
    if not sim_ids:
        return 'none'
    return ', '.join(sim_ids)


def _no_pairs_error(exp_dir, spec, component_dirs, component_maps):
    """Build an actionable error for missing paired vector predictions."""
    details = ', '.join(
        f"{component}: {component_dirs[component]}"
        for component in spec['components']
    )

    inventory_lines = []
    for component in spec['components']:
        inventory_lines.append(
            f"  {component}: {_format_sim_list(sorted(component_maps[component]))}"
        )

    commands = '\n'.join(
        f"  python src/inference.py --experiments-dir {exp_dir} "
        f"--param {spec['component_paths'][component].input_param}"
        for component in spec['components']
    )

    message = (
        "No paired pred_sim_*.npy files found for both vector components. "
        f"Checked {details}.\n"
        "Available predictions:\n"
        + '\n'.join(inventory_lines)
        + "\n"
    )
    message += (
        "Vector rendering requires both components for the same simulation ID "
        "under the same experiment root.\n"
        "Generate matching predictions first, for example:\n"
        f"{commands}"
    )
    return FileNotFoundError(message)


def _collect_prediction_pairs(exp_dir, spec):
    """Return paired prediction files shared by both vector components."""
    component_maps = {}
    component_dirs = {}
    for component, component_paths in spec['component_paths'].items():
        pred_dir = component_paths.prediction_dir
        component_dirs[component] = pred_dir
        component_maps[component] = _prediction_map(component_paths)

    shared_sim_ids = sorted(
        set.intersection(*(set(mapping) for mapping in component_maps.values()))
    )
    if not shared_sim_ids:
        raise _no_pairs_error(exp_dir, spec, component_dirs, component_maps)

    paired_files = []
    for sim_id in shared_sim_ids:
        pair = {
            component: component_maps[component][sim_id]
            for component in spec['components']
        }
        paired_files.append((sim_id, pair))

    return paired_files, component_dirs


def _load_reference_trajectory(
    test_dir,
    sim_id,
    n_pred_frames,
    preferred_ref_path=None,
):
    """Load a reference trajectory for one component."""
    ref, dense = _load_rollout_reference(
        test_dir, sim_id, preferred_ref_path=preferred_ref_path,
    )
    return _rollout_reference_trajectory(ref, dense, n_pred_frames)


def _align_component_pair(component_data, ordered_components):
    """Align the two components of a vector field on a shared timeline."""
    comp0, comp1 = ordered_components
    frames, traj0, traj1 = _align_trajectories(
        component_data[comp0][0],
        component_data[comp0][1],
        component_data[comp1][0],
        component_data[comp1][1],
    )
    return frames, traj0, traj1


def _vector_magnitude(comp0, comp1):
    """Return sqrt(comp0**2 + comp1**2) as float64."""
    comp0 = np.asarray(comp0, dtype=np.float64)
    comp1 = np.asarray(comp1, dtype=np.float64)
    return np.sqrt(comp0 ** 2 + comp1 ** 2)


def _mean_squared_magnitude(comp0_traj, comp1_traj):
    """Return the spatial mean of |vector|^2 over time."""
    comp0_traj = np.asarray(comp0_traj, dtype=np.float64)
    comp1_traj = np.asarray(comp1_traj, dtype=np.float64)
    return np.mean(comp0_traj ** 2 + comp1_traj ** 2, axis=(1, 2))


def _relative_l2_error(ref_mag_traj, pred_mag_traj):
    """Relative L2 error (%) of magnitude per timestep.

    Returns 100 * ||ref - pred||_2 / ||ref||_2 for each frame.
    Frames where the reference norm is zero are assigned NaN.
    """
    ref = np.asarray(ref_mag_traj, dtype=np.float64)
    pred = np.asarray(pred_mag_traj, dtype=np.float64)
    diff_norm = np.sqrt(np.sum((ref - pred) ** 2, axis=(-2, -1)))
    ref_norm = np.sqrt(np.sum(ref ** 2, axis=(-2, -1)))
    with np.errstate(invalid='ignore', divide='ignore'):
        error = np.where(ref_norm > 0, 100.0 * diff_norm / ref_norm, np.nan)
    return error


def _build_error_metric(spec, param, overlap_frames, overlap_ref_mag, overlap_pred_mag):
    """Build the error-metric panel data dict (relative L2 error in %)."""
    error_curve = _relative_l2_error(overlap_ref_mag, overlap_pred_mag)
    x_vals, xlabel = _frames_to_plot_x(param, overlap_frames)
    magnitude_label = spec['magnitude_label']
    valid = error_curve[np.isfinite(error_curve)]
    ylim = _curve_ylim([valid]) if valid.size else None
    return {
        'title': f'Relative L2 error of {magnitude_label}',
        'ylabel': 'Error (%)',
        'xlabel': xlabel,
        'frames': overlap_frames,
        'x': x_vals,
        'curve': error_curve,
        'ylim': ylim,
    }


def _timeseries_ylim(curves, log_scale):
    """Return a stable y-limit for linear or log-scaled time-series plots."""
    valid = [np.asarray(curve) for curve in curves if curve is not None]
    if not valid:
        return None

    if log_scale:
        positive = [curve[curve > 0] for curve in valid if np.any(curve > 0)]
        if not positive:
            return None
        return _spectrum_ylim(positive, pad=0.2)

    return _curve_ylim(valid)


def _build_diagnostic(spec, param, sim_id, test_dir,
                      ref_frames, ref_comp0, ref_comp1,
                      pred_frames, pred_comp0, pred_comp1):
    """Build the lower-right time-series diagnostic panel."""
    ref_x, xlabel = _frames_to_plot_x(param, ref_frames)
    pred_x, _ = _frames_to_plot_x(param, pred_frames)

    if spec['family'] == 'magnetic':
        mu = _load_mu(test_dir, sim_id)
        ref_curve = compute_magnetic_dissipation(
            ref_comp0, ref_comp1, mu, param,
        )
        pred_curve = compute_magnetic_dissipation(
            pred_comp0, pred_comp1, mu, param,
        )
        log_scale = True
        title = r'Magnetic dissipation $\epsilon_M$'
        ylabel = r'$\epsilon_M$'
    else:
        ref_curve = _mean_squared_magnitude(ref_comp0, ref_comp1)
        pred_curve = _mean_squared_magnitude(pred_comp0, pred_comp1)
        log_scale = False
        title = spec['fallback_diag_title']
        ylabel = spec['fallback_diag_ylabel']

    return {
        'title': title,
        'ylabel': ylabel,
        'xlabel': xlabel,
        'log_scale': log_scale,
        'ref_frames': ref_frames,
        'ref_x': ref_x,
        'ref_curve': ref_curve,
        'pred_frames': pred_frames,
        'pred_x': pred_x,
        'pred_curve': pred_curve,
        'ylim': _timeseries_ylim([ref_curve, pred_curve], log_scale),
    }


def _build_vector_dataset(pair_paths, sim_id, spec):
    """Load paired component trajectories and derived diagnostics."""
    components = spec['components']
    pred_components = {}
    for component in components:
        pred_traj = _load_optional_prediction_trajectory(pair_paths[component])
        if pred_traj is None:
            raise FileNotFoundError(
                f'Missing prediction trajectory for {pair_paths[component]}.'
            )
        pred_components[component] = pred_traj

    pred_frames, pred_comp0, pred_comp1 = _align_component_pair(
        pred_components, components,
    )

    ref_components = {}
    for component, component_param in spec['component_params'].items():
        test_dir = Path(DATA_DIR) / component_param / 'test'
        ref_components[component] = _load_reference_trajectory(
            test_dir,
            sim_id,
            n_pred_frames=len(pred_frames),
            preferred_ref_path=reference_file(
                spec['component_paths'][component], sim_id,
            ),
        )

    ref_frames, ref_comp0, ref_comp1 = _align_component_pair(
        ref_components, components,
    )

    ref_index = {
        int(frame): idx for idx, frame in enumerate(ref_frames)
    }
    pred_index = {
        int(frame): idx for idx, frame in enumerate(pred_frames)
    }
    overlap_frames = np.array(
        sorted(set(ref_index) & set(pred_index)),
        dtype=np.int64,
    )
    if overlap_frames.size == 0:
        raise ValueError(
            f'No overlapping frames available for sim {sim_id}.'
        )

    overlap_ref_mag = _vector_magnitude(
        np.stack([ref_comp0[ref_index[int(frame)]] for frame in overlap_frames]),
        np.stack([ref_comp1[ref_index[int(frame)]] for frame in overlap_frames]),
    )
    overlap_pred_mag = _vector_magnitude(
        np.stack(
            [pred_comp0[pred_index[int(frame)]] for frame in overlap_frames]
        ),
        np.stack(
            [pred_comp1[pred_index[int(frame)]] for frame in overlap_frames]
        ),
    )

    test_dir = Path(DATA_DIR) / spec['param'] / 'test'
    diagnostic = _build_diagnostic(
        spec,
        spec['param'],
        sim_id,
        test_dir,
        ref_frames,
        ref_comp0,
        ref_comp1,
        pred_frames,
        pred_comp0,
        pred_comp1,
    )

    error_metric = _build_error_metric(
        spec,
        spec['param'],
        overlap_frames,
        overlap_ref_mag,
        overlap_pred_mag,
    )

    return {
        'ref_frames': ref_frames,
        'ref_comp0': ref_comp0,
        'ref_comp1': ref_comp1,
        'ref_index': ref_index,
        'pred_frames': pred_frames,
        'pred_comp0': pred_comp0,
        'pred_comp1': pred_comp1,
        'pred_index': pred_index,
        'overlap_ref_mag': overlap_ref_mag,
        'overlap_pred_mag': overlap_pred_mag,
        'diagnostic': diagnostic,
        'error_metric': error_metric,
    }


def _quiver_grid(shape, stride):
    """Return X/Y grid positions for a downsampled quiver plot."""
    height, width = shape
    y_coords = np.arange(0, height, stride)
    x_coords = np.arange(0, width, stride)
    return np.meshgrid(x_coords + 0.5, y_coords + 0.5)


def _plot_vector_panel(ax, fig, magnitude, comp0, comp1, title,
                       vmin, vmax, stride):
    """Draw a magnitude colormap with vector arrows overlaid."""
    im = ax.pcolormesh(
        magnitude, shading='auto', vmin=vmin, vmax=vmax, cmap='viridis',
    )
    x_grid, y_grid = _quiver_grid(comp0.shape, stride)
    ax.quiver(
        x_grid,
        y_grid,
        comp0[::stride, ::stride],
        comp1[::stride, ::stride],
        color='white',
        alpha=0.85,
        pivot='mid',
        angles='xy',
        scale_units='xy',
        scale=None,
        width=0.003,
        headwidth=3.5,
        headlength=4.5,
        headaxislength=4.0,
    )
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, location='bottom', shrink=0.9, pad=0.08)


def _plot_residual_panel(ax, fig, residual, magnitude_label, eabs):
    """Draw the magnitude residual panel."""
    im = ax.pcolormesh(
        residual,
        shading='auto',
        vmin=-eabs,
        vmax=eabs,
        cmap='RdBu_r',
    )
    ax.set_title(f'Residual {magnitude_label} (Ref - Pred)')
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, location='bottom', shrink=0.9, pad=0.08)


def _plot_magnitude_spectra(ax, ref_mag, pred_mag, magnitude_label,
                            with_reference, ylim=None):
    """Draw the power spectrum of the vector magnitude."""
    if with_reference:
        k_bins, ref_power = compute_power_spectrum(ref_mag)
        ax.loglog(
            k_bins, ref_power, color='steelblue', linewidth=2,
            label='Reference',
        )

    k_bins_pred, pred_power = compute_power_spectrum(pred_mag)
    ax.loglog(
        k_bins_pred,
        pred_power,
        color='darkorange',
        linestyle='--',
        linewidth=2,
        label='Prediction',
    )
    ax.set_xlabel('Wavenumber k')
    ax.set_ylabel('P(k)')
    ax.set_title(f'Power spectrum of {magnitude_label}')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9)
    if ylim is not None:
        ax.set_ylim(ylim)


def _plot_diagnostic(ax, frame, diagnostic):
    """Draw the diagnostic time series with a moving frame marker."""
    ax.plot(
        diagnostic['ref_x'],
        diagnostic['ref_curve'],
        color='steelblue',
        linewidth=2,
        label='Reference',
    )
    ax.plot(
        diagnostic['pred_x'],
        diagnostic['pred_curve'],
        color='darkorange',
        linestyle='--',
        linewidth=2,
        label='Prediction',
    )

    marker_sources = [
        (
            diagnostic['pred_frames'],
            diagnostic['pred_x'],
            diagnostic['pred_curve'],
        ),
        (
            diagnostic['ref_frames'],
            diagnostic['ref_x'],
            diagnostic['ref_curve'],
        ),
    ]
    for marker_frames, marker_x, marker_curve in marker_sources:
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

    if diagnostic['log_scale']:
        ax.set_yscale('log')
    if diagnostic['ylim'] is not None:
        ax.set_ylim(diagnostic['ylim'])
    ax.set_title(diagnostic['title'])
    ax.set_xlabel(diagnostic['xlabel'])
    ax.set_ylabel(diagnostic['ylabel'])
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=9)


def _plot_error_metric(ax, frame, error_metric):
    """Draw the relative L2 error (%) time series with a moving frame marker."""
    ax.plot(
        error_metric['x'],
        error_metric['curve'],
        color='mediumseagreen',
        linewidth=2,
    )

    matches = np.where(error_metric['frames'] == frame)[0]
    if matches.size:
        idx = int(matches[0])
        ax.plot(
            error_metric['x'][idx],
            error_metric['curve'][idx],
            marker='o',
            markersize=12,
            color='red',
            markeredgecolor='white',
            markeredgewidth=1.5,
            linestyle='None',
            label='Current frame',
            zorder=5,
        )

    if error_metric['ylim'] is not None:
        ax.set_ylim(error_metric['ylim'])
    ax.set_title(error_metric['title'])
    ax.set_xlabel(error_metric['xlabel'])
    ax.set_ylabel(error_metric['ylabel'])
    ax.grid(True, alpha=0.3, which='both')


def _figure_title(spec, sim_id, frame, suffix='', prediction_only=False):
    """Build a consistent figure title."""
    vector_symbol = spec['vector_symbol']
    title = (
        f"{vector_symbol}=({spec['components'][0]}, {spec['components'][1]})  "
        f"|  sim {sim_id}  |  frame {frame}"
    )
    time_label = _format_alfven_time_label(spec['param'], frame)
    if time_label:
        title += f'  |  {time_label}'
    if prediction_only:
        title += '  |  prediction only'
    if suffix:
        title += f'  |  {suffix}'
    return title


def _save_frame_with_reference(job):
    """Render a vector comparison frame with reference data."""
    vis_dir = Path(job['vis_dir'])
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1.2])
    fig.suptitle(
        _figure_title(
            job['spec'], job['sim_id'], job['frame'], suffix=job['suffix'],
        ),
        fontsize=12,
    )

    ax_ref = fig.add_subplot(gs[0, 0])
    ax_pred = fig.add_subplot(gs[0, 1])
    ax_res = fig.add_subplot(gs[0, 2])
    ax_spec = fig.add_subplot(gs[1, 0])
    ax_diag = fig.add_subplot(gs[1, 1])
    ax_error = fig.add_subplot(gs[1, 2])

    _plot_vector_panel(
        ax_ref,
        fig,
        job['ref_mag'],
        job['ref_comp0'],
        job['ref_comp1'],
        f"Reference {job['spec']['vector_symbol']}",
        job['vmin'],
        job['vmax'],
        job['quiver_stride'],
    )
    _plot_vector_panel(
        ax_pred,
        fig,
        job['pred_mag'],
        job['pred_comp0'],
        job['pred_comp1'],
        f"Prediction {job['spec']['vector_symbol']}",
        job['vmin'],
        job['vmax'],
        job['quiver_stride'],
    )
    _plot_residual_panel(
        ax_res,
        fig,
        job['ref_mag'] - job['pred_mag'],
        job['spec']['magnitude_label'],
        job['eabs'],
    )
    _plot_magnitude_spectra(
        ax_spec,
        job['ref_mag'],
        job['pred_mag'],
        job['spec']['magnitude_label'],
        with_reference=True,
        ylim=job['spec_ylim'],
    )
    _plot_diagnostic(ax_diag, job['frame'], job['diagnostic'])
    _plot_error_metric(ax_error, job['frame'], job['error_metric'])

    plt.savefig(vis_dir / job['fname'], dpi=100)
    plt.close(fig)


def _save_frame_prediction_only(job):
    """Render a prediction-only frame when no reference is available."""
    vis_dir = Path(job['vis_dir'])
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1.2])
    fig.suptitle(
        _figure_title(
            job['spec'],
            job['sim_id'],
            job['frame'],
            suffix=job['suffix'],
            prediction_only=True,
        ),
        fontsize=12,
    )

    ax_blank_left = fig.add_subplot(gs[0, 0])
    ax_pred = fig.add_subplot(gs[0, 1])
    ax_blank_right = fig.add_subplot(gs[0, 2])
    ax_spec = fig.add_subplot(gs[1, 0])
    ax_diag = fig.add_subplot(gs[1, 1])
    ax_error = fig.add_subplot(gs[1, 2])

    _plot_vector_panel(
        ax_pred,
        fig,
        job['pred_mag'],
        job['pred_comp0'],
        job['pred_comp1'],
        f"Prediction {job['spec']['vector_symbol']}",
        job['vmin'],
        job['vmax'],
        job['quiver_stride'],
    )
    for ax in (ax_blank_left, ax_blank_right):
        ax.set_visible(False)

    _plot_magnitude_spectra(
        ax_spec,
        None,
        job['pred_mag'],
        job['spec']['magnitude_label'],
        with_reference=False,
        ylim=job['spec_ylim'],
    )
    _plot_diagnostic(ax_diag, job['frame'], job['diagnostic'])
    ax_error.set_visible(False)

    plt.savefig(vis_dir / job['fname'], dpi=100)
    plt.close(fig)


def _render_frame_job(job):
    """Picklable worker entrypoint for vector frame rendering."""
    if job['has_reference']:
        _save_frame_with_reference(job)
    else:
        _save_frame_prediction_only(job)
    return job['fname']


def _run_vector_jobs(jobs, desc, workers=None):
    """Render jobs serially or in parallel."""
    if not jobs:
        return 0

    if workers and workers > 1:
        ctx = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=ctx,
        ) as pool:
            futures = [pool.submit(_render_frame_job, job) for job in jobs]
            for _ in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=desc,
                unit='frame',
                leave=False,
            ):
                pass
    else:
        for job in tqdm(jobs, desc=desc, unit='frame', leave=False):
            _render_frame_job(job)
    return len(jobs)


def _simulation_limits(dataset, selected_frames):
    """Compute stable global color and residual limits for one simulation."""
    pred_idx = dataset['pred_index']
    pred_mag = _vector_magnitude(
        np.stack([dataset['pred_comp0'][pred_idx[int(f)]] for f in selected_frames]),
        np.stack([dataset['pred_comp1'][pred_idx[int(f)]] for f in selected_frames]),
    )

    overlap = sorted(set(selected_frames.tolist()) & set(dataset['ref_index']))
    if overlap:
        overlap_frames = np.asarray(overlap, dtype=np.int64)
        ref_idx = dataset['ref_index']
        ref_mag = _vector_magnitude(
            np.stack([dataset['ref_comp0'][ref_idx[int(f)]] for f in overlap_frames]),
            np.stack([dataset['ref_comp1'][ref_idx[int(f)]] for f in overlap_frames]),
        )
        pred_overlap_mag = _vector_magnitude(
            np.stack(
                [dataset['pred_comp0'][pred_idx[int(f)]] for f in overlap_frames]
            ),
            np.stack(
                [dataset['pred_comp1'][pred_idx[int(f)]] for f in overlap_frames]
            ),
        )
        vmin = float(min(pred_mag.min(), ref_mag.min()))
        vmax = float(max(pred_mag.max(), ref_mag.max()))
        eabs = float(np.max(np.abs(ref_mag - pred_overlap_mag)))
    else:
        vmin = float(pred_mag.min())
        vmax = float(pred_mag.max())
        eabs = 0.0

    return vmin, vmax, eabs


def _spectrum_limits(dataset, selected_frames):
    """Compute a stable power-spectrum y-limit over rendered frames."""
    spectra = []
    pred_idx = dataset['pred_index']
    ref_idx = dataset['ref_index']
    for frame in selected_frames:
        pred_mag = _vector_magnitude(
            dataset['pred_comp0'][pred_idx[int(frame)]],
            dataset['pred_comp1'][pred_idx[int(frame)]],
        )
        _, pred_power = compute_power_spectrum(pred_mag)
        spectra.append(pred_power)

        if int(frame) in ref_idx:
            ref_mag = _vector_magnitude(
                dataset['ref_comp0'][ref_idx[int(frame)]],
                dataset['ref_comp1'][ref_idx[int(frame)]],
            )
            _, ref_power = compute_power_spectrum(ref_mag)
            spectra.append(ref_power)

    return _spectrum_ylim(spectra)


def _frame_jobs(dataset, spec, sim_id, vis_dir, force, max_frames,
                quiver_stride):
    """Build render jobs for all selected frames of one simulation."""
    selected_frames = dataset['pred_frames']
    if max_frames is not None:
        selected_frames = selected_frames[:max_frames]

    vmin, vmax, eabs = _simulation_limits(dataset, selected_frames)
    spec_ylim = _spectrum_limits(dataset, selected_frames)
    ref_idx = dataset['ref_index']
    pred_idx = dataset['pred_index']

    # Stable y-limits for the error metric panel across all rendered frames.
    error_metric = dataset['error_metric']
    valid_errors = error_metric['curve'][np.isfinite(error_metric['curve'])]
    error_ylim = _curve_ylim([valid_errors]) if valid_errors.size else None
    error_metric_stable = dict(error_metric)
    error_metric_stable['ylim'] = error_ylim

    jobs = []
    skipped = 0
    file_prefix = spec['file_prefix']
    for frame in selected_frames:
        fname = f'{file_prefix}_frame_{int(frame):04d}_sim_{sim_id}.png'
        if (vis_dir / fname).exists() and not force:
            skipped += 1
            continue

        pred_index = pred_idx[int(frame)]
        pred_comp0 = dataset['pred_comp0'][pred_index].copy()
        pred_comp1 = dataset['pred_comp1'][pred_index].copy()
        pred_mag = _vector_magnitude(pred_comp0, pred_comp1)

        job = {
            'vis_dir': str(vis_dir),
            'fname': fname,
            'spec': spec,
            'sim_id': sim_id,
            'frame': int(frame),
            'suffix': '',
            'vmin': vmin,
            'vmax': vmax,
            'eabs': eabs,
            'spec_ylim': spec_ylim,
            'diagnostic': dataset['diagnostic'],
            'error_metric': error_metric_stable,
            'quiver_stride': quiver_stride,
            'pred_comp0': pred_comp0,
            'pred_comp1': pred_comp1,
            'pred_mag': pred_mag,
            'has_reference': int(frame) in ref_idx,
        }

        if int(frame) in ref_idx:
            ref_index = ref_idx[int(frame)]
            ref_comp0 = dataset['ref_comp0'][ref_index].copy()
            ref_comp1 = dataset['ref_comp1'][ref_index].copy()
            job['ref_comp0'] = ref_comp0
            job['ref_comp1'] = ref_comp1
            job['ref_mag'] = _vector_magnitude(ref_comp0, ref_comp1)

        jobs.append(job)

    return jobs, skipped


def visualize_prediction_pair(sim_id, pair_paths, spec, exp_dir, vis_dir,
                              force, max_frames=None, workers=None,
                              quiver_stride=8):
    """Render all frames for one paired prediction file set."""
    dataset = _build_vector_dataset(pair_paths, sim_id, spec)
    jobs, skipped = _frame_jobs(
        dataset,
        spec,
        sim_id,
        vis_dir,
        force,
        max_frames,
        quiver_stride,
    )
    desc = f'  sim {sim_id}'
    total = _run_vector_jobs(jobs, desc=desc, workers=workers)
    return total, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--param',
        type=str,
        required=True,
        help='One vector component leaf or path, e.g. bx or idefix/numpy/t20/bx.',
    )
    parser.add_argument(
        '--experiments-dir',
        type=str,
        default=EXPERIMENTS_DIR,
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Regenerate PNGs and movies even if they already exist.',
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        metavar='N',
        help='Render only the first N predicted frames per simulation.',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        metavar='N',
        help='Number of worker processes for parallel PNG rendering.',
    )
    parser.add_argument(
        '--quiver-stride',
        type=int,
        default=8,
        metavar='N',
        help='Subsample stride for quiver arrows (default: 8).',
    )
    opt = parser.parse_args()

    if opt.quiver_stride <= 0:
        raise ValueError('--quiver-stride must be a positive integer.')
    if opt.max_frames is not None and opt.max_frames <= 0:
        raise ValueError('--max-frames must be a positive integer.')

    spec = _resolve_vector_spec(opt.experiments_dir, opt.param)
    exp_dir = Path(opt.experiments_dir)
    vis_dir = _vector_output_dir(exp_dir, spec)
    vis_dir.mkdir(parents=True, exist_ok=True)

    paired_files, component_dirs = _collect_prediction_pairs(exp_dir, spec)

    print(f"Processing vector family: {spec['family']}")
    if spec['param'] != opt.param:
        print(f"Resolved data parameter: {spec['param']}")
    print(f"Experiment layout: {spec['paths'].layout}")
    print(
        'Component prediction directories: '
        + ', '.join(
            f"{component} -> {component_dirs[component]}"
            for component in spec['components']
        )
    )
    print(f'Output directory: {vis_dir}')
    print(f'Found {len(paired_files)} paired prediction file(s)')
    if opt.workers and opt.workers > 1:
        print(f'Parallel rendering: {opt.workers} worker processes')
    else:
        print('Rendering: serial (use --workers N for parallelism)')

    total_pngs = 0
    skipped_pngs = 0

    for sim_id, pair_paths in tqdm(paired_files, desc='Predictions', unit='file'):
        total, skipped = visualize_prediction_pair(
            sim_id,
            pair_paths,
            spec,
            exp_dir,
            vis_dir,
            opt.force,
            max_frames=opt.max_frames,
            workers=opt.workers,
            quiver_stride=opt.quiver_stride,
        )
        total_pngs += total
        skipped_pngs += skipped

    print(
        f'\nDone. Generated {total_pngs} PNGs, skipped {skipped_pngs} '
        f'already existing, in {vis_dir}/'
    )

    print('\nRendering movies...')
    prefix = spec['file_prefix']
    for sim_id, _ in paired_files:
        render_movie(
            vis_dir,
            png_glob=f'{prefix}_frame_*_sim_{sim_id}.png',
            movie_name=f'{prefix}_movie_sim_{sim_id}.mp4',
            force=opt.force,
        )


if __name__ == '__main__':
    main()
