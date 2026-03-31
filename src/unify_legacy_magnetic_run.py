"""Copy legacy bx/by experiment artifacts into one run-scoped root.

This is a one-off migration utility for older magnetic-field experiments that
were trained separately in legacy nested directories such as:

    experiments/64_30_autoreg--2026-29-03/idefix/numpy/t20/by/
    experiments/64_30_autoreg--2026-30-03/idefix/numpy/t20/bx/

The current pipeline expects both components under one shared run root so that
`inference.py`, `viz_scalar.py`, and especially `viz_vector.py` can resolve a
single experiment layout:

    <run>/
      manifest.json
      params/
        bx/
          model_best.pt
          model_64_30.pt
          training_state.pt
          loss_64_30.npy
          train.log
          predictions/
            teacher_forced/
            rollout/
          renders/
        by/
          ...
      vector/
        magnetic/

This script discovers the legacy `bx` and `by` sources, copies the recognized
artifacts into the run-scoped destination, and writes a modern manifest via
`experiment_layout.write_manifest(...)`.

Examples
--------

Use the current `experiments/64_30_autoreg*` sources and copy them into a new
shared run root:

    python src/unify_legacy_magnetic_run.py \
        --destination experiments/64_30_magfield_v2

Overwrite an existing destination root:

    python src/unify_legacy_magnetic_run.py \
        --destination experiments/64_30_magfield \
        --force

Provide explicit source directories instead of glob discovery:

    python src/unify_legacy_magnetic_run.py \
        --source-dir experiments/64_30_autoreg--2026-29-03 \
        --source-dir experiments/64_30_autoreg--2026-30-03 \
        --destination experiments/64_30_magfield_v2
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil

from experiment_layout import (
    ExperimentParamPaths,
    ensure_param_layout,
    prediction_files,
    prediction_mode_dir,
    reference_dir,
    resolve_experiment_param,
    resolve_vector_output_dir,
    write_manifest,
)


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / 'data'
EXPERIMENTS_DIR = ROOT_DIR / 'experiments'
MAGNETIC_COMPONENTS = ('bx', 'by')
CHECKPOINT_FILE_NAMES = (
    'model_64_30.pt',
    'model_best.pt',
    'training_state.pt',
    'loss_64_30.npy',
    'train.log',
)


@dataclass(frozen=True)
class CopyStats:
    checkpoints: int = 0
    predictions_teacher_forced: int = 0
    predictions_rollout: int = 0
    references: int = 0
    renders: int = 0


@dataclass(frozen=True)
class LegacyComponentSource:
    exp_dir: Path
    param: str


def _resolve_cli_path(path_str: str, root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _iter_source_dirs(args: argparse.Namespace) -> list[Path]:
    if args.source_dir:
        source_dirs = [
            _resolve_cli_path(path_str, args.experiments_root)
            for path_str in args.source_dir
        ]
    else:
        source_dirs = sorted(args.experiments_root.glob(args.source_glob))

    destination = _resolve_cli_path(args.destination, args.experiments_root)
    return [
        source_dir.resolve()
        for source_dir in source_dirs
        if source_dir.resolve() != destination
    ]


def _legacy_component_candidates(exp_dir: Path, component: str) -> list[str]:
    candidates = set()
    for marker in ('checkpoints', 'visualizations'):
        for marker_dir in exp_dir.rglob(marker):
            parent = marker_dir.parent
            if not parent.is_dir() or parent.name != component:
                continue
            if (
                (parent / 'checkpoints').is_dir()
                or (parent / 'visualizations').is_dir()
            ):
                candidates.add(parent.relative_to(exp_dir).as_posix())
    return sorted(candidates)


def _pick_legacy_component_candidate(candidates: list[str], component: str
                                     ) -> str | None:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    preferred_suffixes = (
        f'idefix/numpy/t20/{component}',
        f'numpy/t20/{component}',
        f't20/{component}',
        f'idefix/numpy/{component}',
        f'numpy/{component}',
        component,
    )
    for suffix in preferred_suffixes:
        preferred = [candidate for candidate in candidates
                     if candidate.endswith(suffix)]
        if len(preferred) == 1:
            return preferred[0]

    joined = ', '.join(candidates)
    raise ValueError(f"Ambiguous legacy '{component}' candidates: {joined}")


def _discover_component_sources(
    source_dirs: list[Path],
) -> dict[str, LegacyComponentSource]:
    sources: dict[str, list[LegacyComponentSource]] = {
        component: [] for component in MAGNETIC_COMPONENTS
    }
    for source_dir in source_dirs:
        if not source_dir.is_dir():
            raise FileNotFoundError(f'Source directory not found: {source_dir}')
        for component in MAGNETIC_COMPONENTS:
            candidates = _legacy_component_candidates(source_dir, component)
            selected = _pick_legacy_component_candidate(candidates, component)
            if selected is not None:
                sources[component].append(
                    LegacyComponentSource(exp_dir=source_dir, param=selected)
                )

    resolved_sources = {}
    for component, matches in sources.items():
        if not matches:
            joined = ', '.join(str(path) for path in source_dirs)
            raise FileNotFoundError(
                f"No legacy '{component}' source found in: {joined}"
            )
        if len(matches) > 1:
            joined = ', '.join(
                f'{match.exp_dir}:{match.param}' for match in matches
            )
            raise ValueError(
                f"Ambiguous '{component}' sources found: {joined}"
            )
        resolved_sources[component] = matches[0]
    return resolved_sources


def _copy_file(src: Path, dst: Path, *, force: bool, dry_run: bool) -> bool:
    if dst.exists():
        if not force:
            raise FileExistsError(
                f'Destination already exists: {dst} '
                '(re-run with --force to overwrite)'
            )
        if not dry_run:
            dst.unlink()

    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return True


def _parse_prediction_path(path: Path) -> tuple[str, bool]:
    stem = path.stem
    if not stem.startswith('pred_sim_'):
        raise ValueError(f'Unrecognized prediction filename: {path.name}')

    suffix = stem[len('pred_sim_'):]
    is_rollout = suffix.endswith('_rollout')
    sim_id = suffix.removesuffix('_rollout')
    if not sim_id:
        raise ValueError(f'Could not parse simulation id from {path.name}')
    return sim_id, is_rollout


def _prediction_destination(dest_paths: ExperimentParamPaths, sim_id: str,
                            is_rollout: bool, *, create: bool) -> Path:
    suffix = '_rollout' if is_rollout else ''
    mode_dir = prediction_mode_dir(
        dest_paths, is_rollout=is_rollout, create=create,
    )
    return mode_dir / f'pred_sim_{sim_id}{suffix}.npy'


def _copy_checkpoints(source_paths, dest_paths, *, force, dry_run) -> int:
    copied = 0
    for file_name in CHECKPOINT_FILE_NAMES:
        src = source_paths.checkpoint_dir / file_name
        dst = dest_paths.checkpoint_dir / file_name
        if not src.exists():
            continue
        if _copy_file(src, dst, force=force, dry_run=dry_run):
            copied += 1
    return copied


def _copy_visualization_tree(source_paths, dest_paths, *, force, dry_run):
    tf_count = 0
    rollout_count = 0
    ref_count = 0
    render_count = 0

    if not source_paths.render_dir.exists():
        return tf_count, rollout_count, ref_count, render_count

    for src in sorted(source_paths.render_dir.rglob('*')):
        if not src.is_file():
            continue

        name = src.name
        if name.startswith('pred_sim_') and src.suffix == '.npy':
            sim_id, is_rollout = _parse_prediction_path(src)
            dst = _prediction_destination(
                dest_paths, sim_id, is_rollout, create=not dry_run,
            )
            _copy_file(src, dst, force=force, dry_run=dry_run)
            if is_rollout:
                rollout_count += 1
            else:
                tf_count += 1
            continue

        if name.startswith('ref_sim_') and src.suffix == '.npy':
            dst_ref_dir = reference_dir(dest_paths, split='test', create=not dry_run)
            if dst_ref_dir is None:
                raise ValueError(
                    f'Run-scoped destination expected for references: '
                    f'{dest_paths.exp_dir}'
                )
            dst = dst_ref_dir / name
            _copy_file(src, dst, force=force, dry_run=dry_run)
            ref_count += 1
            continue

        relative = src.relative_to(source_paths.render_dir)
        dst = dest_paths.render_dir / relative
        _copy_file(src, dst, force=force, dry_run=dry_run)
        render_count += 1

    return tf_count, rollout_count, ref_count, render_count


def _copy_component(source: LegacyComponentSource, destination: Path,
                    component: str, *, force: bool, dry_run: bool
                    ) -> tuple[ExperimentParamPaths, CopyStats]:
    source_paths = resolve_experiment_param(
        source.exp_dir, source.param, DATA_DIR,
    )
    dest_paths = resolve_experiment_param(
        destination, source_paths.data_param, DATA_DIR, create=True,
    )
    if dest_paths.layout != 'run':
        raise ValueError(
            'Destination must resolve to a run-scoped experiment root, got '
            f'{dest_paths.exp_dir}.'
        )
    if not dry_run:
        ensure_param_layout(dest_paths)

    checkpoints = _copy_checkpoints(
        source_paths, dest_paths, force=force, dry_run=dry_run,
    )
    tf_count, rollout_count, ref_count, render_count = _copy_visualization_tree(
        source_paths, dest_paths, force=force, dry_run=dry_run,
    )
    stats = CopyStats(
        checkpoints=checkpoints,
        predictions_teacher_forced=tf_count,
        predictions_rollout=rollout_count,
        references=ref_count,
        renders=render_count,
    )
    return dest_paths, stats


def _write_component_manifest(dest_paths, source: LegacyComponentSource,
                              component_stats: CopyStats,
                              all_sources: dict[str, LegacyComponentSource]
                              ) -> None:
    write_manifest(
        dest_paths,
        metadata={
            'migration': {
                'script': Path(__file__).name,
                'mode': 'copy',
                'sources': {
                    component: {
                        'experiment': str(source.exp_dir),
                        'param': source.param,
                    }
                    for component, source in all_sources.items()
                },
            },
            'params': {
                dest_paths.param_key: {
                    'migration': {
                        'source_experiment': str(source.exp_dir),
                        'source_param': source.param,
                        'copied': {
                            'checkpoints': component_stats.checkpoints,
                            'predictions_teacher_forced': (
                                component_stats.predictions_teacher_forced
                            ),
                            'predictions_rollout': (
                                component_stats.predictions_rollout
                            ),
                            'references': component_stats.references,
                            'renders': component_stats.renders,
                        },
                    },
                },
            },
        },
    )


def _shared_prediction_keys(destination: Path) -> set[tuple[str, bool]]:
    component_keys = []
    for component in MAGNETIC_COMPONENTS:
        paths = resolve_experiment_param(destination, component, DATA_DIR)
        keys = {
            _parse_prediction_path(path)
            for path in prediction_files(paths)
        }
        component_keys.append(keys)
    return set.intersection(*component_keys)


def _print_summary(destination: Path,
                   component_sources: dict[str, LegacyComponentSource],
                   component_stats: dict[str, CopyStats], dry_run: bool) -> None:
    action = 'Would copy' if dry_run else 'Copied'
    print(f'{action} legacy magnetic artifacts into: {destination}')
    for component in MAGNETIC_COMPONENTS:
        stats = component_stats[component]
        print(
            f'  {component}: source={component_sources[component].exp_dir}'
            f' ({component_sources[component].param}) | '
            f'checkpoints={stats.checkpoints}, '
            f'teacher_forced={stats.predictions_teacher_forced}, '
            f'rollout={stats.predictions_rollout}, '
            f'references={stats.references}, '
            f'renders={stats.renders}'
        )
    if dry_run:
        return

    shared_keys = _shared_prediction_keys(destination)
    if shared_keys:
        shared_labels = ', '.join(
            f'{sim_id}{" rollout" if is_rollout else ""}'
            for sim_id, is_rollout in sorted(shared_keys)
        )
        print(f'  shared prediction pairs: {shared_labels}')
        return

    print(
        '  warning: no shared bx/by prediction pairs were copied. '
        'The unified run is still valid for future inference, but '
        '`viz_vector.py` needs matching sim IDs in the same mode for both '
        'components.'
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Copy legacy bx/by experiment artifacts into one run-scoped root '
            'compatible with the current inference and visualization pipeline.'
        )
    )
    parser.add_argument(
        '--source-dir',
        action='append',
        default=[],
        help=(
            'Explicit legacy experiment directory. Repeat this flag to pass '
            'multiple source roots. Relative paths resolve under '
            '--experiments-root.'
        ),
    )
    parser.add_argument(
        '--source-glob',
        type=str,
        default='64_30_autoreg*',
        help=(
            'Glob used under --experiments-root when no --source-dir flags are '
            'provided (default: 64_30_autoreg*).'
        ),
    )
    parser.add_argument(
        '--experiments-root',
        type=Path,
        default=EXPERIMENTS_DIR,
        help='Root directory for resolving relative experiment paths.',
    )
    parser.add_argument(
        '--destination',
        type=str,
        required=True,
        help=(
            'Destination run root. Relative paths resolve under '
            '--experiments-root.'
        ),
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite destination files when they already exist.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print what would be copied without writing any files.',
    )
    args = parser.parse_args()

    args.experiments_root = args.experiments_root.resolve()
    destination = _resolve_cli_path(args.destination, args.experiments_root)
    source_dirs = _iter_source_dirs(args)
    if not source_dirs:
        raise FileNotFoundError(
            f'No source directories found under {args.experiments_root} '
            f'for glob {args.source_glob!r}.'
        )

    component_sources = _discover_component_sources(source_dirs)
    component_stats: dict[str, CopyStats] = {}

    if not args.dry_run:
        resolve_vector_output_dir(destination, 'magnetic').mkdir(
            parents=True, exist_ok=True,
        )

    for component, source_dir in component_sources.items():
        dest_paths, stats = _copy_component(
            source_dir, destination, component,
            force=args.force, dry_run=args.dry_run,
        )
        component_stats[component] = stats
        if not args.dry_run:
            _write_component_manifest(
                dest_paths, source_dir, stats, component_sources,
            )

    _print_summary(destination, component_sources, component_stats, args.dry_run)


if __name__ == '__main__':
    main()
