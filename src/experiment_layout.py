from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
ROOT_EXPERIMENTS_DIR = ROOT_DIR / 'experiments'

MANIFEST_NAME = 'manifest.json'
RUN_PARAMS_DIRNAME = 'params'
RUN_VECTOR_DIRNAME = 'vector'
RUN_RENDER_DIRNAME = 'renders'

LEGACY_LAYOUT = 'legacy'
RUN_LAYOUT = 'run'


@dataclass(frozen=True)
class ExperimentParamPaths:
    exp_dir: Path
    layout: str
    input_param: str
    param_key: str
    data_param: str
    param_dir: Path
    prediction_dir: Path
    render_dir: Path
    checkpoint_dir: Path
    model_path: Path
    best_model_path: Path
    training_state_path: Path
    loss_path: Path
    log_path: Path


def is_run_layout(exp_dir: str | Path) -> bool:
    exp_path = Path(exp_dir)
    return (
        (exp_path / MANIFEST_NAME).exists()
        or (exp_path / RUN_PARAMS_DIRNAME).is_dir()
    )


def resolve_experiment_param(
    exp_dir: str | Path,
    param: str,
    data_dir: str | Path,
    *,
    create: bool = False,
) -> ExperimentParamPaths:
    exp_path = Path(exp_dir)
    data_path = Path(data_dir)
    layout = _select_layout(exp_path, create=create)
    manifest = load_manifest(exp_path) if layout == RUN_LAYOUT else {}
    data_param = _resolve_data_param(
        exp_path, param, data_path, layout=layout, manifest=manifest,
    )
    param_key = Path(data_param).name

    if layout == RUN_LAYOUT:
        param_dir = exp_path / RUN_PARAMS_DIRNAME / param_key
        prediction_dir = param_dir
        render_dir = param_dir / RUN_RENDER_DIRNAME
        checkpoint_dir = param_dir
    else:
        param_dir = exp_path / Path(data_param)
        prediction_dir = param_dir / 'visualizations'
        render_dir = prediction_dir
        checkpoint_dir = param_dir / 'checkpoints'

    return ExperimentParamPaths(
        exp_dir=exp_path,
        layout=layout,
        input_param=param,
        param_key=param_key,
        data_param=data_param,
        param_dir=param_dir,
        prediction_dir=prediction_dir,
        render_dir=render_dir,
        checkpoint_dir=checkpoint_dir,
        model_path=checkpoint_dir / 'model_64_30.pt',
        best_model_path=checkpoint_dir / 'model_best.pt',
        training_state_path=checkpoint_dir / 'training_state.pt',
        loss_path=checkpoint_dir / 'loss_64_30.npy',
        log_path=checkpoint_dir / 'train.log',
    )


def ensure_param_layout(paths: ExperimentParamPaths) -> None:
    paths.param_dir.mkdir(parents=True, exist_ok=True)
    paths.render_dir.mkdir(parents=True, exist_ok=True)
    if paths.layout == LEGACY_LAYOUT:
        paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)


def load_manifest(exp_dir: str | Path) -> dict[str, Any]:
    manifest_path = Path(exp_dir) / MANIFEST_NAME
    if not manifest_path.exists():
        return {}

    with open(manifest_path) as handle:
        return json.load(handle)


def write_manifest(
    paths: ExperimentParamPaths,
    metadata: dict[str, Any] | None = None,
) -> Path | None:
    if paths.layout != RUN_LAYOUT:
        return None

    manifest_path = paths.exp_dir / MANIFEST_NAME
    manifest = load_manifest(paths.exp_dir)
    now = _timestamp()
    if not manifest:
        manifest = {
            'layout_version': 1,
            'layout': 'run-centric-v1',
            'created_at': now,
        }

    manifest['updated_at'] = now
    manifest.setdefault(
        'artifacts',
        {
            'params_dir': RUN_PARAMS_DIRNAME,
            'vector_dir': RUN_VECTOR_DIRNAME,
            'render_dirname': RUN_RENDER_DIRNAME,
        },
    )
    params = manifest.setdefault('params', {})
    params[paths.param_key] = {
        'data_param': paths.data_param,
        'param_dir': str(
            Path(RUN_PARAMS_DIRNAME) / paths.param_key
        ),
        'render_dir': str(
            Path(RUN_PARAMS_DIRNAME) / paths.param_key / RUN_RENDER_DIRNAME
        ),
        'prediction_glob': 'pred_sim_*.npy',
    }

    if metadata:
        _deep_merge(manifest, metadata)

    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write('\n')
    return manifest_path


def prediction_file(
    paths: ExperimentParamPaths,
    sim_id: str,
    *,
    is_rollout: bool = False,
) -> Path:
    suffix = '_rollout' if is_rollout else ''
    return paths.prediction_dir / f'pred_sim_{sim_id}{suffix}.npy'


def prediction_files(paths: ExperimentParamPaths) -> list[Path]:
    return sorted(paths.prediction_dir.glob('pred_sim_*.npy'))


def resolve_vector_output_dir(
    exp_dir: str | Path,
    family: str,
    *,
    data_param: str | None = None,
) -> Path:
    exp_path = Path(exp_dir)
    if is_run_layout(exp_path):
        return exp_path / RUN_VECTOR_DIRNAME / family

    dirname = f'{family}_vector_visualizations'
    if data_param is None:
        return exp_path / dirname

    parent = Path(data_param).parent
    if str(parent) == '.':
        return exp_path / dirname
    return exp_path / parent / dirname


def resolve_param_key(
    exp_dir: str | Path,
    param: str,
    data_dir: str | Path,
) -> str:
    return resolve_experiment_param(exp_dir, param, data_dir).param_key


def _select_layout(exp_dir: Path, *, create: bool) -> str:
    if is_run_layout(exp_dir):
        return RUN_LAYOUT
    if _looks_like_legacy_experiment_root(exp_dir):
        return LEGACY_LAYOUT
    if create and exp_dir.resolve() != ROOT_EXPERIMENTS_DIR.resolve():
        return RUN_LAYOUT
    return LEGACY_LAYOUT


def _resolve_data_param(
    exp_dir: Path,
    param: str,
    data_dir: Path,
    *,
    layout: str,
    manifest: dict[str, Any],
) -> str:
    param_path = Path(param)
    param_norm = param_path.as_posix()
    leaf = param_path.name

    if layout == RUN_LAYOUT:
        manifest_params = manifest.get('params', {})
        if leaf in manifest_params:
            return manifest_params[leaf]['data_param']
        if len(param_path.parts) > 1:
            return param_norm
        return discover_data_param(data_dir, leaf)

    direct_candidate = exp_dir / param_path
    if _looks_like_legacy_param_dir(direct_candidate):
        return param_norm

    candidates = _legacy_param_candidates(exp_dir, param_norm)
    if candidates:
        if len(candidates) > 1:
            joined = ', '.join(candidates)
            raise ValueError(
                f"Parameter '{param}' is ambiguous under {exp_dir}. "
                f"Matches: {joined}. Pass the full nested path or a specific "
                '--experiments-dir run root.'
            )
        return candidates[0]

    if len(param_path.parts) > 1:
        return param_norm
    return discover_data_param(data_dir, leaf)


def discover_data_param(data_dir: str | Path, param: str) -> str:
    data_path = Path(data_dir)
    param_path = Path(param)
    param_norm = param_path.as_posix()

    direct = data_path / param_path
    if _looks_like_dataset_param_dir(direct):
        return param_norm

    leaf = param_path.name
    matches = sorted({
        candidate.relative_to(data_path).as_posix()
        for candidate in data_path.rglob(leaf)
        if _looks_like_dataset_param_dir(candidate)
    })
    if not matches:
        raise FileNotFoundError(
            f"No dataset parameter directory matching '{param}' exists under "
            f'{data_path}.'
        )
    if len(matches) > 1:
        preferred = _preferred_data_matches(matches, leaf)
        if len(preferred) == 1:
            return preferred[0]
        joined = ', '.join(matches)
        raise ValueError(
            f"Parameter leaf '{param}' is ambiguous under {data_path}. "
            f"Matches: {joined}. Pass the full dataset path."
        )
    return matches[0]


def _legacy_param_candidates(exp_dir: Path, param: str) -> list[str]:
    param_path = Path(param)
    leaf = param_path.name
    candidates = set()

    for marker in ('checkpoints', 'visualizations'):
        for marker_dir in exp_dir.rglob(marker):
            parent = marker_dir.parent
            if not _looks_like_legacy_param_dir(parent):
                continue

            relative = parent.relative_to(exp_dir).as_posix()
            if len(param_path.parts) > 1:
                if relative == param or relative.endswith(f'/{param}'):
                    candidates.add(relative)
            elif parent.name == leaf:
                candidates.add(relative)

    return sorted(candidates)


def _looks_like_dataset_param_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any((path / split).is_dir() for split in ('train', 'val', 'test'))


def _looks_like_legacy_param_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (
        (path / 'checkpoints').is_dir()
        or (path / 'visualizations').is_dir()
    )


def _looks_like_legacy_experiment_root(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False

    for marker in ('checkpoints', 'visualizations'):
        if any(path.rglob(marker)):
            return True
    return False


def _deep_merge(dest: dict[str, Any], src: dict[str, Any]) -> None:
    for key, value in src.items():
        if (
            key in dest
            and isinstance(dest[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(dest[key], value)
        else:
            dest[key] = value


def _preferred_data_matches(matches: list[str], leaf: str) -> list[str]:
    preferred_suffixes = (
        f'idefix/numpy/t20/{leaf}',
        f'numpy/t20/{leaf}',
        f't20/{leaf}',
    )
    for suffix in preferred_suffixes:
        preferred = [match for match in matches if match.endswith(suffix)]
        if preferred:
            return preferred
    return []


def _timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'
