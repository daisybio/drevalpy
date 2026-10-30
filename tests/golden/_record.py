"""Record the golden baseline: predictions and metrics of every zoo model on one fixed fold.

Run ``python -m tests.golden._record`` from the repository root (with ``src`` on ``PYTHONPATH``) to
rewrite ``tests/golden/data/``. Record only on a commit whose numerics are the reference; the
refactor packages compare against the baseline, they never re-record it.

Every call into drevalpy that trains or predicts goes through :func:`train_and_predict` and
:func:`round_trip_predict`. When ``ModelSpec``, ``DRPModel``, the HPO entry points or the archive
format change signature, adapt these two functions and nothing else: the baseline stays valid as
long as the numbers they produce do not change.

:func:`train_and_predict` mirrors ``drevalpy._single._execute`` step by step (the same scopes, seeds
and HPO call), because ``drevalpy.single`` does not hand back the fitted model the round trip needs.
The recorder checks that both give bit-identical predictions, so the baseline is anchored to the
public ``drevalpy.single`` path.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import json
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from upath import UPath

#: Directory holding the baseline: ``predictions.npz`` and ``baseline.json``.
DATA_DIR: Final = UPath(__file__).resolve().parent / "data"
PREDICTIONS_FILE: Final = DATA_DIR / "predictions.npz"
BASELINE_FILE: Final = DATA_DIR / "baseline.json"

DATASET_SEED: Final = 0
SPLIT_MODE: Final = "LPO"
N_SPLITS: Final = 5
SPLIT_SEED: Final = 42
FOLD_INDEX: Final = 0
SEED: Final = 42
HPO_MODEL: Final = "ElasticNet"
HPO_TRIALS: Final = 2
HPO_SEED: Final = 7
#: Baseline key of the tuned ElasticNet run.
HPO_KEY: Final = f"{HPO_MODEL}[hpo]"
#: Tolerance for models whose predictions are not bit-identical across two runs in one process.
ATOL: Final = 1e-6

#: Shorter training for the models that take minutes at their defaults (PharmaFormer about 19 minutes,
#: RandomForestAEMolGNet 13, PaccMann 8, DIPK 2.5), so the gate stays well under 15 minutes. The code
#: path is the one the defaults take; only the number of epochs (or ensemble members) shrinks. Keys
#: are the short public hyperparameter names, which stay valid user input after the refactor.
OVERRIDES: Final[dict[str, dict[str, Any]]] = {
    "DIPK": {"epochs": 3, "epochs_autoencoder": 10},
    "EnsembleMF": {"n_ensemble": 3, "max_epochs": 30},
    "PaccMann": {"epochs": 1},
    "PharmaFormer": {"epochs": 2},
    "RandomForestAEMolGNet": {"epochs_autoencoder": 10},
}

#: Every opt-in extra of ``drevalpy.synthetic``, so each zoo model finds the views it reads.
DATASET_OPTIONS: Final[dict[str, Any]] = {
    "omics": ("gene_expression", "proteomics", "mutations", "methylation", "copy_number_variation_gistic"),
    "sample_views": ("pathway_features",),
    "drug_views": ("morgan_fingerprint", "chemberta", "bpe_smiles", "smilesvec"),
    "uns_payloads": ("drug_graphs", "pathways_gmt", "sparsego", "bpe_codes"),
    "extra_layers": ("AUC", "IC50"),
}


@dataclass(frozen=True)
class GoldenRun:
    """Outcome of one golden fit: test predictions in scope order, metrics and the fitted model."""

    predictions: np.ndarray
    metrics: dict[str, float]
    best_hyperparameters: dict[str, Any]
    model: Any


def build_dataset():
    """Build the golden dataset: ``drevalpy.synthetic`` with every extra view on."""
    import drevalpy

    return drevalpy.synthetic(seed=DATASET_SEED, **{key: list(value) for key, value in DATASET_OPTIONS.items()})


def golden_split(dataset):
    """Return fold :data:`FOLD_INDEX` of the :data:`SPLIT_MODE` split of *dataset*."""
    import drevalpy

    return drevalpy.split(dataset, SPLIT_MODE, n_splits=N_SPLITS, random_state=SPLIT_SEED)[FOLD_INDEX]


def zoo_names() -> list[str]:
    """Return the sorted names of the built-in zoo presets."""
    from drevalpy.models.zoo import list_zoo_names

    return list_zoo_names(include_external=False)


def train_and_predict(
    model_name: str,
    dataset,
    split,
    *,
    hyperparameter_tuning: bool = False,
    overrides: dict[str, Any] | None = None,
    hpo_num_trials: int = HPO_TRIALS,
    hpo_random_state: int = HPO_SEED,
    seed: int = SEED,
) -> GoldenRun:
    """Train zoo model *model_name* on *split* the way ``drevalpy.single`` does and predict its test scope.

    The single place that trains: adapt it when the model or tuning API changes.

    :param model_name: Zoo preset name.
    :param dataset: Golden dataset.
    :param split: Fold to train and test on.
    :param hyperparameter_tuning: Whether to tune, as ``drevalpy.single(hyperparameter_tuning=...)``.
    :param overrides: Short public hyperparameter names and values that replace defaults, untuned only.
    :param hpo_num_trials: Number of tuning trials.
    :param hpo_random_state: Seed of the tuning sampler.
    :param seed: Root seed of the run.
    :returns: Predictions, metrics, chosen hyperparameters and the fitted model.
    """
    from drevalpy._utils.seed import fit_seed, seed_everything
    from drevalpy.evaluation._scoring import run_metrics
    from drevalpy.experiment._execution import build_execution
    from drevalpy.models import DRPModel, ModelSpec
    from drevalpy.models.tuning.config import HPOConfig
    from drevalpy.models.tuning.hpo import hpam_tune
    from drevalpy.types.data.batch.response_batch import ResponseBatch

    spec = ModelSpec.from_name(model_name)
    execution = build_execution(
        spec,
        dataset,
        split,
        hyperparameter_tuning=hyperparameter_tuning,
        hpo_metric="RMSE",
        hpo_num_trials=hpo_num_trials,
        hpo_random_state=hpo_random_state,
        seed=seed,
    )
    scopes = execution.scopes
    fit_rng_seed = fit_seed(execution.seed, execution.identity)
    if hyperparameter_tuning:
        seed_everything(fit_rng_seed)
        best, _ = hpam_tune(
            model_spec=spec,
            dataset=dataset,
            train_scope=scopes.training,
            val_scope=scopes.validation,
            early_stopping_scope=scopes.stopping,
            model_checkpoint_dir=None,
            hpo_config=HPOConfig(n_trials=hpo_num_trials, metric="RMSE", random_state=hpo_random_state),
            split_index=execution.identity.fold_index,
        )
    else:
        best = spec.with_hyperparameters(overrides or {}).hyperparameters
    model = DRPModel(spec, best)
    seed_everything(fit_rng_seed)
    with tempfile.TemporaryDirectory() as checkpoint_dir:
        model.train(
            dataset=dataset,
            scope=scopes.final_training,
            early_stopping_scope=scopes.stopping,
            model_checkpoint_dir=checkpoint_dir,
        )
    predictions = np.asarray(model.predict(dataset=dataset, scope=scopes.test))
    metrics = run_metrics(predictions, ResponseBatch.from_dataset(dataset, scopes.test).response)
    return GoldenRun(predictions, dict(metrics), model.hyperparameters, model)


def round_trip_predict(run: GoldenRun, dataset, split) -> np.ndarray:
    """Save the fitted model of *run* to a temporary archive, load it and predict the test scope again.

    :param run: Fitted golden run.
    :param dataset: Golden dataset.
    :param split: Fold the model was trained on.
    :returns: Predictions of the reloaded model, in test-scope order.
    """
    from drevalpy.models import DRPModel

    with tempfile.TemporaryDirectory() as directory:
        path = f"{directory}/model.zip"
        run.model.save(path)
        loaded = DRPModel.load(path)
    return np.asarray(loaded.predict(dataset, split.test))


def single_predictions(model_name: str, dataset, split, *, hyperparameter_tuning: bool) -> np.ndarray:
    """Predict through the public ``drevalpy.single``, to anchor :func:`train_and_predict` to it."""
    import drevalpy

    result = drevalpy.single(
        drevalpy.ModelSpec.from_name(model_name),
        dataset,
        split,
        hyperparameter_tuning=hyperparameter_tuning,
        hpo_num_trials=HPO_TRIALS,
        hpo_random_state=HPO_SEED,
        seed=SEED,
    )
    return np.asarray(result.predictions)


def hyperparameter_values(hyperparameters: dict[str, Any]) -> list[str]:
    """Return the sorted ``repr`` of every hyperparameter value, independent of the key form."""
    return sorted(repr(value) for value in hyperparameters.values())


def metrics_equal(actual: dict[str, float], expected: dict[str, float]) -> bool:
    """Whether two metric dicts have the same keys and bit-identical values, NaN equal to NaN."""
    return actual.keys() == expected.keys() and all(
        np.array_equal(np.float64(actual[key]), np.float64(expected[key]), equal_nan=True) for key in expected
    )


def predictions_match(actual: np.ndarray, expected: np.ndarray, *, exact: bool) -> bool:
    """Compare predictions bit for bit, or within :data:`ATOL` when *exact* is false."""
    if actual.shape != expected.shape:
        return False
    if exact:
        return bool(np.array_equal(actual, expected, equal_nan=True))
    return bool(np.allclose(actual, expected, rtol=0, atol=ATOL, equal_nan=True))


@contextmanager
def _artifact_watch() -> Iterator[set[str]]:
    """Collect the names of the downloadable artifacts (pretrained weights, PPI tables) read inside the block.

    A model that reads one runs offline only from a warm ``DREVALPY_CACHE_DIR``, so it is marked ``network``.
    """
    from drevalpy.data import artifacts as module

    seen: set[str] = set()
    original = module._cache_root

    def watched():
        seen.add(str(sys._getframe(1).f_locals.get("name", "?")))
        return original()

    module._cache_root = watched
    try:
        yield seen
    finally:
        module._cache_root = original


def _record_one(name: str, dataset, split, *, hyperparameter_tuning: bool) -> tuple[np.ndarray, dict[str, Any]]:
    """Run one model three ways and return its predictions and baseline entry."""
    overrides = {} if hyperparameter_tuning else OVERRIDES.get(name, {})
    start = time.perf_counter()
    options = {"hyperparameter_tuning": hyperparameter_tuning, "overrides": overrides}
    with _artifact_watch() as artifacts:
        first = train_and_predict(name, dataset, split, **options)
    seconds = time.perf_counter() - start
    second = train_and_predict(name, dataset, split, **options)
    deterministic = predictions_match(first.predictions, second.predictions, exact=True) and metrics_equal(
        first.metrics, second.metrics
    )
    if not deterministic and not predictions_match(first.predictions, second.predictions, exact=False):
        raise RuntimeError(f"{name}: two runs differ by more than atol={ATOL}")
    if not overrides:
        public = single_predictions(name, dataset, split, hyperparameter_tuning=hyperparameter_tuning)
        if not predictions_match(public, first.predictions, exact=deterministic):
            raise RuntimeError(f"{name}: train_and_predict disagrees with drevalpy.single")
    loaded = round_trip_predict(first, dataset, split)
    round_trip_exact = predictions_match(loaded, first.predictions, exact=True)
    if not round_trip_exact and not predictions_match(loaded, first.predictions, exact=False):
        raise RuntimeError(f"{name}: reloaded predictions differ by more than atol={ATOL}")
    entry = {
        "model": name,
        "hyperparameter_tuning": hyperparameter_tuning,
        "hyperparameter_overrides": overrides,
        "artifacts": sorted(artifacts),
        "deterministic": deterministic,
        "round_trip_exact": round_trip_exact,
        "metrics": first.metrics,
        "best_hyperparameter_values": hyperparameter_values(first.best_hyperparameters),
        "best_hyperparameters_at_record": {key: repr(value) for key, value in first.best_hyperparameters.items()},
        "seconds": round(seconds, 1),
    }
    return first.predictions, entry


def _git_commit() -> str:
    git = shutil.which("git")
    if git is None:
        return "unknown"
    try:
        return subprocess.run(  # noqa: S603 - fixed argv, resolved git binary
            [git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=DATA_DIR.parent
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _load_existing() -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, str]]:
    """Return the predictions, model entries and exclusions of the baseline on disk, or empty ones."""
    if not (PREDICTIONS_FILE.exists() and BASELINE_FILE.exists()):
        return {}, {}, {}
    with PREDICTIONS_FILE.open("rb") as handle, np.load(handle) as archive:
        predictions = {key: archive[key] for key in archive.files}
    with BASELINE_FILE.open(encoding="utf-8") as handle:
        baseline = json.load(handle)
    return predictions, baseline["models"], baseline["network_only"]


def record(names: list[str] | None = None) -> dict[str, Any]:
    """Record the baseline and write it.

    :param names: Baseline keys to re-record (zoo names, or :data:`HPO_KEY`), merged into the baseline
        on disk; ``None`` records the whole zoo plus the tuned ElasticNet from scratch.
    :returns: The written baseline metadata.
    """
    dataset = build_dataset()
    split = golden_split(dataset)
    predictions, models, excluded = _load_existing() if names else ({}, {}, {})
    keys = names or [*zoo_names(), HPO_KEY]
    jobs = [(key, HPO_MODEL, True) if key == HPO_KEY else (key, key, False) for key in keys]
    for key, name, tuned in jobs:
        print(f"[golden] {key} ...", file=sys.stderr, flush=True)
        predictions.pop(key, None)
        models.pop(key, None)
        excluded.pop(key, None)
        try:
            predictions[key], models[key] = _record_one(name, dataset, split, hyperparameter_tuning=tuned)
        except Exception as exc:  # noqa: BLE001 - a model that cannot run offline is recorded as excluded
            traceback.print_exc()
            excluded[key] = f"{type(exc).__name__}: {exc}"[:500]
            continue
        print(f"[golden] {key}: {models[key]['seconds']}s", file=sys.stderr, flush=True)
    baseline = {
        "drevalpy_commit": _git_commit(),
        "dataset": {"seed": DATASET_SEED, **{key: list(value) for key, value in DATASET_OPTIONS.items()}},
        "split": {"mode": SPLIT_MODE, "n_splits": N_SPLITS, "random_state": SPLIT_SEED, "fold_index": FOLD_INDEX},
        "seed": SEED,
        "hpo": {"model": HPO_MODEL, "n_trials": HPO_TRIALS, "random_state": HPO_SEED, "metric": "RMSE"},
        "atol": ATOL,
        "models": models,
        "network_only": excluded,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with PREDICTIONS_FILE.open("wb") as handle:
        np.savez(handle, **predictions)
    with BASELINE_FILE.open("w", encoding="utf-8") as handle:
        json.dump(baseline, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return baseline


if __name__ == "__main__":
    result = record(sys.argv[1:] or None)
    print(json.dumps({"recorded": sorted(result["models"]), "network_only": result["network_only"]}, indent=2))
