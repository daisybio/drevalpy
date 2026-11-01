"""Final model: train one model on every pair of a dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from drevalpy._utils.n_jobs import check_n_jobs
from drevalpy.data import CurveQuality, load
from drevalpy.data.splitters import observed_mask
from drevalpy.errors import InvalidInputError
from drevalpy.experiment._execution import build_execution
from drevalpy.experiment._fit import fit
from drevalpy.log import get_logger
from drevalpy.models.drp_model import DRPModel
from drevalpy.models.spec import ModelSpec
from drevalpy.types import Fold, PairMask
from drevalpy.types.data.dataset import Dataset

if TYPE_CHECKING:
    from drevalpy.models.response_transformation import ResponseTransformer

logger = get_logger(__name__)

#: Fraction of the pairs held out for tuning and early stopping, as the ``validation_ratio`` of a split.
VALIDATION_RATIO = 0.1


def _holdout_folds(dataset: Dataset, quality: CurveQuality | None, seed: int) -> Fold:
    """Split all usable pairs into a training set and the validation slice a run would set aside.

    :param dataset: Dataset whose usable pairs are the final training set.
    :param quality: Curve-quality rule a pair must pass, or ``None`` for the default rule.
    :param seed: Root seed of the holdout draw.
    :returns: One fold with an empty test set.
    :raises InvalidInputError: If fewer than two pairs are usable.
    """
    observed = observed_mask(dataset, CurveQuality.default() if quality is None else quality)
    pairs = np.argwhere(observed)
    if len(pairs) < 2:
        raise InvalidInputError(f"Dataset '{dataset.name}' has {len(pairs)} usable pair(s); training needs at least 2.")
    n_val = max(1, int(len(pairs) * VALIDATION_RATIO))
    order = np.random.default_rng(seed).permutation(len(pairs))
    val = PairMask.from_pairs(pairs[order[:n_val]], observed.shape)
    train = PairMask.from_pairs(pairs[order[n_val:]], observed.shape)
    nothing = PairMask(np.zeros(observed.shape, dtype=bool))
    return Fold(train, nothing, val, {"split_mode": "LPO", "fold_index": 0})


def train(
    model: ModelSpec | str,
    dataset: Dataset | str,
    *,
    hyperparameter_tuning: bool = True,
    hpo_metric: str = "RMSE",
    hpo_num_trials: int = 16,
    hpo_random_state: int | None = None,
    response_transformation: ResponseTransformer | None = None,
    quality: CurveQuality | None = None,
    seed: int = 42,
    n_jobs: int = 1,
) -> DRPModel:
    """Train a model on every usable pair of a dataset, for later use on new data.

    Unlike :func:`drevalpy.run` there is no test set. A random 10% of the pairs is set aside as a
    validation slice, as in a run. Tuning, when enabled, scores on it. A model that stops early
    keeps a quarter of it as its early-stopping set, and every other pair is a training pair. A
    single-drug model fits one estimator per drug through its own routing. Save the result with
    :meth:`~drevalpy.models.DRPModel.save` and apply it with :meth:`~drevalpy.models.DRPModel.load`
    and :meth:`~drevalpy.models.DRPModel.predict`. The fitted response transformation is part of
    the checkpoint.

    :param model: Model recipe, or the zoo name or recipe string of one.
    :param dataset: Dataset, or the name or ``.h5mu`` path of one.
    :param hyperparameter_tuning: Whether to tune the hyperparameters; without it the model uses
        its default hyperparameters.
    :param hpo_metric: Metric to optimize when tuning.
    :param hpo_num_trials: Number of tuning trials.
    :param hpo_random_state: Seed of the tuning sampler; defaults to ``seed``.
    :param response_transformation: Optional unfitted response transformation, cloned for every fit.
    :param quality: Curve-quality rule a pair must pass to be used; ``None`` means
        ``CurveQuality.default()``, as for :func:`drevalpy.run`.
    :param seed: Root seed of the validation slice, the tuning and the final fit.
    :param n_jobs: Number of worker processes a single-drug model fits its per-drug models in, during tuning and in the
        final fit; ``-1`` uses all CPUs. The model does not depend on it, and other models ignore it.
    :returns: The fitted model.
    """
    check_n_jobs(n_jobs)
    spec = ModelSpec.from_name(model) if isinstance(model, str) else model
    data = load(dataset) if isinstance(dataset, str) else dataset
    execution = build_execution(
        spec,
        data,
        _holdout_folds(data, quality, seed),
        hyperparameter_tuning=hyperparameter_tuning,
        hpo_metric=hpo_metric,
        hpo_num_trials=hpo_num_trials,
        hpo_random_state=seed if hpo_random_state is None else hpo_random_state,
        seed=seed,
    )
    logger.info("Training %s on %d pairs", spec.name, len(execution.scopes.final_training))
    return fit(execution, data, response_transformation=response_transformation, n_jobs=n_jobs).model
