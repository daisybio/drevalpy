"""Final model: train one model on every pair of a dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from drevalpy.data import CurveQuality, load
from drevalpy.data.splitters import observed_mask
from drevalpy.errors import InvalidInputError
from drevalpy.experiment._execution import ExecutionSpec, build_execution
from drevalpy.log import get_logger
from drevalpy.models.drp_model import DRPModel
from drevalpy.models.spec import ModelSpec
from drevalpy.models.tuning.config import build_experiment_hpo_config
from drevalpy.models.tuning.hpo import hpam_tune
from drevalpy.types import SplitMask, SplitMasks
from drevalpy.types.data.dataset import Dataset
from drevalpy.utils.seed import fit_seed, seed_everything

if TYPE_CHECKING:
    from sklearn.base import TransformerMixin

logger = get_logger(__name__)

#: Fraction of the pairs held out for tuning and early stopping, as the ``validation_ratio`` of a split.
VALIDATION_RATIO = 0.1


def _holdout_execution(spec: ModelSpec, dataset: Dataset, quality: CurveQuality | None, seed: int) -> ExecutionSpec:
    """Build the scopes of a final fit: all usable pairs, with the validation slice a run would set aside.

    :param spec: Model recipe, which decides whether an early-stopping set is reserved.
    :param dataset: Dataset whose usable pairs are the final training set.
    :param quality: Curve-quality rule a pair must pass, or ``None`` for the default rule.
    :param seed: Root seed of the holdout draw and of the fit.
    :returns: The execution decisions.
    :raises InvalidInputError: If fewer than two pairs are usable.
    """
    observed = observed_mask(dataset, CurveQuality.default() if quality is None else quality)
    pairs = np.argwhere(observed)
    if len(pairs) < 2:
        raise InvalidInputError(f"Dataset '{dataset.name}' has {len(pairs)} usable pair(s); training needs at least 2.")
    n_val = max(1, int(len(pairs) * VALIDATION_RATIO))
    order = np.random.default_rng(seed).permutation(len(pairs))
    val = SplitMask.from_pairs(pairs[order[:n_val]], observed.shape)
    train = SplitMask.from_pairs(pairs[order[n_val:]], observed.shape)
    nothing = SplitMask(np.zeros(observed.shape, dtype=bool))
    folds = SplitMasks(train, nothing, val, {"split_mode": "LPO", "fold_index": 0})
    return build_execution(
        spec,
        dataset,
        folds,
        hyperparameter_tuning=False,
        hpo_metric="RMSE",
        hpo_num_trials=0,
        hpo_random_state=seed,
        seed=seed,
    )


def train(
    model: ModelSpec | str,
    dataset: Dataset | str,
    *,
    hyperparameter_tuning: bool = True,
    hpo_metric: str = "RMSE",
    hpo_num_trials: int = 16,
    hpo_random_state: int | None = None,
    response_transformation: TransformerMixin | None = None,
    quality: CurveQuality | None = None,
    seed: int = 42,
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
    :param response_transformation: Optional unfitted target transformer, cloned for every fit.
    :param quality: Curve-quality rule a pair must pass to be used; ``None`` means
        ``CurveQuality.default()``, as for :func:`drevalpy.run`.
    :param seed: Root seed of the validation slice, the tuning and the final fit.
    :returns: The fitted model.
    """
    spec = ModelSpec.from_name(model) if isinstance(model, str) else model
    data = load(dataset) if isinstance(dataset, str) else dataset
    execution = _holdout_execution(spec, data, quality, seed)
    scopes, identity = execution.scopes, execution.identity
    fit_rng_seed = fit_seed(seed, identity)
    if hyperparameter_tuning:
        seed_everything(fit_rng_seed)
        hyperparameters, _ = hpam_tune(
            model_spec=spec,
            mudataset=data,
            train_scope=scopes.training,
            val_scope=scopes.validation,
            early_stopping_scope=scopes.stopping,
            response_transformation=response_transformation,
            metric=hpo_metric,
            hpo_config=build_experiment_hpo_config(
                hpo_metric,
                n_trials=hpo_num_trials,
                random_state=seed if hpo_random_state is None else hpo_random_state,
            ),
            split_index=0,
        )
    else:
        hyperparameters = spec.default_hyperparameters
    logger.info(
        "Training %s on %d pairs with hyperparameters %s", spec.name, len(scopes.final_training), hyperparameters
    )
    fitted = DRPModel(spec, hyperparameters, response_transformation=response_transformation)
    seed_everything(fit_rng_seed)
    fitted.train(data, scopes.final_training, early_stopping_scope=scopes.stopping)
    return fitted
