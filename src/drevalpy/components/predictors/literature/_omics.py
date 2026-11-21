"""Three-omic data handling and the per-drug predictor shell shared by MOLIR and SuperFELTR.

Both are single-drug models over the same three sample views (:data:`OMIC_BLOCK_NAMES`, in encoder
order), and both need:

* :class:`OmicsSplit`, one split's entity-level matrices with the pair-level response and indices,
  and :func:`make_omics_loaders` to turn splits into loaders: reshape the response to a column,
  index the three matrices by the same pair index array, drop the last incomplete training batch,
  keep it for validation. The three validation matrices are one optional :class:`OmicsSplit`, so a
  partly given validation set is unrepresentable;
* :func:`fit_omics_model`, the fit both run: build the loaders, then an early-stopped CPU fit with W&B
  as the only logger. Each caller reloads the best checkpoint its own way;
* the matrices of a batch (:func:`omic_matrices`), the early-stopping split over them
  (:func:`validation_split`) and the feature names a model was trained on (:func:`omic_feature_names`),
  whose columns :func:`aligned_pair_matrices` realigns at predict time;
* :class:`SingleDrugOmicsPredictor`, the block contract, early-stopping support and recorded feature
  names on top of :class:`~drevalpy.components.predictors.abstract.single_drug.SingleDrugPredictor`.

Nothing here imports ``torch``: ``make_pair_loader`` and ``run_lightning_fit`` defer it, and both
predictors are registered on ``import drevalpy``, so the training stack must stay out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np

from drevalpy.components.contracts.feature_format import BlockSpec
from drevalpy.components.predictors._lightning_training import LightningRun, run_lightning_fit
from drevalpy.components.predictors.abstract.single_drug import SingleDrugPredictor
from drevalpy.components.predictors.inputs import Inputs

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pytorch_lightning as pl
    from torch.utils.data import DataLoader
    from upath import UPath

    from drevalpy.types.data.predictor_input import PredictorInput

#: The three sample views both models consume, in encoder order.
OMIC_BLOCK_NAMES = ("gene_expression", "mutations", "copy_number_variation_gistic")


@dataclass(frozen=True)
class OmicsSplit:
    """One split's entity-level omics, plus the pair-level response and indices.

    The three matrices are entity-level (``[n_entities, n_features]``); ``pair_idx``
    maps each pair onto a row of all three at once.
    """

    gene_expression: np.ndarray
    mutations: np.ndarray
    copy_number: np.ndarray
    response: np.ndarray
    pair_idx: np.ndarray


def omic_matrices(data: PredictorInput) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the three sample views out of *data*.

    :param data: Batch carrying all three blocks.
    :returns: The entity-level expression, mutation and copy number matrices as float32.
    """
    expression, mutations, copy_number = (
        np.asarray(data.sample.blocks[name].values, dtype=np.float32) for name in OMIC_BLOCK_NAMES
    )
    return expression, mutations, copy_number


def omic_feature_names(data: PredictorInput) -> dict[str, tuple[str, ...] | None]:
    """Record the feature names of the three sample views of *data*.

    :param data: Batch carrying all three blocks.
    :returns: Feature names (``None`` when a block has none) keyed by block name.
    """
    return {name: data.sample.blocks[name].feature_names for name in OMIC_BLOCK_NAMES}


def validation_split(matrices: tuple[np.ndarray, np.ndarray, np.ndarray], data: PredictorInput) -> OmicsSplit | None:
    """Build the early-stopping split over the same entity matrices as training.

    :param matrices: The batch's entity-level omic matrices, as returned by :func:`omic_matrices`.
    :param data: Training batch carrying the early-stopping response.
    :returns: The validation split, or ``None`` when there are fewer than two early-stopping pairs.
    """
    stopping = data.stopping
    if stopping is None or len(stopping) < 2:
        return None
    return OmicsSplit(
        *matrices,
        response=np.asarray(stopping.response, dtype=np.float32),
        pair_idx=stopping.sample_pair_idx,
    )


def aligned_pair_matrices(
    data: PredictorInput,
    feature_names: dict[str, tuple[str, ...] | None],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand the three views to pair level and realign them onto the trained order.

    :param data: Batch to score.
    :param feature_names: Feature names the model was trained on, keyed by block name.
    :returns: Tuple of ``(expression, mutation, cnv)`` pair-level matrices.
    """
    expression, mutations, copy_number = (
        _realign_omic_matrix(
            np.asarray(data.sample.blocks[name].values[data.sample_pair_idx], dtype=np.float32),
            feature_names.get(name),
            data.sample.blocks[name].feature_names,
        )
        for name in OMIC_BLOCK_NAMES
    )
    return expression, mutations, copy_number


def _realign_omic_matrix(
    values: np.ndarray,
    model_features: Sequence[str] | None,
    current_features: Sequence[str] | None,
) -> np.ndarray:
    """Align the columns of a prediction-time omic matrix to the feature order the model was trained on.

    A feature the model was trained on but the data lacks is filled with zeros, unlike the featurizers'
    ``view_matrix``, which raises: a per-drug model scored on a dataset without all its features still predicts.

    :param values: Omic feature matrix in incoming column order.
    :param model_features: Feature names the model was trained on; ``None`` when unknown.
    :param current_features: Feature names of the incoming data; ``None`` when unknown.
    :returns: *values* itself when the names are unknown or equal, else a matrix with the columns of
        *model_features*, zero where the incoming data lacks the feature.
    """
    if model_features is None or current_features is None or list(model_features) == list(current_features):
        return values
    realigned = np.zeros((values.shape[0], len(model_features)))
    position = {feature: i for i, feature in enumerate(current_features)}
    for column, feature in enumerate(model_features):
        if feature in position:
            realigned[:, column] = values[:, position[feature]]
    return realigned


def make_omics_loaders(
    train: OmicsSplit,
    val: OmicsSplit | None,
    batch_size: int,
) -> tuple[DataLoader, DataLoader | None]:
    """Build the train and (optional) validation loaders for a three-omic model.

    :param train: Training split.
    :param val: Validation split, or ``None`` to train without one.
    :param batch_size: Mini-batch size for both loaders.
    :returns: The training loader and the validation loader, the latter ``None``
        when *val* is ``None``.
    """
    train_loader = _loader(train, batch_size, drop_last=True)
    val_loader = None if val is None else _loader(val, batch_size, drop_last=False)
    return train_loader, val_loader


def _loader(split: OmicsSplit, batch_size: int, *, drop_last: bool) -> DataLoader:
    """Build one loader over the three omic views of *split*.

    :param split: The split to iterate.
    :param batch_size: Mini-batch size.
    :param drop_last: Whether to drop a trailing incomplete batch.
    :returns: A loader yielding ``(expression, mutations, copy_number, response)``.
    """
    from drevalpy.components.predictors._pair_loader import make_pair_loader

    response = split.response
    response_column = response.reshape(-1, 1) if response.ndim == 1 else response
    return make_pair_loader(
        (split.gene_expression, split.pair_idx),
        (split.mutations, split.pair_idx),
        (split.copy_number, split.pair_idx),
        response=response_column,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
    )


def fit_omics_model(
    model: pl.LightningModule,
    train: OmicsSplit,
    val: OmicsSplit | None,
    *,
    batch_size: int,
    epochs: int,
    patience: int,
    checkpoint_dir: str | UPath,
    **run_options: Any,
) -> pl.callbacks.ModelCheckpoint:
    r"""Fit *model* on the CPU, early-stopping on the validation loss (the training loss without *val*).

    :param model: MOLIR model, SuperFELTR encoder or SuperFELTR regressor.
    :param train: Training split.
    :param val: Validation split, or ``None``.
    :param batch_size: Mini-batch size of both loaders.
    :param epochs: Upper bound on training epochs.
    :param patience: Epochs without improvement before early stopping.
    :param checkpoint_dir: Directory the best epoch is checkpointed into.
    :param \**run_options: Further :class:`LightningRun` fields.
    :returns: The checkpoint callback holding the best epoch's path and score.
    """
    train_loader, val_loader = make_omics_loaders(train, val, batch_size)
    checkpoint = run_lightning_fit(
        model,
        train_loader,
        val_loader,
        LightningRun(
            max_epochs=epochs,
            patience=patience,
            checkpoint_dir=checkpoint_dir,
            devices=1,
            default_logger=False,
            **run_options,
        ),
    )
    return cast("pl.callbacks.ModelCheckpoint", checkpoint)


class SingleDrugOmicsPredictor(SingleDrugPredictor):
    """Per-drug multi-omics predictor shell shared by MOLIR and SuperFELTR.

    A subclass trains one model per drug (``_fit_drug``) whose ``predict`` takes the three
    pair-level omic matrices; the shell realigns their columns onto the feature names seen in
    training before scoring.
    """

    sample_inputs: ClassVar[Inputs] = Inputs.blocks(*(BlockSpec(name) for name in OMIC_BLOCK_NAMES))
    supports_early_stopping: ClassVar[bool] = True

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Initialize predictor with optional hyperparameter overrides.

        :param hyperparameters: Optional hyperparameter overrides.
        """
        super().__init__(hyperparameters)
        self._feature_names: dict[str, tuple[str, ...] | None] = {}

    def _fit(self, data: PredictorInput) -> None:
        """Record the omic feature names of *data*, then train one model per drug.

        :param data: Training batch with all required sample omics blocks.
        """
        self._feature_names = omic_feature_names(data)
        super()._fit(data)

    def _predict_drug(self, drug_id: str, model: Any, data: PredictorInput) -> np.ndarray:
        """Score the pairs of one drug on omic columns realigned to the training order.

        :param drug_id: Drug whose pairs *data* holds.
        :param model: The drug's fitted model.
        :param data: The drug's pairs to score.
        :returns: Predicted responses, one per pair.
        """
        return np.atleast_1d(model.predict(*aligned_pair_matrices(data, self._feature_names)))
