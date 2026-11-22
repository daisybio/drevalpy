"""Three-omic data handling and the per-drug predictor shell shared by MOLIR and SuperFELTR.

Both are single-drug models over the same three sample views (:data:`OMIC_BLOCK_NAMES`, in encoder
order), and both need:

* :func:`make_omics_loaders`, turning a :class:`PredictorInput` (and its ``stopping`` pairs, when at
  least two exist) into train and validation loaders over the three sample matrices, dropping the last
  incomplete training batch and keeping it for validation;
* :func:`fit_omics_model`, the fit both run: build the loaders, then an early-stopped fit at the predictor's
  accelerator level, which is ``NONE`` (the CPU) for both, with W&B as the only logger. Each caller reloads the
  best checkpoint its own way;
* the feature names a model was trained on (:func:`omic_feature_names`), whose columns
  :func:`aligned_pair_matrices` realigns at predict time;
* :class:`SingleDrugOmicsPredictor`, the block contract, early-stopping support and recorded feature
  names on top of :class:`~drevalpy.components.predictors.abstract.single_drug.SingleDrugPredictor`.

Nothing here imports ``torch``: ``pair_loader`` and ``run_lightning_fit`` defer it, and both
predictors are registered on ``import drevalpy``, so the training stack must stay out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np

from drevalpy.components.predictors._lightning_training import LightningRun, run_lightning_fit
from drevalpy.components.predictors.abstract.single_drug import SingleDrugPredictor
from drevalpy.components.predictors.inputs import Inputs
from drevalpy.types.data.feature_format import BlockSpec

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pytorch_lightning as pl
    from torch.utils.data import DataLoader

    from drevalpy.types.data.predictor_input import PredictorInput
    from drevalpy.types.enums import Accelerator

#: The three sample views both models consume, in encoder order.
OMIC_BLOCK_NAMES = ("gene_expression", "mutations", "copy_number_variation_gistic")


def omic_feature_names(data: PredictorInput) -> dict[str, tuple[str, ...] | None]:
    """Record the feature names of the three sample views of *data*.

    :param data: Batch carrying all three blocks.
    :returns: Feature names (``None`` when a block has none) keyed by block name.
    """
    return {name: data.sample.blocks[name].feature_names for name in OMIC_BLOCK_NAMES}


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


def make_omics_loaders(data: PredictorInput, batch_size: int) -> tuple[DataLoader, DataLoader | None]:
    """Build the train and (optional) validation loaders for a three-omic model.

    Validation is omitted when *data* carries fewer than two early-stopping pairs.

    :param data: Training batch carrying all three omic sample blocks.
    :param batch_size: Mini-batch size for both loaders.
    :returns: The training loader and the validation loader, the latter ``None``
        when *data* has fewer than two early-stopping pairs.
    """
    from drevalpy.components.predictors._pair_loader import pair_loader

    matrices = tuple(data.sample.blocks[name].values for name in OMIC_BLOCK_NAMES)
    train_loader = pair_loader(data, *matrices, batch_size=batch_size, shuffle=False, drop_last=True)
    stopping = data.stopping
    val_loader = (
        None
        if stopping is None or len(stopping) < 2
        else pair_loader(stopping, *matrices, batch_size=batch_size, shuffle=False, drop_last=False)
    )
    return train_loader, val_loader


def fit_omics_model(
    model: pl.LightningModule,
    data: PredictorInput,
    *,
    batch_size: int,
    epochs: int,
    patience: int,
    accelerator: Accelerator,
    **run_options: Any,
) -> pl.callbacks.ModelCheckpoint:
    r"""Fit *model* at the predictor's accelerator level, early-stopping on the validation loss.

    Without early-stopping pairs it early-stops on the training loss.

    :param model: MOLIR model, SuperFELTR encoder or SuperFELTR regressor.
    :param data: Training batch carrying all three omic sample blocks and optional early-stopping pairs.
    :param batch_size: Mini-batch size of both loaders.
    :param epochs: Upper bound on training epochs.
    :param patience: Epochs without improvement before early stopping.
    :param accelerator: The registered level of the predictor; ``NONE``, as MOLIR and SuperFELTR are registered,
        keeps the fit on the CPU.
    :param \**run_options: Further :class:`LightningRun` fields.
    :returns: The checkpoint callback holding the best epoch's path and score.
    """
    train_loader, val_loader = make_omics_loaders(data, batch_size)
    checkpoint = run_lightning_fit(
        model,
        train_loader,
        val_loader,
        LightningRun(
            max_epochs=epochs,
            patience=patience,
            checkpoint_dir=data.training_context.checkpoint_dir,
            accelerator=accelerator,
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
