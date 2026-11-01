"""Three-omic pair loaders and the Lightning fit shared by MOLIR and SuperFELTR.

``molir/utils.py`` and ``superfeltr/utils.py`` built their train and validation
loaders with byte-identical code: reshape the response to a column, index the
``gene_expression``/``mutations``/``copy_number`` matrices by the same pair index
array, drop the last incomplete training batch, keep it for validation.

Both used to take the three validation matrices as separately optional arguments and
raise when only some were given. :class:`OmicsSplit` makes that partial state
unrepresentable, so the runtime guard is gone rather than shared.

:func:`fit_omics_model` is the fit both run: build the loaders, then an early-stopped
CPU fit with W&B as the only logger. Each caller reloads the best checkpoint its own way.

Nothing here imports ``torch``: ``make_pair_loader`` and ``run_lightning_fit`` defer it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from drevalpy.components.predictors._lightning_training import LightningRun, run_lightning_fit

if TYPE_CHECKING:
    import numpy as np
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader
    from upath import UPath


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
            accelerator="cpu",  # the models are too small to profit from a GPU
            default_logger=False,
            **run_options,
        ),
    )
    return cast("pl.callbacks.ModelCheckpoint", checkpoint)
