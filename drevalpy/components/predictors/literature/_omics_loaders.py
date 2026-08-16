"""Three-omic pair loaders shared by MOLIR and SuperFELTR.

``molir/utils.py`` and ``superfeltr/utils.py`` built their train and validation
loaders with byte-identical code: reshape the response to a column, index the
``gene_expression``/``mutations``/``copy_number`` matrices by the same pair index
array, drop the last incomplete training batch, keep it for validation.

Both used to take the three validation matrices as separately optional arguments and
raise when only some were given. :class:`OmicsSplit` makes that partial state
unrepresentable, so the runtime guard is gone rather than shared.

Nothing here imports ``torch``: ``make_pair_loader`` defers it. The leading
underscore keeps the module out of
``registry/_builtins.py::_discover_modules``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from torch.utils.data import DataLoader


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
    from drevalpy.types.data.tensor_data import make_pair_loader

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
