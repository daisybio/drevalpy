"""Resolution of the two accepted call shapes of ``DRPModel.train``.

``train`` accepts either ``(mudataset, scope)`` - the Dataset path every caller
inside the library uses - or ``(output, cell_line_input, drug_input)``, the
ResponseBatch path kept for hand-rolled models. Both may be passed positionally
or by keyword, and the second positional slot additionally accepts a
``SplitMasks`` in place of a ``SplitMask``.

Untangling that lives here rather than on ``DRPModel`` because it reads no
instance state: it is a pure function from a call site's arguments to a
:class:`TrainCallArgs`, which is what lets ``train`` itself be a two-branch
dispatch instead of a run of ``isinstance`` checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from drevalpy.types import SplitMask, SplitMasks
from drevalpy.types.data.dataset import Dataset


@dataclass(frozen=True, slots=True)
class TrainCallArgs:
    """One resolved ``train`` call, with the two input forms told apart.

    Exactly one of :attr:`is_dataset_form` / :attr:`is_feature_source_form` is
    true for a well-formed call; neither is for a call missing its inputs, which
    is what ``train`` reports as a ``TypeError``.
    """

    mudataset: Dataset | None = None
    scope: SplitMask | None = None
    early_stopping_scope: SplitMask | None = None
    output: Any = None
    cell_line_input: Any = None
    drug_input: Any = None

    @property
    def is_dataset_form(self) -> bool:
        """Whether the call carries a ``Dataset`` and the scope to train on.

        :returns: ``True`` when the Dataset path applies.
        """
        return self.mudataset is not None and self.scope is not None

    @property
    def is_feature_source_form(self) -> bool:
        """Whether the call carries a response batch and a cell-line source.

        :returns: ``True`` when the ResponseBatch path applies.
        """
        return self.output is not None and self.cell_line_input is not None


def _first_positional(value: Any, output: Any) -> tuple[Dataset | None, Any]:
    """Assign the first positional argument to whichever form it belongs to.

    :param value: The first positional argument.
    :param output: The ``output`` keyword, returned unchanged for a ``Dataset``.
    :returns: Pair of ``(mudataset, output)``.
    """
    if isinstance(value, Dataset):
        return value, output
    return None, value


def _second_positional(
    value: Any,
    split: SplitMasks | None,
    cell_line_input: Any,
) -> tuple[SplitMask | None, SplitMasks | None, Any]:
    """Assign the second positional argument to whichever form it belongs to.

    :param value: The second positional argument.
    :param split: The ``split`` keyword, returned unchanged unless *value* is one.
    :param cell_line_input: The ``cell_line_input`` keyword, likewise.
    :returns: Triple of ``(scope, split, cell_line_input)``.
    """
    if isinstance(value, SplitMask):
        return value, split, cell_line_input
    if isinstance(value, SplitMasks):
        return None, value, cell_line_input
    return None, split, value


def resolve_train_args(
    first_positional: Any = None,
    second_positional: Any = None,
    drug_input: Any = None,
    *,
    mudataset: Dataset | None = None,
    split: SplitMasks | None = None,
    scope: SplitMask | None = None,
    early_stopping_scope: SplitMask | None = None,
    output: Any = None,
    cell_line_input: Any = None,
) -> TrainCallArgs:
    """Resolve one ``train`` call into named, form-tagged arguments.

    A keyword always wins over the positional slot it duplicates. A ``SplitMasks``
    is narrowed to its ``train`` mask, and its ``val`` mask becomes the
    early-stopping scope when it selects anything.

    :param first_positional: ``mudataset`` or ``output``, by type.
    :param second_positional: ``scope``, ``split`` or ``cell_line_input``, by type.
    :param drug_input: Drug feature source for the ResponseBatch form.
    :param mudataset: Dataset carrying responses and features.
    :param split: Full set of split masks; narrowed to ``scope`` here.
    :param scope: Mask selecting the training pairs.
    :param early_stopping_scope: Mask selecting the early-stopping pairs.
    :param output: Response batch for the ResponseBatch form.
    :param cell_line_input: Cell-line feature source for the ResponseBatch form.
    :returns: The resolved call arguments.
    """
    if mudataset is None and first_positional is not None:
        mudataset, output = _first_positional(first_positional, output)
    if scope is None and second_positional is not None:
        scope, split, cell_line_input = _second_positional(second_positional, split, cell_line_input)
    if scope is None and split is not None:
        scope = split.train
        if split.val.any():
            early_stopping_scope = split.val
    return TrainCallArgs(
        mudataset=mudataset,
        scope=scope,
        early_stopping_scope=early_stopping_scope,
        output=output,
        cell_line_input=cell_line_input,
        drug_input=drug_input,
    )
