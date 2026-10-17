"""Eval-time pair inference shared by the predictors that run their own torch loop.

``pharmaformer``, ``precily`` and ``sparsego`` each resolved the pair indices, built
an eval ``make_pair_loader`` and accumulated predictions under ``torch.no_grad()``
themselves. Only the forward call genuinely differed, which is why it is passed in
rather than unified here.

``torch`` is imported inside the entry points. Every caller lives in a
``predictor.py`` that ``drevalpy.registry`` imports on ``import drevalpy``, so a
module-scope import would put the training stack back on the CLI startup path. See
``tests/test_import_cost_policy.py``.

The leading underscore keeps the module out of
``registry/_builtins.py::_discover_modules``, which imports every public ``*.py`` in
a component directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    import numpy as np
    import torch

    from drevalpy.types.data.batch.model_input_batch import ModelInputBatch


@dataclass(frozen=True)
class PairEvalSpec:
    """One eval pass: which entity matrices to index, how to batch, where to run.

    The two block groups are kept apart because they are indexed by different arrays:
    cell-line blocks by ``cell_line_pair_idx``, drug blocks by ``drug_pair_idx``.
    """

    cell_line_blocks: Sequence[np.ndarray]
    drug_blocks: Sequence[np.ndarray]
    batch_size: int
    device: torch.device


def predict_pairs(
    model: Any,
    batch: ModelInputBatch,
    spec: PairEvalSpec,
    forward: Callable[..., torch.Tensor] | None = None,
) -> np.ndarray:
    """Score every pair in *batch* with *model* in eval mode, in pair order.

    :param model: Fitted torch module; switched to eval mode before the pass.
    :param batch: Featurized pairs to score.
    :param spec: Blocks, batch size and target device for this pass.
    :param forward: Called with one tensor per block; defaults to ``model(*tensors)``.
    :returns: One predicted response per pair, flattened.
    """
    import numpy as np
    import torch

    run_forward = model if forward is None else forward
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for tensors in _iter_pair_batches(batch, spec):
            outputs = run_forward(*tensors)
            chunks.append(outputs.detach().cpu().numpy().reshape(-1))
    if not chunks:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(chunks).astype(np.float64)


def concatenated_forward(model: Any) -> Callable[..., torch.Tensor]:
    """Wrap *model* so it receives the per-block tensors concatenated feature-wise.

    :param model: Torch module taking a single ``[batch, sum(dims)]`` tensor.
    :returns: A forward callable accepting one tensor per block.
    """
    import torch

    def forward(*tensors: torch.Tensor) -> torch.Tensor:
        return model(torch.cat(tensors, dim=1))

    return forward


def require_drug_pair_idx(drug_pair_idx: np.ndarray | None) -> np.ndarray:
    """Narrow a batch's ``drug_pair_idx`` to non-``None``.

    :param drug_pair_idx: The batch's drug pair indices, possibly ``None``.
    :returns: The same array.
    :raises RuntimeError: If the batch carries no drug pair indices.
    """
    if drug_pair_idx is None:
        raise RuntimeError("drug_pair_idx is required for this predictor")
    return drug_pair_idx


def _iter_pair_batches(batch: ModelInputBatch, spec: PairEvalSpec) -> Iterator[tuple[Any, ...]]:
    """Yield mini-batches of entity features for every pair in *batch*, in order.

    :param batch: Featurized pairs to score.
    :param spec: Blocks, batch size and target device for this pass.
    :yields: One tuple of device-resident tensors per mini-batch.
    """
    from drevalpy.types.data.tensor_data import make_pair_loader

    drug_pair_idx = require_drug_pair_idx(batch.drug_pair_idx)
    cell_line_pair_idx = batch.cell_line_pair_idx

    loader = make_pair_loader(
        *((values, cell_line_pair_idx) for values in spec.cell_line_blocks),
        *((values, drug_pair_idx) for values in spec.drug_blocks),
        batch_size=spec.batch_size,
        shuffle=False,
    )
    for tensors in loader:
        yield tuple(tensor.to(spec.device) for tensor in tensors)
