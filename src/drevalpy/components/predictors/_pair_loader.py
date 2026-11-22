"""Torch loaders that look up entity features per pair, and the eval pass that drains one.

:func:`pair_loader` is the loader over the pairs of a :class:`PredictorInput`: training pairs come
from ``data``, early-stopping pairs from ``data.stopping``, so no caller picks indices and response
apart by hand. :func:`predict_pairs` scores pairs with it; only the forward call differs between
predictors, which is why it is passed in.

``torch`` is imported inside the entry points. Predictor modules import this one, and
``drevalpy.registry`` imports those modules when it registers builtins, so a module-scope
``import torch`` would put ~0.35s on the startup path of every CLI invocation. See
``tests/test_import_cost_policy.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from drevalpy.types.data.predictor_input import PredictorInput


def require_drug_pair_idx(drug_pair_idx: np.ndarray | None) -> np.ndarray:
    """Narrow a batch's ``drug_pair_idx`` to non-``None``.

    :param drug_pair_idx: The batch's drug pair indices, possibly ``None``.
    :returns: The same array.
    :raises RuntimeError: If the batch carries no drug pair indices.
    """
    if drug_pair_idx is None:
        raise RuntimeError("drug_pair_idx is required for this predictor")
    return drug_pair_idx


def pair_loader(
    data: PredictorInput,
    *sample: np.ndarray,
    drug: np.ndarray | None = None,
    batch_size: int,
    shuffle: bool,
    drop_last: bool = False,
    response: bool = True,
) -> DataLoader:
    r"""Create the loader over the pairs of *data*, gathering each batch's entity rows directly.

    The loader yields ``(*sample, drug, response)`` tensors, leaving out *drug* when ``None`` and,
    without *response*, the response. Pass ``data.stopping`` for the early-stopping pairs.

    :param data: Pairs to iterate, supplying the pair indices and the response.
    :param \*sample: Entity-level sample matrices, each indexed by ``data.sample_pair_idx``.
    :param drug: Entity-level drug matrix, indexed by ``data.drug_pair_idx``; ``None`` to leave it out.
    :param batch_size: Mini-batch size.
    :param shuffle: Whether to shuffle the pairs each epoch.
    :param drop_last: Whether to drop the last incomplete batch.
    :param response: Whether to yield the float32 response of each pair as the last tensor.
    :returns: A DataLoader over the pairs of *data*.
    """
    import torch
    from torch.utils.data import DataLoader

    sample_idx = torch.as_tensor(data.sample_pair_idx, dtype=torch.long)
    specs = [(torch.as_tensor(matrix, dtype=torch.float32), sample_idx) for matrix in sample]
    if drug is not None:
        drug_idx = torch.as_tensor(require_drug_pair_idx(data.drug_pair_idx), dtype=torch.long)
        specs.append((torch.as_tensor(drug, dtype=torch.float32), drug_idx))
    target = torch.as_tensor(data.response, dtype=torch.float32) if response else None

    def gather(batch: list[int]) -> tuple[torch.Tensor, ...]:
        rows = torch.as_tensor(batch, dtype=torch.long)
        feats = tuple(matrix[idx[rows]] for matrix, idx in specs)
        return feats if target is None else (*feats, target[rows])

    return DataLoader(
        cast("Any", range(len(data))),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=gather,
    )


def collect_predictions(model: Any, batches: Iterable[Any], forward: Callable[[Any], torch.Tensor]) -> np.ndarray:
    """Run *forward* over every batch with *model* in eval mode and gradients off.

    :param model: Torch module; switched to eval mode before the pass.
    :param batches: Loader batches, in pair order.
    :param forward: Maps one batch to the model's output for it.
    :returns: The outputs flattened and concatenated as float64; empty when there are no batches.
    """
    import numpy as np
    import torch

    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in batches:
            chunks.append(forward(batch).detach().cpu().numpy().reshape(-1))
    if not chunks:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(chunks).astype(np.float64)


def predict_pairs(
    model: Any,
    data: PredictorInput,
    sample: np.ndarray,
    drug: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    forward: Callable[..., torch.Tensor] | None = None,
) -> np.ndarray:
    """Score every pair in *data* with *model* in eval mode, in pair order.

    :param model: Fitted torch module; switched to eval mode before the pass.
    :param data: Featurized pairs to score.
    :param sample: Entity-level sample matrix, indexed by ``sample_pair_idx``.
    :param drug: Entity-level drug matrix, indexed by ``drug_pair_idx``.
    :param batch_size: Mini-batch size.
    :param device: Device the batches are moved to.
    :param forward: Called with the sample and drug tensors; defaults to ``model(sample, drug)``.
    :returns: One predicted response per pair, flattened.
    """
    loader = pair_loader(data, sample, drug=drug, batch_size=batch_size, shuffle=False, response=False)
    run_forward = model if forward is None else forward
    batches = (tuple(tensor.to(device) for tensor in tensors) for tensors in loader)
    return collect_predictions(model, batches, lambda tensors: run_forward(*tensors))


def concatenated_forward(model: Any) -> Callable[..., torch.Tensor]:
    """Wrap *model* so it receives the per-block tensors concatenated feature-wise.

    :param model: Torch module taking a single ``[batch, sum(dims)]`` tensor.
    :returns: A forward callable accepting one tensor per block.
    """
    import torch

    def forward(*tensors: torch.Tensor) -> torch.Tensor:
        return model(torch.cat(tensors, dim=1))

    return forward


def declared_blocks(predictor: Any, data: PredictorInput) -> tuple[np.ndarray, np.ndarray]:
    """Return the entity-level values of the one block each side of *predictor* declares.

    :param predictor: Predictor whose ``sample_inputs`` and ``drug_inputs`` each name a single block.
    :param data: Batch to read the blocks from.
    :returns: Tuple of ``(sample_block, drug_block)`` values.
    """
    (sample_spec,) = predictor.sample_inputs.specs
    (drug_spec,) = predictor.drug_inputs.specs
    return data.sample.blocks[sample_spec.name].values, data.drug.blocks[drug_spec.name].values
