"""Torch loaders that look up entity features per pair, and the eval pass that drains one.

:func:`pair_loader` is the loader over the pairs of a :class:`PredictorInput`: training pairs come
from ``data``, early-stopping pairs from ``data.stopping``, so no caller picks indices and response
apart by hand. :func:`predict_pairs` scores pairs with it; only the forward call differs between
predictors, which is why it is passed in. :func:`make_pair_loader` is the layer underneath, for
splits that are not a ``PredictorInput`` (the three-omic ``OmicsSplit``).

``torch`` is imported inside the entry points. Predictor modules import this one, and
``drevalpy.registry`` imports those modules when it registers builtins, so a module-scope
``import torch`` would put ~0.35s on the startup path of every CLI invocation. See
``tests/test_import_cost_policy.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from drevalpy.types.data.predictor_input import PredictorInput


class IndexedPairDataset:
    """Looks up entity-level features by pair index on each access.

    Instead of materializing a full pair-level feature matrix upfront, this
    dataset stores compact entity-level matrices and performs the lookup
    per-pair in ``__getitem__``.

    Deliberately not a ``torch.utils.data.Dataset`` subclass: that base class
    contributes only ``__add__``, which nothing here uses, and inheriting from it
    would require ``torch`` at class-definition time. ``DataLoader`` accepts any
    object with ``__getitem__`` and ``__len__`` as a map-style dataset.
    """

    def __init__(
        self,
        *feature_specs: tuple[np.ndarray, np.ndarray],
        response: np.ndarray | None = None,
    ) -> None:
        r"""Initialize with entity matrices and corresponding pair indices.

        :param \*feature_specs: Each positional arg is a tuple of
            ``(entity_matrix, pair_index_array)`` where entity_matrix has shape
            ``[n_entities, d]`` and pair_index_array has shape ``[n_pairs]``.
        :param response: Optional pair-level response array of shape ``[n_pairs]``.
        """
        import torch

        self._matrices = [torch.as_tensor(m, dtype=torch.float32) for m, _ in feature_specs]
        self._indices = [torch.as_tensor(i, dtype=torch.long) for _, i in feature_specs]
        self._response = torch.as_tensor(response, dtype=torch.float32) if response is not None else None
        self._n_pairs = int(self._indices[0].shape[0]) if self._indices else 0

    def __len__(self) -> int:
        """Return number of pairs.

        :returns: Dataset length.
        """
        return self._n_pairs

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        """Look up features for pair *idx*.

        :param idx: Pair index.
        :returns: Tuple of feature tensors, optionally followed by the response scalar.
        """
        feats = tuple(m[i[idx]] for m, i in zip(self._matrices, self._indices, strict=True))
        if self._response is not None:
            return (*feats, self._response[idx])
        return feats


def make_pair_loader(
    *feature_specs: tuple[np.ndarray, np.ndarray],
    response: np.ndarray | None = None,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    r"""Create a DataLoader that lazily indexes entity features per mini-batch.

    :param \*feature_specs: Each positional arg is ``(entity_matrix, pair_indices)``.
    :param response: Optional pair-level response vector.
    :param batch_size: Mini-batch size.
    :param shuffle: Whether to shuffle each epoch.
    :param drop_last: Whether to drop the last incomplete batch.
    :returns: A DataLoader yielding tuples of tensors.
    """
    from torch.utils.data import DataLoader

    ds = IndexedPairDataset(*feature_specs, response=response)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


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
    sample: np.ndarray | None,
    drug: np.ndarray | None,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool = False,
    response: bool = True,
) -> DataLoader:
    """Create the loader over the pairs of *data*, looking up each pair's entity rows in *sample* and *drug*.

    The loader yields ``(sample, drug, response)`` tensors, leaving out the matrices given as ``None``
    and, without *response*, the response. Pass ``data.stopping`` for the early-stopping pairs.

    :param data: Pairs to iterate, supplying the pair indices and the response.
    :param sample: Entity-level sample matrix, indexed by ``data.sample_pair_idx``; ``None`` to leave it out.
    :param drug: Entity-level drug matrix, indexed by ``data.drug_pair_idx``; ``None`` to leave it out.
    :param batch_size: Mini-batch size.
    :param shuffle: Whether to shuffle the pairs each epoch.
    :param drop_last: Whether to drop the last incomplete batch.
    :param response: Whether to yield the float32 response of each pair as the last tensor.
    :returns: A DataLoader over the pairs of *data*.
    """
    import numpy as np

    specs: list[tuple[np.ndarray, np.ndarray]] = []
    if sample is not None:
        specs.append((sample, data.sample_pair_idx))
    if drug is not None:
        specs.append((drug, require_drug_pair_idx(data.drug_pair_idx)))
    return make_pair_loader(
        *specs,
        response=np.asarray(data.response, dtype=np.float32) if response else None,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
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
    loader = pair_loader(data, sample, drug, batch_size=batch_size, shuffle=False, response=False)
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
