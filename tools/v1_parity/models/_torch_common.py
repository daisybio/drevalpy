"""Helpers shared by the torch model cases (every neural model except PaccMann and Precily).

Like the case modules, this one is imported by both interpreters, so only the standard library, numpy and
``tools.v1_parity`` are imported at module scope; ``torch`` and ``drevalpy`` come in inside the functions.
The leading underscore keeps it out of ``model_stems()``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

from tools.v1_parity._bridge import Case


def draw_weights(network: Any, rng: np.random.Generator, prefix: str, scale: float = 0.4) -> dict[str, np.ndarray]:
    """Draw one float32 array per floating-point ``state_dict`` entry of a (v2) network.

    Batch norm entries get values in a sane range (scale around 1, positive variance) so that the eval
    mode is a real transformation rather than a division by a tiny number.

    :param network: Torch module whose ``state_dict`` names and shapes define the keys.
    :param rng: Generator of the case.
    :param prefix: Array-name prefix, for example ``forward/small/w/``.
    :param scale: Standard deviation of ordinary weights.
    :returns: ``{prefix + key: array}``.
    """
    from torch.nn.modules.batchnorm import _BatchNorm

    norm_modules = {name for name, module in network.named_modules() if isinstance(module, _BatchNorm)}
    arrays: dict[str, np.ndarray] = {}
    for key, tensor in sorted(network.state_dict().items()):
        if not tensor.is_floating_point():
            continue
        shape = tuple(tensor.shape)
        owner, _, leaf = key.rpartition(".")
        if owner in norm_modules and leaf == "running_var":
            values = 0.5 + rng.random(shape)
        elif owner in norm_modules and leaf == "running_mean":
            values = 0.2 * rng.normal(size=shape)
        elif owner in norm_modules and leaf == "weight":
            values = 1.0 + 0.2 * rng.normal(size=shape)
        else:
            values = scale * rng.normal(size=shape)
        arrays[f"{prefix}{key}"] = values.astype(np.float32)
    return arrays


def load_weights(network: Any, case: Case, prefix: str) -> None:
    """Load the case's arrays under *prefix* into *network* with ``strict=True``.

    Non-float buffers (``num_batches_tracked``) keep their initial values.

    :param network: Torch module to fill.
    :param case: The shared case.
    :param prefix: Array-name prefix used by :func:`draw_weights`.
    """
    import torch

    state = network.state_dict()
    state.update({key[len(prefix) :]: torch.as_tensor(v) for key, v in case.arrays.items() if key.startswith(prefix)})
    network.load_state_dict(state, strict=True)


def forward_pair(network: Any, inputs: tuple[Any, ...], seed: int) -> dict[str, np.ndarray]:
    """Run a network in eval mode and in train mode (batch statistics, dropout under *seed*).

    :param network: Network holding the case's weights.
    :param inputs: Positional tensors passed to ``network``.
    :param seed: Torch seed set before the train-mode pass.
    :returns: ``eval`` and ``train`` outputs as float64 arrays.
    """
    import torch

    network.eval()
    with torch.no_grad():
        evaluated = network(*inputs).detach().cpu().numpy()
    network.train()
    torch.manual_seed(seed)
    with torch.no_grad():
        trained = network(*inputs).detach().cpu().numpy()
    network.eval()
    return {"eval": np.asarray(evaluated, dtype=np.float64), "train": np.asarray(trained, dtype=np.float64)}


@contextmanager
def inline_loaders(*modules: Any) -> Iterator[None]:
    """Make ``DataLoader`` in the given 1.6.1 modules single-process, without changing what it yields.

    1.6.1 hard-codes worker processes (up to 8, persistent) into its loaders, which the 4-CPU machine
    cannot afford. The workers do not touch the main process's torch RNG beyond the ``base_seed`` draw
    that a single-process iterator makes too, so the shuffle order, the dropout masks and the
    initialization are unchanged.

    :param modules: Modules that have a module-level ``DataLoader`` name.
    :yields: Nothing; the patch is undone on exit.
    """
    from torch.utils.data import DataLoader

    def single_process(*args: Any, **kwargs: Any) -> Any:
        kwargs["num_workers"] = 0
        kwargs.pop("persistent_workers", None)
        kwargs.pop("prefetch_factor", None)
        return DataLoader(*args, **kwargs)

    originals = [module.DataLoader for module in modules]
    for module in modules:
        module.DataLoader = single_process
    try:
        yield
    finally:
        for module, original in zip(modules, originals, strict=True):
            module.DataLoader = original


def restrict_v1_genes(
    cell_lines: Any, gene_names: np.ndarray, gene_list: np.ndarray, view: str = "gene_expression"
) -> None:
    """Select and reorder the expression of a 1.6.1 ``FeatureDataset`` to a gene list, in place.

    1.6.1 applies the gene list while it reads the CSVs, which the in-memory harness skips, so the
    selection is done here with the same result: the columns in list order.

    :param cell_lines: 1.6.1 ``FeatureDataset`` holding *view* for every sample.
    :param gene_names: Gene name of every column.
    :param gene_list: Genes to keep, in list order.
    :param view: View to restrict.
    """
    names = list(gene_names)
    columns = [names.index(gene) for gene in gene_list]
    for views in cell_lines.features.values():
        views[view] = views[view][columns]
    cell_lines.meta_info[view] = np.array(gene_list)


def shuffled_gene_names(
    rng: np.random.Generator, gene_list: np.ndarray, n_extra: int, prefix: str = "XTRA"
) -> np.ndarray:
    """Return the genes of a list plus extras in shuffled order, so a column selection is exercised.

    :param rng: Generator of the case.
    :param gene_list: Genes of the shipped list.
    :param n_extra: Number of extra genes that are not in the list.
    :param prefix: Name prefix of the extra genes.
    :returns: Shuffled unicode array.
    """
    names = np.concatenate([gene_list, np.array([f"{prefix}{i:03d}" for i in range(n_extra)])])
    return names[rng.permutation(len(names))]


def v2_model(name: str, hyperparameters: dict[str, Any], **component_overrides: dict[str, Any]) -> Any:
    """Build a v2 ``DRPModel`` from the zoo preset *name*, replacing options of its components.

    :param name: Zoo preset name.
    :param hyperparameters: Hyperparameters set explicitly, so no default leaks in.
    :param component_overrides: ``sample_featurizer`` / ``drug_featurizer`` / ``predictor`` mapping to a dict
        with an optional new ``name`` and ``options`` (merged into the preset's options).
    :returns: The untrained model.
    """
    from drevalpy.models.config import ModelConfig
    from drevalpy.models.drp_model import DRPModel
    from drevalpy.models.spec import ModelSpec
    from drevalpy.models.zoo import get_zoo_config

    base = get_zoo_config(name).model_dump(mode="python")
    for slot, change in component_overrides.items():
        current = base[slot]
        base[slot] = {
            **current,
            "name": change.get("name", current["name"]),
            "options": {**(current.get("options") or {}), **change.get("options", {})},
        }
    return DRPModel(ModelSpec(name, ModelConfig.model_validate(base)), hyperparameters)


@contextmanager
def capture_fit(predictor_class: Any, sink: dict[str, Any]) -> Iterator[None]:
    """Store the ``ModelInputBatch`` that a v2 predictor's ``_fit`` receives in ``sink["batch"]``.

    :param predictor_class: Predictor class whose ``_fit`` is wrapped.
    :param sink: Dictionary receiving the batch.
    :yields: Nothing; the patch is undone on exit.
    """
    original = predictor_class._fit

    def capture(self: Any, batch: Any) -> None:
        sink["batch"] = batch
        original(self, batch)

    predictor_class._fit = capture
    try:
        yield
    finally:
        predictor_class._fit = original


@contextmanager
def prediction_mode_checkpoints() -> Iterator[None]:
    """Let ``torch.load(weights_only=True)`` read checkpoints whose hyperparameters hold a ``PredictionMode``.

    ``build_components`` adds ``prediction_mode`` to the hyperparameters of every predictor, and the MOLIR and
    SuperFELTR networks store those in their Lightning checkpoints (``save_hyperparameters``). The reload of the
    best epoch (``utils.torch_io.load_state_dict``, ``weights_only=True``) then fails on the enum. The cases work
    around it so that the training can be compared; the failure is a finding of the cases.

    :yields: Nothing; the allow-list entry is removed on exit.
    """
    import torch

    from drevalpy.types.enums.prediction_mode import PredictionMode

    with torch.serialization.safe_globals([PredictionMode]):
        yield
