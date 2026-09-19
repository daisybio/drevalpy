"""Independent numerical parity audit for the full DIPK and SparseGO networks.

Upstream classes are parsed out of their original scripts so importing the scripts
cannot trigger their training, CLI, logging, or global RNG side effects.
"""

from __future__ import annotations

import ast
import json
import os
from types import SimpleNamespace
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch import nn
from torch.nn import functional
from upath import UPath

ROOT = UPath(os.environ.get("UPSTREAM_REFERENCE_ROOT", ".")).resolve()
RESULT = UPath(os.environ.get("EXPANDED_RESULTS_PATH", "expanded-reference.json")).resolve()


def extract_nodes(path: UPath, names: set[str], namespace: dict[str, Any]) -> dict[str, Any]:
    """Compile only selected top-level functions/classes from a source file."""
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102 - selected definitions only
    return namespace


def maximum_error(left: torch.Tensor, right: torch.Tensor) -> float:
    """Audit helper."""
    return float(torch.max(torch.abs(left.detach() - right.detach())).cpu())


def assert_close(left: torch.Tensor, right: torch.Tensor, *, label: str) -> float:
    """Audit helper."""
    torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-7)
    error = maximum_error(left, right)
    if not np.isfinite(error):
        raise AssertionError(f"{label} produced a non-finite error")
    return error


def compare_gradients(left: nn.Module, right: nn.Module) -> dict[str, Any]:
    """Audit helper."""
    left_params = dict(left.named_parameters())
    right_params = dict(right.named_parameters())
    if left_params.keys() != right_params.keys():
        raise AssertionError((left_params.keys() - right_params.keys(), right_params.keys() - left_params.keys()))
    none_left = sorted(name for name, parameter in left_params.items() if parameter.grad is None)
    none_right = sorted(name for name, parameter in right_params.items() if parameter.grad is None)
    if none_left != none_right:
        raise AssertionError((none_left, none_right))
    errors = {
        name: assert_close(left_params[name].grad, right_params[name].grad, label=f"gradient {name}")
        for name in left_params
        if left_params[name].grad is not None
    }
    return {
        "active_parameter_tensors": len(errors),
        "grad_none": none_left,
        "max_gradient_error": max(errors.values(), default=0.0),
    }


def compare_parameters(left: nn.Module, right: nn.Module) -> float:
    """Audit helper."""
    left_params = dict(left.named_parameters())
    right_params = dict(right.named_parameters())
    if left_params.keys() != right_params.keys():
        raise AssertionError("parameter names differ")
    return max(
        (assert_close(left_params[name], right_params[name], label=f"parameter {name}") for name in left_params),
        default=0.0,
    )


def dense_batch(flat: torch.Tensor, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimal to_dense_batch equivalent for ordered graph batches used in this audit."""
    batch_size = int(batch.max().item()) + 1
    counts = torch.bincount(batch, minlength=batch_size)
    max_atoms = int(counts.max().item())
    dense = flat.new_zeros((batch_size, max_atoms, flat.shape[-1]))
    mask = torch.zeros((batch_size, max_atoms), dtype=torch.bool, device=flat.device)
    offsets = torch.zeros(batch_size, dtype=torch.long, device=flat.device)
    for atom, graph_index in zip(flat, batch, strict=True):
        graph = int(graph_index.item())
        offset = int(offsets[graph].item())
        dense[graph, offset] = atom
        mask[graph, offset] = True
        offsets[graph] += 1
    return dense, mask


def upstream_dipk_classes() -> dict[str, Any]:
    """Audit helper."""
    dipk = ROOT / "DIPK" / "Task1" / "fold=0_model=2"
    mha_namespace: dict[str, Any] = {"torch": torch, "nn": nn}
    extract_nodes(dipk / "Model_MHA.py", {"MultiHeadAttentionLayer"}, mha_namespace)
    namespace = {
        "torch": torch,
        "nn": nn,
        "F": functional,
        "to_dense_batch": dense_batch,
        "MultiHeadAttentionLayer": mha_namespace["MultiHeadAttentionLayer"],
        "features_dim_gene": 512,
        "features_dim_bionic": 512,
        "DEVICE": torch.device("cpu"),
    }
    return extract_nodes(dipk / "Model.py", {"AttentionLayer", "DenseLayers", "Predictor"}, namespace)


def audit_dipk() -> dict[str, Any]:
    """Audit helper."""
    from drevalpy.components.predictors.literature.dipk.model_utils import Predictor as LocalPredictor

    upstream = upstream_dipk_classes()
    torch.manual_seed(1729)
    source_model = upstream["Predictor"](
        embedding_dim=768,
        heads=1,
        fc_layer_num=3,
        fc_layer_dim=[64, 32, 16, 8, 4, 2],
        dropout_rate=0.2,
    )
    local_model = LocalPredictor(
        heads=1,
        fc_layer_num=3,
        fc_layer_dim=[64, 32, 16, 8, 4, 2],
        dropout_rate=0.2,
    )
    local_model.load_state_dict(source_model.state_dict(), strict=True)

    generator = torch.Generator().manual_seed(2718)
    flat_atoms = torch.randn(8, 768, generator=generator)
    graph_batch = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2], dtype=torch.long)
    graph = SimpleNamespace(batch=graph_batch)
    padded_atoms, mask = dense_batch(flat_atoms, graph_batch)
    gene = torch.randn(3, 512, generator=generator)
    bionic = torch.randn(3, 512, generator=generator)
    targets = torch.tensor([[0.25], [-0.75], [1.5]])

    source_model.eval()
    local_model.eval()
    with torch.no_grad():
        source_attention = source_model.attention_layer(flat_atoms, graph, gene, bionic)
        local_attention = local_model.attention_layer(padded_atoms, mask, gene, bionic)
        attention_error = assert_close(source_attention, local_attention, label="DIPK attention")
        source_dense = source_model.dense_layers(source_attention, gene, bionic)
        local_dense = local_model.dense_layers(local_attention, gene, bionic)
        dense_error = assert_close(source_dense, local_dense, label="DIPK dense head")
        source_output = source_model(flat_atoms, graph, gene, bionic)
        local_output = local_model(padded_atoms, gene, bionic, mask)
        forward_error = assert_close(source_output, local_output, label="DIPK forward")
        source_mse = functional.mse_loss(source_output, targets)
        local_mse = functional.mse_loss(local_output, targets)
        mse_error = assert_close(source_mse, local_mse, label="DIPK MSE")

    source_model.train()
    local_model.train()
    source_model.zero_grad(set_to_none=True)
    local_model.zero_grad(set_to_none=True)
    torch.manual_seed(314159)
    source_train_output = source_model(flat_atoms, graph, gene, bionic)
    source_loss = functional.mse_loss(source_train_output, targets)
    source_loss.backward()
    torch.manual_seed(314159)
    local_train_output = local_model(padded_atoms, gene, bionic, mask)
    local_loss = functional.mse_loss(local_train_output, targets)
    local_loss.backward()
    train_forward_error = assert_close(source_train_output, local_train_output, label="DIPK train forward")
    train_mse_error = assert_close(source_loss, local_loss, label="DIPK train MSE")
    gradient_report = compare_gradients(source_model, local_model)

    expected_unused = [
        f"dense_layers.fc_layers.{index}.{suffix}" for index in range(3, 7) for suffix in ("bias", "weight")
    ]
    if gradient_report["grad_none"] != sorted(expected_unused):
        raise AssertionError((gradient_report["grad_none"], sorted(expected_unused)))
    source_optimizer = torch.optim.SGD(source_model.parameters(), lr=0.025)
    local_optimizer = torch.optim.SGD(local_model.parameters(), lr=0.025)
    source_optimizer.step()
    local_optimizer.step()
    post_step_error = compare_parameters(source_model, local_model)

    return {
        "source": {
            "repository": "user15632/DIPK",
            "commit": "0edfc151fbbdb7864d577865f9541a689d706c5b",
            "files": ["Task1/fold=0_model=2/Model.py", "Task1/fold=0_model=2/Model_MHA.py"],
        },
        "adaptations": [
            "Parsed network and attention classes only; training side effects were excluded.",
            "Set Model_DAE.ldim to its source value 512.",
            "Adapted ragged graph input to local padded tensors and masks.",
            "Used heads=1 because the selected upstream AttentionLayer hard-codes n_heads=1.",
        ],
        "shape": list(source_output.shape),
        "mse": float(source_mse),
        "intermediate_max_abs_error": {"attention": attention_error, "dense_head": dense_error},
        "forward_max_abs_error": forward_error,
        "mse_abs_error": mse_error,
        "train_forward_max_abs_error": train_forward_error,
        "train_mse_abs_error": train_mse_error,
        **gradient_report,
        "post_sgd_parameter_max_abs_error": post_step_error,
    }


class LightningModuleShim(nn.Module):
    """Audit helper."""

    @property
    def device(self) -> torch.device:
        """Audit helper."""
        parameter = next(self.parameters(), None)
        return torch.device("cpu") if parameter is None else parameter.device

    def save_hyperparameters(self, **_: Any) -> None:
        """Audit helper."""
        return None


class MeanMetricShim(nn.Module):
    """Audit helper."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Audit helper."""
        return value


class TorchSparseShim:
    """Equivalent coalesce operation implemented with native PyTorch COO."""

    @staticmethod
    def coalesce(
        indices: torch.Tensor, values: torch.Tensor, rows: int, columns: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Coalesce sparse connectivity indices."""
        matrix = torch.sparse_coo_tensor(indices, values, (rows, columns)).coalesce()
        return matrix.indices(), matrix.values()


def upstream_sparsego_symbols() -> dict[str, Any]:
    """Audit helper."""
    source = ROOT / "SparseGO" / "code"
    sparse_namespace: dict[str, Any] = {
        "math": __import__("math"),
        "warnings": __import__("warnings"),
        "np": np,
        "torch": torch,
        "nn": nn,
        "torch_sparse": TorchSparseShim,
    }
    extract_nodes(source / "sparse_linear_new.py", {"SparseLinearNew"}, sparse_namespace)
    utils_namespace: dict[str, Any] = {"np": np, "pd": pd}
    extract_nodes(source / "utils.py", {"sort_pairs", "pairs_in_layers", "create_index"}, utils_namespace)
    model_namespace = {
        "os": os,
        "np": np,
        "torch": torch,
        "nn": nn,
        "sparse": sparse,
        "LightningModule": LightningModuleShim,
        "MeanMetric": MeanMetricShim,
        "SparseLinearNew": sparse_namespace["SparseLinearNew"],
        "create_index": utils_namespace["create_index"],
    }
    extract_nodes(source / "train_cv.py", {"SparseGO"}, model_namespace)
    return {**model_namespace, **{f"utils_{key}": value for key, value in utils_namespace.items()}}


def compare_connections(left: list[np.ndarray], right: list[np.ndarray]) -> None:
    """Audit helper."""
    if len(left) != len(right):
        raise AssertionError((len(left), len(right)))
    for index, (left_layer, right_layer) in enumerate(zip(left, right, strict=True)):
        np.testing.assert_array_equal(left_layer, right_layer, err_msg=f"ontology layer {index}")


def hook_outputs(model: nn.Module, names: list[str]) -> tuple[dict[str, torch.Tensor], list[Any]]:
    """Audit helper."""
    outputs: dict[str, torch.Tensor] = {}
    modules = dict(model.named_modules())
    handles = []
    for name in names:
        handles.append(
            modules[name].register_forward_hook(
                lambda _module, _inputs, output, name=name: outputs.__setitem__(name, output.detach().clone())
            )
        )
    return outputs, handles


def audit_sparsego() -> dict[str, Any]:
    """Audit helper."""
    from drevalpy.components.predictors.literature.sparsego.algorithm import SparseGONetwork
    from drevalpy.components.predictors.literature.sparsego.utils import (
        pairs_in_layers as local_pairs_in_layers,
    )
    from drevalpy.components.predictors.literature.sparsego.utils import (
        sort_pairs as local_sort_pairs,
    )

    upstream = upstream_sparsego_symbols()
    gene_mapping = {f"G{i}": i for i in range(4)}
    gene_term_pairs = np.array([["B", "G0"], ["B", "G1"], ["A", "G2"], ["ROOT", "G3"]])
    term_pairs = np.array([["ROOT", "A"], ["A", "B"]])
    graph = nx.DiGraph()
    graph.add_edges_from(term_pairs)
    source_sorted, source_levels, source_level_numbers = upstream["utils_sort_pairs"](
        gene_term_pairs, term_pairs, graph, gene_mapping
    )
    local_sorted, local_levels, local_level_numbers = local_sort_pairs(gene_term_pairs, term_pairs, graph, gene_mapping)
    np.testing.assert_array_equal(source_sorted, local_sorted)
    if source_levels != local_levels or source_level_numbers != local_level_numbers:
        raise AssertionError("ontology levels differ")
    source_connections = upstream["utils_pairs_in_layers"](source_sorted, source_levels, source_level_numbers)
    local_connections = local_pairs_in_layers(local_sorted, local_levels, local_level_numbers)
    compare_connections(source_connections, local_connections)

    common = {
        "num_neurons_per_GO": 2,
        "num_neurons_per_final_GO": 3,
        "num_neurons_drug": [5, 3],
        "num_neurons_final": 4,
        "layer_connections": source_connections,
        "p_drop_final": 0.0,
        "p_drop_genes": 0.0,
        "p_drop_terms": 0.0,
        "p_drop_drugs": 0.0,
    }
    cell_features = np.zeros((2, 4), dtype=np.float32)
    drug_features = np.zeros((2, 6), dtype=np.float32)
    torch.manual_seed(4242)
    source_model = upstream["SparseGO"](
        output_folder="",
        fold="",
        input_type="expression",
        learning_rate=0.01,
        decay_rate=0.0,
        optimizer_type="SGD",
        momentum=0.0,
        loss_type="MSELoss",
        cell_features=cell_features,
        drug_features=drug_features,
        gene2id_mapping_ont=gene_mapping,
        **common,
    )
    local_model = SparseGONetwork(
        layer_connections=local_connections,
        num_neurons_per_go=common["num_neurons_per_GO"],
        num_neurons_per_final_go=common["num_neurons_per_final_GO"],
        num_neurons_drug=common["num_neurons_drug"],
        num_neurons_final=common["num_neurons_final"],
        drug_dim=drug_features.shape[1],
        gene2id_mapping=gene_mapping,
        p_drop_final=0.0,
        p_drop_genes=0.0,
        p_drop_terms=0.0,
        p_drop_drugs=0.0,
    )
    local_model.load_state_dict(source_model.state_dict(), strict=True)

    generator = torch.Generator().manual_seed(5150)
    inputs = torch.randn(5, 10, generator=generator)
    targets = torch.tensor([[0.2], [-0.4], [0.8], [1.1], [-0.7]])
    intermediate_names = [
        "genes_terms_sparse_linear_1",
        "GO_terms_sparse_linear_1",
        "GO_terms_sparse_linear_2",
        "drug_linear_layer_1",
        "drug_linear_layer_2",
        "final_linear_layer",
        "final_aux_linear_layer",
        "final_linear_layer_output",
    ]
    source_intermediates, source_handles = hook_outputs(source_model, intermediate_names)
    local_intermediates, local_handles = hook_outputs(local_model, intermediate_names)
    source_model.eval()
    local_model.eval()
    with torch.no_grad():
        source_output = source_model(inputs)
        local_output = local_model(inputs)
    for handle in source_handles + local_handles:
        handle.remove()
    intermediate_errors = {
        name: assert_close(source_intermediates[name], local_intermediates[name], label=f"SparseGO {name}")
        for name in intermediate_names
    }
    forward_error = assert_close(source_output, local_output, label="SparseGO forward")
    source_mse = source_model.criterion(source_output, targets)
    local_mse = functional.mse_loss(local_output, targets)
    mse_error = assert_close(source_mse, local_mse, label="SparseGO MSE")

    source_model.train()
    local_model.train()
    source_model.zero_grad(set_to_none=True)
    local_model.zero_grad(set_to_none=True)
    source_train_output = source_model(inputs)
    local_train_output = local_model(inputs)
    train_forward_error = assert_close(source_train_output, local_train_output, label="SparseGO train forward")
    source_loss = source_model.criterion(source_train_output, targets)
    local_loss = functional.mse_loss(local_train_output, targets)
    train_mse_error = assert_close(source_loss, local_loss, label="SparseGO train MSE")
    source_loss.backward()
    local_loss.backward()
    gradient_report = compare_gradients(source_model, local_model)
    if gradient_report["grad_none"]:
        raise AssertionError(f"unexpected inactive SparseGO parameters: {gradient_report['grad_none']}")
    source_optimizer = torch.optim.SGD(source_model.parameters(), lr=0.01)
    local_optimizer = torch.optim.SGD(local_model.parameters(), lr=0.01)
    source_optimizer.step()
    local_optimizer.step()
    post_step_error = compare_parameters(source_model, local_model)

    return {
        "source": {
            "repository": "KatynaSada/SparseGO_lightning",
            "commit": "8cec017405bddc52e3963e5807f6e62f8c224f1d",
            "files": ["code/train_cv.py", "code/sparse_linear_new.py", "code/utils.py"],
        },
        "adaptations": [
            "Parsed ontology wiring only; training side effects were excluded.",
            "Used parameter-free module shims for construction, forward, and criterion.",
            "Used equivalent sparse index coalescing for fixed connectivity.",
            "Compared source and local layer connections.",
        ],
        "ontology": {
            "levels": source_levels,
            "layer_connections": [layer.tolist() for layer in source_connections],
            "wiring_equal": True,
        },
        "shape": list(source_output.shape),
        "mse": float(source_mse),
        "intermediate_max_abs_error": intermediate_errors,
        "forward_max_abs_error": forward_error,
        "mse_abs_error": mse_error,
        "train_forward_max_abs_error": train_forward_error,
        "train_mse_abs_error": train_mse_error,
        **gradient_report,
        "post_sgd_parameter_max_abs_error": post_step_error,
    }


def main() -> None:
    """Audit helper."""
    torch.set_num_threads(1)
    report = {
        "schema_version": 1,
        "purpose": "Independent full-network parity against pinned upstream source trees",
        "dipk": audit_dipk(),
        "sparsego": audit_sparsego(),
    }
    RESULT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(RESULT)


if __name__ == "__main__":
    main()
