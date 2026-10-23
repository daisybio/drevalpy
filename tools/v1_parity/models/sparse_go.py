"""SparseGO parity case: 1.6.1 ``SparseGO`` against the v2 ``SparseGO`` zoo preset.

The case holds a tiny synthetic ontology of four levels (genes, leaf terms, two middle terms, a root) that has
the awkward shapes of the real one: a gene annotated to two terms, a gene annotated directly to a middle term
(a non-adjacent gene-term edge) and a root that is also the direct parent of a leaf term (a non-adjacent
term-term edge), so that ``pairs_in_layers`` has to insert virtual nodes. ``gene2ind`` indexes a subset of
the expression genes in an order that differs from the expression columns, and the ontology file annotates one
gene that ``gene2ind`` does not index.

Steps:

``structure/{layers,gene_order,gene_index,sparse_indices,sparse_shapes}``
    The ontology as the loaders read it: the per-layer (parent, child) pairs of ``pairs_in_layers`` (1.6.1:
    ``load_mapping`` / ``load_ontology`` / ``sort_pairs`` / ``pairs_in_layers`` on the written files; v2: the
    same functions behind ``resolve_ontology_metadata`` on the text in ``mdata.uns['sparsego']``), the gene
    order and indices, and the connectivity (``indices`` buffer, shape) of every ``SparseLinearNew`` of a network
    built from it. All exact.
``forward/<config>/{eval,train}``
    The two ``SparseGONetwork`` classes with identical weights on the same probe, in eval mode and in train mode
    (batch statistics and dropout under the same torch seed). The weights are drawn in ``build_case`` per
    ``state_dict`` key of the v2 network and loaded with ``strict=True`` on both sides.
``<variant>/{sample_ids,gene_order,features_sample,features_drug,drug_dim}``
    What enters the network: the samples' expression (or mutations) restricted and reordered to the ``gene2ind``
    order, the fingerprints and the fingerprint width that the model sets in its hyperparameters. 1.6.1 reads
    them in the real ``load_cell_line_features`` / ``load_drug_features`` from CSV files written by the case.
``<variant>/{params/*,predictions}``
    A short seeded CPU training and the test predictions: SGD with momentum and the per-epoch learning-rate decay.
    The network is built at the start of ``train`` on both sides, so one seed gives the same initial weights.

``tail_batch/completes``
    Prediction with a batch size that leaves a single row in the last batch (deviation).

Variants: ``expression`` (``input_type`` expression) and ``mutations`` (binary mutation matrix, ``input_type``
mutations).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from tools.v1_parity._bridge import Case, seed_all, tabular_case_arrays
from tools.v1_parity.compare import DEVIATION, StepSpec
from tools.v1_parity.models._torch_common import (
    capture_fit,
    draw_weights,
    forward_pair,
    load_weights,
    v2_model,
)

NAME = "SparseGO"

N_SAMPLES = 24
N_DRUGS = 5
N_BITS = 10
N_ONTOLOGY_GENES = 12
N_EXTRA_GENES = 5
PROBE_ROWS = 9
DROPOUT_SEED = 19
DATASET = "parity"
FORWARD_CONFIGS: dict[str, dict[str, Any]] = {
    "base": {
        "num_neurons_per_GO": 3,
        "num_neurons_per_final_GO": 4,
        "num_neurons_drug": [6, 4],
        "num_neurons_final": 5,
        "p_drop_genes": 0.2,
        "p_drop_terms": 0.2,
        "p_drop_drugs": 0.2,
        "p_drop_final": 0.1,
    },
    "single": {
        "num_neurons_per_GO": 1,
        "num_neurons_per_final_GO": 2,
        "num_neurons_drug": [5],
        "num_neurons_final": 3,
        "p_drop_genes": 0.3,
        "p_drop_terms": 0.3,
        "p_drop_drugs": 0.3,
        "p_drop_final": 0.0,
    },
}
TRAINING = {
    "num_neurons_per_GO": 3,
    "num_neurons_per_final_GO": 4,
    "num_neurons_drug": [6, 4],
    "num_neurons_final": 5,
    "p_drop_genes": 0.2,
    "p_drop_terms": 0.2,
    "p_drop_drugs": 0.2,
    "p_drop_final": 0.1,
    "learning_rate": 0.05,
    "decay_rate": 0.05,
    "momentum": 0.9,
    "epochs": 5,
    "batch_size": 16,
}
VARIANTS: dict[str, dict[str, Any]] = {
    "expression": {"input_type": "expression", "view": "gene_expression", "seed": 3},
    "mutations": {"input_type": "mutations", "view": "mutations", "seed": 4},
}
RECORDED_KEYS = (
    "genes_terms_sparse_linear_1.weights",
    "genes_terms_sparse_linear_1.bias",
    "genes_terms_batchnorm.running_mean",
    "GO_terms_sparse_linear_2.weights",
    "drug_linear_layer_1.weight",
    "final_linear_layer.weight",
    "final_batchnorm_layer.running_var",
    "final_linear_layer_output.weight",
    "final_linear_layer_output.bias",
)

TAIL_EXPLANATION = (
    "1.6.1 collects `outputs.squeeze()` per prediction batch and concatenates them. A last batch of one row "
    "squeezes to a zero-dimensional array and `np.concatenate` raises `ValueError: zero-dimensional arrays cannot "
    "be concatenated`; the same happens for a dataset of one pair. v2 predicts through `predict_pairs`, which "
    "keeps the batch dimension. Case: the batch size is one less than the number of test pairs."
)

FINDINGS = [
    "The ontology reading is the 1.6.1 one: `load_mapping`, `load_ontology`, `sort_pairs` and `pairs_in_layers` "
    "produce the same per-layer pairs (virtual nodes for non-adjacent edges included) and the same connectivity "
    "(`indices` buffers) from the same files. v2 reads the text from `mdata.uns['sparsego']` "
    "(`{'ontology': ..., 'gene2ind': ...}`) instead of `sparseGO_ont.txt` / `gene2ind.txt` next to the CSVs, and "
    "validates more (it raises on terms without genes, several roots, several components, non-contiguous gene "
    "indices; 1.6.1 prints and goes on).",
    "The network, the SGD loop (momentum 0.9, learning rate `lr / (1 + decay_rate * epoch)` set per epoch) and "
    "the network construction at the start of `train` are the 1.6.1 ones: with the same seed the initial weights, "
    "the shuffling and the dropout masks agree and the trained parameters and predictions agree at the `train` "
    "class.",
    "v2 defaults (`SparseGO` predictor, commit 174ceea7, B5) are the first grid point "
    "of the 1.6.1 `hyperparameters.yaml` (`num_neurons_per_final_GO` 30, `num_neurons_drug` [100, 50, 6], "
    "`p_drop_*` 0.15, `epochs` 400, `batch_size` 20000), not the defaults of the 1.6.1 class (6, [200, 100, 50], "
    "0.1, 100, 10000). The case sets every hyperparameter explicitly on both sides.",
    "v2 selects the ontology genes by name from the `gene_expression` / `mutations` view and raises if one is "
    "missing, as the 1.6.1 loader does; the preset has no gene list (1.6.1 loads `gene_list=None`, all columns, "
    "then restricts to `gene2ind`).",
    "The model keeps the last epoch on both sides (no early stopping, no checkpoint).",
]


def _ontology_rows(indexed: list[str]) -> list[tuple[str, str, str]]:
    """Return the ontology of the case over the genes that ``gene2ind`` indexes (plus one it does not).

    :param indexed: Gene names; ``indexed[i]`` is annotated as listed below.
    :returns: ``(parent, child, kind)`` rows in file order.
    """
    g = indexed
    rows = [("ROOT", "M1", "default"), ("ROOT", "M2", "default"), ("ROOT", "T3", "default")]
    rows += [("M1", "T1", "default"), ("M1", "T2", "default"), ("M2", "T2", "default"), ("M2", "T3", "default")]
    rows += [("T1", g[i], "gene") for i in (0, 1, 2, 3)]
    rows += [("T2", g[i], "gene") for i in (3, 4, 5, 6)]
    rows += [("T3", g[i], "gene") for i in (7, 8, 9, 11)]
    rows += [("M1", g[10], "gene"), ("T1", "NOT_INDEXED", "gene")]
    return rows


def _write_ontology_files(directory: Path, case: Case) -> None:
    """Write ``sparseGO_ont.txt`` and ``gene2ind.txt`` as 1.6.1 reads them.

    :param directory: Dataset directory.
    :param case: The shared case.
    """
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "sparseGO_ont.txt").write_text(_ontology_text(case))
    (directory / "gene2ind.txt").write_text(_gene2ind_text(case))


def _ontology_text(case: Case) -> str:
    return "".join(f"{a}\t{b}\t{kind}\n" for a, b, kind in case.arrays["ontology_rows"])


def _gene2ind_text(case: Case) -> str:
    return "".join(f"{index}\t{gene}\n" for index, gene in enumerate(case.arrays["gene2ind_genes"]))


def _layers_array(layers: list[np.ndarray]) -> np.ndarray:
    """Flatten per-layer pair arrays into ``(n, 3)`` rows ``[layer, parent, child]``.

    :param layers: Output of ``pairs_in_layers``.
    :returns: Unicode array.
    """
    return np.array([[str(i), str(a), str(b)] for i, layer in enumerate(layers) for a, b in layer])


def _sparse_connectivity(network: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``indices`` of every sparse layer (rows ``[layer, row, col]``) and the layer shapes.

    :param network: A ``SparseGONetwork``.
    :returns: ``(indices, shapes)`` as int64 arrays.
    """
    indices, shapes = [], []
    layers = [m for m in network.modules() if hasattr(m, "indices") and hasattr(m, "out_features")]
    for number, layer in enumerate(layers):
        values = layer.indices.cpu().numpy()
        indices.append(np.column_stack([np.full(values.shape[1], number), values.T]))
        shapes.append([layer.out_features, layer.in_features])
    return np.concatenate(indices).astype(np.int64), np.array(shapes, dtype=np.int64)


def _v2_structure(case: Case) -> tuple[list[np.ndarray], dict[str, int], list[str]]:
    from drevalpy.components.predictors.literature.sparsego.utils import resolve_ontology_metadata

    return resolve_ontology_metadata({"ontology": _ontology_text(case), "gene2ind": _gene2ind_text(case)})


def _v1_structure(case: Case) -> tuple[list[np.ndarray], dict[str, int]]:
    from drevalpy.models.SparseGO import utils

    with tempfile.TemporaryDirectory() as directory:
        _write_ontology_files(Path(directory), case)
        mapping = utils.load_mapping(str(Path(directory) / "gene2ind.txt"))
        graph, terms, annotations = utils.load_ontology(str(Path(directory) / "sparseGO_ont.txt"), mapping)
        pairs, levels, numbers = utils.sort_pairs(annotations, terms, graph, mapping)
        return utils.pairs_in_layers(pairs, levels, numbers), mapping


def _build_v2_network(connections: list, mapping: dict, setting: dict[str, Any]) -> Any:
    from drevalpy.components.predictors.literature.sparsego.algorithm import SparseGONetwork

    return SparseGONetwork(
        layer_connections=connections,
        num_neurons_per_go=setting["num_neurons_per_GO"],
        num_neurons_per_final_go=setting["num_neurons_per_final_GO"],
        num_neurons_drug=setting["num_neurons_drug"],
        num_neurons_final=setting["num_neurons_final"],
        drug_dim=N_BITS,
        gene2id_mapping=mapping,
        p_drop_final=setting["p_drop_final"],
        p_drop_genes=setting["p_drop_genes"],
        p_drop_terms=setting["p_drop_terms"],
        p_drop_drugs=setting["p_drop_drugs"],
    )


def _build_v1_network(connections: list, mapping: dict, setting: dict[str, Any]) -> Any:
    from drevalpy.models.SparseGO.sparsego import SparseGONetwork

    return SparseGONetwork(layer_connections=connections, drug_dim=N_BITS, gene2id_mapping=mapping, **setting)


def build_case(seed: int) -> Case:
    """Draw the tabular inputs, the ontology files' content and the forward probes.

    :param seed: Seed of the input generator.
    :returns: The case.
    :raises ValueError: If the ontology does not have the intended three layers.
    """
    rng = np.random.default_rng(seed + 1)
    n_genes = N_ONTOLOGY_GENES + N_EXTRA_GENES
    arrays = tabular_case_arrays(seed, n_samples=N_SAMPLES, n_drugs=N_DRUGS, n_genes=n_genes, n_bits=N_BITS)
    ontology_genes = np.array([f"G{i:02d}" for i in range(N_ONTOLOGY_GENES)])
    names = np.concatenate([ontology_genes, [f"XTRA{i:02d}" for i in range(N_EXTRA_GENES)]])
    arrays["gene_names"] = names[rng.permutation(len(names))]
    indexed = ontology_genes[rng.permutation(N_ONTOLOGY_GENES)]
    arrays["gene2ind_genes"] = indexed
    arrays["ontology_rows"] = np.array(_ontology_rows(list(indexed)))
    arrays["fingerprints"] = rng.normal(size=arrays["fingerprints"].shape).astype(np.float32)
    arrays["mutations"] = (rng.random(arrays["expression"].shape) < 0.4).astype(np.float32)

    case = Case(model=NAME, seed=seed, arrays=arrays, meta={"variants": VARIANTS, "forward": FORWARD_CONFIGS})
    connections, mapping, _ = _v2_structure(case)
    if len(connections) != 3:
        msg = f"expected three layers, got {len(connections)}"
        raise ValueError(msg)
    for config, setting in FORWARD_CONFIGS.items():
        network = _build_v2_network(connections, mapping, setting)
        arrays.update(draw_weights(network, rng, f"forward/{config}/w/"))
        arrays[f"forward/{config}/probe"] = rng.normal(size=(PROBE_ROWS, len(mapping) + N_BITS)).astype(np.float32)
    return case


def steps() -> list[StepSpec]:
    """Declare the compared outputs.

    :returns: Structure, forward, feature, training and prediction steps.
    """
    specs = [
        StepSpec("structure/layers", "exact", description="per-layer (parent, child) pairs of pairs_in_layers"),
        StepSpec("structure/gene_order", "exact", description="gene names ordered by their gene2ind index"),
        StepSpec("structure/gene_index", "exact", description="gene2ind index of each gene in that order"),
        StepSpec("structure/sparse_indices", "exact", description="(layer, row, col) of every sparse weight"),
        StepSpec("structure/sparse_shapes", "exact", description="(out, in) features of every sparse layer"),
    ]
    specs += [
        StepSpec(f"forward/{config}/{mode}", "forward", description=f"{config} network, {mode} mode output")
        for config in FORWARD_CONFIGS
        for mode in ("eval", "train")
    ]
    for variant in VARIANTS:
        table = [
            ("sample_ids", "exact", "ids of the samples whose features enter the network"),
            ("gene_order", "exact", "genes of the feature columns"),
            ("features_sample", "exact", "features of those samples restricted and ordered to gene2ind"),
            ("features_drug", "exact", "fingerprints of the drugs"),
            ("drug_dim", "exact", "fingerprint width written to the hyperparameters"),
            *[(f"params/{key}", "train", f"{key} after training") for key in RECORDED_KEYS],
            ("predictions", "train", "test predictions after the seeded short training"),
        ]
        specs += [
            StepSpec(f"{variant}/{quantity}", tolerance, description=f"{variant}: {text}")
            for quantity, tolerance, text in table
        ]
    specs.append(
        StepSpec(
            "tail_batch/completes",
            "exact",
            scope=DEVIATION,
            description="prediction with a last batch of a single row completes (1) or raises (0)",
            explanation=TAIL_EXPLANATION,
        )
    )
    return specs


def _structure_arrays(layers: list[np.ndarray], mapping: dict[str, int], network: Any) -> dict[str, np.ndarray]:
    order = sorted(mapping, key=mapping.__getitem__)
    indices, shapes = _sparse_connectivity(network)
    return {
        "structure/layers": _layers_array(layers),
        "structure/gene_order": np.array(order),
        "structure/gene_index": np.array([mapping[g] for g in order], dtype=np.int64),
        "structure/sparse_indices": indices,
        "structure/sparse_shapes": shapes,
    }


def _forward_outputs(build: Any, layers: list, mapping: dict, case: Case) -> dict[str, np.ndarray]:
    import torch

    out: dict[str, np.ndarray] = {}
    for config, setting in case.meta["forward"].items():
        network = build(layers, mapping, setting)
        load_weights(network, case, f"forward/{config}/w/")
        probe = torch.as_tensor(case.arrays[f"forward/{config}/probe"])
        for mode, values in forward_pair(network, (probe,), DROPOUT_SEED).items():
            out[f"forward/{config}/{mode}"] = values
    return out


def _hyperparameters(setting: dict[str, Any]) -> dict[str, Any]:
    return {**TRAINING, "input_type": setting["input_type"]}


def _variant_case(case: Case, setting: dict[str, Any]) -> Case:
    """Return the case with the variant's omics matrix under ``expression``.

    :param case: The shared case.
    :param setting: Variant entry.
    :returns: A case whose ``expression`` array is the expression or the mutation matrix.
    """
    key = "expression" if setting["input_type"] == "expression" else "mutations"
    return Case(
        model=case.model, seed=case.seed, arrays={**case.arrays, "expression": case.arrays[key]}, meta=case.meta
    )


def _write_v1_tree(root: Path, case: Case, setting: dict[str, Any]) -> None:
    """Write the CSV and text files that the 1.6.1 loaders read for one variant.

    :param root: Data path.
    :param case: The shared case (``expression`` holds the variant's matrix).
    :param setting: Variant entry.
    """
    import pandas as pd

    dataset = root / DATASET
    _write_ontology_files(dataset, case)
    omics = pd.DataFrame(
        case.arrays["expression"].astype(np.float64), index=case.arrays["sample_ids"], columns=case.arrays["gene_names"]
    )
    omics.index.name = "cell_line_name"
    omics.to_csv(dataset / f"{setting['view']}.csv")
    (dataset / "drug_fingerprints").mkdir()
    fingerprints = pd.DataFrame(case.arrays["fingerprints"].T.astype(np.float64), columns=case.arrays["drug_ids"])
    fingerprints.to_csv(dataset / "drug_fingerprints" / "pubchem_id_to_demorgan_128_map.csv", index=False)


def run_v1(case: Case) -> dict[str, np.ndarray]:
    """Run the structure, the forward probes and every variant with the 1.6.1 ``SparseGO`` model.

    :param case: The shared case.
    :returns: Arrays keyed by the step names.
    """
    from drevalpy.models.SparseGO.sparsego import SparseGOModel

    from tools.v1_parity._v1_runner import tabular_inputs

    layers, mapping = _v1_structure(case)
    out = _structure_arrays(layers, mapping, _build_v1_network(layers, mapping, FORWARD_CONFIGS["base"]))
    out.update(_forward_outputs(_build_v1_network, layers, mapping, case))

    for index, (variant, setting) in enumerate(case.meta["variants"].items()):
        variant_case = _variant_case(case, setting)
        data = tabular_inputs(variant_case, feature_dtype="float32", expression_view=setting["view"])
        with tempfile.TemporaryDirectory() as directory:
            _write_v1_tree(Path(directory), variant_case, setting)
            model = SparseGOModel()
            model.build_model(_hyperparameters(setting))
            loaded = model.load_cell_line_features(directory, DATASET)
            drugs = model.load_drug_features(directory, DATASET)

        seed_all(setting["seed"] + 100 * index)
        model.train(data.train, loaded, drugs)
        predictions = model.predict(data.test.cell_line_ids, data.test.drug_ids, loaded, drugs)

        used = np.unique(data.train.cell_line_ids)
        out[f"{variant}/sample_ids"] = used.astype(str)
        out[f"{variant}/gene_order"] = np.array(model.ontology_gene_order)
        out[f"{variant}/features_sample"] = loaded.get_feature_matrix(setting["view"], used).astype(np.float32)
        out[f"{variant}/features_drug"] = drugs.get_feature_matrix("fingerprints", np.unique(data.train.drug_ids))
        out[f"{variant}/drug_dim"] = np.array(model.hyperparameters["drug_dim"])
        out[f"{variant}/predictions"] = np.asarray(predictions, dtype=np.float64)
        state = model.model.state_dict()
        for key in RECORDED_KEYS:
            out[f"{variant}/params/{key}"] = state[key].cpu().numpy()
        if index == 0:
            model.hyperparameters["batch_size"] = len(data.test.response) - 1
            try:
                model.predict(data.test.cell_line_ids, data.test.drug_ids, loaded, drugs)
                out["tail_batch/completes"] = np.array(1)
            except ValueError:
                out["tail_batch/completes"] = np.array(0)
    return out


def run_v2(case: Case) -> dict[str, np.ndarray]:
    """Run the structure, the forward probes and every variant with the v2 ``SparseGO`` preset.

    :param case: The shared case.
    :returns: Arrays keyed like those of :func:`run_v1`.
    """
    from drevalpy.components.predictors.literature.sparsego.predictor import SparseGOPredictor
    from tools.v1_parity._v2_runner import tabular_inputs

    layers, mapping, _ = _v2_structure(case)
    out = _structure_arrays(layers, mapping, _build_v2_network(layers, mapping, FORWARD_CONFIGS["base"]))
    out.update(_forward_outputs(_build_v2_network, layers, mapping, case))

    for index, (variant, setting) in enumerate(case.meta["variants"].items()):
        variant_case = _variant_case(case, setting)
        data = tabular_inputs(variant_case, expression_view=setting["view"])
        data.dataset.mdata.uns["sparsego"] = {"ontology": _ontology_text(case), "gene2ind": _gene2ind_text(case)}
        model = v2_model(NAME, _hyperparameters(setting))
        captured: dict[str, Any] = {}
        seed_all(setting["seed"] + 100 * index)
        with capture_fit(SparseGOPredictor, captured):
            model.train(data.dataset, data.train)
        predictions = model.predict(data.dataset, data.test)

        batch = captured["batch"]
        block = batch.sample.blocks[setting["view"]]
        sample_order = np.argsort(batch.sample.entity_ids.astype(str), kind="stable")
        drug_order = np.argsort(batch.drug.entity_ids.astype(str), kind="stable")
        predictor = model._pipeline._predictor
        out[f"{variant}/sample_ids"] = batch.sample.entity_ids[sample_order].astype(str)
        out[f"{variant}/gene_order"] = np.array(predictor._ontology_gene_order)
        out[f"{variant}/features_sample"] = np.asarray(block.values, dtype=np.float32)[sample_order]
        out[f"{variant}/features_drug"] = np.asarray(batch.drug.blocks["fingerprints"].values)[drug_order]
        out[f"{variant}/drug_dim"] = np.array(predictor._hyperparameters["drug_dim"])
        out[f"{variant}/predictions"] = np.asarray(predictions, dtype=np.float64)
        state = predictor._model.state_dict()
        for key in RECORDED_KEYS:
            out[f"{variant}/params/{key}"] = state[key].cpu().numpy()
        if index == 0:
            predictor._hyperparameters["batch_size"] = len(case.arrays["test_pairs"]) - 1
            model.predict(data.dataset, data.test)
            out["tail_batch/completes"] = np.array(1)
    return out
