"""DrugGNN parity case: 1.6.1 ``DrugGNN`` against the v2 ``DrugGNN`` zoo preset.

Steps:

``graphs/*``
    The drug graphs built from real SMILES (aspirin, an amino acid, an indole, a salt, a charged amide ...):
    node features ``x``, ``edge_index`` and ``edge_attr`` of each molecule, concatenated, with the node and
    edge counts per molecule. 1.6.1 builds them in ``datasets/featurizer/create_drug_graphs._smiles_to_graph``
    (a script run before training), v2 in ``drug_graph._smiles_to_graph``. All exact.
``graphs/single_atom_shapes``
    The shapes of ``edge_index`` and ``edge_attr`` for ``[Na+]``, a molecule without bonds (deviation).
``forward/<config>/{eval,train}``
    The two ``DrugGraphNet`` classes with identical weights on a batch of those graphs plus random sample
    features, in eval mode and in train mode (dropout under the same torch seed). The weights are drawn in
    ``build_case`` per ``state_dict`` key of the v2 network and loaded with ``strict=True`` on both sides.
``<variant>/{sample_ids,features_sample,graphs_x}``
    What enters the network: the selected landmark genes in list order (neither preset scales them) and the
    node features of each training drug's graph. The case holds ``landmark_genes`` (the preset's and the
    1.6.1 default) in shuffled order plus extra genes, so the v2 selection is exercised; 1.6.1 selects the
    list while loading CSVs, which the in-memory harness skips, so its side receives the matrix reordered to the list.
``<variant>/{params/*,predictions}``
    A short seeded CPU training and the test predictions.

Variants: ``no_es`` (no early-stopping set, both sides keep the last epoch), ``es_improving`` (with one, and a
validation loss that improves every epoch, so the last epoch is the best one) and ``es`` (a large learning rate
makes the validation loss rise at the end: 1.6.1 keeps the last epoch, v2 restores the best one, a deviation).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tools.v1_parity._bridge import Case, seed_all, tabular_case_arrays
from tools.v1_parity.compare import DEVIATION, PARITY, StepSpec
from tools.v1_parity.models._torch_common import (
    capture_fit,
    draw_weights,
    forward_pair,
    load_weights,
    restrict_v1_genes,
    shuffled_gene_names,
    v2_model,
)

NAME = "DrugGNN"

GENE_LIST = "landmark_genes"
N_EXTRA_GENES = 8
N_SAMPLES = 24
STOP_EVERY = 5
PROBE_ROWS = 7
DROPOUT_SEED = 13
SMILES = (
    "CCO",
    "CC(=O)Oc1ccccc1C(=O)O",
    "C[C@@H](N)C(=O)O",
    "ClCCBr",
    "c1ccc2c(c1)[nH]c1ccccc12",
    "CC(=O)[O-].[Na+]",
)
EDGE_SMILES = ("[Na+]", "C#N", "O=C1CCC(=O)N1Cl")
N_CELL_FEATURES_FORWARD = 9
HIDDEN_FORWARD = 6

VARIANTS: dict[str, dict[str, Any]] = {
    "no_es": {
        "hidden_dim": 6,
        "dropout": 0.2,
        "learning_rate": 0.01,
        "epochs": 3,
        "batch_size": 8,
        "early_stopping": False,
        "seed": 1,
    },
    "es_improving": {
        "hidden_dim": 6,
        "dropout": 0.2,
        "learning_rate": 0.01,
        "epochs": 3,
        "batch_size": 8,
        "early_stopping": True,
        "improving": True,
        "seed": 3,
    },
    "es": {
        "hidden_dim": 6,
        "dropout": 0.2,
        "learning_rate": 0.3,
        "epochs": 3,
        "batch_size": 8,
        "early_stopping": True,
        "seed": 2,
    },
}
_NON_HYPERPARAMETERS = ("early_stopping", "improving", "seed")
FORWARD_CONFIGS: dict[str, dict[str, Any]] = {
    "dropout": {"hidden_dim": HIDDEN_FORWARD, "dropout": 0.3},
    "no_dropout": {"hidden_dim": 4, "dropout": 0.0},
}
BEST_EPOCH_EXPLANATION = (
    "1.6.1 trains with a default Lightning `ModelCheckpoint` that is never restored, so its predictions come "
    "from the last epoch; v2 restores the checkpoint with the best validation loss before predicting (documented "
    "in `docs/explanation/literature_models.rst`). The two differ whenever the validation loss of the last epoch "
    "is not the minimum, which the large learning rate of this variant makes happen."
)
SINGLE_ATOM_EXPLANATION = (
    "1.6.1 builds `edge_index` with `torch.tensor([]).t()` for a molecule without bonds, which has shape (0,), "
    "not (2, 0); a graph convolution would fail on it. v2 returns an explicit (2, 0) tensor, so a single-atom "
    "drug (an ion) can be used. Its empty `edge_attr` has shape (0, 6) whereas a real bond has 7 edge features "
    "(`edge_attr` is not read by the network)."
)

FINDINGS = [
    "The network and the Lightning module are the 1.6.1 ones apart from docstrings and the removal of the metrics "
    "mixin, so a `state_dict` loads into both with `strict=True` and the forward pass agrees at the `forward` "
    "class in eval and in train mode.",
    "The drug graphs of real molecules agree exactly (161 node features, the same one-hot layout and bond "
    "order). v2 additionally supports a molecule without bonds (deviation `graphs/single_atom_shapes`).",
    "The v2 preset (commit 174ceea7, B5) is `landmarkGenes` with "
    "`gene_list_stem: landmark_genes`, `standardize: false`, `arcsinh: false`: the 867-gene list of 1.6.1 "
    "(`DrugGNN.gene_list`) with no scaling, as in 1.6.1. Before B5 it was `landmarkGenesReduced` (270 genes). "
    "The case does not override the preset; the selected raw columns agree (`*/features_sample`).",
    "The v2 predictor defaults since B5 (174ceea7) (`epochs` 100, `batch_size` 1024) follow 1.6.1; before them they "
    "were 2 and 8. 1.6.1 uses the same `batch_size` default of 1024 for training but 32 for the validation and "
    "prediction loaders when the key is absent; v2 uses the one `batch_size` for all three (no numeric effect: "
    "the loss is averaged over the validation set with batch-size weights). The case sets all hyperparameters "
    "explicitly on both sides, and 1.6.1's `num_workers` to 0.",
    "1.6.1 early-stops on the validation loss with patience 5 and keeps the last epoch; v2 restores the best "
    "epoch (deviation `es/*`). Both early-stop only when a validation set is given; without one v2 attaches no "
    "checkpoint, so the last epoch is kept as in 1.6.1 (`no_es/*` agree).",
    "Both sides draw the shuffling permutation after the module initialization and a DataLoader base seed; with "
    "`num_workers=0` the torch RNG order is identical.",
]


def _graph_arrays(graphs: list[Any], prefix: str) -> dict[str, np.ndarray]:
    """Concatenate the tensors of a list of ``torch_geometric`` graphs.

    :param graphs: ``Data`` objects.
    :param prefix: Output-name prefix.
    :returns: ``x``, ``edge_index``, ``edge_attr`` and the node and edge counts per graph.
    """
    return {
        f"{prefix}x": np.concatenate([g.x.numpy() for g in graphs]),
        f"{prefix}edge_index": np.concatenate([g.edge_index.numpy().reshape(2, -1) for g in graphs], axis=1),
        f"{prefix}edge_attr": np.concatenate([g.edge_attr.numpy().reshape(-1, 7) for g in graphs]),
        f"{prefix}n_nodes": np.array([g.x.shape[0] for g in graphs]),
        f"{prefix}n_edges": np.array([g.edge_index.numel() // 2 for g in graphs]),
    }


def _forward_arrays(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Draw weights and probes for every forward configuration from the v2 network's shapes.

    :param rng: Generator of the case.
    :returns: Arrays ``forward/<config>/w/<key>``, ``.../cells`` and ``.../graph_index``.
    """
    from drevalpy.components.featurizers.drug.drug_graph import _smiles_to_graph
    from drevalpy.components.predictors.literature.druggnn.algorithm import DrugGraphNet

    n_nodes = _smiles_to_graph(SMILES[0]).x.shape[1]
    arrays: dict[str, np.ndarray] = {}
    for config, setting in FORWARD_CONFIGS.items():
        network = DrugGraphNet(n_nodes, N_CELL_FEATURES_FORWARD, **setting)
        arrays.update(draw_weights(network, rng, f"forward/{config}/w/", scale=0.3))
        arrays[f"forward/{config}/cells"] = rng.normal(size=(PROBE_ROWS, N_CELL_FEATURES_FORWARD)).astype(np.float32)
        arrays[f"forward/{config}/graph_index"] = rng.integers(0, len(SMILES), size=PROBE_ROWS)
    return arrays


def build_case(seed: int) -> Case:
    """Draw the tabular inputs (list genes in shuffled order plus extras), the SMILES and the forward probes.

    :param seed: Seed of the input generator.
    :returns: The case.
    :raises ValueError: If a drug occurs in no training pair.
    """
    from drevalpy.components.featurizers.sample.gene_lists import resolve_gene_list

    gene_list = np.array(resolve_gene_list(GENE_LIST))
    rng = np.random.default_rng(seed + 1)
    names = shuffled_gene_names(rng, gene_list, N_EXTRA_GENES)
    arrays = tabular_case_arrays(seed, n_samples=N_SAMPLES, n_drugs=len(SMILES), n_genes=len(names), n_bits=4)
    arrays["expression"] = np.log1p(arrays["expression"]).astype(np.float32)
    arrays["gene_names"] = names
    arrays["gene_list"] = gene_list
    arrays["smiles"] = np.array(SMILES)
    arrays["edge_smiles"] = np.array(EDGE_SMILES)
    arrays["stop_mask"] = np.arange(len(arrays["train_pairs"])) % STOP_EVERY == STOP_EVERY - 1
    if len(np.unique(arrays["train_pairs"][:, 1])) != len(SMILES):
        msg = "every drug must occur in the training pairs"
        raise ValueError(msg)
    arrays.update(_forward_arrays(rng))
    meta = {"variants": VARIANTS, "forward": FORWARD_CONFIGS, "gene_list": GENE_LIST}
    return Case(model=NAME, seed=seed, arrays=arrays, meta=meta)


def _recorded_keys() -> list[str]:
    """Name the parameters compared after training.

    :returns: Keys of the ``DrugGraphNet`` ``state_dict``.
    """
    return [
        "conv1.lin.weight",
        "conv1.bias",
        "conv3.lin.weight",
        "drug_embed_fc.weight",
        "cell_fc1.weight",
        "combiner_fc2.bias",
        "output_fc.weight",
    ]


def steps() -> list[StepSpec]:
    """Declare the compared outputs.

    :returns: Graph, forward, feature, training and prediction steps.
    """
    specs = [
        StepSpec("graphs/x", "exact", description="node features of the molecular graphs"),
        StepSpec("graphs/edge_index", "exact", description="edges of the molecular graphs"),
        StepSpec("graphs/edge_attr", "exact", description="edge features of the molecular graphs"),
        StepSpec("graphs/n_nodes", "exact", description="atoms per molecule"),
        StepSpec("graphs/n_edges", "exact", description="directed edges per molecule"),
        StepSpec(
            "graphs/single_atom_shapes",
            "exact",
            scope=DEVIATION,
            description="shapes of edge_index and edge_attr (separated by -1) for a molecule without bonds",
            explanation=SINGLE_ATOM_EXPLANATION,
        ),
    ]
    specs += [
        StepSpec(f"forward/{config}/{mode}", "forward", description=f"{config} network, {mode} mode output")
        for config in FORWARD_CONFIGS
        for mode in ("eval", "train")
    ]
    for variant, setting in VARIANTS.items():
        table = [
            ("sample_ids", "exact", "ids of the samples whose features enter the network"),
            ("features_sample", "exact", "landmark-gene expression of those samples, in list order"),
            ("graphs_x", "exact", "node features of the graphs of the drugs, in id order"),
        ]
        specs += [
            StepSpec(name=f"{variant}/{quantity}", tolerance=tolerance, scope=PARITY, description=f"{variant}: {text}")
            for quantity, tolerance, text in table
        ]
        deviates = setting["early_stopping"] and not setting.get("improving", False)
        checks = [(f"params/{key}", f"{key} after training") for key in _recorded_keys()]
        checks.append(("predictions", "test predictions after the seeded short training"))
        for quantity, text in checks:
            specs.append(
                StepSpec(
                    name=f"{variant}/{quantity}",
                    tolerance="train",
                    scope=DEVIATION if deviates else PARITY,
                    description=f"{variant}: {text}",
                    explanation=BEST_EPOCH_EXPLANATION if deviates else "",
                )
            )
    return specs


def _forward_outputs(network_class: Any, graph_of: Any, case: Case) -> dict[str, np.ndarray]:
    """Run the forward probes of every configuration through *network_class*.

    :param network_class: ``DrugGraphNet`` of either side.
    :param graph_of: ``smiles -> Data`` of that side.
    :param case: The shared case.
    :returns: ``forward/<config>/{eval,train}``.
    """
    import torch
    from torch_geometric.data import Batch

    graphs = [graph_of(str(smiles)) for smiles in case.arrays["smiles"]]
    out: dict[str, np.ndarray] = {}
    for config, setting in case.meta["forward"].items():
        network = network_class(graphs[0].x.shape[1], N_CELL_FEATURES_FORWARD, **setting)
        load_weights(network, case, f"forward/{config}/w/")
        index = case.arrays[f"forward/{config}/graph_index"]
        batch = Batch.from_data_list([graphs[i] for i in index])
        cells = torch.as_tensor(case.arrays[f"forward/{config}/cells"])
        for mode, values in forward_pair(network, (batch, cells), DROPOUT_SEED).items():
            out[f"forward/{config}/{mode}"] = values
    return out


def _graph_outputs(graph_of: Any, case: Case) -> dict[str, np.ndarray]:
    """Build the graphs of the case's molecules and the single-atom edge case.

    :param graph_of: ``smiles -> Data`` of one side.
    :param case: The shared case.
    :returns: ``graphs/*`` arrays.
    """
    graphs = [graph_of(str(smiles)) for smiles in [*case.arrays["smiles"], *case.arrays["edge_smiles"]]]
    out = _graph_arrays(graphs, "graphs/")
    ion = graph_of("[Na+]")
    out["graphs/single_atom_shapes"] = np.array([*ion.edge_index.shape, -1, *ion.edge_attr.shape])
    return out


def _hyperparameters(setting: dict[str, Any]) -> dict[str, Any]:
    """Drop the variant keys that are not model hyperparameters.

    :param setting: Variant entry.
    :returns: Hyperparameters shared by both sides.
    """
    return {key: value for key, value in setting.items() if key not in _NON_HYPERPARAMETERS}


def run_v1(case: Case) -> dict[str, np.ndarray]:
    """Run the graphs, the forward probes and every variant with the 1.6.1 ``DrugGNN`` model.

    :param case: The shared case.
    :returns: Arrays keyed by the step names.
    """
    from drevalpy.datasets.dataset import FeatureDataset
    from drevalpy.datasets.featurizer.create_drug_graphs import _smiles_to_graph
    from drevalpy.models.DrugGNN.drug_gnn import DrugGNN, DrugGraphNet

    from tools.v1_parity._v1_runner import tabular_inputs

    out = _graph_outputs(_smiles_to_graph, case)
    out.update(_forward_outputs(DrugGraphNet, _smiles_to_graph, case))

    data = tabular_inputs(case, feature_dtype="float32")
    restrict_v1_genes(data.cell_line_input, case.arrays["gene_names"], case.arrays["gene_list"])
    drugs = FeatureDataset(
        features={
            str(did): {"drug_graph": _smiles_to_graph(str(smiles))}
            for did, smiles in zip(case.arrays["drug_ids"], case.arrays["smiles"], strict=True)
        }
    )
    stop = case.arrays["stop_mask"]

    for index, (variant, setting) in enumerate(case.meta["variants"].items()):
        fit_set, stop_set = data.train.copy(), data.train.copy()
        if setting["early_stopping"]:
            fit_set.mask(~stop)
            stop_set.mask(stop)
            used = np.unique(np.concatenate([fit_set.cell_line_ids, stop_set.cell_line_ids]))
        else:
            stop_set = None
            used = np.unique(fit_set.cell_line_ids)
        model = DrugGNN()
        model.build_model({**_hyperparameters(setting), "num_workers": 0, "gene_list": case.meta["gene_list"]})
        seed_all(setting["seed"] + 100 * index)
        model.train(fit_set, data.cell_line_input, drugs, stop_set)
        predictions = model.predict(data.test.cell_line_ids, data.test.drug_ids, data.cell_line_input, drugs)

        out[f"{variant}/sample_ids"] = used.astype(str)
        out[f"{variant}/features_sample"] = data.cell_line_input.get_feature_matrix("gene_expression", used)
        out[f"{variant}/graphs_x"] = np.concatenate(
            [drugs.features[d]["drug_graph"].x.numpy() for d in sorted(drugs.features)]
        )
        out[f"{variant}/predictions"] = np.asarray(predictions, dtype=np.float64)
        state = model.model.model.state_dict()
        for key in _recorded_keys():
            out[f"{variant}/params/{key}"] = state[key].cpu().numpy()
    return out


def run_v2(case: Case) -> dict[str, np.ndarray]:
    """Run the graphs, the forward probes and every variant with the v2 ``DrugGNN`` preset.

    :param case: The shared case.
    :returns: Arrays keyed like those of :func:`run_v1`.
    """
    from drevalpy.components.featurizers.drug.drug_graph import _smiles_to_graph
    from drevalpy.components.predictors.literature.druggnn.algorithm import DrugGraphNet
    from drevalpy.components.predictors.literature.druggnn.predictor import DrugGNNPredictor
    from drevalpy.types.data.split_mask import SplitMask
    from tools.v1_parity._v2_runner import tabular_inputs

    out = _graph_outputs(_smiles_to_graph, case)
    out.update(_forward_outputs(DrugGraphNet, _smiles_to_graph, case))

    data = tabular_inputs(case)
    data.dataset.response.var["canonical_smiles"] = list(case.arrays["smiles"].astype(str))
    data.dataset.mdata.uns["drug_graphs"] = {}
    shape = (len(case.arrays["sample_ids"]), len(case.arrays["drug_ids"]))
    stop = case.arrays["stop_mask"]

    for index, (variant, setting) in enumerate(case.meta["variants"].items()):
        if setting["early_stopping"]:
            fit_scope = SplitMask.from_pairs(case.arrays["train_pairs"][~stop], shape)
            stop_scope = SplitMask.from_pairs(case.arrays["train_pairs"][stop], shape)
        else:
            fit_scope, stop_scope = SplitMask.from_pairs(case.arrays["train_pairs"], shape), None
        model = v2_model(NAME, {**_hyperparameters(setting), "precision": 32})
        captured: dict[str, Any] = {}
        seed_all(setting["seed"] + 100 * index)
        with capture_fit(DrugGNNPredictor, captured):
            model.train(data.dataset, fit_scope, early_stopping_scope=stop_scope)
        predictions = model.predict(data.dataset, data.test)

        batch = captured["batch"]
        sample_order = np.argsort(batch.sample.entity_ids.astype(str), kind="stable")
        drug_order = np.argsort(batch.drug.entity_ids.astype(str), kind="stable")
        graphs = batch.drug.blocks["drug_graph"].values
        out[f"{variant}/sample_ids"] = batch.sample.entity_ids[sample_order].astype(str)
        out[f"{variant}/features_sample"] = batch.sample.blocks["gene_expression"].values[sample_order]
        out[f"{variant}/graphs_x"] = np.concatenate([graphs[i].x.numpy() for i in drug_order])
        out[f"{variant}/predictions"] = np.asarray(predictions, dtype=np.float64)
        state = model._pipeline._predictor._model.model.state_dict()
        for key in _recorded_keys():
            out[f"{variant}/params/{key}"] = state[key].cpu().numpy()
    return out
