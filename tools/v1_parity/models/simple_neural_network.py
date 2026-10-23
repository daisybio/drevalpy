"""SimpleNeuralNetwork parity case: 1.6.1 ``SimpleNeuralNetwork`` against the v2 ``SimpleNeuralNetwork`` preset.

Steps:

``forward/<config>/{eval,train}``
    The 1.6.1 ``FeedForwardNetwork`` and the v2 one with identical weights on the same probe, in eval mode and
    in train mode (batch statistics and dropout under the same torch seed). The weights are drawn in
    ``build_case`` per ``state_dict`` key of the v2 network and loaded with ``strict=True`` on both sides.
``<variant>/{sample_ids,features_sample,features_drug}``
    What enters the network: the arcsinh + ``StandardScaler`` expression (fit on the distinct training
    samples on both sides) and the fingerprints. The case holds the ``landmark_genes`` list (the
    preset's list and the 1.6.1 default) in shuffled order plus extra genes, so the gene-list selection is
    exercised; 1.6.1 selects the list while loading CSVs, which the in-memory harness skips, so its side
    receives the matrix reordered to the list.
``<variant>/{params/*,predictions}``
    A short seeded CPU training and the test predictions. 1.6.1's loaders are made single-process
    (``inline_loaders``); the base seed draw of the DataLoader iterator is the same single draw either way.

Variants: ``default`` (two hidden layers, so a batch norm layer is in the forward path; explicit early-stopping
set), ``es_plain`` (one hidden layer, early stopping) and ``no_es`` (one hidden layer, none: the training loss
is monitored). The fingerprints are continuous random values: a binary column that is constant in a batch has
the same zero gradient as the bias in front of a batch norm layer, and Adam turns its rounding noise into steps.
"""

from __future__ import annotations

import tempfile
from typing import Any

import numpy as np

from tools.v1_parity._bridge import Case, seed_all, tabular_case_arrays
from tools.v1_parity.compare import DEVIATION, PARITY, StepSpec
from tools.v1_parity.models._torch_common import (
    capture_fit,
    draw_weights,
    forward_pair,
    inline_loaders,
    load_weights,
    restrict_v1_genes,
    shuffled_gene_names,
    v2_model,
)

NAME = "SimpleNeuralNetwork"

GENE_LIST = "landmark_genes"
N_EXTRA_GENES = 12
N_SAMPLES = 24
N_DRUGS = 5
N_BITS = 10
STOP_EVERY = 5
PROBE_ROWS = 7
PROBE_INPUT_DIM = 14
DROPOUT_SEED = 11

VARIANTS: dict[str, dict[str, Any]] = {
    "default": {
        "units_per_layer": [12, 6],
        "dropout_prob": 0.3,
        "max_epochs": 4,
        "batch_size": 16,
        "patience": 5,
        "early_stopping": True,
        "seed": 3,
    },
    "es_plain": {
        "units_per_layer": [8],
        "dropout_prob": 0.2,
        "max_epochs": 4,
        "batch_size": 16,
        "patience": 2,
        "early_stopping": True,
        "seed": 4,
    },
    "no_es": {
        "units_per_layer": [8],
        "dropout_prob": 0.1,
        "max_epochs": 3,
        "batch_size": 16,
        "patience": 5,
        "early_stopping": False,
        "seed": 5,
    },
}
_NON_HYPERPARAMETERS = ("early_stopping", "seed", "batch_size", "patience")
FORWARD_CONFIGS: dict[str, dict[str, Any]] = {
    "two_hidden": {"units_per_layer": [10, 6], "dropout_prob": 0.3},
    "one_hidden": {"units_per_layer": [8], "dropout_prob": 0.1},
}

PRE_BATCH_NORM = ("params/fully_connected_layers.0.bias", "params/batch_norm_layers.0.running_mean", "predictions")
BATCH_NORM_EXPLANATION = (
    "A linear bias in front of a batch norm layer has an exactly zero gradient, so what Adam sees is float32 "
    "rounding noise, and Adam rescales it to a full learning-rate step (the same mechanism as in the PaccMann "
    "case). The running mean behind that bias, and through it the eval-mode predictions, inherit the drift: the "
    "EMA has not yet forgotten the bias after a few steps. The weights, the batch norm scale and the second layer "
    "do not depend on the bias in train mode and agree at the `train` class. Variants without a batch norm layer "
    "in the forward path (a single hidden layer: 1.6.1 and v2 both skip it) reproduce the predictions. Measured on "
    "2026-10-23 (variant `default`): training v2 itself on expression scaled by 1 + 3e-7 moves its test "
    "predictions by 6.0e-4, the bias by 6.1e-3 and the running mean by 3.2e-3, against 5.0e-4, 4.3e-3 and 1.6e-3 "
    "between 1.6.1 and v2."
)

FINDINGS = [
    "With a batch norm layer in the forward path (`units_per_layer` with two or more entries) a seeded fit does "
    "not reproduce the 1.6.1 eval-mode predictions to the `train` class (deviation steps of variant `default`); "
    "the cause is the zero gradient of the bias in front of the batch norm layer, not a logic difference "
    "(see `BATCH_NORM_EXPLANATION`). It is a property of the architecture that v2 inherits unchanged.",
    "The network is the 1.6.1 network, so a `state_dict` of one loads into the other with `strict=True` and the "
    "forward pass agrees at the `forward` class in eval and in train mode.",
    "The scaler is the 1.6.1 one: arcsinh, then a `StandardScaler` fit on the distinct training samples "
    "(`np.unique(output.cell_line_ids)`); the v2 `scaledGeneExpression` fits on the distinct samples too "
    "(`fit_on_unique_ids`).",
    "The v2 preset (commit 174ceea7, B5) sets `gene_list: landmark_genes` on "
    "`scaledGeneExpression`, the 1.6.1 default (`SimpleNeuralNetwork.gene_list`). The case does not override "
    "it: the v2 side reads the list from the preset, the 1.6.1 side from its class default, and the selected "
    "columns agree. Before B5 the preset had no gene list.",
    "The v2 predictor defaults (since B5 (174ceea7)) are the first grid point of 1.6.1: `units_per_layer` "
    "[32, 16, 8, 4], `dropout_prob` 0.3, `max_epochs` 100; before them they were [512, 256, 128], 0.2 and 50. "
    "The batch size (16) and the patience (5) were hard-coded in 1.6.1 and are v2 hyperparameters. The case sets "
    "all of them explicitly on both sides, so it does not depend on the defaults.",
    "1.6.1 loaders use 8 persistent worker processes; the case makes them single-process, which does not "
    "change the main-process RNG order. The drop-last rule agrees (`len > batch_size` against "
    "`batch_size < n_pairs` after clipping the batch size to the number of pairs).",
]


def _forward_arrays(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Draw weights and probes for every forward configuration from the v2 network's shapes.

    :param rng: Generator of the case.
    :returns: Arrays ``forward/<config>/w/<key>`` and ``forward/<config>/probe``.
    """
    from drevalpy.components.predictors.neural_network.network import FeedForwardNetwork

    arrays: dict[str, np.ndarray] = {}
    for config, setting in FORWARD_CONFIGS.items():
        network = FeedForwardNetwork(dict(setting), PROBE_INPUT_DIM)
        arrays.update(draw_weights(network, rng, f"forward/{config}/w/"))
        arrays[f"forward/{config}/probe"] = rng.normal(size=(PROBE_ROWS, PROBE_INPUT_DIM)).astype(np.float32)
    return arrays


def build_case(seed: int) -> Case:
    """Draw the tabular inputs (list genes in shuffled order plus extras) and the forward probes.

    :param seed: Seed of the input generator.
    :returns: The case.
    :raises ValueError: If a drug occurs in no training pair.
    """
    from drevalpy.components.featurizers.sample.gene_lists import resolve_gene_list

    gene_list = np.array(resolve_gene_list(GENE_LIST))
    rng = np.random.default_rng(seed + 1)
    names = shuffled_gene_names(rng, gene_list, N_EXTRA_GENES)
    arrays = tabular_case_arrays(seed, n_samples=N_SAMPLES, n_drugs=N_DRUGS, n_genes=len(names), n_bits=N_BITS)
    arrays["fingerprints"] = rng.normal(size=arrays["fingerprints"].shape).astype(np.float32)
    arrays["gene_names"] = names
    arrays["gene_list"] = gene_list
    arrays["stop_mask"] = np.arange(len(arrays["train_pairs"])) % STOP_EVERY == STOP_EVERY - 1
    if len(np.unique(arrays["train_pairs"][:, 1])) != N_DRUGS:
        msg = "every drug must occur in the training pairs"
        raise ValueError(msg)
    arrays.update(_forward_arrays(rng))
    meta = {"variants": VARIANTS, "forward": FORWARD_CONFIGS, "gene_list": GENE_LIST}
    return Case(model=NAME, seed=seed, arrays=arrays, meta=meta)


def _recorded_keys(setting: dict[str, Any]) -> list[str]:
    """Name the parameters compared after training.

    :param setting: Variant entry.
    :returns: ``state_dict`` keys present in the variant's network.
    """
    keys = ["fully_connected_layers.0.weight", "fully_connected_layers.0.bias"]
    if len(setting["units_per_layer"]) > 1:
        keys.append("fully_connected_layers.1.weight")
    return [*keys, "batch_norm_layers.0.running_mean", "batch_norm_layers.0.weight"]


def steps() -> list[StepSpec]:
    """Declare the compared outputs.

    :returns: Forward, feature, training and prediction steps.
    """
    specs = [
        StepSpec(f"forward/{config}/{mode}", "forward", description=f"{config} network, {mode} mode output")
        for config in FORWARD_CONFIGS
        for mode in ("eval", "train")
    ]
    for variant, setting in VARIANTS.items():
        table = [
            ("sample_ids", "exact", "ids of the samples whose features enter the network"),
            ("features_sample", "forward", "arcsinh + StandardScaler expression of those samples"),
            ("features_drug", "exact", "fingerprints of the drugs"),
            *[(f"params/{key}", "train", f"{key} after training") for key in _recorded_keys(setting)],
            ("predictions", "train", "test predictions after the seeded short training"),
        ]
        uses_norm = len(setting["units_per_layer"]) > 1
        for quantity, tolerance, text in table:
            drifts = uses_norm and quantity in PRE_BATCH_NORM
            specs.append(
                StepSpec(
                    name=f"{variant}/{quantity}",
                    tolerance=tolerance,
                    scope=DEVIATION if drifts else PARITY,
                    description=f"{variant}: {text}",
                    explanation=BATCH_NORM_EXPLANATION if drifts else "",
                )
            )
    return specs


def _probe_outputs(network_class: Any, case: Case) -> dict[str, np.ndarray]:
    """Run the forward probes through *network_class* with the case's weights.

    :param network_class: ``FeedForwardNetwork`` of either side.
    :param case: The shared case.
    :returns: ``forward/<config>/{eval,train}``.
    """
    import torch

    out: dict[str, np.ndarray] = {}
    for config, setting in case.meta["forward"].items():
        network = network_class(dict(setting), PROBE_INPUT_DIM)
        load_weights(network, case, f"forward/{config}/w/")
        probe = torch.as_tensor(case.arrays[f"forward/{config}/probe"])
        for mode, values in forward_pair(network, (probe,), DROPOUT_SEED).items():
            out[f"forward/{config}/{mode}"] = values
    return out


def _hyperparameters(setting: dict[str, Any]) -> dict[str, Any]:
    """Return the keys that are hyperparameters of the 1.6.1 model.

    :param setting: Variant entry.
    :returns: ``units_per_layer``, ``dropout_prob`` and ``max_epochs``.
    """
    return {key: value for key, value in setting.items() if key not in _NON_HYPERPARAMETERS}


def run_v1(case: Case) -> dict[str, np.ndarray]:
    """Run the forward probes and every variant with the 1.6.1 ``SimpleNeuralNetwork`` model.

    :param case: The shared case.
    :returns: Arrays keyed by the step names.
    """
    from drevalpy.models.SimpleNeuralNetwork import simple_neural_network, utils

    from tools.v1_parity._v1_runner import tabular_inputs

    out = _probe_outputs(utils.FeedForwardNetwork, case)
    data = tabular_inputs(case, feature_dtype="float32")
    restrict_v1_genes(data.cell_line_input, case.arrays["gene_names"], case.arrays["gene_list"])
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
        model = simple_neural_network.SimpleNeuralNetwork()
        model.build_model(_hyperparameters(setting))
        seed_all(setting["seed"] + 100 * index)
        with inline_loaders(utils), tempfile.TemporaryDirectory() as checkpoints:
            model.train(fit_set, data.cell_line_input, data.drug_input, stop_set, model_checkpoint_dir=checkpoints)
        predictions = model.predict(data.test.cell_line_ids, data.test.drug_ids, data.cell_line_input, data.drug_input)

        raw = data.cell_line_input.get_feature_matrix("gene_expression", used)
        out[f"{variant}/sample_ids"] = used.astype(str)
        out[f"{variant}/features_sample"] = model.gene_expression_scaler.transform(np.arcsinh(raw)).astype(np.float32)
        out[f"{variant}/features_drug"] = data.drug_input.get_feature_matrix("fingerprints", case.arrays["drug_ids"])
        out[f"{variant}/predictions"] = np.asarray(predictions, dtype=np.float64)
        state = model.model.state_dict()
        for key in _recorded_keys(setting):
            out[f"{variant}/params/{key}"] = state[key].cpu().numpy()
    return out


def run_v2(case: Case) -> dict[str, np.ndarray]:
    """Run the forward probes and every variant with the v2 ``SimpleNeuralNetwork`` preset.

    :param case: The shared case.
    :returns: Arrays keyed like those of :func:`run_v1`.
    """
    from drevalpy.components.predictors.neural_network.network import FeedForwardNetwork
    from drevalpy.components.predictors.neural_network.predictor import NeuralNetworkPredictor
    from drevalpy.types.data.split_mask import SplitMask
    from tools.v1_parity._v2_runner import tabular_inputs

    out = _probe_outputs(FeedForwardNetwork, case)
    data = tabular_inputs(case)
    shape = (len(case.arrays["sample_ids"]), len(case.arrays["drug_ids"]))
    stop = case.arrays["stop_mask"]

    for index, (variant, setting) in enumerate(case.meta["variants"].items()):
        if setting["early_stopping"]:
            fit_scope = SplitMask.from_pairs(case.arrays["train_pairs"][~stop], shape)
            stop_scope = SplitMask.from_pairs(case.arrays["train_pairs"][stop], shape)
        else:
            fit_scope, stop_scope = SplitMask.from_pairs(case.arrays["train_pairs"], shape), None
        model = v2_model(
            NAME,
            {**_hyperparameters(setting), "batch_size": setting["batch_size"], "patience": setting["patience"]},
        )
        captured: dict[str, Any] = {}
        seed_all(setting["seed"] + 100 * index)
        with capture_fit(NeuralNetworkPredictor, captured):
            model.train(data.dataset, fit_scope, early_stopping_scope=stop_scope)
        predictions = model.predict(data.dataset, data.test)

        batch = captured["batch"]
        sample_order = np.argsort(batch.sample.entity_ids.astype(str), kind="stable")
        drug_order = np.argsort(batch.drug.entity_ids.astype(str), kind="stable")
        out[f"{variant}/sample_ids"] = batch.sample.entity_ids[sample_order].astype(str)
        out[f"{variant}/features_sample"] = batch.sample.matrix[sample_order]
        out[f"{variant}/features_drug"] = batch.drug.matrix[drug_order]
        out[f"{variant}/predictions"] = np.asarray(predictions, dtype=np.float64)
        state = model._pipeline._predictor._model.state_dict()
        for key in _recorded_keys(setting):
            out[f"{variant}/params/{key}"] = state[key].cpu().numpy()
    return out
