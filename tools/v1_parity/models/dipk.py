"""DIPK parity case: 1.6.1 ``DIPK`` against the v2 ``DIPK`` zoo preset.

Steps:

``forward/<config>/{eval,train}``
    The 1.6.1 ``Predictor`` network and the v2 one with identical weights on the same probe (padded MolGNet
    features with a mask, expression embedding, BIONIC features), in eval mode and in train mode (attention and
    dense dropout under the same torch seed). The weights are drawn in ``build_case`` per ``state_dict`` key of
    the v2 network and loaded with ``strict=True`` on both sides.
``<variant>/zscore`` and ``<variant>/zscore_loader``
    The per-sample preprocessing in front of the autoencoder: ``log2(x + 1)`` for the linear-TPM datasets
    only, then ``(x - mean) / std(ddof=1)`` of every sample profile, on the landmark genes in list order. 1.6.1
    does this inside ``DIPKModel.load_cell_line_features`` while it reads the CSVs. ``zscore`` runs the same
    lambdas on the float32 matrix of the case (the path the training uses, because v2 holds float32); the
    ``zscore_loader`` step is what the real 1.6.1 loader returns from CSV files written by the case (float64),
    compared with the v2 preprocessing at the ``forward`` class. The v2 side has one pipeline, so both names
    hold its one result.
``<variant>/{sample_ids,drug_ids,molgnet,bionic}``
    What enters the network besides the embedding: the sample and drug ids, the MolGNet tensors of the drugs
    (written to ``MolGNet_<drug>.csv`` for 1.6.1) and the BIONIC features. 1.6.1 builds the latter in the real
    ``load_bionic_features`` from CSV files; v2 in the ``bionic`` featurizer, whose PPI table is replaced by the
    same synthetic table (``_load_ppi_data`` is the only network access).
``<variant>/{features_sample,params/*,predictions}``
    The autoencoder embedding of the samples (512 values, which depends on the autoencoder fit: unique training
    samples, noise, batch norm statistics), the parameters after the seeded short training and the test
    predictions. The predictor weights are not drawn from the torch RNG: 1.6.1 builds the network in
    ``build_model``, before the autoencoder, and v2 builds it in ``_fit``, after it, so a seed cannot give both
    the same initial weights. Both sides load the case's weights, and v2 builds under a saved and restored RNG
    state so that the later draws (shuffling, dropout) are in the same order as in 1.6.1.

``<variant>/features_sample_matrix``
    The embedding as v2 computes it (the whole matrix in one encoder call) against 1.6.1's row-wise one. They
    differ by about 1e-6 (different matrix kernels for a single row and for a matrix). The training itself
    feeds the v2 network the row-wise values (see ``_rowwise_encoding``), otherwise Adam turns that difference
    into a different trajectory through the weights whose gradient is almost zero.

Variants: ``default`` (log-scale dataset, complete PPI table), ``tpm`` (the dataset is linear TPM: 1.6.1
decides by the dataset name ``CCLE``, v2 by the ``expression_unit`` of the modality) and ``ppi_gaps`` (a few
genes of ``gene_list_sel.txt`` have no PPI embedding: the BIONIC aggregation differs, a deviation).
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from tools.v1_parity._bridge import Case, seed_all, tabular_case_arrays
from tools.v1_parity.compare import DEVIATION, PARITY, StepSpec
from tools.v1_parity.models._torch_common import (
    capture_fit,
    forward_pair,
    load_weights,
    restrict_v1_genes,
    shuffled_gene_names,
    v2_model,
)

NAME = "DIPK"

GENE_LIST = "landmark_genes"
N_EXTRA_GENES = 8
N_SAMPLES = 24
STOP_EVERY = 5
N_DRUGS = 4
ATOMS = (5, 8, 6, 11)
MOLGNET_DIM = 768
EMBED_DIM = 512
N_SELECTED = 300
N_PPI_GAPS = 30
N_PPI_EXTRA = 20
PROBE_ROWS = 5
PROBE_ATOMS = 7
DROPOUT_SEED = 17
INIT_PREFIX = "forward/heads2_fc3/w/"

_PREDICTOR: dict[str, Any] = {
    "heads": 2,
    "fc_layer_num": 3,
    "fc_layer_dim": [256, 128, 64, 32, 16, 1],
    "dropout_rate": 0.3,
}
_AUTOENCODER: dict[str, Any] = {
    "epochs_autoencoder": 3,
    "lr_autoencoder": 1e-3,
    "patience_autoencoder": 3,
    "batch_size_autoencoder": 8,
}
_TRAINING: dict[str, Any] = {"lr": 1e-3, "epochs": 3, "batch_size": 8, "patience": 10}
VARIANTS: dict[str, dict[str, Any]] = {
    "default": {**_PREDICTOR, **_AUTOENCODER, **_TRAINING, "dataset": "GDSC2", "seed": 1},
    "tpm": {**_PREDICTOR, **_AUTOENCODER, **_TRAINING, "dataset": "CCLE", "seed": 2, "tpm": True},
    "ppi_gaps": {**_PREDICTOR, **_AUTOENCODER, **_TRAINING, "dataset": "GDSC2", "seed": 3, "gaps": True},
}
_NON_HYPERPARAMETERS = ("dataset", "seed", "tpm", "gaps")

FORWARD_CONFIGS: dict[str, dict[str, Any]] = {
    "heads2_fc3": _PREDICTOR,
    "heads1_fc2": {"heads": 1, "fc_layer_num": 2, "fc_layer_dim": [256, 128, 64, 32, 16, 1], "dropout_rate": 0.1},
}
GAPS_EXPLANATION = (
    "1.6.1 takes the top `gene_add_num` (256) genes of `gene_list_sel.txt` by expression first and only then drops "
    "those without a PPI embedding, so a gene without one uses up a slot and the mean runs over fewer than 256 "
    "genes. v2 restricts the list to the genes that have an embedding before it takes the top 256. The two agree "
    "whenever every gene of the list has an embedding (variants `default` and `tpm`); with 30 of 300 genes "
    "missing they average different gene sets, and the trained parameters and the predictions downstream differ "
    "(the autoencoder embedding does not use BIONIC)."
)

FINDINGS = [
    "The predictor network is the 1.6.1 network apart from docstrings and type hints, so a `state_dict` loads into "
    "both with `strict=True` and the forward pass agrees at the `forward` class in eval and in train mode. The "
    "autoencoder (encoder, decoder, loss, noise `np.random.binomial(1, 0.2)`) is also unchanged.",
    "The v2 preset (commit 174ceea7, B5) is `dipkGeneExpression` with "
    "`gene_list: landmark_genes` plus `bionic` and `molgnet`. 1.6.1 has two defaults: the class attribute "
    "`gene_list` is `gene_expression_intersection` (2269 genes) but its `hyperparameters.yaml` sets "
    "`gene_list: landmark_genes`, which every experiment run uses; the preset follows the latter.",
    "The v2 preprocessing (`log2(x + 1)` for linear units only, then the per-sample z-score with `ddof=1`) "
    "reproduces `load_cell_line_features`: the float32 path is exact and the float64 CSV loader agrees at the "
    "`forward` class. 1.6.1 chose the log step by the dataset name (`CCLE`, `CTRPv1`, `CTRPv2`); v2 reads the "
    "`expression_unit` of the modality and takes a source without a unit as log scale.",
    "The autoencoder is fit on the unique training samples and validated on the unique early-stopping samples "
    "in both versions. v2 additionally moves the last training row to the validation rows when "
    "`len(train) % batch_size == 1` (`avoid_singleton_batch`), which 1.6.1 does not: there a trailing "
    "mini-batch of one row makes `BatchNorm1d` raise in training mode. The case keeps every batch larger.",
    "BIONIC: the PPI table and the gene list are the real artifacts in both versions (replaced here by a synthetic "
    "table). The aggregation is identical when every gene of the list has an embedding; with gaps v2 and 1.6.1 "
    "average different gene sets (deviation `ppi_gaps/*`, see the explanation there).",
    "v2 cannot read a precomputed ragged `molgnet_features` view: `Dataset.get_drug_features` goes through "
    "`_aligned_fetch`, which needs a 2-D float matrix, and raises `ValueError: setting an array element with a "
    "sequence` for the object array of per-drug tensors (the featurizer only catches `KeyError`). A `.h5mu` "
    "cannot store such a ragged `varm` entry either, so v2 always computes MolGNet from the SMILES with the "
    "`MolGNet.pt` artifact during the fit. 1.6.1 reads one `MolGNet_<drug>.csv` per drug. The case feeds the "
    "same tensors through the patched `_compute_molgnet_embedding`.",
    "Training loop: Adam without weight decay, MSE on the squeezed outputs, the best validation epoch "
    "checkpointed and reloaded, early stop when the epochs without improvement reach `patience`, the "
    "early-stopping loader shuffled, as in 1.6.1. The seeded runs agree at the `train` class once the "
    "predictor weights are loaded (see the module docstring for why a seed alone cannot do that).",
    "The attention key bias (`fc_k.bias`) has an exactly zero gradient (softmax is invariant to a constant "
    "shift per query), so Adam turns float32 rounding noise into full steps on it; it is not compared. The "
    "output does not depend on it.",
]


def _weights(network: Any, rng: np.random.Generator, prefix: str) -> dict[str, np.ndarray]:
    """Draw one float32 array per ``state_dict`` entry with a standard deviation of ``1 / sqrt(fan_in)``.

    :param network: v2 ``Predictor`` whose names and shapes define the keys.
    :param rng: Generator of the case.
    :param prefix: Array-name prefix.
    :returns: ``{prefix + key: array}``.
    """
    arrays: dict[str, np.ndarray] = {}
    for key, tensor in sorted(network.state_dict().items()):
        shape = tuple(tensor.shape)
        std = 1.0 / np.sqrt(shape[1]) if key.endswith(".weight") and len(shape) == 2 else 0.05
        arrays[f"{prefix}{key}"] = (std * rng.normal(size=shape)).astype(np.float32)
    return arrays


def _forward_arrays(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Draw weights and probes for every forward configuration from the v2 network's shapes.

    :param rng: Generator of the case.
    :returns: Arrays ``forward/<config>/w/<key>``, ``.../molgnet``, ``.../mask``, ``.../gene`` and ``.../bionic``.
    """
    from drevalpy.components.predictors.literature.dipk.model_utils import Predictor

    arrays: dict[str, np.ndarray] = {}
    for config, setting in FORWARD_CONFIGS.items():
        arrays.update(_weights(Predictor(**setting), rng, f"forward/{config}/w/"))
        lengths = rng.integers(2, PROBE_ATOMS + 1, size=PROBE_ROWS)
        lengths[0] = PROBE_ATOMS
        mask = np.arange(PROBE_ATOMS)[None, :] < lengths[:, None]
        molgnet = (0.5 * rng.normal(size=(PROBE_ROWS, PROBE_ATOMS, MOLGNET_DIM))).astype(np.float32)
        arrays[f"forward/{config}/molgnet"] = molgnet * mask[:, :, None]
        arrays[f"forward/{config}/mask"] = mask
        arrays[f"forward/{config}/gene"] = (0.5 * rng.normal(size=(PROBE_ROWS, EMBED_DIM))).astype(np.float32)
        arrays[f"forward/{config}/bionic"] = (0.5 * rng.normal(size=(PROBE_ROWS, EMBED_DIM))).astype(np.float32)
    return arrays


def _ppi_arrays(rng: np.random.Generator, gene_list: np.ndarray) -> dict[str, np.ndarray]:
    """Draw the synthetic PPI table and the gene selection of the BIONIC step.

    :param rng: Generator of the case.
    :param gene_list: Genes of the landmark list.
    :returns: ``sel_genes`` (the selection file), ``ppi_genes`` and ``ppi_values`` of the complete table,
        ``gap_genes`` (selection genes the ``ppi_gaps`` table lacks).
    """
    order = rng.permutation(len(gene_list))
    selected = gene_list[order[:N_SELECTED]]
    outside = gene_list[order[N_SELECTED : N_SELECTED + N_PPI_EXTRA]]
    ppi_genes = rng.permutation(np.concatenate([selected, outside]))
    values = rng.normal(size=(len(ppi_genes), EMBED_DIM)).astype(np.float32)
    return {
        "sel_genes": selected,
        "ppi_genes": ppi_genes,
        "ppi_values": values,
        "gap_genes": rng.choice(selected, size=N_PPI_GAPS, replace=False),
    }


def build_case(seed: int) -> Case:
    """Draw the tabular inputs, MolGNet tensors, PPI table and forward probes.

    :param seed: Seed of the input generator.
    :returns: The case.
    :raises ValueError: If a drug occurs in no training pair.
    """
    from drevalpy.components.featurizers.sample.gene_lists import resolve_gene_list

    gene_list = np.array(resolve_gene_list(GENE_LIST))
    rng = np.random.default_rng(seed + 1)
    names = shuffled_gene_names(rng, gene_list, N_EXTRA_GENES)
    arrays = tabular_case_arrays(seed, n_samples=N_SAMPLES, n_drugs=N_DRUGS, n_genes=len(names), n_bits=4)
    arrays["gene_names"] = names
    arrays["gene_list"] = gene_list
    arrays["stop_mask"] = np.arange(len(arrays["train_pairs"])) % STOP_EVERY == STOP_EVERY - 1
    if len(np.unique(arrays["train_pairs"][:, 1])) != N_DRUGS:
        msg = "every drug must occur in the training pairs"
        raise ValueError(msg)
    arrays["molgnet_atoms"] = np.array(ATOMS)
    arrays["molgnet_values"] = rng.normal(size=(sum(ATOMS), MOLGNET_DIM)).astype(np.float32)
    arrays.update(_ppi_arrays(rng, gene_list))
    arrays.update(_forward_arrays(rng))
    meta = {"variants": VARIANTS, "forward": FORWARD_CONFIGS, "gene_list": GENE_LIST}
    return Case(model=NAME, seed=seed, arrays=arrays, meta=meta)


def _recorded_keys() -> list[str]:
    """Name the parameters compared after training.

    :returns: Keys of the ``Predictor`` ``state_dict``.
    """
    return [
        "attention_layer.fc_layer_0.weight",
        "attention_layer.fc_layer_1.bias",
        "attention_layer.attention_0.fc_q.weight",
        "attention_layer.attention_0.fc_v.weight",
        "attention_layer.attention_1.fc_o.weight",
        "dense_layers.fc_layer_0.weight",
        "dense_layers.fc_input.weight",
        "dense_layers.fc_layers.0.weight",
        "dense_layers.fc_layers.2.bias",
        "dense_layers.fc_output.weight",
        "dense_layers.fc_output.bias",
    ]


def steps() -> list[StepSpec]:
    """Declare the compared outputs.

    :returns: Forward, preprocessing, feature, training and prediction steps.
    """
    specs = [
        StepSpec(f"forward/{config}/{mode}", "forward", description=f"{config} network, {mode} mode output")
        for config in FORWARD_CONFIGS
        for mode in ("eval", "train")
    ]
    for variant, setting in VARIANTS.items():
        gaps = bool(setting.get("gaps"))
        table = [
            ("sample_ids", "exact", "ids of the samples whose features enter the network", PARITY),
            ("drug_ids", "exact", "ids of the drugs whose MolGNet tensors enter the network", PARITY),
            ("molgnet", "exact", "MolGNet tensors of the drugs (concatenated, in id order)", PARITY),
            ("zscore", "exact", "log2 (linear datasets) and per-sample z-score of the landmark genes, float32", PARITY),
            ("zscore_loader", "forward", "the same from the real 1.6.1 CSV loader (float64)", PARITY),
            ("bionic", "forward", "BIONIC features of the samples", DEVIATION if gaps else PARITY),
            ("features_sample", "train", "autoencoder embedding of the samples after the seeded fit", PARITY),
            ("features_sample_matrix", "forward", "the same embedding encoded as one matrix, as v2 does", PARITY),
        ]
        table += [(f"params/{key}", "train", f"{key} after training", PARITY) for key in _recorded_keys()]
        table.append(("predictions", "train", "test predictions after the seeded short training", PARITY))
        for quantity, tolerance, text, scope in table:
            downstream = gaps and (quantity.startswith("params/") or quantity == "predictions")
            if downstream:
                scope = DEVIATION
            specs.append(
                StepSpec(
                    name=f"{variant}/{quantity}",
                    tolerance=tolerance,
                    scope=scope,
                    description=f"{variant}: {text}",
                    explanation=GAPS_EXPLANATION if scope == DEVIATION else "",
                )
            )
    return specs


def _forward_outputs(network_class: Any, case: Case) -> dict[str, np.ndarray]:
    """Run the forward probes of every configuration through *network_class*.

    :param network_class: ``Predictor`` of either side.
    :param case: The shared case.
    :returns: ``forward/<config>/{eval,train}``.
    """
    import torch

    out: dict[str, np.ndarray] = {}
    for config, setting in case.meta["forward"].items():
        network = network_class(**setting)
        load_weights(network, case, f"forward/{config}/w/")
        inputs = (
            torch.as_tensor(case.arrays[f"forward/{config}/molgnet"]),
            torch.as_tensor(case.arrays[f"forward/{config}/gene"]),
            torch.as_tensor(case.arrays[f"forward/{config}/bionic"]),
            torch.as_tensor(case.arrays[f"forward/{config}/mask"]),
        )
        for mode, values in forward_pair(network, inputs, DROPOUT_SEED).items():
            out[f"forward/{config}/{mode}"] = values
    return out


def _hyperparameters(setting: dict[str, Any]) -> dict[str, Any]:
    """Drop the variant keys that are not model hyperparameters.

    :param setting: Variant entry.
    :returns: Hyperparameters shared by both sides.
    """
    return {key: value for key, value in setting.items() if key not in _NON_HYPERPARAMETERS}


def _ppi_table(case: Case, gaps: bool) -> tuple[np.ndarray, list[str], set[str]]:
    """Return the PPI embeddings, their gene names and the selection set of a variant.

    :param case: The shared case.
    :param gaps: Whether the genes in ``gap_genes`` are missing from the table.
    :returns: ``(values float64, gene names, selection set)``; the selection file always lists every gene.
    """
    names = [str(g) for g in case.arrays["ppi_genes"]]
    values = case.arrays["ppi_values"].astype(np.float64)
    if gaps:
        missing = {str(g) for g in case.arrays["gap_genes"]}
        keep = [i for i, g in enumerate(names) if g not in missing]
        names, values = [names[i] for i in keep], values[keep]
    return values, names, {str(g) for g in case.arrays["sel_genes"]}


def _write_ppi_files(directory: Path, case: Case, gaps: bool) -> None:
    """Write the PPI table and the gene selection file that both versions read.

    :param directory: Target directory.
    :param case: The shared case.
    :param gaps: Whether the table lacks the ``gap_genes``.
    """
    import pandas as pd

    values, names, selected = _ppi_table(case, gaps)
    pd.DataFrame(values, index=names).to_csv(directory / "human_ppi_features.tsv", sep="\t")
    (directory / "gene_list_sel.txt").write_text("".join(f"{g}\n" for g in sorted(selected)))


def _write_v1_tree(root: Path, case: Case, setting: dict[str, Any]) -> None:
    """Write the CSV files that the 1.6.1 loaders read for one variant.

    :param root: Data path of the variant.
    :param case: The shared case.
    :param setting: Variant entry.
    """
    import pandas as pd

    dataset = root / setting["dataset"]
    (dataset / "DIPK_features" / "Drugs").mkdir(parents=True)
    (root / "meta" / "gene_lists").mkdir(parents=True)
    expression = pd.DataFrame(
        case.arrays["expression"].astype(np.float64), index=case.arrays["sample_ids"], columns=case.arrays["gene_names"]
    )
    expression.insert(0, "cellosaurus_id", [f"CVCL_{i:04d}" for i in range(len(expression))])
    expression.index.name = "cell_line_name"
    expression.to_csv(dataset / "gene_expression.csv")
    pd.DataFrame({"Symbol": case.arrays["gene_list"]}).to_csv(root / "meta" / "gene_lists" / f"{GENE_LIST}.csv")
    _write_ppi_files(dataset / "DIPK_features", case, bool(setting.get("gaps")))
    start = 0
    for drug, atoms in zip(case.arrays["drug_ids"], case.arrays["molgnet_atoms"], strict=True):
        block = case.arrays["molgnet_values"][start : start + atoms].astype(np.float64)
        pd.DataFrame(block).to_csv(dataset / "DIPK_features" / "Drugs" / f"MolGNet_{drug}.csv", sep="\t")
        start += int(atoms)


def run_v1(case: Case) -> dict[str, np.ndarray]:
    """Run the forward probes and every variant with the 1.6.1 ``DIPK`` model.

    :param case: The shared case.
    :returns: Arrays keyed by the step names.
    """
    import copy

    from drevalpy.models.DIPK.dipk import DIPKModel
    from drevalpy.models.DIPK.model_utils import Predictor

    from tools.v1_parity._v1_runner import tabular_inputs

    out = _forward_outputs(Predictor, case)

    data = tabular_inputs(case, feature_dtype="float32")
    restrict_v1_genes(data.cell_line_input, case.arrays["gene_names"], case.arrays["gene_list"])
    stop = case.arrays["stop_mask"]
    sample_ids = np.array(sorted(data.cell_line_input.features))

    for index, (variant, setting) in enumerate(case.meta["variants"].items()):
        fit_set, stop_set = data.train.copy(), data.train.copy()
        fit_set.mask(~stop)
        stop_set.mask(stop)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_v1_tree(root, case, setting)
            model = DIPKModel()
            model.build_model({**_hyperparameters(setting), "gene_list": case.meta["gene_list"]})
            load_weights(model.model, case, INIT_PREFIX)
            loaded = model.load_cell_line_features(str(root), setting["dataset"])
            drugs = model.load_drug_features(str(root), setting["dataset"])

        float32 = copy.deepcopy(data.cell_line_input)
        if setting.get("tpm"):
            float32.apply(function=lambda x: np.log2(x + 1), view="gene_expression")
        float32.apply(function=lambda x: (x - np.mean(x)) / np.std(x, ddof=1), view="gene_expression")
        out[f"{variant}/zscore"] = float32.get_feature_matrix("gene_expression", sample_ids)
        out[f"{variant}/zscore_loader"] = loaded.get_feature_matrix("gene_expression", sample_ids).astype(np.float32)
        for sample_id in sample_ids:
            loaded.features[sample_id]["gene_expression"] = float32.features[sample_id]["gene_expression"]

        seed_all(setting["seed"] + 100 * index)
        model.train(fit_set, loaded, drugs, stop_set, model_checkpoint_dir=str(Path(tempfile.mkdtemp())))
        predictions = model.predict(data.test.cell_line_ids, data.test.drug_ids, loaded, drugs)

        used = np.unique(np.concatenate([fit_set.cell_line_ids, stop_set.cell_line_ids]))
        used_drugs = np.unique(np.concatenate([fit_set.drug_ids, stop_set.drug_ids]))
        out[f"{variant}/sample_ids"] = used.astype(str)
        out[f"{variant}/drug_ids"] = used_drugs.astype(str)
        out[f"{variant}/molgnet"] = np.concatenate([drugs.features[d]["molgnet_features"] for d in used_drugs])
        out[f"{variant}/bionic"] = loaded.get_feature_matrix("bionic_features", used).astype(np.float32)
        out[f"{variant}/features_sample"] = loaded.get_feature_matrix("gene_expression", used)
        out[f"{variant}/features_sample_matrix"] = out[f"{variant}/features_sample"]
        out[f"{variant}/predictions"] = np.asarray(predictions, dtype=np.float64)
        state = model.model.state_dict()
        for key in _recorded_keys():
            out[f"{variant}/params/{key}"] = state[key].cpu().numpy()
    return out


@contextmanager
def _case_initial_weights(case: Case) -> Iterator[None]:
    """Make the v2 predictor start from the case's weights without moving the torch RNG.

    :param case: The shared case.
    :yields: Nothing; the patch is undone on exit.
    """
    import torch

    from drevalpy.components.predictors.literature.dipk.predictor import DIPKPredictor

    original = DIPKPredictor._build_model

    def build(self: Any) -> Any:
        state = torch.get_rng_state()
        network = original(self)
        load_weights(network, case, INIT_PREFIX)
        torch.set_rng_state(state)
        return network

    DIPKPredictor._build_model = build
    try:
        yield
    finally:
        DIPKPredictor._build_model = original


@contextmanager
def _synthetic_ppi(case: Case, gaps: bool) -> Iterator[None]:
    """Serve the case's PPI files to the ``bionic`` featurizer instead of the downloaded artifacts.

    The featurizer's own ``_load_ppi_data`` still parses the files, so the parse is the one 1.6.1 applies to
    the same text.

    :param case: The shared case.
    :param gaps: Whether the table lacks the ``gap_genes``.
    :yields: Nothing; the patch is undone on exit.
    """
    from drevalpy.components.featurizers.sample import bionic

    original = bionic.get_artifact
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _write_ppi_files(root, case, gaps)
        bionic.get_artifact = lambda name: root / name
        try:
            yield
        finally:
            bionic.get_artifact = original


@contextmanager
def _rowwise_encoding(sink: list[tuple[np.ndarray, np.ndarray]]) -> Iterator[None]:
    """Encode the samples one profile at a time, like 1.6.1, and keep what the whole-matrix call returns.

    1.6.1 encodes every sample on its own (``FeatureDataset.apply`` hands a single vector to the encoder),
    v2 encodes the whole matrix in one call. The two evaluate the first layer with different matrix kernels
    and differ by about 1e-6, which Adam turns into a different trajectory through the weights whose
    gradient is almost zero. The case therefore feeds the network the row-wise values on the v2 side and
    records the batched ones in *sink* for the ``features_sample_matrix`` step.

    :param sink: List receiving ``(row-wise, whole-matrix)`` results of every encoder call.
    :yields: Nothing; the patch is undone on exit.
    """
    from drevalpy.components.predictors.literature.dipk import gene_expression_encoder

    original = gene_expression_encoder.encode_gene_expression

    def encode(matrix: np.ndarray, encoder: Any) -> np.ndarray:
        rows = np.stack([original(row, encoder) for row in matrix])
        sink.append((rows, original(matrix, encoder)))
        return rows

    gene_expression_encoder.encode_gene_expression = encode
    try:
        yield
    finally:
        gene_expression_encoder.encode_gene_expression = original


@contextmanager
def _synthetic_molgnet(case: Case) -> Iterator[None]:
    """Replace the MolGNet network of the ``molgnet`` featurizer by the case's tensors.

    v2 cannot read a precomputed ragged ``molgnet_features`` view (``Dataset.get_drug_features`` assumes a 2-D
    float matrix), so the featurizer always computes the tensor from the SMILES. The case uses the drug id as
    the SMILES and returns that drug's tensor from the patched ``_compute_molgnet_embedding``.

    :param case: The shared case.
    :yields: Nothing; the patch is undone on exit.
    """
    from drevalpy.components.featurizers.drug import molgnet

    tensors: dict[str, np.ndarray] = {}
    start = 0
    for drug, atoms in zip(case.arrays["drug_ids"], case.arrays["molgnet_atoms"], strict=True):
        tensors[str(drug)] = case.arrays["molgnet_values"][start : start + atoms]
        start += int(atoms)
    original = molgnet._compute_molgnet_embedding
    molgnet._compute_molgnet_embedding = lambda smiles: tensors[smiles]
    try:
        yield
    finally:
        molgnet._compute_molgnet_embedding = original


def run_v2(case: Case) -> dict[str, np.ndarray]:
    """Run the forward probes and every variant with the v2 ``DIPK`` preset.

    :param case: The shared case.
    :returns: Arrays keyed like those of :func:`run_v1`.
    """
    from drevalpy.components.featurizers.sample.dipk_gene_expression import DIPKGeneExpressionFeaturizer
    from drevalpy.components.predictors.literature.dipk.model_utils import Predictor
    from drevalpy.components.predictors.literature.dipk.predictor import DIPKPredictor
    from drevalpy.types.data.feature_source import SampleFeatureSource
    from drevalpy.types.data.split_mask import SplitMask
    from tools.v1_parity._v2_runner import tabular_inputs

    out = _forward_outputs(Predictor, case)

    data = tabular_inputs(case)
    data.dataset.mdata.mod["response"].var["canonical_smiles"] = [str(d) for d in case.arrays["drug_ids"]]
    shape = (len(case.arrays["sample_ids"]), len(case.arrays["drug_ids"]))
    stop = case.arrays["stop_mask"]
    names = [str(g) for g in case.arrays["gene_names"]]
    columns = [names.index(str(g)) for g in case.arrays["gene_list"]]
    sample_ids = np.array(sorted(case.arrays["sample_ids"]))
    order = [list(case.arrays["sample_ids"]).index(s) for s in sample_ids]

    for index, (variant, setting) in enumerate(case.meta["variants"].items()):
        expression = data.dataset.mdata.mod["gene_expression"]
        if setting.get("tpm"):
            expression.uns["expression_unit"] = "tpm"
        else:
            expression.uns.pop("expression_unit", None)
        fit_scope = SplitMask.from_pairs(case.arrays["train_pairs"][~stop], shape)
        stop_scope = SplitMask.from_pairs(case.arrays["train_pairs"][stop], shape)
        model = v2_model(NAME, _hyperparameters(setting))

        raw = case.arrays["expression"][order][:, columns].astype(np.float32)
        featurizer = DIPKGeneExpressionFeaturizer(gene_list=case.meta["gene_list"])
        scored = featurizer._preprocess(SampleFeatureSource(data.dataset, sample_ids), raw)
        out[f"{variant}/zscore"] = scored
        out[f"{variant}/zscore_loader"] = scored

        captured: dict[str, Any] = {}
        encoded: list[tuple[np.ndarray, np.ndarray]] = []
        seed_all(setting["seed"] + 100 * index)
        with (
            capture_fit(DIPKPredictor, captured),
            _rowwise_encoding(encoded),
            _case_initial_weights(case),
            _synthetic_ppi(case, bool(setting.get("gaps"))),
            _synthetic_molgnet(case),
        ):
            model.train(data.dataset, fit_scope, early_stopping_scope=stop_scope)
            predictions = model.predict(data.dataset, data.test)

        batch = captured["batch"]
        sample_order = np.argsort(batch.sample.entity_ids.astype(str), kind="stable")
        drug_order = np.argsort(batch.drug.entity_ids.astype(str), kind="stable")
        out[f"{variant}/sample_ids"] = batch.sample.entity_ids[sample_order].astype(str)
        out[f"{variant}/drug_ids"] = batch.drug.entity_ids[drug_order].astype(str)
        tensors = batch.drug.blocks["molgnet_features"].values
        out[f"{variant}/molgnet"] = np.concatenate([np.asarray(tensors[i], dtype=np.float32) for i in drug_order])
        out[f"{variant}/bionic"] = batch.sample.blocks["bionic_features"].values[sample_order]
        rows = batch.sample.blocks["gene_expression"].values
        out[f"{variant}/features_sample"] = rows[sample_order]
        whole = next(
            matrix for single, matrix in encoded if single.shape == rows.shape and np.array_equal(single, rows)
        )
        out[f"{variant}/features_sample_matrix"] = whole[sample_order]
        out[f"{variant}/predictions"] = np.asarray(predictions, dtype=np.float64)
        state = model._pipeline._predictor._model.state_dict()
        for key in _recorded_keys():
            out[f"{variant}/params/{key}"] = state[key].cpu().numpy()
    return out
