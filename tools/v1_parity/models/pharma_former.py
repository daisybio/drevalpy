"""PharmaFormer parity case: 1.6.1 ``PharmaFormer`` against the v2 ``PharmaFormer`` zoo preset.

Steps:

``forward/<config>/{eval,train}``
    The 1.6.1 ``CombinedModel`` and the v2 one with identical weights on the same probe, in eval mode and in
    train mode (transformer and head dropout under the same torch seed). The weights are drawn in
    ``build_case`` per ``state_dict`` key of the v2 network and loaded with ``strict=True`` on both sides.
``<variant>/{sample_ids,features_sample}``
    What enters the network on the expression side: the landmark genes in list order after the
    ``StandardScaler`` then ``MinMaxScaler`` of PharmaFormer, fit on the unique training samples. The case
    holds ``landmark_genes`` (the preset's and the 1.6.1 default) in shuffled order plus extra genes, so the
    v2 selection is exercised; 1.6.1 selects the list while loading CSVs, which the in-memory harness
    skips, so its side receives the matrix reordered to the list.
``<variant>/{drug_ids,bpe}``
    The BPE-encoded SMILES rows of the training drugs. 1.6.1 reads them from a CSV written by
    ``datasets/featurizer/create_pharmaformer_drug_embeddings.py``, which learns the BPE codes on every drug of
    the dataset; the case runs that script and reads the CSV back. v2 learns the codes in the drug
    featurizer from the training drugs.
``<variant>/bpe_test``
    The BPE rows of the drugs of the test pairs, which for ``unseen_drug`` include a drug the v2 codes never saw.
``<variant>/{params/*,predictions}``
    A short seeded CPU training and the test predictions.

Variants: ``default`` (explicit early-stopping set, the validation loss improves), ``patience`` (a large
learning rate and patience 1 make the loop stop early and reload the best epoch on both sides) and
``unseen_drug`` (one drug is absent from the training pairs, so v2 learns the BPE codes without it while
1.6.1 learned them on every drug: a deviation of the codes and of everything downstream).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
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

NAME = "PharmaFormer"

GENE_LIST = "landmark_genes"
N_EXTRA_GENES = 8
N_SAMPLES = 24
STOP_EVERY = 5
PROBE_ROWS = 6
DROPOUT_SEED = 11
BPE_SYMBOLS = 40
BPE_LENGTH = 128
SMILES = (
    "CCO",
    "CC(=O)Oc1ccccc1C(=O)O",
    "C[C@@H](N)C(=O)O",
    "ClCCBr",
    "c1ccc2c(c1)[nH]c1ccccc12",
    "CC(=O)[O-].[Na+]",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
)

_NETWORK: dict[str, Any] = {
    "gene_hidden_size": 16,
    "drug_hidden_size": 16,
    "feature_dim": 8,
    "nhead": 2,
    "num_layers": 2,
    "dim_feedforward": 24,
    "dropout": 0.1,
}
VARIANTS: dict[str, dict[str, Any]] = {
    "default": {**_NETWORK, "lr": 1e-3, "epochs": 3, "batch_size": 8, "patience": 10, "seed": 1},
    "patience": {**_NETWORK, "lr": 0.05, "epochs": 5, "batch_size": 8, "patience": 1, "seed": 3},
    "unseen_drug": {**_NETWORK, "lr": 1e-3, "epochs": 3, "batch_size": 9, "patience": 10, "seed": 2, "unseen": True},
}
_NON_HYPERPARAMETERS = ("seed", "unseen")

FORWARD_CONFIGS: dict[str, dict[str, Any]] = {
    "two_layers": {**_NETWORK, "gene_input_size": 9},
    "wide": {
        "gene_input_size": 7,
        "gene_hidden_size": 24,
        "drug_hidden_size": 8,
        "feature_dim": 4,
        "nhead": 4,
        "num_layers": 1,
        "dim_feedforward": 12,
        "dropout": 0.3,
    },
}
UNSEEN_EXPLANATION = (
    "1.6.1 learns the BPE codes once on every drug of the dataset (`create_pharmaformer_drug_embeddings`, a "
    "script run before any split), so a drug that is only tested is encoded with codes that saw its SMILES. "
    "v2 learns them in the drug featurizer from the drugs of the training fold only (documented in "
    "`docs/explanation/literature_models.rst`), so the same SMILES is encoded differently and every number "
    "downstream differs."
)

FINDINGS = [
    "The network is the 1.6.1 network apart from docstrings, so a `state_dict` loads into both with "
    "`strict=True` and the forward pass agrees at the `forward` class in eval and in train mode.",
    "The v2 preset (commit 174ceea7, B5) is `pharmaFormerGeneExpression` with "
    "`gene_list: landmark_genes`, the 867-gene list of 1.6.1 (`PharmaFormerModel.gene_list`). The case does not "
    "override it; the selected columns agree (`*/features_sample`). The v2 featurizer fits both scalers on the "
    "unique training samples (`fit_on_unique_ids`), as 1.6.1 fits them on `np.unique(output.cell_line_ids)`.",
    "The training loop is the 1.6.1 loop: Adam without weight decay, MSE on `outputs.squeeze()`, the best "
    "validation epoch checkpointed and reloaded, early stop when the epochs without improvement reach "
    "`patience`. Both sides draw the shuffling permutation after the module initialization and a DataLoader "
    "base seed, so the torch RNG order is identical and the seeded runs agree at the `train` class.",
    "Both versions use one `dropout` value for the encoder layers and the head and no extra dropout (the "
    "documented omission concerns the published fine-tuning variant, which neither 1.6.1 nor v2 implements).",
    "The BPE codes are learned on every drug in 1.6.1 and on the training drugs in v2; with all drugs "
    "in training the codes and the encoded rows are identical (`*/bpe`), with a drug held out of training "
    "its test row and the predictions on it differ (deviation `unseen_drug/bpe_test` and `unseen_drug/predictions`; "
    "the rows of the training drugs and the trained parameters still agree in this case because the held-out "
    "drug does not change their merges).",
    "Both predictors require early-stopping data; v2 raises `ValueError` without it like 1.6.1.",
    "1.6.1 crashes when the last batch of an epoch (training or early-stopping) holds a single pair: "
    "`outputs.squeeze()` is 0-d and `np.concatenate` of the collected predictions raises `ValueError` "
    "(measured with 17 early-stopping pairs and batch size 8). v2 trains the same expression through "
    "`run_epoch`, which collects no predictions and so does not crash, but the 0-d against `(1,)` MSE "
    "broadcasts with a `UserWarning` as in 1.6.1. The case keeps every batch larger than one pair.",
]


def _forward_arrays(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Draw weights and probes for every forward configuration from the v2 network's shapes.

    :param rng: Generator of the case.
    :returns: Arrays ``forward/<config>/w/<key>``, ``.../genes`` and ``.../bpe``.
    """
    from drevalpy.components.predictors.literature.pharmaformer.model_utils import CombinedModel

    arrays: dict[str, np.ndarray] = {}
    for config, setting in FORWARD_CONFIGS.items():
        network = CombinedModel(**setting)
        arrays.update(draw_weights(network, rng, f"forward/{config}/w/", scale=0.3))
        arrays[f"forward/{config}/genes"] = rng.random(size=(PROBE_ROWS, setting["gene_input_size"])).astype(np.float32)
        arrays[f"forward/{config}/bpe"] = rng.integers(0, 120, size=(PROBE_ROWS, BPE_LENGTH)).astype(np.float32)
    return arrays


def build_case(seed: int) -> Case:
    """Draw the tabular inputs (list genes in shuffled order plus extras), the SMILES and the forward probes.

    :param seed: Seed of the input generator.
    :returns: The case.
    :raises ValueError: If a drug occurs in no training pair or the unseen drug has no test pair.
    """
    from drevalpy.components.featurizers.sample.gene_lists import resolve_gene_list

    gene_list = np.array(resolve_gene_list(GENE_LIST))
    rng = np.random.default_rng(seed + 1)
    names = shuffled_gene_names(rng, gene_list, N_EXTRA_GENES)
    arrays = tabular_case_arrays(seed, n_samples=N_SAMPLES, n_drugs=len(SMILES), n_genes=len(names), n_bits=4)
    arrays["gene_names"] = names
    arrays["gene_list"] = gene_list
    arrays["smiles"] = np.array(SMILES)
    arrays["stop_mask"] = np.arange(len(arrays["train_pairs"])) % STOP_EVERY == STOP_EVERY - 1
    if len(np.unique(arrays["train_pairs"][:, 1])) != len(SMILES):
        msg = "every drug must occur in the training pairs"
        raise ValueError(msg)
    unseen = int(arrays["test_pairs"][-1, 1])
    arrays["unseen_drug"] = np.array(unseen)
    arrays["unseen_stop_mask"] = (
        np.arange(int((arrays["train_pairs"][:, 1] != unseen).sum())) % STOP_EVERY == STOP_EVERY - 1
    )
    arrays.update(_forward_arrays(rng))
    meta = {"variants": VARIANTS, "forward": FORWARD_CONFIGS, "gene_list": GENE_LIST}
    return Case(model=NAME, seed=seed, arrays=arrays, meta=meta)


def _recorded_keys() -> list[str]:
    """Name the parameters compared after training.

    :returns: Keys of the ``CombinedModel`` ``state_dict``.
    """
    return [
        "feature_extractor.gene_fc1.weight",
        "feature_extractor.gene_fc2.bias",
        "feature_extractor.smiles_fc.weight",
        "transformer.transformer_encoder.layers.0.self_attn.in_proj_weight",
        "transformer.transformer_encoder.layers.0.self_attn.out_proj.weight",
        "transformer.transformer_encoder.layers.0.linear1.weight",
        "transformer.transformer_encoder.layers.1.norm2.weight",
        "transformer.output.0.weight",
        "transformer.output.3.bias",
    ]


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
        deviates = bool(setting.get("unseen"))
        table = [
            ("sample_ids", "exact", "ids of the samples whose features enter the network", PARITY),
            ("features_sample", "forward", "scaled landmark-gene expression of those samples, in list order", PARITY),
            ("drug_ids", "exact", "ids of the drugs whose BPE rows enter the network", PARITY),
            ("bpe", "exact", "BPE-encoded SMILES rows of the training drugs, in id order", PARITY),
            ("bpe_test", "exact", "BPE rows of the drugs of the test pairs, in id order", PARITY),
        ]
        checks = [(f"params/{key}", "train", f"{key} after training") for key in _recorded_keys()]
        table += [(q, t, text, PARITY) for q, t, text in checks]
        table.append(("predictions", "train", "test predictions after the seeded short training", PARITY))
        for quantity, tolerance, text, scope in table:
            if deviates and quantity in ("bpe_test", "predictions"):
                scope = DEVIATION
            specs.append(
                StepSpec(
                    name=f"{variant}/{quantity}",
                    tolerance=tolerance,
                    scope=scope,
                    description=f"{variant}: {text}",
                    explanation=UNSEEN_EXPLANATION if scope == DEVIATION else "",
                )
            )
    return specs


def _forward_outputs(network_class: Any, case: Case) -> dict[str, np.ndarray]:
    """Run the forward probes of every configuration through *network_class*.

    :param network_class: ``CombinedModel`` of either side.
    :param case: The shared case.
    :returns: ``forward/<config>/{eval,train}``.
    """
    import torch

    out: dict[str, np.ndarray] = {}
    for config, setting in case.meta["forward"].items():
        network = network_class(**setting)
        load_weights(network, case, f"forward/{config}/w/")
        genes = torch.as_tensor(case.arrays[f"forward/{config}/genes"])
        bpe = torch.as_tensor(case.arrays[f"forward/{config}/bpe"])
        for mode, values in forward_pair(network, (genes, bpe), DROPOUT_SEED).items():
            out[f"forward/{config}/{mode}"] = values
    return out


def _hyperparameters(setting: dict[str, Any]) -> dict[str, Any]:
    """Drop the variant keys that are not model hyperparameters.

    :param setting: Variant entry.
    :returns: Hyperparameters shared by both sides.
    """
    return {key: value for key, value in setting.items() if key not in _NON_HYPERPARAMETERS}


def _v1_bpe_rows(case: Case) -> dict[str, np.ndarray]:
    """Run the 1.6.1 drug-embedding script on every SMILES of the case and read its CSV back.

    :param case: The shared case.
    :returns: ``{drug_id: encoded row}``.
    """
    import tempfile
    from pathlib import Path

    import pandas as pd
    from drevalpy.datasets.featurizer.create_pharmaformer_drug_embeddings import create_pharmaformer_drug_embeddings

    with tempfile.TemporaryDirectory() as root:
        directory = Path(root) / "toy"
        directory.mkdir()
        pd.DataFrame({"pubchem_id": case.arrays["drug_ids"], "canonical_smiles": case.arrays["smiles"]}).to_csv(
            directory / "drug_smiles.csv", index=False
        )
        create_pharmaformer_drug_embeddings(root, "toy", num_symbols=BPE_SYMBOLS, max_length=BPE_LENGTH)
        table = pd.read_csv(directory / "drug_bpe_smiles.csv", dtype={"pubchem_id": str})
    return {str(row[0]): np.asarray(row[1:], dtype=np.float64) for row in table.itertuples(index=False)}


def run_v1(case: Case) -> dict[str, np.ndarray]:
    """Run the forward probes and every variant with the 1.6.1 ``PharmaFormer`` model.

    :param case: The shared case.
    :returns: Arrays keyed by the step names.
    """
    import tempfile

    from drevalpy.datasets.dataset import FeatureDataset
    from drevalpy.models.PharmaFormer.model_utils import CombinedModel
    from drevalpy.models.PharmaFormer.pharmaformer import PharmaFormerModel

    from tools.v1_parity._v1_runner import tabular_inputs

    out = _forward_outputs(CombinedModel, case)

    data = tabular_inputs(case, feature_dtype="float32")
    restrict_v1_genes(data.cell_line_input, case.arrays["gene_names"], case.arrays["gene_list"])
    rows = _v1_bpe_rows(case)
    drugs = FeatureDataset(
        features={str(did): {"bpe_smiles": rows[str(did)]} for did in case.arrays["drug_ids"]},
    )
    unseen = int(case.arrays["unseen_drug"])

    for index, (variant, setting) in enumerate(case.meta["variants"].items()):
        fit_set, stop_set = data.train.copy(), data.train.copy()
        stop = case.arrays["stop_mask"]
        if setting.get("unseen"):
            keep = fit_set.drug_ids != str(case.arrays["drug_ids"][unseen])
            fit_set.mask(keep)
            stop_set.mask(keep)
            stop = case.arrays["unseen_stop_mask"]
        fit_set.mask(~stop)
        stop_set.mask(stop)
        model = PharmaFormerModel()
        model.build_model({**_hyperparameters(setting), "gene_list": case.meta["gene_list"]})
        seed_all(setting["seed"] + 100 * index)
        with tempfile.TemporaryDirectory() as checkpoints:
            model.train(fit_set, data.cell_line_input, drugs, stop_set, model_checkpoint_dir=checkpoints)
        predictions = model.predict(data.test.cell_line_ids, data.test.drug_ids, data.cell_line_input, drugs)

        used = np.unique(np.concatenate([fit_set.cell_line_ids, stop_set.cell_line_ids]))
        used_drugs = np.unique(np.concatenate([fit_set.drug_ids, stop_set.drug_ids]))
        gex = data.cell_line_input.get_feature_matrix("gene_expression", used)
        scaled = model.gene_expression_normalizer.transform(model.gene_expression_scaler.transform(gex))
        out[f"{variant}/sample_ids"] = used.astype(str)
        out[f"{variant}/features_sample"] = scaled.astype(np.float32)
        out[f"{variant}/drug_ids"] = used_drugs.astype(str)
        out[f"{variant}/bpe"] = np.stack([rows[str(d)] for d in used_drugs])
        out[f"{variant}/bpe_test"] = np.stack([rows[str(d)] for d in np.unique(data.test.drug_ids)])
        out[f"{variant}/predictions"] = np.asarray(predictions, dtype=np.float64)
        state = model.model.state_dict()
        for key in _recorded_keys():
            out[f"{variant}/params/{key}"] = state[key].cpu().numpy()
    return out


@contextmanager
def _capture_predict(sink: dict[str, Any]) -> Iterator[None]:
    """Store the ``ModelInputBatch`` that the v2 predictor scores in ``sink["predict_batch"]``.

    :param sink: Dictionary receiving the batch.
    :yields: Nothing; the patch is undone on exit.
    """
    from drevalpy.components.predictors.literature.pharmaformer.predictor import PharmaFormerPredictor

    original = PharmaFormerPredictor._predict

    def capture(self: Any, batch: Any) -> Any:
        sink["predict_batch"] = batch
        return original(self, batch)

    PharmaFormerPredictor._predict = capture
    try:
        yield
    finally:
        PharmaFormerPredictor._predict = original


def run_v2(case: Case) -> dict[str, np.ndarray]:
    """Run the forward probes and every variant with the v2 ``PharmaFormer`` preset.

    :param case: The shared case.
    :returns: Arrays keyed like those of :func:`run_v1`.
    """
    from drevalpy.components.predictors.literature.pharmaformer.model_utils import CombinedModel
    from drevalpy.components.predictors.literature.pharmaformer.predictor import PharmaFormerPredictor
    from drevalpy.types.data.split_mask import SplitMask
    from tools.v1_parity._v2_runner import tabular_inputs

    out = _forward_outputs(CombinedModel, case)

    data = tabular_inputs(case)
    data.dataset.response.var["canonical_smiles"] = list(case.arrays["smiles"].astype(str))
    shape = (len(case.arrays["sample_ids"]), len(case.arrays["drug_ids"]))
    unseen = int(case.arrays["unseen_drug"])

    for index, (variant, setting) in enumerate(case.meta["variants"].items()):
        pairs, stop = case.arrays["train_pairs"], case.arrays["stop_mask"]
        if setting.get("unseen"):
            pairs, stop = pairs[pairs[:, 1] != unseen], case.arrays["unseen_stop_mask"]
        fit_scope = SplitMask.from_pairs(pairs[~stop], shape)
        stop_scope = SplitMask.from_pairs(pairs[stop], shape)
        model = v2_model(
            NAME,
            _hyperparameters(setting),
            drug_featurizer={"options": {"num_symbols": BPE_SYMBOLS, "max_length": BPE_LENGTH}},
        )
        captured: dict[str, Any] = {}
        seed_all(setting["seed"] + 100 * index)
        with capture_fit(PharmaFormerPredictor, captured), _capture_predict(captured):
            model.train(data.dataset, fit_scope, early_stopping_scope=stop_scope)
            predictions = model.predict(data.dataset, data.test)

        batch = captured["batch"]
        sample_order = np.argsort(batch.sample.entity_ids.astype(str), kind="stable")
        drug_order = np.argsort(batch.drug.entity_ids.astype(str), kind="stable")
        out[f"{variant}/sample_ids"] = batch.sample.entity_ids[sample_order].astype(str)
        out[f"{variant}/features_sample"] = batch.sample.blocks["gene_expression"].values[sample_order]
        out[f"{variant}/drug_ids"] = batch.drug.entity_ids[drug_order].astype(str)
        out[f"{variant}/bpe"] = batch.drug.blocks["bpe_smiles"].values[drug_order].astype(np.float64)
        scored = captured["predict_batch"]
        test_ids = np.unique(case.arrays["drug_ids"][case.arrays["test_pairs"][:, 1]])
        position = {str(d): i for i, d in enumerate(scored.drug.entity_ids)}
        rows = scored.drug.blocks["bpe_smiles"].values
        out[f"{variant}/bpe_test"] = np.stack([rows[position[str(d)]] for d in test_ids]).astype(np.float64)
        out[f"{variant}/predictions"] = np.asarray(predictions, dtype=np.float64)
        state = model._pipeline._predictor._model.state_dict()
        for key in _recorded_keys():
            out[f"{variant}/params/{key}"] = state[key].cpu().numpy()
    return out
