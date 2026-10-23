"""PaccMann parity case: 1.6.1 ``PaccMann`` against the v2 ``PaccMann`` zoo preset.

Four layers are compared.

``tokenizer/*``, ``vocabulary/*``, ``encoding/*``, ``randomized/*``
    The SMILES helpers on tricky strings (``Cl``, ``Br``, ``[C@@H]``, ``[nH]``, ``%12`` ring closures,
    both slashes, a salt, and one string longer than the padding): the tokens, the vocabulary, the
    padded rows and the randomized SMILES of the variant bank. All exact. The 1.6.1 helpers are
    the ``_tokenize_smiles``, ``_build_smiles_vocab``, ``_encode_smiles`` and ``_randomize_smiles``
    of ``PaccMann/paccmann.py``; the v2 ones are in ``drug/_smiles_tokenizer.py``.
``forward/<config>/*``
    The 1.6.1 ``PaccMannNetwork`` and the v2 one with identical weights on the same probe, in eval
    mode and in train mode (batch norm statistics, and dropout under the same torch seed). The weights
    are drawn in ``build_case`` per ``state_dict`` key of the v2 network and loaded with
    ``strict=True`` on both sides, so a missing or extra parameter fails the case.
``<variant>/{vocabulary_size,bank,features_sample,sample_ids,drug_ids}``
    What enters the network after the featurizers: the vocabulary size, the encoded variant bank per
    distinct training SMILES (v2 rows reordered to the sorted SMILES, which is the 1.6.1 order), and
    the scaled expression. The case holds the shipped ``gene_list_paccmann_network_prop_reduced`` in
    shuffled order plus extra genes, so the v2 gene-list selection is exercised; 1.6.1 selects the list
    while loading CSVs, which the in-memory harness skips, so its side receives the matrix reordered
    to the list.
``<variant>/{predictions,<parameters>}``
    A three-epoch training from the seeded initialization and the test predictions.

Variants: ``default`` (explicit early-stopping set), ``no_es`` (none: full budget, last weights) and
``plain`` (no SMILES augmentation, no batch norm).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tools.v1_parity._bridge import Case, seed_all, tabular_case_arrays
from tools.v1_parity.compare import DEVIATION, PARITY, StepSpec

NAME = "PaccMann"

GENE_LIST = "gene_list_paccmann_network_prop_reduced"
N_EXTRA_GENES = 20
N_SAMPLES = 24
STOP_EVERY = 6
PADDING = 64
N_VARIANTS = 20
SMILES = (
    "CCO",
    "CC(=O)Oc1ccccc1C(=O)O",
    "C[C@@H](N)C(=O)O",
    "ClCCBr",
    "c1ccc2c(c1)[nH]c1ccccc12",
    "C1CC%12CC%12C1",
    "C/C=C\\C",
    "CC(=O)[O-].[Na+]",
    "CC(C)C[C@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)CNC(=O)[C@H](CC(C)C)NC(=O)[C@@H](N)Cc1ccc(O)cc1)C(=O)O",
)

_SMALL_NETWORK: dict[str, Any] = {
    "filters": [4, 4, 4],
    "molecule_heads": [2, 1, 1, 1],
    "gene_heads": [1, 2, 1, 1],
    "smiles_embedding_size": 6,
    "smiles_attention_size": 8,
    "gene_attention_size": 1,
    "stacked_dense_hidden_sizes": [16, 8],
    "molecule_temperature": 1.0,
    "gene_temperature": 1.0,
}

VARIANTS: dict[str, dict[str, Any]] = {
    "default": {
        **_SMALL_NETWORK,
        "epochs": 3,
        "patience": 10,
        "batch_size": 32,
        "learning_rate": 3e-3,
        "weight_decay": 1e-5,
        "dropout": 0.3,
        "batch_norm": True,
        "augment": True,
        "early_stopping": True,
        "seed": 0,
    },
    "no_es": {
        **_SMALL_NETWORK,
        "epochs": 2,
        "patience": 10,
        "batch_size": 40,
        "learning_rate": 3e-3,
        "weight_decay": 0.0,
        "dropout": 0.3,
        "batch_norm": True,
        "augment": True,
        "early_stopping": False,
        "seed": 4,
    },
    "plain": {
        **_SMALL_NETWORK,
        "epochs": 3,
        "patience": 10,
        "batch_size": 32,
        "learning_rate": 3e-3,
        "weight_decay": 0.0,
        "dropout": 0.2,
        "batch_norm": False,
        "augment": False,
        "early_stopping": True,
        "seed": 2,
    },
}
_NON_HYPERPARAMETERS = ("augment", "early_stopping", "seed")

_FORWARD_BASE: dict[str, Any] = {
    "smiles_vocabulary_size": 14,
    "smiles_embedding_size": 4,
    "smiles_padding_length": 20,
    "number_of_genes": 9,
    "smiles_attention_size": 6,
    "stacked_dense_hidden_sizes": [10, 6],
    "dropout": 0.5,
}
FORWARD_CONFIGS: dict[str, dict[str, Any]] = {
    "batch_norm": {
        **_FORWARD_BASE,
        "filters": [3, 3, 3],
        "molecule_heads": [2, 1, 1, 1],
        "gene_heads": [1, 2, 1, 1],
        "batch_norm": True,
    },
    "plain": {
        **_FORWARD_BASE,
        "filters": [2, 3, 2],
        "molecule_heads": [1, 1, 2, 1],
        "gene_heads": [2, 1, 1, 1],
        "batch_norm": False,
        "gene_attention_size": 2,
        "molecule_temperature": 0.7,
        "gene_temperature": 1.3,
    },
}
PRE_BATCH_NORM_BIAS = "dense_layers.0.0.bias"
BATCH_NORM_EXPLANATION = (
    "A linear bias in front of a batch norm layer has an exactly zero gradient, so what Adam sees is float32 "
    "rounding noise, and Adam rescales it to a full learning-rate step. The scaled expression differs by one "
    "float32 rounding (about 5e-7) between 1.6.1 and v2, which is enough to send that bias, the running means "
    "behind it and, through them, the eval-mode predictions of a 3-epoch fit to a different place. Measured "
    "on 2026-10-23: training 1.6.1 itself on expression scaled by 1 + 3e-7 moves its test predictions by "
    "2.2e-4 (default) and 2.2e-3 (no_es), against 1.0e-3 and 1.6e-3 between 1.6.1 and v2. Every other "
    "parameter and the batch-norm-free variant agree at the train class."
)
PROBE_ROWS = 6
DROPOUT_SEED = 7

FINDINGS = [
    "The network is the 1.6.1 network (only `strict=True` was added to the `zip` calls), so the weights of the "
    "forward case load into both sides with identical `state_dict` keys; the seeded initialization and the "
    "dropout draws coincide as long as the training loops consume the torch RNG in the same order.",
    "The shipped gene list is trimmed: `gene_list_paccmann_network_prop` has 1955 genes in v2, not 1957, because "
    "the re-curated cell-line screens lack PRRC2B and S100A7L2. The case uses the reduced list "
    "(`gene_list_paccmann_network_prop_reduced`) on both sides, so the trim itself is not exercised here.",
    "v2 holds features as float32 and the scaler is fit on float32 values; 1.6.1 also casts the expression to "
    "float32 before its scaler, so the scaled matrices are compared at the `forward` class.",
    "The scaler is fit on the training pairs (a sample once per pair) on both sides: 1.6.1 fits it on the "
    "expression of `output.cell_line_ids`; v2 reproduces that with `landmarkGenes(fit_on_pairs=True)`.",
    "The variant bank is drawn at fit, from the sorted distinct SMILES of the training drugs, with one "
    "`default_rng(42)` stream, as in 1.6.1; the vocabulary is built from the flattened bank. A drug outside "
    "the fit gets its own SMILES in every bank slot in v2. The padding length is the featurizer option "
    "`padding_length` (1.6.1: hyperparameter `smiles_padding_length`; without it 1.6.1 used the longest "
    "variant, v2 always 512 by default) and `augment_smiles` is the option `augment`.",
    "Each batch row draws its variant with `torch.randint` on the global torch RNG, after the shuffling draw, "
    "as in 1.6.1. v2 evaluates early stopping on the plain (non-augmented) SMILES, as 1.6.1 did.",
    "RDKit versions differ between the two environments, which could change `MolToSmiles` output on rare "
    "molecules; the randomized-SMILES steps record whether it did.",
    "With batch norm, a 3-epoch fit does not reproduce the 1.6.1 test predictions to the `train` class (deviation "
    "steps `default/predictions` and `no_es/predictions`): the float32 rounding difference of the scaled "
    "expression is amplified through the bias in front of each batch norm layer, whose gradient is exactly zero "
    "(see `BATCH_NORM_EXPLANATION`). Every other parameter agrees at the `forward` class, and the variant without "
    "batch norm reproduces the predictions at the `forward` class.",
    "1.6.1 `drop_last` rule (drop a trailing batch of one only when more than one batch exists) is kept.",
]


def _network_config(setting: dict[str, Any]) -> dict[str, Any]:
    """Return the keyword arguments of ``PaccMannConfig`` for a forward configuration.

    :param setting: Entry of ``FORWARD_CONFIGS``.
    :returns: Keyword arguments.
    """
    return dict(setting)


def _forward_weights(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Draw weights and probes for every forward configuration from the v2 network's parameter shapes.

    :param rng: Generator of the case.
    :returns: Arrays ``forward/<config>/w/<name>``, ``.../smiles`` and ``.../genes``.
    """
    from drevalpy.components.predictors.literature.paccmann.network import PaccMannConfig, PaccMannNetwork

    arrays: dict[str, np.ndarray] = {}
    for name, setting in FORWARD_CONFIGS.items():
        network = PaccMannNetwork(PaccMannConfig(**_network_config(setting)))
        for key, tensor in sorted(network.state_dict().items()):
            if not tensor.is_floating_point():
                continue
            shape = tuple(tensor.shape)
            if key.endswith("running_var"):
                values = 0.5 + rng.random(shape)
            elif key.endswith("running_mean"):
                values = 0.2 * rng.normal(size=shape)
            elif key.endswith("batch_norm.weight") or ".batch_norm.weight" in key or key == "batch_norm.weight":
                values = 1.0 + 0.2 * rng.normal(size=shape)
            else:
                values = 0.4 * rng.normal(size=shape)
            arrays[f"forward/{name}/w/{key}"] = values.astype(np.float32)
        arrays[f"forward/{name}/smiles"] = rng.integers(
            0, setting["smiles_vocabulary_size"], size=(PROBE_ROWS, setting["smiles_padding_length"])
        )
        arrays[f"forward/{name}/genes"] = rng.normal(size=(PROBE_ROWS, setting["number_of_genes"])).astype(np.float32)
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
    names = np.concatenate([gene_list, np.array([f"XTRA{i:03d}" for i in range(N_EXTRA_GENES)])])
    names = names[rng.permutation(len(names))]
    arrays = tabular_case_arrays(seed, n_samples=N_SAMPLES, n_drugs=len(SMILES), n_genes=len(names), n_bits=4)
    arrays["gene_names"] = names
    arrays["gene_list"] = gene_list
    arrays["smiles"] = np.array(SMILES)
    arrays["stop_mask"] = np.arange(len(arrays["train_pairs"])) % STOP_EVERY == STOP_EVERY - 1
    if len(np.unique(arrays["train_pairs"][:, 1])) != len(SMILES):
        msg = "every drug must occur in the training pairs"
        raise ValueError(msg)
    arrays.update(_forward_weights(rng))
    meta = {"variants": VARIANTS, "forward": FORWARD_CONFIGS, "gene_list": GENE_LIST}
    return Case(model=NAME, seed=seed, arrays=arrays, meta=meta)


def _network_keys(setting: dict[str, Any]) -> list[str]:
    """Name the parameters compared after training.

    :param setting: Variant hyperparameters.
    :returns: Parameter names present in the variant's network.
    """
    keys = ["smiles_embedding.weight", "convolutions.0.convolve.weight", "output_layer.weight", "output_layer.bias"]
    if setting["batch_norm"]:
        keys.append("batch_norm.running_mean")
    return keys


def _keys_to_record(setting: dict[str, Any]) -> list[str]:
    """Name every parameter written to the outputs, the compared ones and the pre-batch-norm bias.

    :param setting: Variant hyperparameters.
    :returns: Parameter names.
    """
    return [*_network_keys(setting), *([PRE_BATCH_NORM_BIAS] if setting["batch_norm"] else [])]


def steps() -> list[StepSpec]:
    """Declare the compared outputs.

    :returns: Helper, forward, feature, training and prediction steps.
    """
    specs = [
        StepSpec("tokenizer/tokens", "exact", description="atom-level tokens of the tricky SMILES"),
        StepSpec("vocabulary/tokens", "exact", description="vocabulary, tokens in index order"),
        StepSpec("encoding/own", "exact", description=f"padded rows (length {PADDING}, one cut off) of each SMILES"),
        StepSpec("randomized/smiles", "exact", description=f"{N_VARIANTS} randomized SMILES per molecule (seed 42)"),
        StepSpec("randomized/encoded", "exact", description="the randomized SMILES, encoded"),
    ]
    for config in FORWARD_CONFIGS:
        specs += [
            StepSpec(f"forward/{config}/{mode}", "forward", description=f"{config} network, {mode} mode output")
            for mode in ("eval", "train")
        ]
    for variant, setting in VARIANTS.items():
        table = [
            ("sample_ids", "exact", "ids of the samples whose features enter the network"),
            ("drug_ids", "exact", "ids of the drugs whose tokens enter the network"),
            ("vocabulary_size", "exact", "size of the vocabulary built from the variant bank"),
            ("bank", "exact", "encoded variant bank per distinct training SMILES, in sorted-SMILES order"),
            ("features_sample", "forward", "scaled expression of the fit and early-stopping samples"),
        ]
        table += [(f"params/{key}", "train", f"{key} after training") for key in _network_keys(setting)]
        if not setting["batch_norm"]:
            table.append(("predictions", "train", "test predictions after the seeded short training"))
        specs += [
            StepSpec(name=f"{variant}/{quantity}", tolerance=tolerance, scope=PARITY, description=f"{variant}: {text}")
            for quantity, tolerance, text in table
        ]
        if setting["batch_norm"]:
            specs += [
                StepSpec(
                    name=f"{variant}/{quantity}",
                    tolerance="train",
                    scope=DEVIATION,
                    description=f"{variant}: {text}",
                    explanation=BATCH_NORM_EXPLANATION,
                )
                for quantity, text in (
                    ("predictions", "test predictions after the seeded short training"),
                    (f"params/{PRE_BATCH_NORM_BIAS}", "bias of the linear layer in front of a batch norm layer"),
                )
            ]
    return specs


def _forward_outputs(network: Any, case: Case, config: str) -> dict[str, np.ndarray]:
    """Run the probe through a network in eval mode and in train mode.

    :param network: 1.6.1 or v2 ``PaccMannNetwork`` holding the case's weights.
    :param case: The shared case.
    :param config: Forward configuration name.
    :returns: ``forward/<config>/eval`` and ``forward/<config>/train``.
    """
    import torch

    smiles = torch.as_tensor(case.arrays[f"forward/{config}/smiles"], dtype=torch.long)
    genes = torch.as_tensor(case.arrays[f"forward/{config}/genes"])
    network.eval()
    with torch.no_grad():
        evaluated = network(smiles, genes).numpy()
    network.train()
    torch.manual_seed(DROPOUT_SEED)
    with torch.no_grad():
        trained = network(smiles, genes).numpy()
    network.eval()
    return {f"forward/{config}/eval": evaluated, f"forward/{config}/train": trained}


def _load_weights(network: Any, case: Case, config: str) -> None:
    """Load the case's weights of *config* into *network* with ``strict=True``.

    :param network: Network to fill; its non-float buffers keep their initial values.
    :param case: The shared case.
    :param config: Forward configuration name.
    """
    import torch

    prefix = f"forward/{config}/w/"
    state = network.state_dict()
    state.update({key[len(prefix) :]: torch.as_tensor(v) for key, v in case.arrays.items() if key.startswith(prefix)})
    network.load_state_dict(state, strict=True)


def _helper_outputs(
    tokenize: Any, build_vocabulary: Any, encode: Any, randomize: Any, case: Case
) -> dict[str, np.ndarray]:
    """Run the SMILES helpers of one side; the callables hide the 1.6.1 / v2 differences.

    :param tokenize: ``smiles -> tokens``.
    :param build_vocabulary: ``smiles list -> {token: index}``.
    :param encode: ``(smiles list, vocabulary, padding) -> int array``.
    :param randomize: ``(smiles list, n_variants) -> variants per molecule``.
    :param case: The shared case.
    :returns: Arrays of the helper steps.
    """
    smiles = [str(value) for value in case.arrays["smiles"]]
    vocabulary = build_vocabulary(smiles)
    variants = randomize(smiles, N_VARIANTS)
    flat = [variant for molecule in variants for variant in molecule]
    return {
        "tokenizer/tokens": np.array(["|".join(tokenize(value)) for value in smiles]),
        "vocabulary/tokens": np.array([token for token, _ in sorted(vocabulary.items(), key=lambda item: item[1])]),
        "encoding/own": encode(smiles, vocabulary, PADDING),
        "randomized/smiles": np.array(variants),
        "randomized/encoded": encode(flat, build_vocabulary(flat), PADDING).reshape(len(smiles), N_VARIANTS, PADDING),
    }


def _hyperparameters(setting: dict[str, Any]) -> dict[str, Any]:
    """Drop the variant keys that are not model hyperparameters.

    :param setting: Variant entry.
    :returns: Hyperparameters shared by both sides.
    """
    return {key: value for key, value in setting.items() if key not in _NON_HYPERPARAMETERS}


def _sorted_bank(bank: np.ndarray, smiles: np.ndarray) -> np.ndarray:
    """Order bank rows by their SMILES, which is the order 1.6.1 builds the bank in.

    :param bank: Bank with one row per drug.
    :param smiles: SMILES of those rows.
    :returns: Rows reordered to the sorted distinct SMILES.
    """
    return bank[np.argsort(smiles.astype(str), kind="stable")]


def run_v1(case: Case) -> dict[str, np.ndarray]:
    """Run the helpers, the forward probes and every variant with the 1.6.1 ``PaccMann`` model.

    :param case: The shared case.
    :returns: Arrays keyed by the step names.
    """
    from drevalpy.datasets.dataset import FeatureDataset
    from drevalpy.models.PaccMann.network import PaccMannConfig, PaccMannNetwork
    from drevalpy.models.PaccMann.paccmann import PaccMann, _randomize_smiles, _tokenize_smiles

    from tools.v1_parity._v1_runner import tabular_inputs

    helper = PaccMann()

    def build_vocabulary(smiles_list: list[str]) -> dict[str, int]:
        helper.smiles_to_idx = {"<pad>": 0, "<unk>": 1}
        helper._build_smiles_vocab(smiles_list)
        return dict(helper.smiles_to_idx)

    def encode(smiles_list: list[str], vocabulary: dict[str, int], padding: int) -> np.ndarray:
        helper.smiles_to_idx, helper.smiles_padding_length = vocabulary, padding
        return helper._encode_smiles(smiles_list)

    out = _helper_outputs(_tokenize_smiles, build_vocabulary, encode, _randomize_smiles, case)
    for config, setting in case.meta["forward"].items():
        network = PaccMannNetwork(PaccMannConfig(**_network_config(setting)))
        _load_weights(network, case, config)
        out.update(_forward_outputs(network, case, config))

    data = tabular_inputs(case)
    names = list(case.arrays["gene_names"])
    columns = [names.index(g) for g in case.arrays["gene_list"]]
    for views in data.cell_line_input.features.values():
        views["gene_expression"] = views["gene_expression"][columns]
    data.cell_line_input.meta_info["gene_expression"] = np.array(case.arrays["gene_list"])
    drugs = FeatureDataset(
        features={
            str(did): {"smiles": np.array([str(smiles)], dtype=object)}
            for did, smiles in zip(case.arrays["drug_ids"], case.arrays["smiles"], strict=True)
        },
        meta_info={"smiles": ["canonical_smiles"]},
    )
    drug_rows = np.unique(case.arrays["train_pairs"][:, 1])
    stop = case.arrays["stop_mask"]

    for index, (variant, setting) in enumerate(case.meta["variants"].items()):
        fit_set, stop_set = data.train.copy(), data.train.copy()
        if setting["early_stopping"]:
            fit_set.mask(~stop)
            stop_set.mask(stop)
        else:
            stop_set = None
        model = PaccMann()
        hyperparameters = {
            **_hyperparameters(setting),
            "smiles_padding_length": PADDING,
            "augment_smiles": setting["augment"],
        }
        model.build_model(hyperparameters)
        banks: list[list[list[str]]] = []
        original = model._build_smiles_variants

        def record(unique: list[str], _original: Any = original, _sink: list[list[list[str]]] = banks) -> Any:
            _sink.append(_original(unique))
            return _sink[-1]

        model._build_smiles_variants = record  # type: ignore[method-assign]
        seed_all(setting["seed"] + 100 * index)
        model.train(fit_set, data.cell_line_input, drugs, stop_set)
        predictions = model.predict(data.test.cell_line_ids, data.test.drug_ids, data.cell_line_input, drugs)

        bank = np.array(
            [model._encode_smiles(molecule) for molecule in banks[0]]
        )  # (molecules, variants, padding), sorted SMILES
        used = np.unique(np.concatenate([fit_set.cell_line_ids, stop_set.cell_line_ids if stop_set else []]))
        gex = data.cell_line_input.get_feature_matrix("gene_expression", used)
        scaled = model.gene_expression_scaler.transform(np.asarray(gex, dtype=np.float32)).astype(np.float32)
        out[f"{variant}/sample_ids"] = used.astype(str)
        out[f"{variant}/drug_ids"] = case.arrays["drug_ids"][drug_rows].astype(str)
        out[f"{variant}/vocabulary_size"] = np.array(len(model.smiles_to_idx))
        out[f"{variant}/bank"] = bank
        out[f"{variant}/features_sample"] = scaled
        out[f"{variant}/predictions"] = np.asarray(predictions, dtype=np.float64)
        state = model.model.state_dict()
        for key in _keys_to_record(setting):
            out[f"{variant}/params/{key}"] = state[key].cpu().numpy()
    return out


def run_v2(case: Case) -> dict[str, np.ndarray]:
    """Run the helpers, the forward probes and every variant with the v2 ``PaccMann`` preset.

    :param case: The shared case.
    :returns: Arrays keyed like those of :func:`run_v1`.
    """
    from drevalpy.components.featurizers.drug._smiles_tokenizer import (
        build_vocabulary,
        encode_smiles,
        randomize_smiles,
        tokenize_smiles,
    )
    from drevalpy.components.predictors.literature.paccmann.network import PaccMannConfig, PaccMannNetwork
    from drevalpy.components.predictors.literature.paccmann.predictor import PaccMannPredictor
    from drevalpy.models.config import DrugFeaturizerConfig, ModelConfig, SampleFeaturizerConfig
    from drevalpy.models.drp_model import DRPModel
    from drevalpy.models.spec import ModelSpec
    from drevalpy.models.zoo import get_zoo_config
    from drevalpy.types.data.split_mask import SplitMask
    from tools.v1_parity._v2_runner import tabular_inputs

    out = _helper_outputs(tokenize_smiles, build_vocabulary, encode_smiles, randomize_smiles, case)
    for config, setting in case.meta["forward"].items():
        network = PaccMannNetwork(PaccMannConfig(**_network_config(setting)))
        _load_weights(network, case, config)
        out.update(_forward_outputs(network, case, config))

    data = tabular_inputs(case)
    data.dataset.response.var["canonical_smiles"] = list(case.arrays["smiles"].astype(str))
    shape = (len(case.arrays["sample_ids"]), len(case.arrays["drug_ids"]))
    stop = case.arrays["stop_mask"]
    base = get_zoo_config(NAME).model_dump(mode="python")

    sample_options = {**base["sample_featurizer"]["options"], "gene_list_stem": case.meta["gene_list"]}

    for index, (variant, setting) in enumerate(case.meta["variants"].items()):
        if setting["early_stopping"]:
            fit_scope = SplitMask.from_pairs(case.arrays["train_pairs"][~stop], shape)
            stop_scope = SplitMask.from_pairs(case.arrays["train_pairs"][stop], shape)
        else:
            fit_scope, stop_scope = SplitMask.from_pairs(case.arrays["train_pairs"], shape), None
        config = ModelConfig.model_validate(
            {
                **base,
                "sample_featurizer": SampleFeaturizerConfig(name="landmarkGenes", options=sample_options),
                "drug_featurizer": DrugFeaturizerConfig(
                    name="smilesTokens",
                    options={"padding_length": PADDING, "n_variants": N_VARIANTS, "augment": setting["augment"]},
                ),
            }
        )
        captured: dict[str, Any] = {}
        original = PaccMannPredictor._fit

        def capture(self: Any, batch: Any, _original: Any = original, _sink: dict[str, Any] = captured) -> None:
            _sink["batch"] = batch
            _original(self, batch)

        PaccMannPredictor._fit = capture  # type: ignore[method-assign]
        try:
            seed_all(setting["seed"] + 100 * index)
            model = DRPModel(ModelSpec(NAME, config), _hyperparameters(setting))
            model.train(data.dataset, fit_scope, early_stopping_scope=stop_scope)
        finally:
            PaccMannPredictor._fit = original  # type: ignore[method-assign]
        predictions = model.predict(data.dataset, data.test)

        batch = captured["batch"]
        drug_order = np.argsort(batch.drug.entity_ids.astype(str), kind="stable")
        drug_smiles = np.array([SMILES[int(str(d)[1:])] for d in batch.drug.entity_ids])
        variants = batch.drug.blocks["smiles_token_variants"]
        n_variants = variants.metadata["n_variants"]
        bank = variants.values.astype(np.int64).reshape(len(drug_smiles), n_variants, -1)
        sample_order = np.argsort(batch.sample.entity_ids.astype(str), kind="stable")
        out[f"{variant}/sample_ids"] = batch.sample.entity_ids[sample_order].astype(str)
        out[f"{variant}/drug_ids"] = batch.drug.entity_ids[drug_order].astype(str)
        out[f"{variant}/vocabulary_size"] = np.array(variants.metadata["vocabulary_size"])
        out[f"{variant}/bank"] = _sorted_bank(bank, drug_smiles)
        out[f"{variant}/features_sample"] = batch.sample.blocks["gene_expression"].values[sample_order]
        out[f"{variant}/predictions"] = np.asarray(predictions, dtype=np.float64)
        state = model._pipeline._predictor._model.state_dict()
        for key in _keys_to_record(setting):
            out[f"{variant}/params/{key}"] = state[key].cpu().numpy()
    return out
