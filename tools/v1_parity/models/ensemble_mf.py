"""EnsembleMF parity case: 1.6.1 ``EnsembleMF`` against the v2 ``EnsembleMF`` zoo preset.

Three layers are compared.

``forward/<config>/*``
    The 1.6.1 ``_MFNet`` and the v2 ``MFNet`` (eval mode) with the same weights on the same random
    probe. The weights are drawn in ``build_case`` under the 1.6.1 ``state_dict`` names; the v2 side
    renames ``cell_encoder``/``cell_bias`` to ``sample_encoder``/``sample_bias`` and loads them
    with ``strict=True``, so a missing or extra parameter fails the case.
``<variant>/{rank,features_*}``
    The rank-transformed matrix before the scaler (exact), the scaled sample matrix and the
    fingerprints that enter the networks. The case holds the 867 shipped ``landmark_genes`` in
    shuffled order plus extra genes, so the v2 gene-list selection is exercised; 1.6.1 selects the
    list while loading CSVs, which the in-memory harness skips, so its side receives the matrix
    reordered to the list.
``<variant>/{member*,predictions}``
    Two members trained for three epochs with an explicit early-stopping set (the ``stop_mask``
    pairs of the training pairs), then the weights that matter most and the test predictions.

Variants: ``default`` (id embedding, interaction head, one residual block) and ``plain`` (no id
embedding, no head, two residual blocks, another seed and batch size).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tools.v1_parity._bridge import Case, record_calls, tabular_case_arrays
from tools.v1_parity.compare import DEVIATION, PARITY, StepSpec

NAME = "EnsembleMF"

N_EXTRA_GENES = 25
N_SAMPLES = 30
N_DRUGS = 6
N_BITS = 16
STOP_EVERY = 6

VARIANTS: dict[str, dict[str, Any]] = {
    "default": {
        "hidden_dim": 16,
        "emb_dim": 8,
        "n_layers": 1,
        "dropout": 0.2,
        "use_mlp_head": True,
        "mlp_hidden": 16,
        "use_drug_id_embedding": True,
        "n_ensemble": 2,
        "learning_rate": 3e-3,
        "weight_decay": 1e-5,
        "batch_size": 64,
        "max_epochs": 3,
        "patience": 25,
        "seed": 0,
    },
    "plain": {
        "hidden_dim": 12,
        "emb_dim": 6,
        "n_layers": 2,
        "dropout": 0.1,
        "use_mlp_head": False,
        "mlp_hidden": 16,
        "use_drug_id_embedding": False,
        "n_ensemble": 2,
        "learning_rate": 3e-3,
        "weight_decay": 1e-5,
        "batch_size": 40,
        "max_epochs": 3,
        "patience": 25,
        "seed": 3,
    },
}

FORWARD_CONFIGS: dict[str, dict[str, Any]] = {
    "full": {
        "sample_in_dim": 11,
        "drug_in_dim": 7,
        "n_drugs": 5,
        "hidden_dim": 10,
        "emb_dim": 6,
        "n_layers": 2,
        "use_drug_id_embedding": True,
        "use_mlp_head": True,
        "mlp_hidden": 9,
    },
    "plain": {
        "sample_in_dim": 11,
        "drug_in_dim": 7,
        "n_drugs": 5,
        "hidden_dim": 10,
        "emb_dim": 6,
        "n_layers": 1,
        "use_drug_id_embedding": False,
        "use_mlp_head": False,
        "mlp_hidden": 9,
    },
}

RENAMES = {"cell_encoder": "sample_encoder", "cell_bias": "sample_bias"}

FINDINGS = [
    "The networks are the 1.6.1 networks with `cell` renamed to `sample` (`cell_encoder`->`sample_encoder`, "
    "`cell_bias`->`sample_bias`). Layer creation order is unchanged, so the seeded initialization and the "
    "dropout draws coincide; the checkpoint of a 1.6.1 net loads after renaming those two prefixes.",
    "v2 featurizes inside the pipeline: `scaledGeneExpression(transform=rank)` fits the sorted training-sample "
    "reference and the scaler on the unique training samples (as 1.6.1 does) and returns float32; 1.6.1 casts to "
    "float32 after scaling in float64. The rank transform itself is compared before the scaler and is exact.",
    "v2 builds the drug id-embedding table over the drugs of the training and early-stopping pairs, 1.6.1 over every "
    "drug with features. The case uses only drugs that occur in those pairs, so the table shapes (and with them the "
    "seeded initialization) coincide.",
    "1.6.1 returns the training mean for a pair whose sample or drug has no features; v2 returns NaN there (the "
    "pipeline's missing-feature rule) and no longer stores the training mean.",
    "Without an early-stopping set 1.6.1 holds out 10% of the pairs with the global torch RNG; v2 uses a "
    "`torch.Generator` seeded from `seed`. The case passes an explicit early-stopping set, which is the "
    "path where the two are comparable.",
    "A `n_bits` hyperparameter on the `fingerprints` featurizer applies only when the fingerprints are computed from "
    "SMILES; a precomputed `morgan_fingerprint` view is used as is (as in this case).",
]


def _state_shapes(config: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    """List the 1.6.1 ``_MFNet.state_dict`` entries and their shapes.

    :param config: Entry of ``FORWARD_CONFIGS``.
    :returns: Shape per parameter name, in the 1.6.1 naming.
    """
    hidden, emb, mlp_hidden = config["hidden_dim"], config["emb_dim"], config["mlp_hidden"]
    shapes: dict[str, tuple[int, ...]] = {}
    for tower, in_dim in (("cell_encoder", config["sample_in_dim"]), ("drug_encoder", config["drug_in_dim"])):
        shapes[f"{tower}.input_proj.weight"] = (hidden, in_dim)
        shapes[f"{tower}.input_proj.bias"] = (hidden,)
        for i in range(config["n_layers"]):
            shapes[f"{tower}.blocks.{i}.weight"] = (hidden, hidden)
            shapes[f"{tower}.blocks.{i}.bias"] = (hidden,)
            shapes[f"{tower}.norms.{i}.weight"] = (hidden,)
            shapes[f"{tower}.norms.{i}.bias"] = (hidden,)
        for head in ("output_proj", "skip_proj"):
            shapes[f"{tower}.{head}.weight"] = (emb, hidden)
            shapes[f"{tower}.{head}.bias"] = (emb,)
    if config["use_drug_id_embedding"]:
        shapes["drug_id_emb.weight"] = (config["n_drugs"], emb)
    for bias in ("cell_bias", "drug_bias"):
        shapes[f"{bias}.weight"] = (1, emb)
        shapes[f"{bias}.bias"] = (1,)
    shapes["global_bias"] = (1,)
    shapes["dot_scale"] = (1,)
    if config["use_mlp_head"]:
        shapes["mlp.0.weight"] = (mlp_hidden, 3 * emb)
        shapes["mlp.0.bias"] = (mlp_hidden,)
        shapes["mlp.3.weight"] = (1, mlp_hidden)
        shapes["mlp.3.bias"] = (1,)
    return shapes


def _forward_arrays(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Draw the weights and the probe of every forward configuration.

    :param rng: Generator of the case.
    :returns: Arrays ``forward/<config>/w/<name>`` and ``forward/<config>/{x_sample,x_drug,rows,sample_idx,drug_idx}``.
    """
    arrays: dict[str, np.ndarray] = {}
    for name, config in FORWARD_CONFIGS.items():
        for key, shape in _state_shapes(config).items():
            center = 1.0 if key == "dot_scale" or ("norms" in key and key.endswith("weight")) else 0.0
            arrays[f"forward/{name}/w/{key}"] = (center + 0.4 * rng.normal(size=shape)).astype(np.float32)
        arrays[f"forward/{name}/x_sample"] = rng.normal(size=(7, config["sample_in_dim"])).astype(np.float32)
        arrays[f"forward/{name}/x_drug"] = rng.normal(size=(config["n_drugs"], config["drug_in_dim"])).astype(
            np.float32
        )
        arrays[f"forward/{name}/rows"] = rng.normal(size=(config["n_drugs"], config["emb_dim"])).astype(np.float32)
        arrays[f"forward/{name}/sample_idx"] = rng.integers(0, 7, size=12)
        arrays[f"forward/{name}/drug_idx"] = rng.integers(0, config["n_drugs"], size=12)
    return arrays


def build_case(seed: int) -> Case:
    """Draw the tabular inputs (landmark genes in shuffled order plus extras) and the forward probes.

    :param seed: Seed of the input generator.
    :returns: The case.
    :raises ValueError: If a drug occurs in neither the fit nor the stopping pairs.
    """
    from drevalpy.components.featurizers.sample.gene_lists import resolve_gene_list

    landmark = np.array(resolve_gene_list("landmark_genes"))
    rng = np.random.default_rng(seed + 1)
    names = np.concatenate([landmark, np.array([f"XTRA{i:03d}" for i in range(N_EXTRA_GENES)])])
    names = names[rng.permutation(len(names))]
    arrays = tabular_case_arrays(seed, n_samples=N_SAMPLES, n_drugs=N_DRUGS, n_genes=len(names), n_bits=N_BITS)
    arrays["gene_names"] = names
    arrays["gene_list"] = landmark
    arrays["stop_mask"] = np.arange(len(arrays["train_pairs"])) % STOP_EVERY == STOP_EVERY - 1
    if len(np.unique(arrays["train_pairs"][:, 1])) != N_DRUGS:
        msg = "every drug must occur in the training pairs so both sides build the same id-embedding table"
        raise ValueError(msg)
    arrays.update(_forward_arrays(rng))
    meta = {"variants": VARIANTS, "forward": FORWARD_CONFIGS, "gene_list": "landmark_genes"}
    return Case(model=NAME, seed=seed, arrays=arrays, meta=meta)


def _member_keys(setting: dict[str, Any]) -> list[str]:
    """Name the (v2-named) parameters compared after training.

    :param setting: Variant hyperparameters.
    :returns: Parameter names present in the variant's network.
    """
    keys = ["sample_encoder.input_proj.weight", "drug_encoder.output_proj.weight", "global_bias", "dot_scale"]
    if setting["use_drug_id_embedding"]:
        keys.append("drug_id_emb.weight")
    return keys


def steps() -> list[StepSpec]:
    """Declare the compared outputs.

    :returns: Forward, feature, training and prediction steps; the float64 features are a deviation.
    """
    specs: list[StepSpec] = []
    for config in FORWARD_CONFIGS:
        specs += [
            StepSpec(
                name=f"forward/{config}/{quantity}",
                tolerance="forward",
                scope=PARITY,
                description=f"{config} network with copied weights: {quantity.replace('_', ' ')}",
            )
            for quantity in ("z_sample", "z_drug", "z_drug_rows", "scores")
        ]
    for variant, setting in VARIANTS.items():
        table = [
            ("sample_ids", "exact", "ids of the samples whose features enter the networks"),
            ("drug_ids", "exact", "ids of the drugs whose features enter the networks"),
            ("rank", "exact", "rank-transformed (percentile) expression before the scaler"),
            ("features_sample", "forward", "scaled sample matrix"),
            ("features_drug", "exact", "fingerprint matrix"),
            ("predictions", "train", "test predictions of the 2-member, 3-epoch ensemble"),
        ]
        table += [
            (f"member{m}/{key}", "train", f"member {m}: {key} after training")
            for m in range(setting["n_ensemble"])
            for key in _member_keys(setting)
        ]
        specs += [
            StepSpec(name=f"{variant}/{quantity}", tolerance=tolerance, scope=PARITY, description=f"{variant}: {text}")
            for quantity, tolerance, text in table
        ]
        specs.append(
            StepSpec(
                name=f"{variant}/features_sample_float64",
                tolerance="exact",
                scope=DEVIATION,
                description=f"{variant}: scaled sample matrix, 1.6.1 in float64 against v2 float32",
                explanation=(
                    "v2 holds features as float32; 1.6.1 scales in float64 and casts for the network only. "
                    "The networks see float32 on both sides (features_sample), so this only records the size."
                ),
            )
        )
    return specs


def _fit_rows(case: Case) -> tuple[np.ndarray, np.ndarray]:
    """Return the sorted sample and drug rows that occur in the fit or stopping pairs.

    :param case: The shared case.
    :returns: Sorted unique sample rows and drug rows.
    """
    pairs = case.arrays["train_pairs"]
    return np.unique(pairs[:, 0]), np.unique(pairs[:, 1])


def _rename(key: str) -> str:
    """Translate a 1.6.1 parameter name into the v2 one.

    :param key: Name in the 1.6.1 ``state_dict``.
    :returns: Name in the v2 ``state_dict``.
    """
    prefix, _, rest = key.partition(".")
    return ".".join(filter(None, [RENAMES.get(prefix, prefix), rest]))


def _forward_outputs(net: Any, case: Case, config: str) -> dict[str, np.ndarray]:
    """Run ``encode`` and ``score_pairs`` of an eval-mode network on the probe of *config*.

    :param net: 1.6.1 ``_MFNet`` or v2 ``MFNet`` holding the case's weights.
    :param case: The shared case.
    :param config: Forward configuration name.
    :returns: Latent factors with and without explicit id-embedding rows, and the pair scores.
    """
    import torch

    def tensor(name: str, dtype: Any = None) -> Any:
        return torch.as_tensor(case.arrays[f"forward/{config}/{name}"], dtype=dtype)

    net.eval()
    with torch.no_grad():
        z_sample, z_drug = net.encode(tensor("x_sample"), tensor("x_drug"))
        _, z_drug_rows = net.encode(tensor("x_sample"), tensor("x_drug"), tensor("rows"))
        scores = net.score_pairs(z_sample[tensor("sample_idx", torch.long)], z_drug[tensor("drug_idx", torch.long)])
    return {
        f"forward/{config}/z_sample": z_sample.numpy(),
        f"forward/{config}/z_drug": z_drug.numpy(),
        f"forward/{config}/z_drug_rows": z_drug_rows.numpy(),
        f"forward/{config}/scores": scores.numpy(),
    }


def _weights(case: Case, config: str) -> dict[str, Any]:
    """Return the case's weights of *config* as torch tensors under the 1.6.1 names.

    :param case: The shared case.
    :param config: Forward configuration name.
    :returns: ``state_dict`` mapping.
    """
    import torch

    prefix = f"forward/{config}/w/"
    return {key[len(prefix) :]: torch.as_tensor(value) for key, value in case.arrays.items() if key.startswith(prefix)}


def run_v1(case: Case) -> dict[str, np.ndarray]:
    """Run the forward probes and train every variant with the 1.6.1 ``EnsembleMF`` model.

    :param case: The shared case.
    :returns: Arrays keyed ``forward/<config>/<quantity>`` and ``<variant>/<quantity>``.
    """
    from drevalpy.models.EnsembleMF.ensemble_mf import EnsembleMF, _MFNet
    from sklearn.preprocessing import StandardScaler

    from tools.v1_parity._v1_runner import tabular_inputs

    out: dict[str, np.ndarray] = {}
    for config, setting in case.meta["forward"].items():
        net = _MFNet(
            cell_in_dim=setting["sample_in_dim"],
            drug_in_dim=setting["drug_in_dim"],
            hidden_dim=setting["hidden_dim"],
            emb_dim=setting["emb_dim"],
            n_layers=setting["n_layers"],
            dropout=0.2,
            n_drugs=setting["n_drugs"],
            use_drug_id_embedding=setting["use_drug_id_embedding"],
            use_mlp_head=setting["use_mlp_head"],
            mlp_hidden=setting["mlp_hidden"],
        )
        net.load_state_dict(_weights(case, config))
        out.update(_forward_outputs(net, case, config))

    data = tabular_inputs(case)
    names = list(case.arrays["gene_names"])
    columns = [names.index(g) for g in case.arrays["gene_list"]]
    for views in data.cell_line_input.features.values():
        views["gene_expression"] = views["gene_expression"][columns]
    data.cell_line_input.meta_info["gene_expression"] = np.array(case.arrays["gene_list"])
    stop = case.arrays["stop_mask"]
    fit_set, stop_set = data.train.copy(), data.train.copy()
    fit_set.mask(~stop)
    stop_set.mask(stop)
    sample_rows, drug_rows = _fit_rows(case)

    for variant, setting in case.meta["variants"].items():
        model = EnsembleMF()
        model.build_model({**setting, "feature_transform": "rank", "gene_list": "landmark_genes"})
        sink: dict[str, Any] = {}
        with record_calls(StandardScaler, "transform", sink, "rank"):
            model.train(fit_set, data.cell_line_input, data.drug_input, stop_set)
        predictions = model.predict(data.test.cell_line_ids, data.test.drug_ids, data.cell_line_input, data.drug_input)
        scaled = model._scaler.transform(sink["rank"])
        out[f"{variant}/sample_ids"] = case.arrays["sample_ids"][sample_rows]
        out[f"{variant}/drug_ids"] = case.arrays["drug_ids"][drug_rows]
        out[f"{variant}/rank"] = sink["rank"][sample_rows]
        out[f"{variant}/features_sample"] = model._x_cell.cpu().numpy()[sample_rows]
        out[f"{variant}/features_sample_float64"] = scaled[sample_rows]
        out[f"{variant}/features_drug"] = model._x_drug.cpu().numpy()[drug_rows]
        out[f"{variant}/predictions"] = np.asarray(predictions, dtype=np.float64)
        for index, net in enumerate(model.nets):
            state = {_rename(key): value for key, value in net.state_dict().items()}
            for key in _member_keys(setting):
                out[f"{variant}/member{index}/{key}"] = state[key].cpu().numpy()
    return out


def run_v2(case: Case) -> dict[str, np.ndarray]:
    """Run the forward probes and train every variant with the v2 ``EnsembleMF`` zoo preset.

    :param case: The shared case.
    :returns: Arrays keyed like those of :func:`run_v1`.
    """
    from drevalpy.components.predictors.literature.ensemblemf.network import MFNet, MFNetConfig
    from drevalpy.components.predictors.literature.ensemblemf.predictor import EnsembleMFPredictor
    from drevalpy.models.drp_model import DRPModel
    from drevalpy.models.spec import ModelSpec
    from drevalpy.types.data.split_mask import SplitMask
    from tools.v1_parity._v2_runner import tabular_inputs

    out: dict[str, np.ndarray] = {}
    for config, setting in case.meta["forward"].items():
        net = MFNet(MFNetConfig(**setting, dropout=0.2))
        net.load_state_dict({_rename(key): value for key, value in _weights(case, config).items()}, strict=True)
        out.update(_forward_outputs(net, case, config))

    data = tabular_inputs(case)
    shape = (len(case.arrays["sample_ids"]), len(case.arrays["drug_ids"]))
    stop = case.arrays["stop_mask"]
    fit_scope = SplitMask.from_pairs(case.arrays["train_pairs"][~stop], shape)
    stop_scope = SplitMask.from_pairs(case.arrays["train_pairs"][stop], shape)
    names = list(case.arrays["gene_names"])
    columns = np.array([names.index(g) for g in case.arrays["gene_list"]])
    sample_rows, _ = _fit_rows(case)

    for variant, setting in case.meta["variants"].items():
        captured: dict[str, Any] = {}
        original = EnsembleMFPredictor._fit

        def capture(self: Any, batch: Any, _original: Any = original, _sink: dict[str, Any] = captured) -> None:
            _sink["batch"] = batch
            _original(self, batch)

        EnsembleMFPredictor._fit = capture  # type: ignore[method-assign]
        try:
            model = DRPModel(ModelSpec.from_name(NAME), dict(setting))
            model.train(data.dataset, fit_scope, early_stopping_scope=stop_scope)
        finally:
            EnsembleMFPredictor._fit = original  # type: ignore[method-assign]
        predictions = model.predict(data.dataset, data.test)

        batch = captured["batch"]
        sample_order = np.argsort(batch.sample.entity_ids.astype(str), kind="stable")
        drug_order = np.argsort(batch.drug.entity_ids.astype(str), kind="stable")
        featurizer = model._pipeline._sample_featurizer
        raw = case.arrays["expression"][sample_rows][:, columns]
        scaled = batch.sample.blocks["gene_expression"].values[sample_order]
        out[f"{variant}/sample_ids"] = batch.sample.entity_ids[sample_order].astype(str)
        out[f"{variant}/drug_ids"] = batch.drug.entity_ids[drug_order].astype(str)
        out[f"{variant}/rank"] = featurizer._transform_values(raw)
        out[f"{variant}/features_sample"] = scaled
        out[f"{variant}/features_sample_float64"] = scaled
        out[f"{variant}/features_drug"] = batch.drug.blocks["fingerprints"].values[drug_order]
        out[f"{variant}/predictions"] = np.asarray(predictions, dtype=np.float64)
        for index, net in enumerate(model._pipeline._predictor._nets):
            state = net.state_dict()
            for key in _member_keys(setting):
                out[f"{variant}/member{index}/{key}"] = state[key].cpu().numpy()
    return out
