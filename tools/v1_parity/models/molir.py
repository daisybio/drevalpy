"""MOLIR parity case: 1.6.1 ``MOLIR`` against the v2 ``MOLIR`` preset.

Steps:

``forward/<config>/{eval,train}``
    The 1.6.1 ``MOLIModel`` and the v2 one with identical weights on the same three probes, in eval mode and in
    train mode (batch statistics and dropout under the same torch seed). The weights are drawn per ``state_dict``
    key of the v2 network and loaded with ``strict=True`` on both sides.
``<variant>/{features_expression,features_mutations,features_cnv,selected_genes}``
    What enters the network for the samples of the last drug: arcsinh + ``StandardScaler`` + variance selection of
    the expression, the mutation and copy number columns of their gene lists. 1.6.1 applies the gene lists while
    loading CSVs, which the in-memory harness skips, so its side receives the columns in list order.
``<variant>/<drug>/{params_*,predictions}``
    A short seeded CPU training per drug and the test predictions. The training of every drug is one model, as
    in 1.6.1 (``experiment.make_model_list``) and in the v2 ``SingleDrugOmicsPredictor``. 1.6.1's loaders are made
    single-process (``inline_loaders``). The triplets of both sides are drawn from the same explicit seed sequence
    (``seeded_triplets``): 1.6.1 draws them from an unseeded generator, v2 from the global NumPy stream, so a
    free-running comparison is not possible.

Variants: ``default`` (gamma 1, early-stopping samples), ``no_es`` (the training loss is monitored), ``gamma``
(gamma 0.5: v2 weights the triplet term, 1.6.1 ignores gamma), ``select`` (fewer expression features than genes:
1.6.1 ranks the variance after standardization, v2 before) and ``ragged`` (the last drug lacks some training
samples: 1.6.1 fits scaler and selection per drug, v2 once on all training samples).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import numpy as np

from tools.v1_parity._bridge import Case, seed_all
from tools.v1_parity.compare import DEVIATION, PARITY, StepSpec
from tools.v1_parity.models._three_omics import (
    DEFAULT_LISTS,
    DRUGS,
    N_DRUGS,
    OMIC_GROUPS,
    build_three_omics_case,
    grouped_params,
    run_v1_drugs,
    seeded_triplets,
    used_sample_ids,
    v1_cell_lines,
    v2_three_omics_dataset,
    variant_pairs,
)
from tools.v1_parity.models._torch_common import (
    capture_fit,
    draw_weights,
    forward_pair,
    inline_loaders,
    load_weights,
    v2_model,
)

NAME = "MOLIR"

PROBE_ROWS = 7
PROBE_DROPOUT_SEED = 13
BASE_HYPERPARAMETERS = {
    "mini_batch": 8,
    "h_dim1": 6,
    "h_dim2": 5,
    "h_dim3": 4,
    "learning_rate": 0.0003,
    "dropout_rate": 0.3,
    "weight_decay": 1e-4,
    "gamma": 1.0,
    "epochs": 4,
    "margin": 1.5,
    "n_gene_expression_features": 5000,
}
VARIANTS: dict[str, dict[str, Any]] = {
    "default": {"hyperparameters": {}, "early_stopping": True, "seed": 3},
    "no_es": {"hyperparameters": {"epochs": 3}, "early_stopping": False, "seed": 4},
    "gamma": {"hyperparameters": {"gamma": 0.5}, "early_stopping": False, "seed": 5, "scope": DEVIATION},
    "select": {
        "hyperparameters": {"n_gene_expression_features": 50, "epochs": 1},
        "early_stopping": True,
        "seed": 6,
        "features_only": True,
        "scope": DEVIATION,
    },
    "ragged": {
        "hyperparameters": {"epochs": 1},
        "early_stopping": True,
        "seed": 7,
        "ragged": True,
        "features_only": True,
        "scope": DEVIATION,
    },
}
FORWARD_CONFIGS: dict[str, dict[str, Any]] = {
    "small": {"widths": (13, 9, 11), "h": (6, 5, 4), "dropout_rate": 0.3},
    "wide": {"widths": (20, 7, 8), "h": (8, 3, 5), "dropout_rate": 0.5},
}

GAMMA_EXPLANATION = (
    "v2 weights the triplet term of the loss by gamma (`gamma * triplet + regression`, the correction recorded in "
    "`training-corrections.json::molir_gamma_weighting`); 1.6.1 reads `gamma` into `self.gamma` and never uses it "
    "(`triplet + regression`). With gamma 1 the two losses are the same, with another value the gradients of the "
    "encoders differ from the first step on. The variant sets gamma 0.5 on both sides; "
    "measured on 2026-10-23, the test predictions differ by up to 0.019 (drug D00, range 0.30) and 0.012 (D01, "
    "range 0.50), correlation 0.9989 and 0.9996."
)
SELECT_EXPLANATION = (
    "1.6.1 ranks the variance of the expression columns after arcsinh and standardization, where every column "
    "has variance 1 up to float rounding, so the top-k columns are decided by rounding noise; v2 ranks the "
    "variance of the arcsinh values before standardizing (documented in `literature_models.rst`). The variant "
    "keeps 50 of the 2269 genes; measured on 2026-10-23, the two sides share 2 of their 50 genes (about 1 by chance)."
)
RAGGED_EXPLANATION = (
    "1.6.1 trains one model per drug and fits the scaler (and the variance selection) inside `train` on the "
    "samples of that drug's training pairs. v2 featurizes once, on the distinct samples of all training pairs, "
    "and then splits the pairs by drug. With a drug that lacks some of the training samples (the last drug here "
    "misses the first six), the scaler of that drug differs from the shared one; the case compares the features "
    "of the samples of the last drug; measured on 2026-10-23, the standardized expression differs by up to 2.8."
)

FINDINGS = [
    "Fixed (v2): a preset-built MOLIR could not finish a fit with an early-stopping scope. `build_components` put "
    "the `PredictionMode` enum into the predictor hyperparameters, `save_hyperparameters()` stored it in the "
    "checkpoint, and the `weights_only=True` reload of the best epoch raised `UnpicklingError`. It now passes the "
    "mode as a plain string; the case needs no allow-list.",
    "The `MOLIModel` of v2 is the 1.6.1 network: a `state_dict` of one loads into the other with `strict=True`, "
    "and eval and train mode (batch statistics, dropout, the `normalize(dim=0)` over the batch) agree at the "
    "`forward` class.",
    "1.6.1 draws the triplets from `np.random.default_rng(None)` (unseeded: runs are not reproducible), v2 from the "
    "global NumPy stream seeded per fit. The case passes the same explicit seed sequence to both (the "
    "`random_seed` argument both functions have), so the training steps compare the algorithm and not the "
    "draws. A free-running 1.6.1 fit cannot be reproduced at all.",
    "v2 weights the triplet term by `gamma`; 1.6.1 ignores `gamma` (variant `gamma`, deviation). With gamma 1 "
    "(variants `default`, `no_es`) the training agrees at the `train` class.",
    "v2 selects the highest-variance expression genes before standardization, 1.6.1 after it (variant `select`, "
    "deviation). With `n_gene_expression_features` at least the gene count both keep all genes in the column "
    "order of the list and agree exactly. Scaling and selection use the training rows only on both sides.",
    "v2 fits the scaler and the selection once on the samples of all training pairs; 1.6.1 fits them per drug on "
    "that drug's samples (variant `ragged`, deviation). The two agree when every drug has the same training "
    "samples, which is the case for the complete response matrix of the other variants.",
    "The v2 preset (commit 174ceea7, B5) carries the three `*_intersection` gene "
    "lists of the 1.6.1 defaults (`MOLIR.default_gene_lists`); the case does not override them and the selected "
    "columns agree. Mutation and copy number columns pass through unchanged on both sides.",
]


def _network_dims(setting: dict[str, Any]) -> tuple[dict[str, Any], tuple[int, ...]]:
    """Return the ``hpams`` dictionary of a forward configuration and its input widths.

    :param setting: Entry of ``FORWARD_CONFIGS``.
    :returns: ``(hpams, widths)``.
    """
    hpams = {
        **BASE_HYPERPARAMETERS,
        "h_dim1": setting["h"][0],
        "h_dim2": setting["h"][1],
        "h_dim3": setting["h"][2],
        "dropout_rate": setting["dropout_rate"],
    }
    return hpams, tuple(setting["widths"])


def _forward_arrays(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Draw weights and probes for every forward configuration from the v2 network's shapes.

    :param rng: Generator of the case.
    :returns: Arrays ``forward/<config>/w/<key>`` and ``forward/<config>/probe_{0,1,2}``.
    """
    from drevalpy.components.predictors.literature.molir.utils import MOLIModel

    arrays: dict[str, np.ndarray] = {}
    for config, setting in FORWARD_CONFIGS.items():
        hpams, widths = _network_dims(setting)
        network = MOLIModel(hpams, *widths)
        arrays.update(draw_weights(network, rng, f"forward/{config}/w/"))
        for index, width in enumerate(widths):
            arrays[f"forward/{config}/probe_{index}"] = rng.normal(size=(PROBE_ROWS, width)).astype(np.float32)
    return arrays


def build_case(seed: int) -> Case:
    """Draw the three omics views, the pair splits and the forward probes.

    :param seed: Seed of the input generator.
    :returns: The case.
    """
    case = build_three_omics_case(NAME, seed, VARIANTS, DEFAULT_LISTS)
    rng = np.random.default_rng(seed + 5)
    case.arrays.update(_forward_arrays(rng))
    case.meta["forward"] = FORWARD_CONFIGS
    return case


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
        scope = setting.get("scope", PARITY)
        explanation = {"select": SELECT_EXPLANATION, "ragged": RAGGED_EXPLANATION}.get(variant, "")
        deviating = {"select": ("selected_genes", "features_expression"), "ragged": ("features_expression",)}
        for quantity, tolerance in (
            ("selected_genes", "exact"),
            ("features_expression", "forward"),
            ("features_mutations", "exact"),
            ("features_cnv", "exact"),
        ):
            deviates = quantity in deviating.get(variant, ())
            specs.append(
                StepSpec(
                    f"{variant}/{quantity}",
                    tolerance,
                    scope=DEVIATION if deviates else PARITY,
                    description=f"{variant}: {quantity.replace('_', ' ')} of the samples of the last drug",
                    explanation=explanation if deviates else "",
                )
            )
        if setting.get("features_only"):
            continue
        for drug in DRUGS:
            for group in OMIC_GROUPS:
                specs.append(
                    StepSpec(
                        f"{variant}/{drug}/params_{group}",
                        "train",
                        scope=scope,
                        description=f"{variant}, {drug}: {group} parameters after training",
                        explanation=GAMMA_EXPLANATION if scope == DEVIATION else "",
                    )
                )
            specs.append(
                StepSpec(
                    f"{variant}/{drug}/predictions",
                    "train",
                    scope=scope,
                    description=f"{variant}, {drug}: test predictions after the seeded short training",
                    explanation=GAMMA_EXPLANATION if scope == DEVIATION else "",
                )
            )
    return specs


def _probe_outputs(network_class: Any, case: Case) -> dict[str, np.ndarray]:
    """Run the forward probes through *network_class* with the case's weights.

    :param network_class: ``MOLIModel`` of either side.
    :param case: The shared case.
    :returns: ``forward/<config>/{eval,train}``.
    """
    import torch

    out: dict[str, np.ndarray] = {}
    for config, setting in case.meta["forward"].items():
        hpams, widths = _network_dims(setting)
        network = network_class(hpams, *widths)
        load_weights(network, case, f"forward/{config}/w/")
        probes = tuple(torch.as_tensor(case.arrays[f"forward/{config}/probe_{i}"]) for i in range(3))
        for mode, values in forward_pair(network, probes, PROBE_DROPOUT_SEED).items():
            out[f"forward/{config}/{mode}"] = values
    return out


def _hyperparameters(setting: dict[str, Any]) -> dict[str, Any]:
    """Return the full hyperparameter dictionary of a variant.

    :param setting: Variant entry.
    :returns: Base hyperparameters with the variant's changes.
    """
    return {**BASE_HYPERPARAMETERS, **setting["hyperparameters"]}


@contextmanager
def _capture_v1_fit(sink: dict[str, Any], drug: str) -> Any:
    """Store the (scaled, selected) features that a 1.6.1 ``MOLIModel.fit`` receives.

    :param sink: Dictionary receiving ``sink[drug]``.
    :param drug: Drug id the fit belongs to.
    :yields: Nothing; the patch is undone on exit.
    """
    from drevalpy.models.MOLIR import utils

    original = utils.MOLIModel.fit

    def fit(self: Any, *args: Any, **kwargs: Any) -> Any:
        train, features = kwargs["output_train"], kwargs["cell_line_input"]
        stop = kwargs.get("output_earlystopping")
        ids = np.unique(np.concatenate([train.cell_line_ids, stop.cell_line_ids if stop is not None else []]))
        views = (
            ("features_expression", "gene_expression"),
            ("features_mutations", "mutations"),
            ("features_cnv", "copy_number_variation_gistic"),
        )
        sink[drug] = {
            "ids": ids.astype(str),
            "genes": np.array(features.meta_info["gene_expression"]).astype(str),
            **{key: np.vstack([features.features[str(i)][view] for i in ids]) for key, view in views},
        }
        return original(self, *args, **kwargs)

    utils.MOLIModel.fit = fit
    try:
        yield
    finally:
        utils.MOLIModel.fit = original


def _feature_outputs(variant: str, case: Case, setting: dict[str, Any], recorded: dict[str, Any]) -> dict[str, Any]:
    """Select the rows of the last drug's samples from a recorded feature set.

    :param variant: Variant name.
    :param case: The shared case.
    :param setting: Variant settings.
    :param recorded: ``ids``, ``genes`` and the three matrices of one fit.
    :returns: ``<variant>/<quantity>`` arrays.
    """
    wanted = used_sample_ids(case, setting, N_DRUGS - 1)
    rows = [list(recorded["ids"]).index(sid) for sid in wanted]
    keys = ("features_expression", "features_mutations", "features_cnv")
    out = {f"{variant}/{key}": recorded[key][rows] for key in keys}
    out[f"{variant}/selected_genes"] = recorded["genes"]
    return out


def run_v1(case: Case) -> dict[str, np.ndarray]:
    """Run the forward probes and every variant with the 1.6.1 ``MOLIR`` model.

    :param case: The shared case.
    :returns: Arrays keyed by the step names.
    """
    from drevalpy.models.MOLIR import molir, utils

    out = _probe_outputs(utils.MOLIModel, case)
    cell_lines = v1_cell_lines(case)

    for variant, setting in case.meta["variants"].items():
        hyperparameters = _hyperparameters(setting)

        def make_model(hyperparameters: dict[str, Any] = hyperparameters) -> Any:
            model = molir.MOLIR()
            model.build_model(dict(hyperparameters))
            return model

        seed_all(setting["seed"])
        with inline_loaders(utils), seeded_triplets(utils):
            predictions, trained = run_v1_drugs(case, setting, cell_lines, make_model, _capture_v1_fit)
        for drug, model in trained["models"].items():
            out.update({f"{variant}/{drug}/{k}": v for k, v in grouped_params(model.model.state_dict()).items()})
        out.update({f"{variant}/{key}": value for key, value in predictions.items()})
        out.update(_feature_outputs(variant, case, setting, trained["sink"][DRUGS[-1]]))
    return out


def run_v2(case: Case) -> dict[str, np.ndarray]:
    """Run the forward probes and every variant with the v2 ``MOLIR`` preset.

    :param case: The shared case.
    :returns: Arrays keyed like those of :func:`run_v1`.
    """
    from drevalpy.components.predictors.literature.molir import utils
    from drevalpy.components.predictors.literature.molir.predictor import MOLIRPredictor
    from drevalpy.types.data.split_mask import SplitMask

    out = _probe_outputs(utils.MOLIModel, case)
    dataset = v2_three_omics_dataset(case)
    shape = (len(case.arrays["sample_ids"]), N_DRUGS)
    test = SplitMask.from_pairs(case.arrays["test_pairs"], shape)

    for variant, setting in case.meta["variants"].items():
        pairs, stop = variant_pairs(case, setting)
        fit_scope = SplitMask.from_pairs(pairs[~stop], shape)
        stop_scope = SplitMask.from_pairs(pairs[stop], shape) if setting["early_stopping"] else None
        model = v2_model(NAME, _hyperparameters(setting))
        captured: dict[str, Any] = {}
        seed_all(setting["seed"])
        with capture_fit(MOLIRPredictor, captured), seeded_triplets(utils):
            model.train(dataset, fit_scope, early_stopping_scope=stop_scope)
        predictions = np.asarray(model.predict(dataset, test), dtype=np.float64)

        batch = captured["batch"]
        blocks = batch.sample.blocks
        recorded = {
            "ids": np.array([str(i) for i in batch.sample.entity_ids]),
            "genes": np.array(blocks["gene_expression"].feature_names).astype(str),
            "features_expression": np.asarray(blocks["gene_expression"].values),
            "features_mutations": np.asarray(blocks["mutations"].values),
            "features_cnv": np.asarray(blocks["copy_number_variation_gistic"].values),
        }
        out.update(_feature_outputs(variant, case, setting, recorded))
        trained = model._pipeline._predictor._models
        for index, drug in enumerate(DRUGS):
            of_drug = case.arrays["test_pairs"][:, 1] == index
            out[f"{variant}/{drug}/predictions"] = np.atleast_1d(predictions[of_drug])
            out.update({f"{variant}/{drug}/{k}": v for k, v in grouped_params(trained[drug].state_dict()).items()})
    return out
