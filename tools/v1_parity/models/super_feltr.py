"""SuperFELTR parity case: 1.6.1 ``SuperFELTR`` against the v2 ``SuperFELTR`` preset.

Steps:

``forward/<omic>/{eval,train}``, ``forward/regressor/{eval,train}``, ``forward/full/predict``
    The 1.6.1 and v2 encoders (one per omic type), the regression head and the whole prediction path
    (``SuperFELTRegressor.predict`` over three loaded encoders) with identical weights on the same probes. The
    weights are drawn per ``state_dict`` key of the v2 modules and loaded with ``strict=True`` on both sides.
``<variant>/{selected_*,features_*}``
    What enters the networks for the samples of the last drug: the per-view variance selection (``arcsinh`` on the
    expression first) of the expression, mutation and copy number columns. The views have no gene list on either
    side (1.6.1's ``gene_lists`` default is empty, the preset names none), so the case holds synthetic genes.
``<variant>/<drug>/{params_*,predictions}``
    A short seeded CPU training per drug (three encoders with the triplet loss and Adam, then the regression head
    with Adagrad) and the test predictions. 1.6.1's loaders are made single-process (``inline_loaders``); the
    triplets of both sides come from the same explicit seed sequence (``seeded_triplets``).

Variants: ``default`` (early-stopping samples), ``no_es``, ``ragged`` (the last drug lacks some training samples:
1.6.1 selects per drug, v2 once) and ``binary`` (0/1 mutations, whose variances tie, so the selection depends on how
``argsort`` breaks ties).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import numpy as np

from tools.v1_parity._bridge import Case, seed_all
from tools.v1_parity.compare import DEVIATION, PARITY, StepSpec
from tools.v1_parity.models._three_omics import (
    DRUGS,
    N_DRUGS,
    OMIC_GROUPS,
    VIEWS,
    build_three_omics_case,
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

NAME = "SuperFELTR"

PROBE_ROWS = 7
PROBE_DROPOUT_SEED = 17
RANGES = (0.1, 1.0)
OMIC_TYPES = {"expression": 12, "mutation": 10, "copy_number_variation_gistic": 9}
BASE_HYPERPARAMETERS = {
    "mini_batch": 8,
    "dropout_rate": 0.3,
    "weight_decay": 0.01,
    "out_dim_expr_encoder": 6,
    "out_dim_mutation_encoder": 5,
    "out_dim_cnv_encoder": 4,
    "epochs": 3,
    "margin": 1.0,
    "learning_rate": 0.005,
    "n_features_per_view": 30,
}
VARIANTS: dict[str, dict[str, Any]] = {
    "default": {"hyperparameters": {}, "early_stopping": True, "seed": 3},
    "no_es": {"hyperparameters": {"epochs": 2}, "early_stopping": False, "seed": 4},
    "ragged": {
        "hyperparameters": {"epochs": 1},
        "early_stopping": True,
        "seed": 7,
        "ragged": True,
        "features_only": True,
        "scope": DEVIATION,
    },
    "binary": {
        "hyperparameters": {"epochs": 1},
        "early_stopping": True,
        "seed": 8,
        "binary": True,
        "features_only": True,
    },
}
FEATURE_QUANTITIES = (
    ("selected_expression", "exact"),
    ("selected_mutations", "exact"),
    ("selected_cnv", "exact"),
    ("features_expression", "forward"),
    ("features_mutations", "exact"),
    ("features_cnv", "exact"),
)
RAGGED_EXPLANATION = (
    "1.6.1 trains one model per drug and fits the variance selection of every view inside `train` on the samples of "
    "that drug's training pairs. v2 featurizes once, on the distinct samples of all training pairs, and then splits "
    "the pairs by drug. With a drug that lacks some of the training samples (the last drug here misses the first "
    "six), the selected columns of that drug differ from the shared selection; the case compares the selected "
    "columns and values of the last drug. Measured on 2026-10-23, the two sides share 22 of 30 expression, "
    "25 of 30 mutation and 29 of 30 copy number columns."
)

FINDINGS = [
    "Fixed (v2): a preset-built SuperFELTR could not finish a fit or predict, for the same reason as MOLIR: the "
    "`PredictionMode` enum in the predictor hyperparameters ended up in the Lightning checkpoints and "
    "`load_from_checkpoint` raised `UnpicklingError`. `build_components` now passes a plain string.",
    "The networks are the 1.6.1 networks: the encoders (linear, batch norm, ReLU, dropout), the regression head and "
    "the whole `predict` path load a 1.6.1 `state_dict` with `strict=True` and agree at the `forward` class.",
    "The regressor holds its three encoders differently. 1.6.1 keeps them in a plain tuple: they are not "
    "parameters of the module, not in its `state_dict`, and Lightning's `train()` does not reach them. v2 "
    "registers them (`nn.ModuleList`), freezes them (`requires_grad_(False)`) and overrides `train()` to keep them "
    "in eval mode. The optimizer sees only the head in both, and the encoders stay in eval mode in both, so the "
    "training agrees; the compared parameters are the head's (`regressor.*`).",
    "1.6.1 draws the triplets of the encoder fits from an unseeded generator, v2 from the global NumPy stream seeded "
    "per fit. The case passes the same explicit seed sequence to both, in the same call order (expression, "
    "mutation and copy number encoder, drug after drug).",
    "The encoder bias is in front of a batch norm layer, so its gradient is zero in exact arithmetic. With Adam "
    "and no weight decay the rounding noise would become full learning-rate steps (see PaccMann, "
    "SimpleNeuralNetwork); the case uses `weight_decay` 0.01 (the 1.6.1 and v2 default), which gives the "
    "bias a deterministic gradient, and continuous mutation and copy number values (a column that is constant in a "
    "batch has the same zero gradient).",
    "Variance selection: both select the top-variance columns per view on the distinct training samples and keep "
    "the dataset column order; the expression is passed through `arcsinh` first. 1.6.1 does that while loading the "
    "CSVs (the harness applies it on the in-memory 1.6.1 side), v2 inside the featurizer. v2 fits it once over "
    "all training pairs, 1.6.1 per drug (variant `ragged`, deviation).",
    "Neither side has a gene list: 1.6.1's `gene_lists` default is empty, the v2 preset `superfeltrOmics` names "
    "none (commit 174ceea7, B5). They agree.",
]


def _hpams(setting: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the full hyperparameter dictionary of a variant.

    :param setting: Variant entry, or ``None`` for the base hyperparameters.
    :returns: Base hyperparameters with the variant's changes.
    """
    return {**BASE_HYPERPARAMETERS, **(setting or {}).get("hyperparameters", {})}


def _modules(module: Any, case: Case | None = None, rng: np.random.Generator | None = None) -> dict[str, Any]:
    """Build the three encoders and the regression head of one implementation.

    With *rng* the weights are drawn into the arrays (v2 side, ``build_case``); with *case* they are loaded.

    :param module: ``utils`` module of the implementation.
    :param case: The shared case holding the weights, or ``None`` when drawing.
    :param rng: Generator that draws the weights, or ``None`` when loading.
    :returns: ``{"encoders": {omic: encoder}, "regressor": regressor, "arrays": drawn arrays}``.
    """
    hpams = _hpams()
    arrays: dict[str, np.ndarray] = {}
    encoders = {}
    for omic, width in OMIC_TYPES.items():
        encoder = module.SuperFELTEncoder(input_size=width, hpams=hpams, omic_type=omic, ranges=RANGES)
        if rng is not None:
            arrays.update(draw_weights(encoder, rng, f"forward/{omic}/w/"))
        else:
            load_weights(encoder, case, f"forward/{omic}/w/")
        encoders[omic] = encoder
    regressor = module.SuperFELTRegressor(input_size=15, hpams=hpams, encoders=tuple(encoders.values()))
    if rng is not None:
        drawn = draw_weights(regressor, rng, "forward/regressor/w/")
        arrays.update({k: v for k, v in drawn.items() if not k.startswith("forward/regressor/w/encoders.")})
    else:
        load_weights(regressor, case, "forward/regressor/w/")
    return {"encoders": encoders, "regressor": regressor, "arrays": arrays}


def build_case(seed: int) -> Case:
    """Draw the three omics views, the pair splits and the forward weights and probes.

    :param seed: Seed of the input generator.
    :returns: The case.
    """
    from drevalpy.components.predictors.literature.superfeltr import utils

    case = build_three_omics_case(NAME, seed, VARIANTS, dict.fromkeys(VIEWS))
    rng = np.random.default_rng(seed + 5)
    case.arrays.update(_modules(utils, rng=rng)["arrays"])
    for omic, width in OMIC_TYPES.items():
        case.arrays[f"forward/{omic}/probe"] = rng.normal(size=(PROBE_ROWS, width)).astype(np.float32)
    case.arrays["forward/regressor/probe"] = rng.normal(size=(PROBE_ROWS, 15)).astype(np.float32)
    case.meta["omic_types"] = OMIC_TYPES
    return case


def steps() -> list[StepSpec]:
    """Declare the compared outputs.

    :returns: Forward, feature, training and prediction steps.
    """
    specs = [
        StepSpec(f"forward/{part}/{mode}", "forward", description=f"{part} forward pass, {mode} mode")
        for part in (*OMIC_TYPES, "regressor")
        for mode in ("eval", "train")
    ]
    specs.append(StepSpec("forward/full/predict", "forward", description="predict() over three loaded encoders"))
    for variant, setting in VARIANTS.items():
        scope = setting.get("scope", PARITY)
        for quantity, tolerance in FEATURE_QUANTITIES:
            specs.append(
                StepSpec(
                    f"{variant}/{quantity}",
                    tolerance,
                    scope=scope,
                    description=f"{variant}: {quantity.replace('_', ' ')} of the samples of the last drug",
                    explanation=RAGGED_EXPLANATION if scope == DEVIATION else "",
                )
            )
        if setting.get("features_only"):
            continue
        for drug in DRUGS:
            specs += [
                StepSpec(
                    f"{variant}/{drug}/params_{group}",
                    "train",
                    description=f"{variant}, {drug}: {group} parameters after training",
                )
                for group in OMIC_GROUPS
            ]
            specs.append(
                StepSpec(
                    f"{variant}/{drug}/predictions",
                    "train",
                    description=f"{variant}, {drug}: test predictions after the seeded short training",
                )
            )
    return specs


def _probe_outputs(module: Any, case: Case) -> dict[str, np.ndarray]:
    """Run the forward probes through the modules of one implementation.

    :param module: ``utils`` module of the implementation.
    :param case: The shared case.
    :returns: ``forward/<part>/{eval,train}`` and ``forward/full/predict``.
    """
    import torch

    built = _modules(module, case=case)
    out: dict[str, np.ndarray] = {}
    for omic, encoder in built["encoders"].items():
        probe = torch.as_tensor(case.arrays[f"forward/{omic}/probe"])
        for mode, values in forward_pair(encoder, (probe,), PROBE_DROPOUT_SEED).items():
            out[f"forward/{omic}/{mode}"] = values
    probe = torch.as_tensor(case.arrays["forward/regressor/probe"])
    for mode, values in forward_pair(built["regressor"], (probe,), PROBE_DROPOUT_SEED).items():
        out[f"forward/regressor/{mode}"] = values
    inputs = [case.arrays[f"forward/{omic}/probe"].astype(np.float32) for omic in OMIC_TYPES]
    out["forward/full/predict"] = np.asarray(built["regressor"].predict(*inputs), dtype=np.float64)
    return out


def _flat(state: dict[str, Any], prefix: str) -> np.ndarray:
    """Concatenate the floating-point ``state_dict`` entries that start with *prefix*.

    :param state: ``state_dict`` of a module.
    :param prefix: Key prefix.
    :returns: One float64 vector, entries in key order.
    """
    parts = [
        state[key].detach().cpu().numpy().astype(np.float64).ravel()
        for key in sorted(state)
        if key.startswith(prefix) and state[key].is_floating_point()
    ]
    return np.concatenate(parts)


def _trained_params(variant: str, drug: str, encoders: tuple[Any, Any, Any], regressor: Any) -> dict[str, np.ndarray]:
    """Read the parameters of the three encoders and of the head.

    :param variant: Variant name.
    :param drug: Drug id.
    :param encoders: Expression, mutation and copy number encoders.
    :param regressor: The regression head module.
    :returns: ``<variant>/<drug>/params_<group>`` arrays.
    """
    out = {
        f"{variant}/{drug}/params_{group}": _flat(encoder.state_dict(), "")
        for group, encoder in zip(("expression", "mutation", "cnv"), encoders, strict=True)
    }
    out[f"{variant}/{drug}/params_regressor"] = _flat(regressor.state_dict(), "regressor.")
    return out


@contextmanager
def _capture_v1_train(sink: dict[str, Any], drug: str) -> Any:
    """Store the (selected) features that a 1.6.1 ``SuperFELTR.train`` leaves in its feature dataset.

    The selectors transform the dataset passed to ``train`` in place, so it is read after the call.

    :param sink: Dictionary receiving ``sink[drug]``.
    :param drug: Drug id the fit belongs to.
    :yields: Nothing; the patch is undone on exit.
    """
    from drevalpy.models.SuperFELTR import superfeltr

    original = superfeltr.SuperFELTR.train

    def train(self: Any, output: Any, cell_line_input: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(self, output, cell_line_input, *args, **kwargs)
        stop = kwargs.get("output_earlystopping", args[1] if len(args) > 1 else None)
        ids = np.unique(np.concatenate([output.cell_line_ids, stop.cell_line_ids if stop is not None else []]))
        views = (
            ("expression", "gene_expression"),
            ("mutations", "mutations"),
            ("cnv", "copy_number_variation_gistic"),
        )
        record: dict[str, Any] = {"ids": ids.astype(str)}
        for key, view in views:
            record[f"features_{key}"] = np.vstack([cell_line_input.features[str(i)][view] for i in ids])
            record[f"selected_{key}"] = np.array(cell_line_input.meta_info[view]).astype(str)
        sink[drug] = record
        return result

    superfeltr.SuperFELTR.train = train
    try:
        yield
    finally:
        superfeltr.SuperFELTR.train = original


def _feature_outputs(variant: str, case: Case, setting: dict[str, Any], recorded: dict[str, Any]) -> dict[str, Any]:
    """Select the rows of the last drug's samples from a recorded feature set.

    :param variant: Variant name.
    :param case: The shared case.
    :param setting: Variant settings.
    :param recorded: ``ids``, ``selected_*`` and ``features_*`` of one fit.
    :returns: ``<variant>/<quantity>`` arrays.
    """
    wanted = used_sample_ids(case, setting, N_DRUGS - 1)
    rows = [list(recorded["ids"]).index(sid) for sid in wanted]
    out = {}
    for quantity, _ in FEATURE_QUANTITIES:
        value = recorded[quantity]
        out[f"{variant}/{quantity}"] = value[rows] if quantity.startswith("features") else value
    return out


def run_v1(case: Case) -> dict[str, np.ndarray]:
    """Run the forward probes and every variant with the 1.6.1 ``SuperFELTR`` model.

    :param case: The shared case.
    :returns: Arrays keyed by the step names.
    """
    from drevalpy.models.MOLIR import utils as molir_utils
    from drevalpy.models.SuperFELTR import superfeltr, utils

    out = _probe_outputs(utils, case)

    for variant, setting in case.meta["variants"].items():
        hyperparameters = _hpams(setting)

        def make_model(hyperparameters: dict[str, Any] = hyperparameters) -> Any:
            model = superfeltr.SuperFELTR()
            model.build_model(dict(hyperparameters))
            return model

        cell_lines = v1_cell_lines(case, setting, arcsinh=True)
        seed_all(setting["seed"])
        with inline_loaders(molir_utils), seeded_triplets(utils):
            predictions, trained = run_v1_drugs(case, setting, cell_lines, make_model, _capture_v1_train)
        for drug, model in trained["models"].items():
            encoders = (model.expr_encoder, model.mut_encoder, model.cnv_encoder)
            out.update(_trained_params(variant, drug, encoders, model.regressor))
        out.update({f"{variant}/{key}": value for key, value in predictions.items()})
        out.update(_feature_outputs(variant, case, setting, trained["sink"][DRUGS[-1]]))
    return out


def run_v2(case: Case) -> dict[str, np.ndarray]:
    """Run the forward probes and every variant with the v2 ``SuperFELTR`` preset.

    :param case: The shared case.
    :returns: Arrays keyed like those of :func:`run_v1`.
    """
    from drevalpy.components.predictors.literature.superfeltr import utils
    from drevalpy.components.predictors.literature.superfeltr.predictor import SuperFELTRPredictor
    from drevalpy.types.data.split_mask import SplitMask

    out = _probe_outputs(utils, case)
    shape = (len(case.arrays["sample_ids"]), N_DRUGS)
    test = SplitMask.from_pairs(case.arrays["test_pairs"], shape)

    for variant, setting in case.meta["variants"].items():
        dataset = v2_three_omics_dataset(case, setting)
        pairs, stop = variant_pairs(case, setting)
        fit_scope = SplitMask.from_pairs(pairs[~stop], shape)
        stop_scope = SplitMask.from_pairs(pairs[stop], shape) if setting["early_stopping"] else None
        model = v2_model(NAME, _hpams(setting))
        captured: dict[str, Any] = {}
        seed_all(setting["seed"])
        with capture_fit(SuperFELTRPredictor, captured), seeded_triplets(utils):
            model.train(dataset, fit_scope, early_stopping_scope=stop_scope)
            predictions = np.asarray(model.predict(dataset, test), dtype=np.float64)

        batch = captured["batch"]
        blocks = batch.sample.blocks
        recorded: dict[str, Any] = {"ids": np.array([str(i) for i in batch.sample.entity_ids])}
        for key, view in (
            ("expression", "gene_expression"),
            ("mutations", "mutations"),
            ("cnv", "copy_number_variation_gistic"),
        ):
            recorded[f"features_{key}"] = np.asarray(blocks[view].values)
            recorded[f"selected_{key}"] = np.array(blocks[view].feature_names).astype(str)
        out.update(_feature_outputs(variant, case, setting, recorded))
        trained = model._pipeline._predictor._models
        for index, drug in enumerate(DRUGS):
            of_drug = case.arrays["test_pairs"][:, 1] == index
            out[f"{variant}/{drug}/predictions"] = np.atleast_1d(predictions[of_drug])
            fitted = trained[drug]
            encoders = (fitted.expr_encoder, fitted.mut_encoder, fitted.cnv_encoder)
            out.update(_trained_params(variant, drug, encoders, fitted.regressor))
    return out
