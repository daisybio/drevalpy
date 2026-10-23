"""RandomForestAEMolGNet parity case: 1.6.1 ``RandomForestAEMolGNet`` against the v2 zoo preset.

The model is the DIPK ablation of 1.6.1: a gene-expression autoencoder, pooled MolGNet node
embeddings and a ``RandomForestRegressor``. 1.6.1 loads expression and MolGNet from CSV files and the
v2 featurizers read a ``Dataset``, so the case compares the three stages on identical in-memory
inputs instead of through the loaders:

``pooling/<pooling>``
    The pooled vectors of drugs with different atom counts (one atom included), for ``mean_max``,
    ``mean`` and ``max``: 1.6.1 ``pool_molgnet`` against v2 ``pool_molgnet``.
``ae/*``
    The latent embedding of a held-out matrix and the first encoder weights after a short seeded
    training on the same matrices. The training matrix has ``n_train % batch_size == 1`` rows, so the
    singleton-batch rule (one row moves to the validation matrix) is exercised on both sides: 1.6.1
    ``train_gene_expession_autoencoder`` plus the rule of ``_fit_or_load_encoder`` against the v2
    ``aeGeneExpression`` featurizer.
``forest/*``
    1.6.1 ``RandomForestRegressor`` (unseeded in 1.6.1, here seeded with ``np.random.seed(42)`` just
    before the fit, which is the state ``RandomState(42)`` starts from) against the v2
    ``RandomForestPredictor`` (``random_state=42``), both with ``n_jobs=1``, on the same float32
    latent and pooled drug matrix. Predictions and feature importances are exact.
``forest/scaled`` (deviation)
    1.6.1 standardizes the latent of the training samples before the forest; v2 does not.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tools.v1_parity._bridge import Case, seed_all
from tools.v1_parity.compare import DEVIATION, StepSpec

NAME = "RandomForestAEMolGNet"

POOLINGS = ("mean_max", "mean", "max")
N_DIM = 16
ATOM_COUNTS = (1, 7, 12, 3, 20, 9)
N_GENES = 40
N_TRAIN, N_VALIDATION, N_TEST = 33, 10, 8
BATCH_SIZE = 16
AE_SETTINGS: dict[str, Any] = {"epochs_autoencoder": 3, "lr_autoencoder": 1e-3, "patience_autoencoder": 50}
FOREST = {"n_estimators": 20, "max_samples": 0.2, "max_depth": 5}
N_LATENT = 24
N_SAMPLES = 30
FOREST_SEED = 42

FINDINGS = [
    "1.6.1 `RandomForest` passes no `random_state`, so every 1.6.1 forest is unseeded and not reproducible. "
    "v2 `RandomForestPredictor` defaults to `random_state=42` (as the plan asks), which also changes "
    "`singleDrugRandomForest` and `MultiViewRandomForest`. Parity is shown by seeding the 1.6.1 global "
    "state with `np.random.seed(42)` before the fit, which equals `RandomState(42)`.",
    "Parity holds with `n_jobs=1`. The preset keeps `n_jobs=-1` (as 1.6.1); with several workers the "
    "predictions differ by one ulp from the sequential run, because the per-tree predictions are summed "
    "in a different order.",
    "1.6.1 `SklearnModel.train` standardizes the encoder latent (`scale_gene_expression`, fitted on the unique "
    "training samples) before the forest; v2 passes the raw latent. A tree split is invariant to a monotone "
    "per-feature rescaling, so only floating-point rounding of the thresholds can differ (deviation step "
    "`forest/scaled`).",
    "1.6.1 applies `log2(clip(x, 0) + 1)` to the TPM datasets by *name* (CCLE, CTRPv1, CTRPv2) and nothing "
    "else; v2 decides from the `expression_unit` of the view (`linear` -> log2), so a log-scale dataset is "
    "never transformed twice. Unlike DIPK there is no per-sample z-score in either version. Not exercised "
    "here: the case feeds the already log-scale matrices.",
    "v2 holds features as float32 end to end; the case uses float32-representable inputs so both sides see "
    "the same numbers.",
    "The v2 encoder cache key is a superset of the 1.6.1 one (config, input width, sorted train and "
    "validation sample ids, gene-order hash) and adds a sha256 of the train and validation matrices, so an "
    "edited expression matrix under unchanged ids cannot hit a stale encoder. The env var is still "
    "`DREVAL_AE_CACHE`. Cache IO is not part of this case.",
    "The v2 MolGNet featurizer pools per drug instead of through a stacked node array, so drugs with equal "
    "atom counts cannot collapse the node matrices to a 3-D array.",
]


def build_case(seed: int) -> Case:
    """Draw node matrices, autoencoder matrices and forest inputs.

    :param seed: Seed of the input generator.
    :returns: The case.
    """
    rng = np.random.default_rng(seed + 11)
    f32 = np.float32
    nodes = rng.normal(size=(sum(ATOM_COUNTS), N_DIM)).astype(f32)
    ae_train = np.maximum(rng.normal(loc=2.0, scale=2.0, size=(N_TRAIN, N_GENES)), 0).astype(f32)
    ae_validation = np.maximum(rng.normal(loc=2.0, scale=2.0, size=(N_VALIDATION, N_GENES)), 0).astype(f32)
    ae_test = np.maximum(rng.normal(loc=2.0, scale=2.0, size=(N_TEST, N_GENES)), 0).astype(f32)
    latent = np.maximum(rng.normal(size=(N_SAMPLES, N_LATENT)), 0).astype(f32)
    n_pairs = 120
    pairs = np.stack([rng.integers(0, N_SAMPLES, n_pairs), rng.integers(0, len(ATOM_COUNTS), n_pairs)], axis=1)
    y = (
        latent[pairs[:, 0], 0] * 2 - latent[pairs[:, 0], 1] + pairs[:, 1] + rng.normal(scale=0.3, size=n_pairs)
    ).astype(f32)
    test_pairs = np.stack([rng.integers(0, N_SAMPLES, 30), rng.integers(0, len(ATOM_COUNTS), 30)], axis=1)
    arrays = {
        "nodes": nodes,
        "atom_counts": np.array(ATOM_COUNTS, dtype=np.int64),
        "ae_train": ae_train,
        "ae_validation": ae_validation,
        "ae_test": ae_test,
        "latent": latent,
        "pairs": pairs.astype(np.int64),
        "test_pairs": test_pairs.astype(np.int64),
        "y": y,
    }
    meta = {"poolings": POOLINGS, "batch_size": BATCH_SIZE, "ae": AE_SETTINGS, "forest": FOREST}
    return Case(model=NAME, seed=seed, arrays=arrays, meta=meta)


def steps() -> list[StepSpec]:
    """Declare the compared outputs.

    :returns: Pooling and forest steps are exact, the autoencoder steps are ``train``.
    """
    specs = [
        StepSpec(name=f"pooling/{pooling}", tolerance="exact", description=f"MolGNet {pooling} pooling, all drugs")
        for pooling in POOLINGS
    ]
    specs += [
        StepSpec(name="ae/encoder_weight", tolerance="train", description="First encoder layer after training"),
        StepSpec(name="ae/latent", tolerance="train", description="Latent embedding of the held-out matrix"),
        StepSpec(name="forest/predictions", tolerance="exact", description="Forest predictions, identical inputs"),
        StepSpec(name="forest/importances", tolerance="exact", description="Forest feature importances"),
        StepSpec(
            name="forest/scaled_predictions",
            tolerance="exact",
            scope=DEVIATION,
            v2_key="forest/predictions",
            description="1.6.1 forest on the standardized latent against the v2 forest on the raw latent",
            explanation=(
                "1.6.1 standardizes the latent before the forest, v2 does not. Splits are invariant to the "
                "monotone rescaling; only rounding of the split thresholds can change a prediction."
            ),
        ),
    ]
    return specs


def _split_nodes(case: Case) -> list[np.ndarray]:
    """Cut the stacked node matrix into one matrix per drug.

    :param case: The shared case.
    :returns: One ``(n_atoms, n_dim)`` float32 matrix per drug.
    """
    bounds = np.cumsum(case.arrays["atom_counts"])[:-1]
    return np.split(case.arrays["nodes"], bounds)


def _forest_matrices(case: Case, pooled: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate latent and pooled drug vectors per pair, as float64 like the estimator inputs.

    :param case: The shared case.
    :param pooled: Pooled ``mean_max`` vector of every drug.
    :returns: Training matrix, test matrix and the training targets.
    """
    latent = case.arrays["latent"].astype(np.float64)
    drugs = np.stack(pooled).astype(np.float64)

    def design(pairs: np.ndarray) -> np.ndarray:
        return np.concatenate([latent[pairs[:, 0]], drugs[pairs[:, 1]]], axis=1)

    return design(case.arrays["pairs"]), design(case.arrays["test_pairs"]), case.arrays["y"].astype(np.float64)


def run_v1(case: Case) -> dict[str, np.ndarray]:
    """Pool, train the autoencoder and fit the forest with the 1.6.1 code.

    :param case: The shared case.
    :returns: Pooled vectors, autoencoder outputs and forest outputs.
    """
    from drevalpy.models.baselines.ae_molgnet_random_forest import pool_molgnet
    from drevalpy.models.DIPK.gene_expression_encoder import (
        encode_gene_expression,
        train_gene_expession_autoencoder,
    )
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    out: dict[str, np.ndarray] = {}
    nodes = _split_nodes(case)
    for pooling in case.meta["poolings"]:
        out[f"pooling/{pooling}"] = np.stack([pool_molgnet(n, pooling).astype(np.float32) for n in nodes])

    train, validation = case.arrays["ae_train"], case.arrays["ae_validation"]
    batch_size = int(case.meta["batch_size"])
    if len(train) % batch_size == 1:  # _fit_or_load_encoder
        validation = np.concatenate([validation, train[-1:]])
        train = train[:-1]
    seed_all(case.seed)
    settings = case.meta["ae"]
    encoder = train_gene_expession_autoencoder(
        train,
        validation,
        epochs_autoencoder=settings["epochs_autoencoder"],
        lr=settings["lr_autoencoder"],
        patience=settings["patience_autoencoder"],
        batch_size=batch_size,
    )
    out["ae/encoder_weight"] = encoder.encoder[0][0].weight.detach().cpu().numpy()
    out["ae/latent"] = np.asarray(encode_gene_expression(case.arrays["ae_test"], encoder))

    pooled = [pool_molgnet(n, "mean_max").astype(np.float32) for n in nodes]
    x_train, x_test, y = _forest_matrices(case, pooled)
    forest = case.meta["forest"]

    def forest_model() -> Any:
        np.random.seed(FOREST_SEED)  # noqa: NPY002 - 1.6.1 passes no random_state
        return RandomForestRegressor(
            n_estimators=forest["n_estimators"],
            criterion="squared_error",
            max_samples=forest["max_samples"],
            max_depth=forest["max_depth"],
            n_jobs=1,
        )

    model = forest_model()
    model.fit(x_train, y)
    out["forest/predictions"] = model.predict(x_test)
    out["forest/importances"] = model.feature_importances_

    scaler = StandardScaler().fit(case.arrays["latent"][np.unique(case.arrays["pairs"][:, 0])].astype(np.float64))
    scaled = scaler.transform(case.arrays["latent"].astype(np.float64))
    scaled_case = Case(model=NAME, seed=case.seed, arrays={**case.arrays, "latent": scaled}, meta=case.meta)
    s_train, s_test, _ = _forest_matrices(scaled_case, pooled)
    scaled_model = forest_model()
    scaled_model.fit(s_train, y)
    out["forest/scaled_predictions"] = scaled_model.predict(s_test)
    return out


def run_v2(case: Case) -> dict[str, np.ndarray]:
    """Pool, train the autoencoder and fit the forest with the v2 components.

    :param case: The shared case.
    :returns: Pooled vectors, autoencoder outputs and forest outputs.
    """
    from drevalpy.components.featurizers.drug.molgnet_pooled import pool_molgnet
    from drevalpy.components.featurizers.sample.ae_gene_expression import AEGeneExpressionFeaturizer
    from drevalpy.components.predictors.literature.dipk.gene_expression_encoder import encode_gene_expression
    from drevalpy.components.predictors.sklearn_models import RandomForestPredictor

    out: dict[str, np.ndarray] = {}
    nodes = _split_nodes(case)
    for pooling in case.meta["poolings"]:
        out[f"pooling/{pooling}"] = np.stack([pool_molgnet(n, pooling).astype(np.float32) for n in nodes])

    train, validation = case.arrays["ae_train"], case.arrays["ae_validation"]
    settings = case.meta["ae"]
    featurizer = AEGeneExpressionFeaturizer(
        gene_list=None, batch_size_autoencoder=int(case.meta["batch_size"]), **settings
    )
    seed_all(case.seed)
    encoder = featurizer._train_or_load(
        train,
        validation,
        np.array([f"T{i}" for i in range(len(train))]),
        np.array([f"V{i}" for i in range(len(validation))]),
    )
    out["ae/encoder_weight"] = encoder.encoder[0][0].weight.detach().cpu().numpy()
    out["ae/latent"] = np.asarray(encode_gene_expression(case.arrays["ae_test"], encoder))

    pooled = [pool_molgnet(n, "mean_max").astype(np.float32) for n in nodes]
    x_train, x_test, y = _forest_matrices(case, pooled)
    estimator = RandomForestPredictor({**case.meta["forest"], "n_jobs": 1})._make_estimator()
    estimator.fit(x_train, y)
    out["forest/predictions"] = estimator.predict(x_test)
    out["forest/importances"] = estimator.feature_importances_
    return out
