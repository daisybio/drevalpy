"""SRMF parity case: 1.6.1 ``SRMF`` against the v2 ``SRMF`` zoo preset.

The solver (``_cmf``, ``_compute_loss``, ``_alg_update``) is the same code on both sides; the case compares what
goes into it and what comes out.

Steps, per variant:

``<variant>/{factor_sample_ids,factor_drug_ids}``
    The samples and drugs that get a latent factor (the rows of ``best_v`` and ``best_u``).
``<variant>/{weights,responses,drug_similarity,sample_similarity}``
    The arguments of ``_cmf``, recorded at the call: the observation mask and the response matrix (drugs x
    samples), the Jaccard similarity of the fingerprints and the Pearson correlation of the expression.
``<variant>/{best_u,best_v,training_mean}``
    The returned latent factors (seeded ``default_rng`` initialization, a few alternating updates) and the mean
    response that is the fallback prediction.
``<variant>/predictions_known``
    Predictions for the test pairs whose sample and drug are both factorized in the fit.
``<variant>/predictions_unseen``
    Predictions for the test pairs with a sample that has no response in training (deviation).

Variants: ``similarity`` (drug and sample similarity regularization, 1.6.1 is given only the features of the
training samples), ``plain`` (``lambda_d`` 0, the 1.6.1 default), ``transductive`` (as ``similarity``, but
1.6.1 is given the features of every sample, which is what its pipeline does) and ``fit_order`` (the order of
the entities the solver sees, which fixes the seeded initial factors; both sides fit in sorted id order).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tools.v1_parity._bridge import Case, tabular_case_arrays
from tools.v1_parity.compare import DEVIATION, StepSpec
from tools.v1_parity.models._torch_common import v2_model

NAME = "SRMF"

N_SAMPLES = 24
N_DRUGS = 5
N_GENES = 15
N_BITS = 12
VARIANTS: dict[str, dict[str, Any]] = {
    "similarity": {
        "hyperparameters": {"K": 3, "lambda_l": 0.05, "lambda_d": 0.1, "lambda_c": 0.08, "max_iter": 6, "seed": 7},
        "all_samples": False,
    },
    "plain": {
        "hyperparameters": {"K": 3, "lambda_l": 0.05, "lambda_d": 0.0, "lambda_c": 0.08, "max_iter": 6, "seed": 8},
        "all_samples": False,
    },
    "transductive": {
        "hyperparameters": {"K": 3, "lambda_l": 0.05, "lambda_d": 0.1, "lambda_c": 0.08, "max_iter": 6, "seed": 9},
        "all_samples": True,
    },
    "fit_order": {
        "hyperparameters": {"K": 3, "lambda_l": 0.05, "lambda_d": 0.1, "lambda_c": 0.08, "max_iter": 6, "seed": 7},
        "all_samples": False,
        "order_only": True,
    },
}
ORDER_QUANTITIES = ("solver_drug_order", "best_u", "best_v", "predictions_known")

SAME_WHEN_TRANSDUCTIVE = ("factor_drug_ids", "drug_similarity", "training_mean")
UNSEEN_EXPLANATION = (
    "A test pair with a sample that has no response in training. v2 fits the factors on the samples of the "
    "training batch only and predicts the training mean for a sample or a drug it has no factor for (documented in "
    "`docs/explanation/literature_models.rst`: if either is unknown the prediction is the mean). 1.6.1 factorizes "
    "every sample of the feature dataset (`np.unique(cell_line_input.identifiers)`): such a sample gets latent "
    "factors from the similarity term alone, and a sample that is absent gets the training mean as its latent "
    "*vector*, so the prediction is `u . mean` (`K * mean^2` when the drug is unknown too)."
)
TRANSDUCTIVE_EXPLANATION = (
    "1.6.1 builds the similarity matrix and the factors over all samples in the feature dataset, including those "
    "without any training response (the pipeline passes the features of the whole dataset). v2 uses the samples "
    "of the training batch, so the sample set, the sample similarity matrix, the initial factors (drawn per row) "
    "and with them every factor differ."
)

FINDINGS = [
    "Fixed (v2): SRMF fitted in the first-seen order of the entities, 1.6.1 in sorted id order (`np.unique`). The "
    "initial factors are drawn row by row from the seeded generator, so the order decides the start point of the "
    "whole optimization. v2 now sorts samples and drugs by id before the solver; variant `fit_order` compares the "
    "factors in stored order, and the harness no longer reorders the arrays.",
    "The solver is the 1.6.1 one and agrees to rounding when both sides factorize the same samples and drugs "
    "(variants `similarity` and `plain`): same seeded `default_rng` initialization, same alternating updates, "
    "same stopping rule. The response pivot, the Jaccard similarity of the fingerprints and the Pearson "
    "correlation of the expression are identical.",
    "1.6.1 is transductive in the sample set: `train` factorizes `np.unique(cell_line_input.identifiers)`, i.e. "
    "every sample of the dataset, also those that occur only in validation or test pairs, and the sample "
    "similarity is computed over all of them. v2 factorizes the samples of the training batch. The drugs are "
    "the same in practice (every drug is trained). Variant `transductive` shows the effect: sample set, "
    "similarity, initial factors and all factors differ.",
    "Prediction fallback: 1.6.1 fills the latent vector of an unknown sample or drug with the training mean "
    "(`np.full((n, K), mean)`), so its prediction is `u . mean` or `K * mean^2`; v2 returns the training mean "
    "(documented). With a transductive sample set 1.6.1 rarely hits the fallback for samples.",
    "The v2 defaults (K 45, lambda_l 0.01, lambda_d 0, lambda_c 0.01, max_iter 50, seed 1) are the 1.6.1 ones; "
    "`n_features` (1036) is in both hyperparameter lists and is not used by either solver. The preset reads "
    "`raw[gene_expression]` with no gene list (1.6.1: `gene_list=None`, every column). The case sets all "
    "hyperparameters explicitly.",
]


def build_case(seed: int) -> Case:
    """Draw the tabular inputs (binary fingerprints, small expression).

    :param seed: Seed of the input generator.
    :returns: The case.
    :raises ValueError: If a drug occurs in no training pair.
    """
    arrays = tabular_case_arrays(seed, n_samples=N_SAMPLES, n_drugs=N_DRUGS, n_genes=N_GENES, n_bits=N_BITS)
    if len(np.unique(arrays["train_pairs"][:, 1])) != N_DRUGS:
        msg = "every drug must occur in the training pairs"
        raise ValueError(msg)
    if not np.isin(arrays["test_pairs"][:, 0], arrays["train_pairs"][:, 0]).any():
        msg = "some test pair needs a sample seen in training"
        raise ValueError(msg)
    return Case(model=NAME, seed=seed, arrays=arrays, meta={"variants": VARIANTS})


def steps() -> list[StepSpec]:
    """Declare the compared outputs.

    :returns: Factor-set, solver-input, factor and prediction steps.
    """
    specs: list[StepSpec] = []
    for variant, setting in VARIANTS.items():
        if setting.get("order_only"):
            specs += _order_steps(variant)
            continue
        transductive = setting["all_samples"]
        table = [
            ("factor_sample_ids", "exact", "samples that get a latent factor"),
            ("factor_drug_ids", "exact", "drugs that get a latent factor"),
            ("solver_sample_order", "exact", "sample order of the arrays passed to _cmf"),
            ("weights", "exact", "observation mask passed to _cmf (drugs x samples)"),
            ("responses", "forward", "response matrix passed to _cmf (drugs x samples)"),
            ("drug_similarity", "forward", "Jaccard similarity of the fingerprints"),
            ("sample_similarity", "forward", "Pearson correlation of the expression"),
            ("best_u", "train", "latent factors of the drugs"),
            ("best_v", "train", "latent factors of the samples"),
            ("training_mean", "forward", "mean training response (fallback prediction)"),
            ("predictions_known", "train", "test predictions for pairs with a factorized sample and drug"),
        ]
        for quantity, tolerance, text in table:
            deviates = transductive and quantity not in SAME_WHEN_TRANSDUCTIVE
            specs.append(
                StepSpec(
                    f"{variant}/{quantity}",
                    tolerance,
                    scope=DEVIATION if deviates else "parity",
                    description=f"{variant}: {text}",
                    explanation=TRANSDUCTIVE_EXPLANATION if deviates else "",
                )
            )
        specs.append(_unseen_step(variant))
    return specs


def _unseen_step(variant: str) -> StepSpec:
    return StepSpec(
        f"{variant}/predictions_unseen",
        "train",
        scope=DEVIATION,
        description=f"{variant}: test predictions for pairs with a sample without training response",
        explanation=UNSEEN_EXPLANATION,
    )


def _order_steps(variant: str) -> list[StepSpec]:
    texts = {
        "solver_drug_order": "drug order of the arrays passed to _cmf",
        "best_u": "latent factors of the drugs (compared by drug id)",
        "best_v": "latent factors of the samples",
        "predictions_known": "test predictions for pairs with a factorized sample and drug",
    }
    return [
        StepSpec(
            f"{variant}/{quantity}",
            "exact" if quantity == "solver_drug_order" else "train",
            description=f"{variant}: {texts[quantity]}",
        )
        for quantity in ORDER_QUANTITIES
    ]


def _hyperparameters(setting: dict[str, Any]) -> dict[str, Any]:
    return dict(setting["hyperparameters"])


def _known_mask(case: Case) -> np.ndarray:
    """Return which test pairs have both their sample and their drug in the training pairs.

    :param case: The shared case.
    :returns: Boolean mask over the test pairs.
    """
    train, test = case.arrays["train_pairs"], case.arrays["test_pairs"]
    return np.isin(test[:, 0], train[:, 0]) & np.isin(test[:, 1], train[:, 1])


def _arrays(variant: str, call: dict[str, Any], factors: Any, mean: float, preds: np.ndarray, case: Case) -> dict:
    known = _known_mask(case)
    best_u, best_v = factors
    preds = np.asarray(preds, dtype=np.float64)
    if case.meta["variants"][variant].get("order_only"):
        return {
            f"{variant}/solver_drug_order": np.array([str(i) for i in call["drug_order"]]),
            f"{variant}/best_u": best_u.to_numpy(dtype=np.float64),
            f"{variant}/best_v": best_v.to_numpy(dtype=np.float64),
            f"{variant}/predictions_known": preds[known],
        }
    return {
        f"{variant}/factor_sample_ids": np.array([str(i) for i in best_v.index]),
        f"{variant}/factor_drug_ids": np.array([str(i) for i in best_u.index]),
        f"{variant}/solver_sample_order": np.array([str(i) for i in call["sample_order"]]),
        f"{variant}/weights": np.asarray(call["w"], dtype=np.float64),
        f"{variant}/responses": np.asarray(call["int_mat"], dtype=np.float64),
        f"{variant}/drug_similarity": np.asarray(call["drug_mat"], dtype=np.float64),
        f"{variant}/sample_similarity": np.asarray(call["cell_mat"], dtype=np.float64),
        f"{variant}/best_u": best_u.to_numpy(dtype=np.float64),
        f"{variant}/best_v": best_v.to_numpy(dtype=np.float64),
        f"{variant}/training_mean": np.array(mean, dtype=np.float64),
        f"{variant}/predictions_known": preds[known],
        f"{variant}/predictions_unseen": preds[~known],
    }


def run_v1(case: Case) -> dict[str, np.ndarray]:
    """Run every variant with the 1.6.1 ``SRMF`` model.

    :param case: The shared case.
    :returns: Arrays keyed by the step names.
    """
    from drevalpy.datasets.dataset import FeatureDataset
    from drevalpy.models.SRMF.srmf import SRMF

    from tools.v1_parity._v1_runner import tabular_inputs

    data = tabular_inputs(case, feature_dtype="float64")
    train_samples = np.unique(data.train.cell_line_ids)
    out: dict[str, np.ndarray] = {}
    for variant, setting in case.meta["variants"].items():
        cells = data.cell_line_input
        if not setting["all_samples"]:
            cells = FeatureDataset(
                features={s: cells.features[s] for s in train_samples}, meta_info=dict(cells.meta_info)
            )
        model = SRMF()
        model.build_model(_hyperparameters(setting))
        call: dict[str, Any] = {}
        solver = model._cmf

        def record(*, _solver: Any = solver, _call: dict[str, Any] = call, **kwargs: Any) -> Any:
            _call.update(kwargs)
            return _solver(**kwargs)

        model._cmf = record
        model.train(data.train, cells, data.drug_input)
        call["drug_order"], call["sample_order"] = model.best_u.index, model.best_v.index
        predictions = model.predict(data.test.cell_line_ids, data.test.drug_ids, cells, data.drug_input)
        out.update(_arrays(variant, call, (model.best_u, model.best_v), float(model.training_mean), predictions, case))
    return out


def _recording_solver(solver: Any, call: dict[str, Any]) -> Any:
    """Wrap ``SRMFPredictor._cmf`` to record the arguments the solver is given.

    :param solver: Original unbound ``_cmf``.
    :param call: Receives the keyword arguments of the call.
    :returns: The replacement function.
    """

    def record(self: Any, **kwargs: Any) -> Any:
        call.update(kwargs)
        return solver(self, **kwargs)

    return record


def run_v2(case: Case) -> dict[str, np.ndarray]:
    """Run every variant with the v2 ``SRMF`` preset.

    :param case: The shared case.
    :returns: Arrays keyed like those of :func:`run_v1`.
    """
    from drevalpy.components.predictors.literature.srmf.predictor import SRMFPredictor
    from tools.v1_parity._v2_runner import tabular_inputs

    data = tabular_inputs(case)
    out: dict[str, np.ndarray] = {}
    for variant, setting in case.meta["variants"].items():
        call: dict[str, Any] = {}
        solver = SRMFPredictor._cmf
        SRMFPredictor._cmf = _recording_solver(solver, call)
        try:
            model = v2_model(NAME, _hyperparameters(setting))
            model.train(data.dataset, data.train)
        finally:
            SRMFPredictor._cmf = solver
        predictions = model.predict(data.dataset, data.test)
        predictor = model._pipeline._predictor
        call["drug_order"], call["sample_order"] = predictor._best_u.index, predictor._best_v.index
        out.update(
            _arrays(variant, call, (predictor._best_u, predictor._best_v), predictor._training_mean, predictions, case)
        )
    return out
