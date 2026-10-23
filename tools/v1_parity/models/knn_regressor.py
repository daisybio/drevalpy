"""KNNRegressor parity case: 1.6.1 ``KNNRegressor`` against the v2 ``KNNRegressor`` zoo preset.

The estimator input, the neighbour sets and distances of the test pairs and the predictions are compared.

``distance``
    ``n_neighbors=7``, ``weights="distance"``.
``uniform``
    ``n_neighbors=4``, ``weights="uniform"``.
``defaults``
    The 1.6.1 first grid point (``n_neighbors=5``, ``weights="distance"``) against the unmodified v2 preset;
    the two are the same configuration.
``binary``
    Binary fingerprints (the real data format) instead of the continuous ones the other variants use.
    Deviation: distance ties are broken differently under float32 and float64.
"""

from __future__ import annotations

import numpy as np

from tools.v1_parity._bridge import Case
from tools.v1_parity.compare import DEVIATION, StepSpec
from tools.v1_parity.models._sklearn_common import (
    build_sklearn_case,
    run_sklearn_v1,
    run_sklearn_v2,
    sklearn_steps,
)

NAME = "KNNRegressor"
V1_MODULE, V1_CLASS, V2_PRESET = "drevalpy.models.baselines.sklearn_models", "KNNRegressor", "KNNRegressor"

VARIANTS: dict[str, dict] = {
    "distance": {
        "v1": {"n_neighbors": 7, "weights": "distance"},
        "v2": {"n_neighbors": 7, "weights": "distance"},
    },
    "uniform": {
        "v1": {"n_neighbors": 4, "weights": "uniform"},
        "v2": {"n_neighbors": 4, "weights": "uniform"},
    },
    "defaults": {
        "v1": {"n_neighbors": 5, "weights": "distance"},
        "v2": {},
    },
    "binary": {
        "v1": {"n_neighbors": 5, "weights": "uniform"},
        "v2": {"n_neighbors": 5, "weights": "uniform"},
        "fingerprints": "fingerprints_binary",
        "scope": DEVIATION,
        "explanation": (
            "With binary fingerprints many candidates tie exactly in distance (the same sample paired "
            "with drugs at the same Hamming distance). Ties are broken by the order of the float "
            "rounding of the distances, which differs between float64 (1.6.1) and float32 (v2), so tied "
            "neighbours, and with them uniform-weight predictions, can differ."
        ),
    },
}

FINDINGS = [
    (
        "Default hyperparameters agree: 1.6.1 first grid point n_neighbors=5, weights=distance (grid "
        "n_neighbors 5/10/3); v2 declares n_neighbors=5, weights=distance (space 3..15)."
    ),
    (
        "The parity variants use continuous drug features so that no two candidates tie. With binary "
        "fingerprints (the real format) distances tie massively and the order of tied neighbours depends on "
        "float rounding, which differs between float64 (1.6.1) and float32 (v2); the `binary` variant records "
        "the effect on the predictions. Ties are inherent to kNN on binary features, not a defect of either "
        "port."
    ),
    (
        "The nearest neighbours are searched on the concatenation of the gene features and the fingerprint "
        "bits, as in 1.6.1; there is no separate scaling of the fingerprints on either side."
    ),
    (
        "1.6.1 selects the `landmark_genes` list while loading the CSVs (hyperparameter `gene_list`, a missing "
        "gene raises); the v2 preset selects the same list with `scaledGeneExpression(gene_list=landmark_genes)`. "
        "The synthetic gene names are in no shipped list, so both sides get a pre-selected matrix and the "
        "selection itself is not compared here."
    ),
    (
        "Features agree to float32 precision (class forward): v2 holds them as float32, 1.6.1 as float64. The "
        "neighbour sets are compared exactly; distances and predictions follow the features."
    ),
]


def build_case(seed: int) -> Case:
    """Draw the shared inputs and record the variants.

    :param seed: Seed of the input generator.
    :returns: The case.
    """
    return build_sklearn_case(
        NAME, seed, VARIANTS, continuous_fingerprints=True, n_samples=60, n_drugs=8, n_genes=50, n_bits=32
    )


def steps() -> list[StepSpec]:
    """Declare the compared outputs.

    :returns: Parity steps for the aligned variants, deviation steps for the others.
    """
    return sklearn_steps(
        VARIANTS,
        attributes={"neighbors": "exact", "distances": "forward"},
        predictions="forward",
        random_state=False,
    )


def run_v1(case: Case) -> dict[str, np.ndarray]:
    """Train and predict with the 1.6.1 model for every variant.

    :param case: The shared case.
    :returns: Arrays keyed ``<variant>/<quantity>``.
    """
    return run_sklearn_v1(case, V1_MODULE, V1_CLASS)


def run_v2(case: Case) -> dict[str, np.ndarray]:
    """Train and predict with the v2 zoo preset for every variant.

    :param case: The shared case.
    :returns: Arrays keyed ``<variant>/<quantity>``.
    """
    return run_sklearn_v2(case, V2_PRESET)
