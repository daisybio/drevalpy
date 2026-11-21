"""Degenerate variants of the synthetic dataset, plus the model support matrix.

:func:`build_partial_coverage_dataset` reproduces the ragged modality coverage of published datasets, which drives
the NaN-filtering path in ``PredictorBase.fit`` and ``PredictorInput.__getitem__``.
"""

from __future__ import annotations

import importlib.util
from typing import Any, Final

import pytest
from _pytest.mark import ParameterSet

from drevalpy.types.data.dataset import Dataset
from tests.synthetic.builders import N_SAMPLES, build_synthetic_dataset

#: Samples covered per public omics name in the partial-coverage variant; omitted views keep full coverage.
PARTIAL_COVERAGE: Final[dict[str, int]] = {
    "gene_expression": N_SAMPLES - 2,
    "proteomics": N_SAMPLES - 3,
    "methylation": N_SAMPLES - 6,
    "copy_number_variation_gistic": N_SAMPLES - 2,
}

#: Global models the fixture drives end to end.
SUPPORTED_GLOBAL_MODELS: Final = (
    "SRMF",
    "SimpleNeuralNetwork[fingerprints]",
    "SimpleNeuralNetwork[chemberta]",
    "MultiViewNeuralNetwork",
    "PharmaFormer",
    "Precily",
    "DrugGNN",
    "SparseGO",
    "PaccMann",
    "EnsembleMF",
    "RandomForestAEMolGNet",
)

#: Single-drug models the fixture is expected to train, predict and round-trip.
SUPPORTED_SINGLE_DRUG_MODELS: Final = (
    "SingleDrugRandomForest[gex]",
    "SingleDrugRandomForest[proteomics]",
    "SingleDrugElasticNet[gex]",
    "SingleDrugElasticNet[proteomics]",
    "SingleDrugKNN",
    "SingleDrugMorgoth",
    "MOLIR",
    "SuperFELTR",
)


#: The hyperparameters that shrink each model to a test-sized configuration: the smallest one that still trains.
#: Keys are zoo model names and values are short hyperparameter names, as :class:`~drevalpy.models.DRPModel` takes
#: them; :func:`small_hyperparameters` is how a test reads the table. The golden baseline keeps its own
#: ``OVERRIDES``, whose values are part of the recorded numbers.
TEST_SIZE_HYPERPARAMETERS: Final[dict[str, dict[str, Any]]] = {
    "DIPK": {"batch_size": 8, "epochs": 1, "epochs_autoencoder": 1, "heads": 1},
    "SimpleNeuralNetwork": {"units_per_layer": [2, 2], "max_epochs": 1},
    "MultiViewNeuralNetwork": {"units_per_layer": [2, 2], "max_epochs": 1},
    # ``gene_hidden_size`` must be a multiple of ``feature_dim`` (128); the smaller transformer cuts the fit by 6x.
    "PharmaFormer": {"epochs": 1, "patience": 2, "gene_hidden_size": 256, "dim_feedforward": 64},
    "Precily": {"epochs": 1, "batch_size": 32},
    "SparseGO": {"epochs": 1, "batch_size": 32},
    "DrugGNN": {"epochs": 1, "batch_size": 32},
    "EnsembleMF": {"n_ensemble": 2, "max_epochs": 2, "hidden_dim": 16, "emb_dim": 8, "mlp_hidden": 8},
    "PaccMann": {
        "epochs": 1,
        "batch_size": 32,
        "filters": [4, 4, 4],
        "molecule_heads": [1, 1, 1, 1],
        "gene_heads": [1, 1, 1, 1],
        "stacked_dense_hidden_sizes": [16],
    },
    "SRMF": {"K": 2, "max_iter": 2},
    "RandomForestAEMolGNet": {"epochs_autoencoder": 1, "n_estimators": 5},
    "AdaBoostDecisionTree": {"max_depth": 2, "min_samples_split": 2, "min_samples_leaf": 2, "n_estimators": 2},
    "MultiViewLightGBM": {"n_estimators": 5, "n_jobs": 1},
    "MultiViewXGBoost": {"n_estimators": 5, "max_depth": 3},
    "GradientBoosting": {"max_iter": 5},
    "RandomForest": {"n_estimators": 2, "max_depth": 2},
    "SingleDrugRandomForest": {"n_estimators": 2, "max_depth": 2},
    "SingleDrugMorgoth": {"n_estimators": 2, "mrmr_k": 3, "min_samples_leaf": 2, "max_depth": 3},
    "MOLIR": {"epochs": 1},
    "SuperFELTR": {"epochs": 1},
}


def small_hyperparameters(model_name: str) -> dict[str, Any]:
    """Return a copy of the test-size hyperparameters of *model_name*; a model not in the table keeps its defaults."""
    return dict(TEST_SIZE_HYPERPARAMETERS.get(model_name, {}))


def predictor_small_hyperparameters(predictor_name: str) -> dict[str, Any]:
    """Return the test-size hyperparameters that the predictor registered as *predictor_name* itself owns.

    They come from the first model of :data:`TEST_SIZE_HYPERPARAMETERS` built on that predictor, without the keys
    that belong to its featurizers; a predictor no listed model builds on keeps its defaults.
    """
    from drevalpy.models.zoo import get_zoo_config
    from drevalpy.registry import predictor

    owned = predictor.get(predictor_name).get_hyperparameter_space()
    for model_name, values in TEST_SIZE_HYPERPARAMETERS.items():
        if get_zoo_config(model_name).predictor.name == predictor_name:
            return {key: value for key, value in values.items() if key in owned}
    return {}


def build_partial_coverage_dataset() -> Dataset:
    """Build a dataset whose omics modalities cover only some of the samples."""
    return build_synthetic_dataset(name="SYNTH_PARTIAL", omics_coverage=PARTIAL_COVERAGE)


#: Skips a test that needs the optional ``morgoth`` extra when it is not installed; CI installs it.
REQUIRES_MORGOTH: Final = pytest.mark.skipif(
    importlib.util.find_spec("morgoth") is None, reason="needs the optional morgoth extra: uv sync --extra morgoth"
)

#: Marks of the zoo models whose predictor lives in an optional extra.
_MODEL_MARKS: Final = {"SingleDrugMorgoth": REQUIRES_MORGOTH}


def model_param(model_name: str) -> ParameterSet:
    """Wrap *model_name* as a pytest parameter whose id is the model name, skipped when its extra is absent."""
    return pytest.param(model_name, id=model_name, marks=_MODEL_MARKS.get(model_name, ()))
