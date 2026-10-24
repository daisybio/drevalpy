"""Degenerate variants of the synthetic dataset, plus the model support matrix.

:func:`build_partial_coverage_dataset` reproduces the ragged modality coverage of published datasets, which drives
the NaN-filtering path in ``PredictorBase.fit`` and ``ModelInputBatch.subset_pairs``.
"""

from __future__ import annotations

from typing import Final

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
    "MOLIR",
    "SuperFELTR",
)


def build_partial_coverage_dataset() -> Dataset:
    """Build a dataset whose omics modalities cover only some of the samples."""
    return build_synthetic_dataset(name="SYNTH_PARTIAL", omics_coverage=PARTIAL_COVERAGE)


def model_param(model_name: str) -> ParameterSet:
    """Wrap *model_name* as a pytest parameter whose id is the model name."""
    return pytest.param(model_name, id=model_name)
