"""Degenerate variants of the synthetic dataset, plus the model support matrix.

The main fixture in :mod:`tests.synthetic.builders` has **complete** modality
coverage, which keeps the common case fast and keeps unrelated failures out of
the model gate. :func:`build_partial_coverage_dataset` covers the other side:
it reproduces the ragged coverage the published datasets have (the smaller toy
dataset carries gene expression for 88 of its 90 samples), which drives the
NaN-filtering path in ``PredictorBase.fit`` and therefore
``ModelInputBatch.subset_pairs`` with a multi-drug mask.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import pytest
from _pytest.mark import ParameterSet

from drevalpy.types.data.dataset import Dataset
from tests.synthetic.builders import N_SAMPLES, build_synthetic_dataset

#: Samples covered per public omics name in the partial-coverage variant.
#: Any omics view left out keeps full coverage.
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

#: DIPK consumes remote pretrained artifacts and has its own network integration gate.
EXCLUDED_MODELS: Final[dict[str, str]] = {
    "DIPK": "Real pretrained MolGNet/PPI artifacts are exercised by the network integration test.",
}

#: Models expected to fail for a named, still-open defect, mapped to that reason.
#:
#: Empty on purpose: the three copy-number models that used to live here
#: (``MultiViewNeuralNetwork``, ``MOLIR``, ``SuperFELTR``) were blocked only
#: because the library passed the public omics name straight to
#: ``Dataset.get_sample_features``. The read sites now resolve through
#: ``OMICS_ACCESSORS``, so all three pass and their markers are retired. The hook
#: stays so the next genuine defect gets a strict marker rather than a skip.
MODEL_DEFECTS: Final[dict[str, str]] = {}

#: No remaining known checkpoint exclusions; learned featurizer state is persisted.
SAVE_LOAD_DEFECTS: Final[dict[str, str]] = {}


def build_partial_coverage_dataset() -> Dataset:
    """Build a dataset whose omics modalities cover only some of the samples.

    :returns: Dataset that drives the predictors' NaN-filtering path.
    """
    return build_synthetic_dataset(name="SYNTH_PARTIAL", omics_coverage=PARTIAL_COVERAGE)


def model_param(model_name: str, *, defects: Mapping[str, str] = MODEL_DEFECTS) -> ParameterSet:
    """Wrap *model_name* as a ``pytest`` parameter, xfailing known defects strictly.

    Strict is the point: when the underlying defect is fixed the test starts
    passing, the ``xpass`` fails the run, and whoever fixed it is told to delete
    the marker instead of leaving a stale exemption behind.

    :param model_name: Model name as the test parametrizes it.
    :param defects: Mapping of model name to the reason it is expected to fail.
    :returns: Parameter carrying a strict xfail marker where applicable.
    """
    reason = defects.get(model_name)
    marks = [pytest.mark.xfail(reason=reason, strict=True)] if reason else []
    return pytest.param(model_name, marks=marks, id=model_name)
