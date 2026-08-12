"""Degenerate variants of the synthetic dataset, plus the model support matrix.

The main fixture in :mod:`tests.synthetic.builders` has **complete** modality
coverage, which keeps the common case fast and keeps unrelated failures out of
the model gate. :func:`build_partial_coverage_dataset` covers the other side:
it reproduces the ragged coverage the published datasets have (the smaller toy
dataset carries gene expression for 88 of its 90 cell lines), which drives the
NaN-filtering path in ``PredictorBase.fit`` and therefore
``ModelInputBatch.subset_pairs`` with a multi-drug mask.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import pytest
from _pytest.mark import ParameterSet

from drevalpy.types.data.dataset import Dataset
from tests.synthetic.builders import N_CELL_LINES, build_synthetic_dataset

#: Cell lines covered per public omics name in the partial-coverage variant.
#: Any omics view left out keeps full coverage.
PARTIAL_COVERAGE: Final[dict[str, int]] = {
    "gene_expression": N_CELL_LINES - 2,
    "proteomics": N_CELL_LINES - 3,
    "methylation": N_CELL_LINES - 6,
    "copy_number_variation_gistic": N_CELL_LINES - 2,
}

#: Global models the fixture drives end to end.
SUPPORTED_GLOBAL_MODELS: Final = (
    "SRMF",
    "SimpleNeuralNetwork[fingerprints]",
    "SimpleNeuralNetwork[chemberta]",
    "MultiViewNeuralNetwork",
    "PharmaFormer",
    "Precily",
)

#: Single-drug models the fixture is expected to train, predict and round-trip.
SUPPORTED_SINGLE_DRUG_MODELS: Final = (
    "SingleDrugRandomForest[gex]",
    "SingleDrugRandomForest[proteomics]",
    "SingleDrugElasticNet[gex]",
    "SingleDrugElasticNet[proteomics]",
    "MOLIR",
    "SuperFELTR",
)

#: Models kept out of the lists above, mapped to the defect that excludes them.
#: All three are pre-existing library bugs, independent of the fixture, and are
#: investigate-only for now, so the reason travels with the exclusion instead of
#: living solely in a planning document. Each reason below was reproduced against
#: the complete-coverage fixture, so it is the observed failure, not a guess.
EXCLUDED_MODELS: Final[dict[str, str]] = {
    "DIPK": (
        "bionic.py fetches human_ppi_features.tsv and gene_list_sel.txt from the artifacts bucket, "
        "which raises PermissionError(403) without AWS credentials -- anonymous reads are refused -- "
        "so the featurizer never reaches the ragged molgnet_features it also needs; uns['dipk'] is "
        "written by the data builder but read by no library code"
    ),
    "DrugGNN": (
        "the drug_graph view stores plain dicts, as a .h5mu must, while "
        "druggnn/predictor.py calls .num_node_features on them, raising AttributeError"
    ),
    "SparseGO": (
        "attach_sparsego_ontology_metadata has no production caller, so read_sparsego_ontology_metadata "
        "never finds the layer_connections key it needs and SparseGOOntologyFeaturizer._fit always "
        "raises 'SparseGO ontology metadata is missing'"
    ),
}

#: Models expected to fail for a named, still-open defect, mapped to that reason.
#:
#: Empty on purpose: the three copy-number models that used to live here
#: (``MultiViewNeuralNetwork``, ``MOLIR``, ``SuperFELTR``) were blocked only
#: because the library passed the public omics name straight to
#: ``Dataset.get_cell_line_features``. The read sites now resolve through
#: ``OMICS_ACCESSORS``, so all three pass and their markers are retired. The hook
#: stays so the next genuine defect gets a strict marker rather than a skip.
MODEL_DEFECTS: Final[dict[str, str]] = {}

#: Models that train and predict but cannot be reloaded from a checkpoint,
#: mapped to the error their round-trip raises. Their set-dependent featurizers
#: (learned BPE merges, GSVA scores) are not part of the saved state.
SAVE_LOAD_DEFECTS: Final[dict[str, str]] = {
    "PharmaFormer": "BpePharmaformerDrugFeaturizer must be fit before transform",
    "Precily": "PathwaysCellLineFeaturizer must be fit before transform",
}


def build_partial_coverage_dataset() -> Dataset:
    """Build a dataset whose omics modalities cover only some of the cell lines.

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
