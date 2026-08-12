"""Deterministic synthetic datasets used in place of downloadable real data.

``builders`` holds the raw-omics MuData factory that the session fixture in
``tests/conftest.py`` exposes; ``variants`` holds the deliberately degenerate
shapes individual tests ask for, such as partial modality coverage; ``results``
holds the result-object factories that stand in for a completed training run.
"""

from __future__ import annotations

from tests.synthetic.builders import (
    BPE_LENGTH,
    BUILTIN_MEASURE,
    CHEMBERTA_DIM,
    CNV_MODALITY,
    FINGERPRINT_BITS,
    N_CELL_LINES,
    N_DRUGS,
    N_GENES,
    N_METHYLATION_SITES,
    N_PATHWAYS,
    N_TISSUES,
    OMICS_MODALITIES,
    RESPONSE_LAYERS,
    SMILESVEC_DIM,
    build_synthetic_dataset,
    synthetic_gene_symbols,
)
from tests.synthetic.results import (
    DEFAULT_DATASET_NAME,
    DEFAULT_MODEL_NAMES,
    DEFAULT_SPLIT_MODE,
    NORMALIZED_METRIC,
    REFERENCE_MODEL,
    make_experiment_result,
    make_metrics,
    make_model_result,
    make_run_result,
)
from tests.synthetic.variants import (
    EXCLUDED_MODELS,
    MODEL_DEFECTS,
    PARTIAL_COVERAGE,
    SAVE_LOAD_DEFECTS,
    SUPPORTED_GLOBAL_MODELS,
    SUPPORTED_SINGLE_DRUG_MODELS,
    build_partial_coverage_dataset,
)

__all__ = [
    "BPE_LENGTH",
    "BUILTIN_MEASURE",
    "CHEMBERTA_DIM",
    "CNV_MODALITY",
    "DEFAULT_DATASET_NAME",
    "DEFAULT_MODEL_NAMES",
    "DEFAULT_SPLIT_MODE",
    "EXCLUDED_MODELS",
    "FINGERPRINT_BITS",
    "MODEL_DEFECTS",
    "NORMALIZED_METRIC",
    "N_CELL_LINES",
    "N_DRUGS",
    "N_GENES",
    "N_METHYLATION_SITES",
    "N_PATHWAYS",
    "N_TISSUES",
    "OMICS_MODALITIES",
    "PARTIAL_COVERAGE",
    "REFERENCE_MODEL",
    "RESPONSE_LAYERS",
    "SAVE_LOAD_DEFECTS",
    "SMILESVEC_DIM",
    "SUPPORTED_GLOBAL_MODELS",
    "SUPPORTED_SINGLE_DRUG_MODELS",
    "build_partial_coverage_dataset",
    "build_synthetic_dataset",
    "make_experiment_result",
    "make_metrics",
    "make_model_result",
    "make_run_result",
    "synthetic_gene_symbols",
]
