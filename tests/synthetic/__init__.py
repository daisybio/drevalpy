"""Deterministic synthetic datasets (``builders``, ``variants``) and result factories (``results``)."""

from __future__ import annotations

from tests.synthetic.builders import (
    BPE_LENGTH,
    BUILTIN_MEASURE,
    CHEMBERTA_DIM,
    CNV_MODALITY,
    FINGERPRINT_BITS,
    N_DRUGS,
    N_GENES,
    N_METHYLATION_SITES,
    N_PATHWAYS,
    N_SAMPLES,
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
    REFERENCE_MODEL,
    make_experiment_result,
    make_metrics,
    make_model_result,
    make_run_result,
)
from tests.synthetic.variants import (
    PARTIAL_COVERAGE,
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
    "FINGERPRINT_BITS",
    "N_SAMPLES",
    "N_DRUGS",
    "N_GENES",
    "N_METHYLATION_SITES",
    "N_PATHWAYS",
    "N_TISSUES",
    "OMICS_MODALITIES",
    "PARTIAL_COVERAGE",
    "REFERENCE_MODEL",
    "RESPONSE_LAYERS",
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
