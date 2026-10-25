"""Raw-omics MuData factory backing the whole test suite.

One call into :func:`drevalpy.synthetic` with every opt-in extra on, so the suite and the library cannot drift.
Only raw data is authored; rdkit-derived views come from the real featurizer, pretrained-weight views are seeded
noise of the correct width.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from drevalpy.components.featurizers.sample.gene_lists import view_gene_union
from drevalpy.data._synthetic import synthetic
from drevalpy.data._synthetic_drugs import (
    BPE_LENGTH,
    CHEMBERTA_DIM,
    DRUG_VIEWS,
    FINGERPRINT_BITS,
    SMILESVEC_DIM,
    UNS_PAYLOADS,
)
from drevalpy.data._synthetic_omics import N_PATHWAYS, SAMPLE_VIEWS
from drevalpy.data._synthetic_omics import gene_symbols as synthetic_gene_symbols
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.views import stored_view_name

__all__ = [
    "BPE_LENGTH",
    "BUILTIN_MEASURE",
    "CHEMBERTA_DIM",
    "CNV_MODALITY",
    "DATASET_NAME",
    "FINGERPRINT_BITS",
    "N_DRUGS",
    "N_GENES",
    "N_METHYLATION_SITES",
    "N_PATHWAYS",
    "N_SAMPLES",
    "N_TISSUES",
    "OMICS_MODALITIES",
    "RESPONSE_LAYERS",
    "SEED",
    "SMILESVEC_DIM",
    "build_synthetic_dataset",
    "synthetic_gene_symbols",
]

#: Smallest shape that keeps downstream libraries happy (at 2x2 sklearn warns and subword-nmt cannot learn BPE).
N_SAMPLES: Final = 24
N_DRUGS: Final = 8

#: ``LTO`` needs at least three tissues; six keeps each fold's training half wide enough for single-drug models.
N_TISSUES: Final = 6

#: Widths of the views the builder produces: each holds every gene of the lists shipped for it.
N_GENES: Final = len(view_gene_union("gene_expression"))
N_METHYLATION_SITES: Final = len(view_gene_union("methylation"))

#: A curve-metric layer curation really emits; ``response.X`` holds pEC50, so this one is derived from ``X``.
BUILTIN_MEASURE: Final = "LN_IC50"
RESPONSE_LAYERS: Final = (BUILTIN_MEASURE, "AUC", "IC50")

DATASET_NAME: Final = "SYNTH"
SEED: Final = 20240612

#: Modality key copy-number data is stored under.
CNV_MODALITY: Final = stored_view_name("copy_number_variation_gistic")

#: Every omics view the fixture carries, by public name.
OMICS_VIEWS: Final = ("gene_expression", "methylation", "mutations", "proteomics", "copy_number_variation_gistic")

#: Every omics modality the fixture carries, keyed as a ``.h5mu`` would.
OMICS_MODALITIES: Final = tuple(stored_view_name(name) for name in OMICS_VIEWS)

#: Sparse response gaps, thin enough that every tissue and drug keeps observed pairs in each fold.
_MISSING_FRACTION: Final = 6 / (N_SAMPLES * N_DRUGS)


def build_synthetic_dataset(
    *,
    name: str = DATASET_NAME,
    omics_coverage: Mapping[str, int] | None = None,
    seed: int = SEED,
) -> Dataset:
    """Build the synthetic raw-omics dataset.

    *omics_coverage* maps public omics names to sample counts (default: full); a reduced count drops trailing
    samples from that modality and so triggers the predictors' NaN-filtering path.
    """
    coverage = {view: int((omics_coverage or {}).get(view, N_SAMPLES)) for view in OMICS_VIEWS}
    return synthetic(
        name=name,
        n_samples=N_SAMPLES,
        n_drugs=N_DRUGS,
        n_tissues=N_TISSUES,
        omics=coverage,
        sample_views=SAMPLE_VIEWS,
        drug_views=DRUG_VIEWS,
        uns_payloads=UNS_PAYLOADS,
        extra_layers=RESPONSE_LAYERS[1:],
        missing_fraction=_MISSING_FRACTION,
        seed=seed,
    )
