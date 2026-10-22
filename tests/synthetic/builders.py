"""Raw-omics MuData factory backing the whole test suite.

The suite used to depend on a gitignored ``data/`` directory holding downloaded
screens, which meant a clean checkout could not run a single test. This module
replaces that with a deterministic, fully in-memory dataset that carries the
same structural slots a published ``.h5mu`` does.

It is one call into :func:`drevalpy.synthetic` with every opt-in extra switched
on, so the suite and the library's own synthetic dataset cannot drift apart. Only
*raw* data is authored there: omics matrices, tissue labels and
``canonical_smiles``. Everything a featurizer can derive from rdkit is derived
by the real featurizer. The remaining views (``chemberta``, ``smilesvec``,
``bpe_smiles``, ``pathway_features``) come from pretrained weights or annotation
downloads in production, so they are seeded noise of the correct width.

Modality names are taken from :data:`drevalpy.types.data.modalities.OMICS_ACCESSORS`
rather than written as literals, so the fixture always stores omics under the
key the published datasets actually use.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from drevalpy.data._synthetic import synthetic
from drevalpy.data._synthetic_drugs import (
    BPE_LENGTH,
    CHEMBERTA_DIM,
    DRUG_VIEWS,
    FINGERPRINT_BITS,
    SMILESVEC_DIM,
    UNS_PAYLOADS,
)
from drevalpy.data._synthetic_omics import N_PATHWAYS, SAMPLE_VIEWS, cpg_ids
from drevalpy.data._synthetic_omics import gene_symbols as synthetic_gene_symbols
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.modalities import OMICS_ACCESSORS, resolve_omics_accessor

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

#: Roughly the smallest shape that keeps every downstream library happy. At 2x2
#: sklearn emits low-bootstrap warnings and subword-nmt refuses to learn BPE
#: codes ("no pair has frequency >= 2"), both of which are errors under -W.
N_SAMPLES: Final = 24
N_DRUGS: Final = 8

#: ``LTO`` runs ``KFold(n_splits=2)`` over the unique tissues and then carves a
#: validation tissue out of the training half, so three is the hard floor. Six
#: leaves four samples per tissue, which keeps each fold's training half wide
#: enough for the single-drug models.
N_TISSUES: Final = 6

N_GENES: Final = 16
N_METHYLATION_SITES: Final = 20

#: A curve-metric layer name curation really emits. ``response.X`` holds pEC50 -
#: curation does not duplicate it as a layer - so this one is derived from ``X``.
BUILTIN_MEASURE: Final = "LN_IC50"
RESPONSE_LAYERS: Final = (BUILTIN_MEASURE, "AUC", "IC50")

DATASET_NAME: Final = "SYNTH"
SEED: Final = 20240612

#: Modality key the fixture stores copy-number data under. Resolved through the
#: accessor map so the fixture follows the datasets, not the public alias.
CNV_MODALITY: Final = resolve_omics_accessor("copy_number_variation_gistic")

#: Every omics modality the fixture carries, keyed by the modality name a
#: ``.h5mu`` would use.
OMICS_MODALITIES: Final = tuple(resolve_omics_accessor(name) for name in OMICS_ACCESSORS)

#: Sparse gaps in the response matrix. Kept deliberately thin: the splitters must
#: still leave every tissue and every drug with observed pairs in each fold.
_MISSING_FRACTION: Final = 6 / (N_SAMPLES * N_DRUGS)


def build_synthetic_dataset(
    *,
    name: str = DATASET_NAME,
    omics_coverage: Mapping[str, int] | None = None,
    seed: int = SEED,
) -> Dataset:
    """Build the synthetic raw-omics :class:`~drevalpy.types.data.dataset.Dataset`.

    :param name: Dataset name recorded on the returned object.
    :param omics_coverage: Optional public-omics-name to sample-count map.
        Any omics view left out is given full coverage. Reducing a count drops
        trailing samples from that modality, which is what makes the
        predictors' NaN-filtering path fire; see
        :mod:`tests.synthetic.variants`.
    :param seed: Seed for every drawn matrix, so the dataset is reproducible.
    :returns: A dataset with complete metadata, five omics modalities, four
        drug views, pathway scores and the auxiliary ``uns`` payloads.
    """
    coverage = {view: int((omics_coverage or {}).get(view, N_SAMPLES)) for view in OMICS_ACCESSORS}
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
        feature_names={"methylation": cpg_ids(N_METHYLATION_SITES)},
        n_features=N_GENES,
        missing_fraction=_MISSING_FRACTION,
        seed=seed,
    )
