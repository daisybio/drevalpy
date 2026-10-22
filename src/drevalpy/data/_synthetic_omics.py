"""Sample-side views of :func:`drevalpy.data.synthetic`: omics matrices and their names.

Every omics view mixes the same hidden sample factors as the response, so a view
carries signal whatever its value range. The view then only shapes the values:
expression and proteomics stay continuous (proteomics floored above zero, as
intensities are), mutations are 0/1 calls on the upper quartile, methylation is a
beta value in ``(0, 1)`` and copy number is a GISTIC-style integer in ``[-2, 2]``.

Feature names are real where the shipped data has a source for them: gene symbols
come from the packaged ``landmark_genes.csv``, so the ``landmarkGenes`` featurizer
selects a non-empty subset, and methylation sites are genomic windows. ``pandas``
and ``anndata`` are imported inside the functions that need them, because
:mod:`drevalpy.data` sits on the critical path of ``import drevalpy``.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    import anndata as ad

    from drevalpy.types.data.dataset import Dataset

#: Names accepted by ``sample_views``: views stored in ``response.obsm`` that would
#: otherwise need an annotation download, filled with seeded noise of the right width.
SAMPLE_VIEWS: Final = ("pathway_features",)
N_PATHWAYS: Final = 2

#: Which axis each omics view is measured over, keyed by public omics name.
_VAR_AXIS: Final[Mapping[str, str]] = {
    "gene_expression": "gene",
    "proteomics": "gene",
    "mutations": "gene",
    "methylation": "cpg",
    "copy_number_variation_gistic": "gene",
}

_OMICS_BASE: Final = 6.0
_OMICS_NOISE_SD: Final = 0.5
#: Intensities are positive, and ``normalizedProteomics`` takes their log10.
_PROTEOMICS_FLOOR: Final = 0.5
#: Standard-normal quantile above which a sample counts as mutated (about 25%).
_MUTATION_CUTOFF: Final = 0.674


def check_choices(label: str, requested: Collection[str], allowed: Collection[str]) -> None:
    """Reject names outside *allowed*, naming every valid choice.

    Args:
        label: What the names are, for the message (e.g. ``"drug_views"``).
        requested: Names the caller passed.
        allowed: Names the builder knows.

    Raises:
        ValueError: If any requested name is not allowed.
    """
    unknown = sorted(set(requested) - set(allowed))
    if unknown:
        msg = f"Unknown {label} {unknown}; choose from {sorted(allowed)}"
        raise ValueError(msg)


def gene_symbols(n: int) -> list[str]:
    """Return the first *n* gene symbols shipped in ``landmark_genes.csv``.

    Args:
        n: Number of symbols to return.

    Returns:
        Ordered, de-duplicated gene symbols.

    Raises:
        ValueError: If the packaged list holds fewer than *n* symbols.
    """
    from drevalpy.components.featurizers.sample.gene_lists import (
        gene_names_from_list_csv,
        resolve_gene_list_path,
    )

    symbols = list(dict.fromkeys(gene_names_from_list_csv(resolve_gene_list_path("landmark_genes"))))
    if len(symbols) < n:
        msg = f"landmark_genes.csv only provides {len(symbols)} symbols, need {n}; pass feature_names instead"
        raise ValueError(msg)
    return symbols[:n]


def cpg_ids(n: int) -> list[str]:
    """Return *n* distinct genomic-window ids of the form ``chr1:1000000-1000800``.

    Args:
        n: Number of methylation sites.

    Returns:
        Site ids, one per methylation feature.
    """
    return [f"chr{index % 22 + 1}:{1_000_000 + index * 977}-{1_000_800 + index * 977}" for index in range(n)]


def omics_coverage(omics: Sequence[str] | Mapping[str, int] | None, n_samples: int) -> dict[str, int]:
    """Normalize the *omics* argument into a public-name to sample-count map.

    Args:
        omics: Public view names, a name to coverage map, or ``None``.
        n_samples: Coverage given to a view named without a count.

    Returns:
        The map; empty for ``None``.
    """
    if omics is None:
        return {}
    if isinstance(omics, Mapping):
        return {name: int(count) for name, count in omics.items()}
    return dict.fromkeys(omics, n_samples)


def view_feature_names(
    public_name: str, feature_names: Sequence[str] | Mapping[str, Sequence[str]] | None, n_features: int
) -> list[str]:
    """Resolve the column names of one omics view.

    Args:
        public_name: Public omics name of the view.
        feature_names: One list for every view, a per-view mapping, or ``None``
            for real gene symbols or methylation sites.
        n_features: How many names to generate when none are given.

    Returns:
        The column names.
    """
    if isinstance(feature_names, Mapping) and public_name in feature_names:
        return list(feature_names[public_name])
    if feature_names is not None and not isinstance(feature_names, Mapping):
        return list(feature_names)
    return cpg_ids(n_features) if _VAR_AXIS.get(public_name) == "cpg" else gene_symbols(n_features)


def _shape_values(public_name: str, signal: np.ndarray) -> np.ndarray:
    """Map the shared continuous signal onto the value range of *public_name*."""
    if public_name == "proteomics":
        return np.maximum(signal, _PROTEOMICS_FLOOR)
    if public_name not in {"mutations", "methylation", "copy_number_variation_gistic"}:
        return signal
    spread = signal.std(axis=0)
    standard = (signal - signal.mean(axis=0)) / np.where(spread > 0, spread, 1.0)
    if public_name == "mutations":
        return (standard > _MUTATION_CUTOFF).astype(np.float32)
    if public_name == "methylation":
        return 1.0 / (1.0 + np.exp(-standard))
    return np.clip(np.rint(standard), -2, 2)


def omics_anndata(
    rng: np.random.Generator,
    public_name: str,
    sample_ids: Sequence[str],
    factors: np.ndarray,
    *,
    n_covered: int,
    feature_names: Sequence[str],
) -> ad.AnnData:
    """Build one omics modality covering the first *n_covered* samples.

    Every feature is the shared hidden factors mixed through a random loading
    matrix, plus noise, so the view carries the sample effect in the response.

    Args:
        rng: Source of randomness.
        public_name: Public omics name, which sets the value range.
        sample_ids: Every sample id of the dataset, in order.
        factors: Hidden sample factors shared with the response.
        n_covered: Number of leading samples the view measures.
        feature_names: Column names.

    Returns:
        The modality.
    """
    import anndata as ad
    import pandas as pd

    loadings = rng.normal(size=(factors.shape[1], len(feature_names)))
    mixed = _OMICS_BASE + factors[:n_covered] @ loadings
    noise = rng.normal(0.0, _OMICS_NOISE_SD, size=mixed.shape)
    return ad.AnnData(
        X=_shape_values(public_name, mixed + noise).astype(np.float32),
        obs=pd.DataFrame(index=pd.Index(list(sample_ids[:n_covered]))),
        var=pd.DataFrame(index=pd.Index(list(feature_names), name="feature")),
    )


def payload_genes(modalities: Mapping[str, ad.AnnData], n_features: int) -> list[str]:
    """Pick the gene list the pathway and ontology payloads are written over.

    Args:
        modalities: Built omics modalities keyed by public name.
        n_features: Number of symbols to take when no gene view was built.

    Returns:
        The var names of the first gene-axis view, else landmark gene symbols.
    """
    for public_name, modality in modalities.items():
        if _VAR_AXIS.get(public_name, "gene") == "gene":
            return list(modality.var_names)
    return gene_symbols(n_features)


def add_sample_views(dataset: Dataset, views: Sequence[str], rng: np.random.Generator) -> None:
    """Fill ``response.obsm`` with the sample views that would need a download.

    Args:
        dataset: Dataset modified in place.
        views: Names from :data:`SAMPLE_VIEWS`.
        rng: Source of the noise.

    Raises:
        ValueError: If a name is not in :data:`SAMPLE_VIEWS`.
    """
    check_choices("sample_views", views, SAMPLE_VIEWS)
    response = dataset.mdata.mod["response"]
    for view in views:
        response.obsm[view] = rng.normal(size=(response.n_obs, N_PATHWAYS)).astype(np.float32)
