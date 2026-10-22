"""Deterministic in-memory synthetic datasets for tutorials, tests and plugins.

Every dataset registered with drevalpy is fetched from a credentialed S3 bucket,
so an offline test run, a tutorial and any third-party plugin's CI cannot use
one. :func:`synthetic` produces a :class:`~drevalpy.types.data.dataset.Dataset`
with the same structural slots a published ``.h5mu`` has, entirely in memory.

The default output carries the ``response`` modality and one
``gene_expression`` view, so the zoo's expression-based models run on it. Pass
*omics* to choose other sample views, or ``()`` for responses only. The opt-in
*sample_views*, *drug_views*, *uns_payloads* and *extra_layers* add what the
rest of the zoo reads; they live in :mod:`drevalpy.data._synthetic_omics` and
:mod:`drevalpy.data._synthetic_drugs`.

Response model
--------------
Real screens show a Simpson-like pattern: drugs differ so much in average
potency that pooled scores look high even when a model ranks samples poorly
within each drug. The simulated responses reproduce it. With ``S`` samples,
``D`` drugs, ``T`` tissues and ``K = 3`` hidden factors::

    y[s, d] = 2 + drug[d] + tissue[t(s)] + z[s] . a + z[s] . w[d] + noise[s, d]
    drug[d]   = 1.5 * Phi^-1((d + 0.5) / D), randomly permuted
    tissue[t] ~ N(0, 0.3^2)
    z[s]      ~ N(0, I_K)                    hidden sample factors
    a         ~ N(0, (0.4 / sqrt(K))^2 I_K)  sample effect carried by the features
    w[d]      ~ N(0, (0.6 / sqrt(K))^2 I_K)  drug x sample interaction
    noise     ~ N(0, 0.5^2)
    omics X[s] = 6 + z[s] . L + N(0, 0.5^2),  L ~ N(0, 1)^(K x n_features)

The drug means are spread by quantile rather than drawn, because with only a
handful of drugs random draws made the pooled score swing between seeds. The
effect sizes are fixed module constants, not parameters. Every omics view mixes
the same ``z``, so features carry signal and permuting a view visibly hurts.

``response.X`` is in pEC50 units and the ``LN_IC50`` layer is derived from it, as
are the optional ``AUC`` and ``IC50`` layers. Views other than expression and
proteomics reshape the same signal into their own value range (0/1 mutation calls,
beta values, GISTIC integers).

``mudata``, ``anndata`` and ``pandas`` are imported inside the functions that
need them: this module is re-exported by :mod:`drevalpy.data`, which sits on the
critical path of ``import drevalpy`` (see ``tests/test_import_cost_policy.py``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import NormalDist
from typing import TYPE_CHECKING, Final

import numpy as np

from drevalpy.data._synthetic_drugs import (
    MAX_SYNTHETIC_DRUGS,
    SMILES,
    add_drug_views,
    add_uns_payloads,
)
from drevalpy.data._synthetic_omics import (
    add_sample_views,
    check_choices,
    omics_anndata,
    omics_coverage,
    payload_genes,
    view_feature_names,
)
from drevalpy.data.utils import TISSUE_IDENTIFIER
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.modalities import resolve_omics_accessor

if TYPE_CHECKING:
    import anndata as ad

#: 60 samples keep the within-drug ranking stable enough for a tutorial to teach
#: the right lesson; at 24 the ranking was a coin flip. Three tissues is the floor
#: for the shipped splitters: ``LTO`` runs ``KFold(n_splits=2)`` over the unique
#: tissues and then carves a validation tissue out of the training half.
N_SAMPLES: Final = 60
N_DRUGS: Final = 8
N_TISSUES: Final = 6
N_FEATURES: Final = 16

#: A curve-metric layer name the curation pipeline really emits. ``response.X``
#: holds pEC50, which curation does not duplicate as a layer, so the dataset
#: derives this one from ``X`` the way a real fit would - see
#: :func:`drevalpy.curation.curate`.
MEASURE: Final = "LN_IC50"

DATASET_NAME: Final = "SYNTHETIC"
SEED: Final = 3

_TISSUES: Final = ("Lung", "Blood", "Skin", "Colon", "Brain", "Breast")

#: Hidden sample factors shared by the response and every omics view.
_N_FACTORS: Final = 3
_DRUG_MEAN_SD: Final = 1.5
_TISSUE_SD: Final = 0.3
_SAMPLE_EFFECT_SD: Final = 0.4
_INTERACTION_SD: Final = 0.6
_NOISE_SD: Final = 0.5
_BASE_PEC50: Final = 2.0


def _punch_holes(matrix: np.ndarray, rng: np.random.Generator, fraction: float) -> None:
    """Blank a fraction of *matrix* in place, leaving every row and column measured.

    Unmeasured pairs are what exercise a component's NaN-filtering path. The
    row/column guarantee matters because a fully unmeasured sample or drug
    makes the leave-one-out splitters produce empty folds.

    Args:
        matrix: Response matrix, modified in place.
        rng: Source of randomness.
        fraction: Share of entries to blank, clamped by the guarantee above.
    """
    n_rows, n_columns = matrix.shape
    if fraction <= 0 or n_rows < 2 or n_columns < 2:
        return
    observed_per_row = np.full(n_rows, n_columns)
    observed_per_column = np.full(n_columns, n_rows)
    candidates = rng.permutation(n_rows * n_columns)
    remaining = int(round(fraction * n_rows * n_columns))
    for flat in candidates:
        if remaining <= 0:
            return
        row, column = divmod(int(flat), n_columns)
        if observed_per_row[row] <= 1 or observed_per_column[column] <= 1:
            continue
        matrix[row, column] = np.nan
        observed_per_row[row] -= 1
        observed_per_column[column] -= 1
        remaining -= 1


def _simulate_response(
    rng: np.random.Generator, *, n_samples: int, n_drugs: int, tissue_index: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Draw the hidden factors and the complete pEC50 matrix they generate.

    Args:
        rng: Source of randomness.
        n_samples: Number of samples.
        n_drugs: Number of drugs.
        tissue_index: Tissue index of each sample.

    Returns:
        The hidden sample factors ``z`` with shape ``(n_samples, K)`` and the
        pEC50 matrix with shape ``(n_samples, n_drugs)``.
    """
    n_tissues = int(tissue_index.max()) + 1
    normal = NormalDist()
    quantiles = [normal.inv_cdf((index + 0.5) / n_drugs) for index in range(n_drugs)]

    factors = rng.normal(size=(n_samples, _N_FACTORS))
    drug_effect = rng.permutation(_DRUG_MEAN_SD * np.asarray(quantiles))
    tissue_effect = rng.normal(0.0, _TISSUE_SD, n_tissues)
    sample_loading = rng.normal(0.0, _SAMPLE_EFFECT_SD / np.sqrt(_N_FACTORS), _N_FACTORS)
    interaction = rng.normal(0.0, _INTERACTION_SD / np.sqrt(_N_FACTORS), (n_drugs, _N_FACTORS))
    sample_effect = tissue_effect[tissue_index] + factors @ sample_loading
    noise = rng.normal(0.0, _NOISE_SD, (n_samples, n_drugs))
    pec50 = _BASE_PEC50 + drug_effect[None, :] + sample_effect[:, None] + factors @ interaction.T + noise
    return factors, pec50.astype(np.float32)


def _response_anndata(
    rng: np.random.Generator,
    *,
    n_samples: int,
    n_drugs: int,
    n_tissues: int,
    missing_fraction: float,
    low_quality_fraction: float,
    extra_layers: Sequence[str],
) -> tuple[ad.AnnData, np.ndarray]:
    """Build the ``response`` modality: the pair matrix plus entity metadata.

    Returns:
        The modality and the hidden sample factors its omics views must mix.
    """
    import anndata as ad
    import pandas as pd

    n_distinct = min(n_tissues, len(_TISSUES))
    sample_ids = [f"SYNTH-{index:03d}" for index in range(n_samples)]
    drug_ids = [f"{100000 + index}" for index in range(n_drugs)]
    tissue_index = np.arange(n_samples) % n_distinct

    factors, matrix = _simulate_response(rng, n_samples=n_samples, n_drugs=n_drugs, tissue_index=tissue_index)
    _punch_holes(matrix, rng, missing_fraction)

    response = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(
            {TISSUE_IDENTIFIER: [_TISSUES[index] for index in tissue_index]},
            index=pd.Index(sample_ids),
        ),
        var=pd.DataFrame(
            {
                "drug_name": [f"synthdrug{index}" for index in range(n_drugs)],
                "canonical_smiles": list(SMILES[:n_drugs]),
            },
            index=pd.Index(drug_ids, name="pubchem_id"),
        ),
    )
    response.layers[MEASURE] = _ln_ic50_from_pec50(matrix)
    for layer_name in extra_layers:
        response.layers[layer_name] = _EXTRA_LAYERS[layer_name](matrix)
    for name, layer in _quality_layers(rng, matrix, low_quality_fraction=low_quality_fraction).items():
        response.layers[name] = layer
    return response, factors


def _ln_ic50_from_pec50(pec50: np.ndarray) -> np.ndarray:
    """Derive an ``LN_IC50`` layer from a pEC50 matrix.

    Uses the ideal-curve identity ``EC50 [uM] = 10 ** (6 - pEC50)``, i.e. the
    ``IC50 == EC50`` case of the closed-form solution
    :func:`drevalpy.curation._postprocess._compute_ic50` inverts. That keeps the
    layer monotone in ``X`` and on a plausible scale without pretending the
    fixture has curve plateaus it never fitted.

    :param pec50: pEC50 matrix, possibly containing NaN.
    :returns: Matching ``LN_IC50`` matrix, NaN preserved.
    """
    return ((6.0 - pec50) * np.log(10.0)).astype(np.float32)


def _auc_from_pec50(pec50: np.ndarray) -> np.ndarray:
    """Derive an ``AUC`` layer in ``(0, 1)``: a more potent pair has a lower area."""
    return (1.0 / (1.0 + np.exp(pec50 - _BASE_PEC50))).astype(np.float32)


def _ic50_from_pec50(pec50: np.ndarray) -> np.ndarray:
    """Derive an ``IC50`` layer in uM from a pEC50 matrix, on the ideal-curve identity."""
    return np.power(10.0, 6.0 - pec50).astype(np.float32)


#: Layers ``extra_layers`` may name, each derived from the pEC50 matrix.
_EXTRA_LAYERS: Final = {"AUC": _auc_from_pec50, "IC50": _ic50_from_pec50}


def _quality_layers(
    rng: np.random.Generator,
    matrix: np.ndarray,
    *,
    low_quality_fraction: float,
) -> dict[str, np.ndarray]:
    """Build the CurveCurator quality layers the splitters filter on.

    Every metric is derived from a single per-pair "is this curve good" draw, so
    the layers agree with each other: a pair that fails ``relevance_score`` does
    not also claim ``R2 = 0.99``. Filtering on any one option therefore selects
    the same pairs, which is what makes a non-default option testable.

    Args:
        rng: Source of randomness.
        matrix: Response matrix, used only for its shape.
        low_quality_fraction: Share of pairs to mark as failing the default
            thresholds. The failing pairs are chosen with the same per-row and
            per-column guarantee ``_punch_holes`` uses, because a quality filter
            that empties a sample breaks the leave-one-out splitters exactly
            as an unmeasured one does.

    Returns:
        Layer name to matrix, ready to assign into ``response.layers``.
    """
    good = np.ones(matrix.shape, dtype=bool)
    if low_quality_fraction > 0:
        holes = np.zeros(matrix.shape, dtype=np.float32)
        _punch_holes(holes, rng, low_quality_fraction)
        good = ~np.isnan(holes)

    def _pick(good_value: float, bad_value: float) -> np.ndarray:
        return np.where(good, good_value, bad_value).astype(np.float32)

    # Values sit far from the thresholds on both sides, so a boundary change in
    # curve_quality_mask cannot silently reclassify a synthetic pair.
    return {
        "relevance_score": _pick(9.0, 0.01),
        "fold_change": _pick(-2.0, -0.01),
        "p_value": _pick(1e-9, 0.9),
        "log_p_value": _pick(9.0, 0.05),
        "f_value": _pick(400.0, 0.5),
        "f_value_sam": _pick(80.0, 0.001),
        "R2": _pick(0.99, 0.02),
        "RMSE": _pick(0.02, 0.8),
        "signal_quality": _pick(1.0, 0.0),
        "slope": _pick(3.0, 0.05),
        "front": _pick(1.0, 0.1),
        "back": _pick(0.05, 5.0),
        "regulation": _pick(-1.0, 0.0),
        # Not filter options, but layers every curated dataset carries: the
        # per-parameter standard errors CurveCurator derives from the fit's
        # Jacobian. A bad curve gets a wide error, matching the draw above.
        "pec50_error": _pick(0.05, 20.0),
        "slope_error": _pick(0.1, 50.0),
        "front_error": _pick(0.01, 5.0),
        "back_error": _pick(0.01, 5.0),
    }


def synthetic(
    *,
    name: str = DATASET_NAME,
    n_samples: int = N_SAMPLES,
    n_drugs: int = N_DRUGS,
    n_tissues: int = N_TISSUES,
    omics: Sequence[str] | Mapping[str, int] | None = ("gene_expression",),
    sample_views: Sequence[str] = (),
    drug_views: Sequence[str] = (),
    uns_payloads: Sequence[str] = (),
    extra_layers: Sequence[str] = (),
    feature_names: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    n_features: int = N_FEATURES,
    missing_fraction: float = 0.05,
    low_quality_fraction: float = 0.0,
    seed: int = SEED,
) -> Dataset:
    """Build a deterministic in-memory dataset with a Simpson-like response structure.

    Drugs differ strongly in average potency, so a baseline predicting each
    drug's mean scores a high pooled correlation but ranks samples no better
    than chance within a drug. Hidden sample factors drive both the response and
    every omics view, so a model reading the features can beat that baseline.

    Everything beyond the response and the default ``gene_expression`` view is
    opt-in, so the default dataset stays small and cheap. The extras are what the
    zoo's other models read: fingerprints, graphs, pathway sets and ontologies.

    Args:
        name: Dataset name recorded on the returned object.
        n_samples: Number of samples on the ``obs`` axis.
        n_drugs: Number of drugs on the ``var`` axis, at most
            :data:`MAX_SYNTHETIC_DRUGS`.
        n_tissues: Number of distinct tissue labels to cycle through, at most
            six.
        omics: Public omics names to add as sample modalities; ``None`` or
            ``()`` adds none. Any of ``gene_expression``, ``proteomics``,
            ``mutations``, ``methylation`` and ``copy_number_variation_gistic``.
            A mapping additionally sets each modality's sample coverage, so a
            value below *n_samples* leaves trailing samples unmeasured and
            exercises a component's NaN handling.
        sample_views: Extra sample views stored in ``response.obsm``:
            ``pathway_features`` (seeded noise).
        drug_views: Drug views stored in ``response.varm``: ``morgan_fingerprint``
            (computed by the real featurizer, so it needs ``rdkit``) and the
            seeded-noise views ``chemberta``, ``bpe_smiles`` and ``smilesvec``.
        uns_payloads: Auxiliary payloads stored in ``mdata.uns``:
            ``drug_graphs`` (computed, needs ``rdkit``), ``pathways_gmt``,
            ``sparsego`` and ``bpe_codes`` (learned, needs ``subword-nmt``).
        extra_layers: Response layers derived from the pEC50 matrix, beyond
            ``LN_IC50``: any of ``AUC`` and ``IC50``.
        feature_names: Column names for every omics modality, or a mapping from
            public omics name to its column names. Defaults to the first
            *n_features* landmark gene symbols, or methylation site ids for
            ``methylation``.
        n_features: Number of generated column names when *feature_names* does
            not name a view.
        missing_fraction: Share of response pairs left unmeasured. Every sample
            and drug keeps at least one measurement regardless.
        low_quality_fraction: Share of response pairs whose curve-quality
            metrics fail the thresholds
            :func:`~drevalpy.data.quality.curve_quality_mask` applies, so the
            built-in splitters drop them. Defaults to ``0.0``, i.e. every curve
            passes and a split is determined by *missing_fraction* alone.
        seed: Seed for every drawn matrix, making the dataset reproducible.

    Returns:
        A dataset carrying a ``response`` modality with sample metadata,
        tissue labels, drug identifiers, drug ``canonical_smiles`` and the
        CurveCurator quality layers, plus any requested omics modality and extra.

    Raises:
        ValueError: If *n_drugs* exceeds :data:`MAX_SYNTHETIC_DRUGS`, if a name in
            *sample_views*, *drug_views*, *uns_payloads* or *extra_layers* is
            unknown, or if the packaged gene list is shorter than *n_features*.
    """
    import mudata as md

    if n_drugs > MAX_SYNTHETIC_DRUGS:
        msg = f"n_drugs={n_drugs} is above the limit of {MAX_SYNTHETIC_DRUGS} synthetic drugs"
        raise ValueError(msg)
    check_choices("extra_layers", extra_layers, _EXTRA_LAYERS)

    rng = np.random.default_rng(seed)
    response, factors = _response_anndata(
        rng,
        n_samples=n_samples,
        n_drugs=n_drugs,
        n_tissues=n_tissues,
        missing_fraction=missing_fraction,
        low_quality_fraction=low_quality_fraction,
        extra_layers=extra_layers,
    )

    built: dict[str, ad.AnnData] = {}
    for public_name, n_covered in omics_coverage(omics, n_samples).items():
        built[public_name] = omics_anndata(
            rng,
            public_name,
            list(response.obs_names),
            factors,
            n_covered=n_covered,
            feature_names=view_feature_names(public_name, feature_names, n_features),
        )

    modalities: dict[str, ad.AnnData] = {"response": response}
    modalities.update({resolve_omics_accessor(public_name): modality for public_name, modality in built.items()})

    md.set_options(pull_on_update=False)
    mdata = md.MuData(modalities)
    mdata.obs = response.obs.copy()
    dataset = Dataset(mdata, name=name)
    add_uns_payloads(dataset, uns_payloads, payload_genes(built, n_features) if uns_payloads else [])
    add_drug_views(dataset, drug_views, rng)
    add_sample_views(dataset, sample_views, rng)
    return dataset
