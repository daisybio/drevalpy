"""Deterministic in-memory synthetic datasets for tutorials, tests and plugins.

Every dataset registered with drevalpy is fetched from a credentialed S3 bucket,
so an offline test run, a tutorial and any third-party plugin's CI cannot use
one. :func:`synthetic` produces a :class:`~drevalpy.types.data.dataset.Dataset`
with the same structural slots a published ``.h5mu`` has, entirely in memory.

The default output carries only the ``response`` modality, which is all an
``entity_id_only`` featurizer needs. Pass *omics* to add sample feature
modalities for featurizers that read a view.

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

``response.X`` is in pEC50 units and the ``LN_IC50`` layer is derived from it.

``mudata``, ``anndata`` and ``pandas`` are imported inside the functions that
need them: this module is re-exported by :mod:`drevalpy.data`, which sits on the
critical path of ``import drevalpy`` (see ``tests/test_import_cost_policy.py``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import NormalDist
from typing import TYPE_CHECKING, Final

import numpy as np

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

#: Real, rdkit-parseable SMILES, so the ``fingerprints`` featurizer works. The
#: simulated responses do not depend on them.
_SMILES: Final = (
    "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC",
    "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4",
    "CS(=O)(=O)CCNCC1=CC2=C(C=C1)N=CN=C2NC3=CC(=C(C=C3)OCC4=CC(=CC=C4)F)Cl",
    "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
    "CNC(=O)C1=NC=CC(=C1)OC2=CC=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F",
    "CCN(CC)CCNC(=O)C1=C(NC(=C1C)C=C2C3=C(C=CC(=C3)F)NC2=O)C",
    "CC1=NC(=CC(=N1)N2CCN(CC2)CCO)NC3=NC=C(S3)C(=O)NC4=C(C=CC=C4Cl)C",
    "CNC(=O)C1=CC=CC=C1SC2=CC3=C(C=C2)C(=NN3)C=CC4=CC=CC=N4",
    "CN(CC1=CN=C2C(=N1)C(=NC(=N2)N)N)C3=CC=C(C=C3)C(=O)NC(CCC(=O)O)C(=O)O",
    "C1=CN(C(=O)N=C1N)C2C(C(C(O2)CO)O)(F)F",
    "C1=C(C(=O)NC(=O)N1)F",
    "CCC(=C(C1=CC=CC=C1)C2=CC=C(C=C2)OCCN(C)C)C3=CC=CC=C3",
    "C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O",
    "CCC1(C2=C(COC1=O)C(=O)N3CC4=CC5=CC=CC=C5N=C4C3=C2)O",
    "C1=CC=C(C=C1)NC(=O)CCCCCCC(=O)NO",
    "C1=CN(C(=O)N=C1N)C2C(C(C(O2)CO)O)O",
    "C1=NC2=C(N1)C(=S)N=CN2",
    "NC(=O)NO",
    "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",
    "CCCC1=NN(C2=C1NC(=NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C",
    "CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F",
    "CN1C(=O)CN=C(C2=C1C=CC(=C2)Cl)C3=CC=CC=C3",
    "CC(C)NCC(COC1=CC=CC2=CC=CC=C21)O",
    "CC(C)NCC(COC1=CC=C(C=C1)CC(=O)N)O",
    "CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O",
    "CC(=O)OC1=CC=CC=C1C(=O)O",
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "CC(=O)NC1=CC=C(C=C1)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CN(C)C(=N)N=C(N)N",
    "CN1CCCC1C2=CN=CC=C2",
)

#: Most drugs :func:`synthetic` can build; one SMILES per drug.
MAX_SYNTHETIC_DRUGS: Final = len(_SMILES)

#: Hidden sample factors shared by the response and every omics view.
_N_FACTORS: Final = 3
_DRUG_MEAN_SD: Final = 1.5
_TISSUE_SD: Final = 0.3
_SAMPLE_EFFECT_SD: Final = 0.4
_INTERACTION_SD: Final = 0.6
_NOISE_SD: Final = 0.5
_BASE_PEC50: Final = 2.0
_OMICS_BASE: Final = 6.0
_OMICS_NOISE_SD: Final = 0.5


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
                "canonical_smiles": list(_SMILES[:n_drugs]),
            },
            index=pd.Index(drug_ids, name="pubchem_id"),
        ),
    )
    response.layers[MEASURE] = _ln_ic50_from_pec50(matrix)
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


def _omics_coverage(omics: Sequence[str] | Mapping[str, int] | None, n_samples: int) -> dict[str, int]:
    """Normalize the *omics* argument into a public-name to coverage map."""
    if omics is None:
        return {}
    if isinstance(omics, Mapping):
        return {name: int(count) for name, count in omics.items()}
    return dict.fromkeys(omics, n_samples)


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


def _omics_anndata(
    rng: np.random.Generator,
    sample_ids: Sequence[str],
    factors: np.ndarray,
    *,
    n_covered: int,
    feature_names: Sequence[str],
) -> ad.AnnData:
    """Build one omics modality covering the first *n_covered* samples.

    Every feature is the shared hidden factors mixed through a random loading
    matrix, plus noise, so the view carries the sample effect in the response.
    """
    import anndata as ad
    import pandas as pd

    covered = list(sample_ids[:n_covered])
    loadings = rng.normal(size=(factors.shape[1], len(feature_names)))
    mixed = _OMICS_BASE + factors[:n_covered] @ loadings
    noise = rng.normal(0.0, _OMICS_NOISE_SD, size=mixed.shape)
    return ad.AnnData(
        X=(mixed + noise).astype(np.float32),
        obs=pd.DataFrame(index=pd.Index(covered)),
        var=pd.DataFrame(index=pd.Index(list(feature_names), name="feature")),
    )


def synthetic(
    *,
    name: str = DATASET_NAME,
    n_samples: int = N_SAMPLES,
    n_drugs: int = N_DRUGS,
    n_tissues: int = N_TISSUES,
    omics: Sequence[str] | Mapping[str, int] | None = None,
    feature_names: Sequence[str] | None = None,
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

    Args:
        name: Dataset name recorded on the returned object.
        n_samples: Number of samples on the ``obs`` axis.
        n_drugs: Number of drugs on the ``var`` axis, at most
            :data:`MAX_SYNTHETIC_DRUGS`.
        n_tissues: Number of distinct tissue labels to cycle through, at most
            six.
        omics: Public omics names to add as sample modalities. A mapping
            additionally sets each modality's sample coverage, so a value
            below *n_samples* leaves trailing samples unmeasured and
            exercises a component's NaN handling.
        feature_names: Column names for every omics modality. Defaults to
            generated ``FEATURE0000`` style names.
        n_features: Number of generated column names when *feature_names* is
            ``None``.
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
        CurveCurator quality layers, plus any requested omics modality.

    Raises:
        ValueError: If *n_drugs* exceeds :data:`MAX_SYNTHETIC_DRUGS`.
    """
    import mudata as md

    if n_drugs > MAX_SYNTHETIC_DRUGS:
        msg = f"n_drugs={n_drugs} is above the limit of {MAX_SYNTHETIC_DRUGS} synthetic drugs"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    response, factors = _response_anndata(
        rng,
        n_samples=n_samples,
        n_drugs=n_drugs,
        n_tissues=n_tissues,
        missing_fraction=missing_fraction,
        low_quality_fraction=low_quality_fraction,
    )
    columns = feature_names if feature_names is not None else [f"FEATURE{index:04d}" for index in range(n_features)]

    modalities: dict[str, ad.AnnData] = {"response": response}
    for public_name, n_covered in _omics_coverage(omics, n_samples).items():
        modalities[resolve_omics_accessor(public_name)] = _omics_anndata(
            rng,
            list(response.obs_names),
            factors,
            n_covered=n_covered,
            feature_names=columns,
        )

    md.set_options(pull_on_update=False)
    mdata = md.MuData(modalities)
    mdata.obs = response.obs.copy()
    return Dataset(mdata, name=name)
