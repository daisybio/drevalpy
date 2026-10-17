"""In-memory synthetic dataset for plugin and component tests.

Every dataset registered with drevalpy is fetched from a credentialed S3 bucket,
so an offline test run - and any third-party plugin's CI - cannot use one. This
builder produces a :class:`~drevalpy.types.data.dataset.Dataset` with the same
structural slots a published ``.h5mu`` has, entirely in memory.

The default output carries only the ``response`` modality, which is all an
``entity_id_only`` featurizer needs. Pass *omics* to add cell-line feature
modalities for featurizers that read a view.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd

from drevalpy.data.utils import CELL_LINE_IDENTIFIER, TISSUE_IDENTIFIER
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.modalities import resolve_omics_accessor

#: Smallest shape that keeps the shipped splitters and predictors happy: ``LTO``
#: runs ``KFold(n_splits=2)`` over the unique tissues and then carves a
#: validation tissue out of the training half, so three tissues is the floor.
N_CELL_LINES: Final = 24
N_DRUGS: Final = 8
N_TISSUES: Final = 6
N_FEATURES: Final = 16

#: A curve-metric layer name the curation pipeline really emits. ``response.X``
#: holds pEC50, which curation does not duplicate as a layer, so the fixture
#: derives this one from ``X`` the way a real fit would - see
#: :func:`drevalpy.curation.curate`.
MEASURE: Final = "LN_IC50"

DATASET_NAME: Final = "SYNTHETIC"
SEED: Final = 20260813

_TISSUES: Final = ("Lung", "Blood", "Skin", "Colon", "Brain", "Breast")


def _punch_holes(matrix: np.ndarray, rng: np.random.Generator, fraction: float) -> None:
    """Blank a fraction of *matrix* in place, leaving every row and column measured.

    Unmeasured pairs are what exercise a component's NaN-filtering path. The
    row/column guarantee matters because a fully unmeasured cell line or drug
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


def _response_anndata(
    rng: np.random.Generator,
    *,
    n_cell_lines: int,
    n_drugs: int,
    n_tissues: int,
    missing_fraction: float,
    low_quality_fraction: float,
) -> ad.AnnData:
    """Build the ``response`` modality: the pair matrix plus entity metadata."""
    cell_line_ids = [f"CVCL_S{index:03d}" for index in range(n_cell_lines)]
    drug_ids = [f"{100000 + index}" for index in range(n_drugs)]
    tissues = [_TISSUES[index % min(n_tissues, len(_TISSUES))] for index in range(n_cell_lines)]

    matrix = rng.normal(2.0, 1.0, size=(n_cell_lines, n_drugs)).astype(np.float32)
    _punch_holes(matrix, rng, missing_fraction)

    response = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(
            {
                CELL_LINE_IDENTIFIER: [f"SYNTH-{index:03d}" for index in range(n_cell_lines)],
                TISSUE_IDENTIFIER: tissues,
            },
            index=pd.Index(cell_line_ids, name="cellosaurus_id"),
        ),
        var=pd.DataFrame(
            {"drug_name": [f"synthdrug{index}" for index in range(n_drugs)]},
            index=pd.Index(drug_ids, name="pubchem_id"),
        ),
    )
    response.layers[MEASURE] = _ln_ic50_from_pec50(matrix)
    for name, layer in _quality_layers(rng, matrix, low_quality_fraction=low_quality_fraction).items():
        response.layers[name] = layer
    return response


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


def _omics_coverage(omics: Sequence[str] | Mapping[str, int] | None, n_cell_lines: int) -> dict[str, int]:
    """Normalize the *omics* argument into a public-name to coverage map."""
    if omics is None:
        return {}
    if isinstance(omics, Mapping):
        return {name: int(count) for name, count in omics.items()}
    return dict.fromkeys(omics, n_cell_lines)


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
            that empties a cell line breaks the leave-one-out splitters exactly
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
    cell_line_ids: Sequence[str],
    *,
    n_covered: int,
    feature_names: Sequence[str],
) -> ad.AnnData:
    """Build one omics modality covering the first *n_covered* cell lines."""
    covered = list(cell_line_ids[:n_covered])
    return ad.AnnData(
        X=rng.normal(6.0, 1.5, size=(len(covered), len(feature_names))).astype(np.float32),
        obs=pd.DataFrame(index=pd.Index(covered, name="cellosaurus_id")),
        var=pd.DataFrame(index=pd.Index(list(feature_names), name="feature")),
    )


def build_synthetic_dataset(
    *,
    name: str = DATASET_NAME,
    n_cell_lines: int = N_CELL_LINES,
    n_drugs: int = N_DRUGS,
    n_tissues: int = N_TISSUES,
    omics: Sequence[str] | Mapping[str, int] | None = None,
    feature_names: Sequence[str] | None = None,
    n_features: int = N_FEATURES,
    missing_fraction: float = 0.05,
    low_quality_fraction: float = 0.0,
    seed: int = SEED,
) -> Dataset:
    """Build a deterministic in-memory dataset for tests.

    Args:
        name: Dataset name recorded on the returned object.
        n_cell_lines: Number of cell lines on the ``obs`` axis.
        n_drugs: Number of drugs on the ``var`` axis.
        n_tissues: Number of distinct tissue labels to cycle through.
        omics: Public omics names to add as cell-line modalities. A mapping
            additionally sets each modality's cell-line coverage, so a value
            below *n_cell_lines* leaves trailing cell lines unmeasured and
            exercises a component's NaN handling.
        feature_names: Column names for every omics modality. Defaults to
            generated ``FEATURE0000`` style names.
        n_features: Number of generated column names when *feature_names* is
            ``None``.
        missing_fraction: Share of response pairs left unmeasured. Every cell
            line and drug keeps at least one measurement regardless.
        low_quality_fraction: Share of response pairs whose curve-quality
            metrics fail the thresholds
            :func:`~drevalpy.data.quality.curve_quality_mask` applies, so the
            built-in splitters drop them. Defaults to ``0.0``, i.e. every curve
            passes and a split is determined by *missing_fraction* alone.
        seed: Seed for every drawn matrix, making the dataset reproducible.

    Returns:
        A dataset carrying a ``response`` modality with cell-line metadata,
        tissue labels, drug identifiers and the CurveCurator quality layers,
        plus any requested omics modality.
    """
    rng = np.random.default_rng(seed)
    response = _response_anndata(
        rng,
        n_cell_lines=n_cell_lines,
        n_drugs=n_drugs,
        n_tissues=n_tissues,
        missing_fraction=missing_fraction,
        low_quality_fraction=low_quality_fraction,
    )
    columns = feature_names if feature_names is not None else [f"FEATURE{index:04d}" for index in range(n_features)]

    modalities: dict[str, ad.AnnData] = {"response": response}
    for public_name, n_covered in _omics_coverage(omics, n_cell_lines).items():
        modalities[resolve_omics_accessor(public_name)] = _omics_anndata(
            rng,
            list(response.obs_names),
            n_covered=n_covered,
            feature_names=columns,
        )

    md.set_options(pull_on_update=False)
    mdata = md.MuData(modalities)
    mdata.obs = response.obs.copy()
    return Dataset(mdata, name=name)
