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

#: Response measure name matching the published datasets.
MEASURE: Final = "LN_IC50_curvecurator"

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
    response.layers[MEASURE] = matrix.copy()
    return response


def _omics_coverage(omics: Sequence[str] | Mapping[str, int] | None, n_cell_lines: int) -> dict[str, int]:
    """Normalize the *omics* argument into a public-name to coverage map."""
    if omics is None:
        return {}
    if isinstance(omics, Mapping):
        return {name: int(count) for name, count in omics.items()}
    return dict.fromkeys(omics, n_cell_lines)


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
        seed: Seed for every drawn matrix, making the dataset reproducible.

    Returns:
        A dataset carrying a ``response`` modality with cell-line metadata,
        tissue labels and drug identifiers, plus any requested omics modality.
    """
    rng = np.random.default_rng(seed)
    response = _response_anndata(
        rng,
        n_cell_lines=n_cell_lines,
        n_drugs=n_drugs,
        n_tissues=n_tissues,
        missing_fraction=missing_fraction,
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
