"""Contains functions to load the GDSC1, GDSC2, CCLE, and Toy datasets."""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from .curvecurator import fit_curves
from .dataset import DrugResponseDataset
from .utils import (
    ALLOWED_MEASURES,
    CELL_LINE_IDENTIFIER,
    DRUG_IDENTIFIER,
    TISSUE_IDENTIFIER,
    download_dataset,
    download_from_url,
    unzip_data,
)


def check_measure(measure_queried: str, measures_data: list[str], dataset_name: str) -> None:
    """
    Check if the queried measure is in the dataset.

    :param measure_queried: The measure to check.
    :param measures_data: The measures in the dataset.
    :param dataset_name: The name of the dataset.
    :raises ValueError: If the measure is not found in the dataset.
    """
    measures_available = set(ALLOWED_MEASURES).intersection(set(measures_data))
    if measure_queried not in measures_data:
        raise ValueError(
            f"Measure '{measure_queried}' not found in dataset {dataset_name}."
            f"Available measures are: {', '.join(measures_available)}."
        )


def _read_response_frame(path_data: str, file_name: str, dataset_name: str) -> pd.DataFrame:
    """
    Download (if needed) and read a dataset's response csv into a cleaned DataFrame.

    Handles the shared loading steps: download of the dataset and tissue-mapping meta,
    reading the csv with string ids, stripping commas from drug ids, and the per-dataset
    tissue overrides. The returned frame keeps every column (e.g. ``Regulation``), so callers
    that filter on curve-curator output can do so before building the dataset.

    :param path_data: Path to the dataset.
    :param file_name: File name of the dataset, e.g., GDSC1.csv
    :param dataset_name: Name of the dataset, e.g., GDSC1.
    :return: the response DataFrame.
    """
    path = os.path.join(path_data, dataset_name, file_name)
    if not os.path.exists(path):
        download_dataset(dataset_name, path_data, redownload=True)
    # tissue mapping is not in TOY play dataset
    meta_path = os.path.join(path_data, "meta", "tissue_mapping.csv")
    if not os.path.exists(meta_path):
        download_dataset("meta", path_data, redownload=True)

    response_data = pd.read_csv(path, dtype={"pubchem_id": str, "cell_line_name": str})
    response_data[DRUG_IDENTIFIER] = response_data[DRUG_IDENTIFIER].str.replace(",", "")
    if dataset_name == "BeatAML2":
        # only has AML patients = blood
        response_data[TISSUE_IDENTIFIER] = "Blood"
    elif dataset_name == "PDX_Bruna":
        # only has breast cancer patients
        response_data[TISSUE_IDENTIFIER] = "Breast"
    return response_data


def _frame_to_dataset(response_data: pd.DataFrame, measure: str, dataset_name: str) -> DrugResponseDataset:
    """
    Build a :class:`DrugResponseDataset` from an already-read response frame.

    :param response_data: response DataFrame, e.g. from :func:`_read_response_frame`.
    :param measure: The name of the column containing the measure to predict.
    :param dataset_name: Name of the dataset, e.g., GDSC1.
    :return: DrugResponseDataset containing response, cell line IDs, drug IDs, and tissues.
    """
    check_measure(measure, list(response_data.columns), dataset_name)
    return DrugResponseDataset(
        response=response_data[measure].values,
        cell_line_ids=response_data[CELL_LINE_IDENTIFIER].values,
        drug_ids=response_data[DRUG_IDENTIFIER].values,
        tissues=response_data[TISSUE_IDENTIFIER].values,
        dataset_name=dataset_name,
    )


def _load_zenodo_dataset(
    path_data: str = "data",
    measure: str = "LN_IC50_curvecurator",
    file_name: str = "dataset_name.csv",
    dataset_name: str = "dataset_name",
) -> DrugResponseDataset:
    """
    Parent function to load_gdsc1, load_gdsc2, ...

    :param path_data: Path to the dataset.
    :param file_name: File name of the dataset, e.g., GDSC1.csv
    :param measure: File name of the dataset, default = "LN_IC50_curvecurator".
    :param dataset_name: Name of the dataset, e.g., GDSC1.
    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs.
    """
    response_data = _read_response_frame(path_data, file_name, dataset_name)
    return _frame_to_dataset(response_data, measure, dataset_name)


def load_gdsc1(
    path_data: str = "data",
    measure: str = "LN_IC50_curvecurator",
) -> DrugResponseDataset:
    """
    Loads the GDSC1 dataset.

    :param path_data: Path to the dataset.
    :param measure: The name of the column containing the measure to predict, default = "LN_IC50_curvecurator"
    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs.
    """
    return _load_zenodo_dataset(path_data=path_data, measure=measure, file_name="GDSC1.csv", dataset_name="GDSC1")


def load_gdsc2(
    path_data: str = "data",
    measure: str = "LN_IC50_curvecurator",
):
    """
    Loads the GDSC2 dataset.

    :param path_data: Path to the dataset.
    :param measure: The name of the column containing the measure to predict, default = "LN_IC50_curvecurator"

    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs.
    """
    return _load_zenodo_dataset(path_data=path_data, measure=measure, file_name="GDSC2.csv", dataset_name="GDSC2")


def load_ccle(
    path_data: str = "data",
    measure: str = "LN_IC50_curvecurator",
) -> DrugResponseDataset:
    """
    Loads the CCLE dataset.

    :param path_data: Path to the dataset.
    :param measure: The name of the column containing the measure to predict, default = "LN_IC50_curvecurator"

    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs.
    """
    return _load_zenodo_dataset(path_data=path_data, measure=measure, file_name="CCLE.csv", dataset_name="CCLE")


def _load_test_data(
    path_data: str = "data", measure: str = "LN_IC50_curvecurator", dataset_name: str = "TOYv1"
) -> DrugResponseDataset:
    # ensure that path_data exists
    Path(path_data).mkdir(parents=True, exist_ok=True)
    test_data_path = "https://github.com/nf-core/test-datasets/raw/refs/heads/drugresponseeval/test_data"
    # first get meta
    meta_path = os.path.join(path_data, "meta")
    if not os.path.exists(meta_path):
        file_url = f"{test_data_path}/meta.zip"
        file_path = Path(path_data) / "meta.zip"
        response_meta = download_from_url(dataset_name="meta", file_url=file_url)
        unzip_data(path_to_zip=file_path, response=response_meta, data_path=path_data)
    # get raw test data
    raw_data_path = os.path.join(path_data, "CTRPv2_sample_test")
    if not os.path.exists(raw_data_path):
        file_url = f"{test_data_path}/CTRPv2_sample_test.zip"
        file_path = Path(path_data) / "CTRPv2_sample_test.zip"
        response_raw = download_from_url(dataset_name="CTRPv2_sample_test", file_url=file_url)
        unzip_data(path_to_zip=file_path, response=response_raw, data_path=path_data)
    file_url = f"{test_data_path}/{dataset_name}.zip"
    file_path = Path(path_data) / f"{dataset_name}.zip"
    response = download_from_url(dataset_name=dataset_name, file_url=file_url)
    unzip_data(path_to_zip=file_path, response=response, data_path=path_data)

    file_name = Path(path_data) / dataset_name / f"{dataset_name}.csv"
    response_data = pd.read_csv(file_name, dtype={"pubchem_id": str, "cell_line_name": str})
    response_data[DRUG_IDENTIFIER] = response_data[DRUG_IDENTIFIER].str.replace(",", "")
    check_measure(measure, list(response_data.columns), dataset_name)
    return DrugResponseDataset(
        response=response_data[measure].values,
        cell_line_ids=response_data[CELL_LINE_IDENTIFIER].values,
        drug_ids=response_data[DRUG_IDENTIFIER].values,
        tissues=response_data[TISSUE_IDENTIFIER].values,
        dataset_name=dataset_name,
    )


def load_toyv1(path_data: str = "data", measure: str = "LN_IC50_curvecurator") -> DrugResponseDataset:
    """
    Loads small Toy dataset, subsampled from CTRPv2.

    :param path_data: Path to the dataset.
    :param measure: The name of the column containing the measure to predict, default = "LN_IC50_curvecurator"

    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs.
    """
    return _load_test_data(path_data=path_data, measure=measure, dataset_name="TOYv1")


def load_toyv2(path_data: str = "data", measure: str = "LN_IC50_curvecurator") -> DrugResponseDataset:
    """
    Loads small Toy dataset, subsampled from GDSC2. Can be used to test cross study prediction.

    :param path_data: Path to the dataset.
    :param measure: The name of the column containing the measure to predict, default = "LN_IC50_curvecurator"

    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs.
    """
    return _load_test_data(path_data=path_data, measure=measure, dataset_name="TOYv2")


def load_ctrpv1(path_data: str = "data", measure: str = "LN_IC50_curvecurator") -> DrugResponseDataset:
    """
    Load CTRPv1 dataset.

    :param path_data: Path to the location of CTRPv1 dataset
    :param measure: The name of the column containing the measure to predict, default = "LN_IC50_curvecurator"

    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs
    """
    return _load_zenodo_dataset(path_data=path_data, measure=measure, file_name="CTRPv1.csv", dataset_name="CTRPv1")


def load_ctrpv2(path_data: str = "data", measure: str = "LN_IC50_curvecurator") -> DrugResponseDataset:
    """
    Load CTRPv2 dataset.

    :param path_data: Path to the location of CTRPv2 dataset
    :param measure: The name of the column containing the measure to predict, default: LN_IC50_curvecurator

    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs
    """
    return _load_zenodo_dataset(path_data=path_data, measure=measure, file_name="CTRPv2.csv", dataset_name="CTRPv2")


@dataclass(frozen=True)
class DrugCurveFilter:
    """
    Keep only drugs with enough reproducible (curve-curated) dose-response curves.

    A drug is kept if its number of *significant* curves (``Regulation`` in
    :attr:`significant_labels`) meets the active threshold. By default the threshold is an
    absolute count (:attr:`min_responders`); set :attr:`min_responder_frac` instead for a
    fraction-of-screened-lines criterion. Exactly one of the two must be set.

    We default to the absolute count rather than a fraction on purpose: a fraction conflates
    truly-dead compounds (prodrugs/non-cytotoxics with ~0 responders anywhere) with
    selective/biomarker-driven drugs (e.g. Venetoclax, Quizartinib) that are flat in most
    lines but have a real cluster of high-quality responders worth keeping. The absolute
    count is a statistical-power floor (how many curves there are to learn a drug's response
    surface from), which is what matters for modelling and is independent of screen size.

    Whole drugs are dropped, never individual (cell, drug) experiments: removing single
    measurements would condition the sample on the outcome (selection-on-outcome leakage).
    """

    min_responders: int | None = None
    min_responder_frac: float | None = None
    significant_labels: tuple[str, ...] = ("down", "up")

    def __post_init__(self) -> None:
        """:raises ValueError: unless exactly one of min_responders / min_responder_frac is set."""
        if (self.min_responders is None) == (self.min_responder_frac is None):
            raise ValueError("Set exactly one of min_responders or min_responder_frac.")

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Return ``frame`` with whole drugs failing the responder criterion removed.

        :param frame: response frame; must carry the CurveCurator ``Regulation`` column.
        :return: the frame restricted to drugs that pass the threshold.
        :raises ValueError: if ``frame`` is not curve-curated (no ``Regulation`` column).
        """
        if "Regulation" not in frame.columns:
            raise ValueError(
                "Drug cleaning needs curve-curated data (a 'Regulation' column), which this "
                "dataset does not have. Load it with a curve-curated measure / dataset first."
            )
        n_sig = frame["Regulation"].isin(self.significant_labels).groupby(frame[DRUG_IDENTIFIER]).sum()
        if self.min_responders is not None:
            keep = n_sig.index[n_sig >= self.min_responders]
        else:
            n_total = frame.groupby(DRUG_IDENTIFIER).size()
            keep = n_sig.index[(n_sig / n_total) >= self.min_responder_frac]
        return frame[frame[DRUG_IDENTIFIER].isin(set(keep))]


def _materialize_variant(path_data: str, base: str, dataset_name: str, frame: pd.DataFrame) -> None:
    """
    Write a derived dataset's filtered response csv and symlink the base dataset's features.

    The filtered ``<dataset_name>/<dataset_name>.csv`` is written from ``frame``; every other
    entry in the base dataset folder (omics, fingerprints, drug graphs, ...) is symlinked so
    feature files are shared rather than duplicated. The base response csv and model caches are
    not linked. No data is uploaded anywhere; the variant is derived locally from the base.

    Idempotent: if the variant csv already exists nothing is rebuilt (delete
    ``<path_data>/<dataset_name>/`` to regenerate after changing a filter).

    :param path_data: Parent data directory, e.g. "data".
    :param base: Name of the base dataset to derive from, e.g. "CTRPv2".
    :param dataset_name: Name of the derived dataset, e.g. "CTRPv2_clean".
    :param frame: the already-filtered response frame to write.
    """
    src_dir = os.path.join(path_data, base)
    dst_dir = os.path.join(path_data, dataset_name)
    dst_csv = os.path.join(dst_dir, f"{dataset_name}.csv")
    if os.path.exists(dst_csv):
        return
    os.makedirs(dst_dir, exist_ok=True)
    frame.to_csv(dst_csv, index=False)
    # share the base dataset's feature files via symlinks (skip its response csv and caches)
    for entry in os.listdir(src_dir):
        low = entry.lower()
        if entry == f"{base}.csv" or "cache" in low or low == ".ds_store":
            continue
        link = os.path.join(dst_dir, entry)
        if os.path.islink(link) or os.path.exists(link):
            continue
        src = os.path.abspath(os.path.join(src_dir, entry))
        try:
            os.symlink(src, link)
        except OSError:
            # symlinks may be unavailable (e.g. Windows without developer mode); copy instead
            if os.path.isdir(src):
                shutil.copytree(src, link)
            else:
                shutil.copy2(src, link)


def _load_filtered_dataset(path_data: str, measure: str, dataset_name: str) -> DrugResponseDataset:
    """
    Load a derived (drug-filtered) dataset, building it from its base on first use.

    ``dataset_name`` must be registered in :data:`DERIVED_DATASETS` as (base, filter). The base
    response frame is read with :func:`_read_response_frame` (reusing the shared download/parse
    logic), the filter drops whole drugs, the variant folder is materialised with shared feature
    symlinks, and the dataset is built with :func:`_frame_to_dataset`.

    :param path_data: Parent data directory, e.g. "data".
    :param measure: The name of the column containing the measure to predict.
    :param dataset_name: A name registered in :data:`DERIVED_DATASETS`.
    :return: DrugResponseDataset containing response, cell line IDs, drug IDs, and tissues.
    """
    base, drug_filter = DERIVED_DATASETS[dataset_name]
    dst_csv = os.path.join(path_data, dataset_name, f"{dataset_name}.csv")
    if os.path.exists(dst_csv):
        frame = _read_response_frame(path_data, f"{dataset_name}.csv", dataset_name)
    else:
        frame = drug_filter.apply(_read_response_frame(path_data, f"{base}.csv", base))
        _materialize_variant(path_data, base, dataset_name, frame)
    return _frame_to_dataset(frame, measure, dataset_name)


# Derived (drug-filtered) datasets: name -> (base dataset, filter). The clean tiers keep drugs
# with at least N reproducible curves; thresholds are deliberately absolute (see DrugCurveFilter).
# The mechanism is dataset-agnostic: register_clean_tiers works for any curve-curated base. A
# percentage-based tier is one line, e.g. DrugCurveFilter(min_responder_frac=0.1).
CTRPV2_CLEAN_MIN_RESPONDERS = {"CTRPv2_clean": 15, "CTRPv2_cleaner": 30, "CTRPv2_cleanest": 50}


def register_clean_tiers(base: str, tiers: dict[str, int]) -> dict[str, "DrugCurveFilter"]:
    """
    Register absolute-threshold clean tiers for a curve-curated base dataset.

    :param base: base dataset name, e.g. "CTRPv2".
    :param tiers: mapping of derived dataset name -> minimum number of reproducible curves.
    :return: the {name: DrugCurveFilter} entries that were added to :data:`DERIVED_DATASETS`.
    """
    added = {name: DrugCurveFilter(min_responders=n) for name, n in tiers.items()}
    DERIVED_DATASETS.update({name: (base, filt) for name, filt in added.items()})
    return added


DERIVED_DATASETS: dict[str, tuple[str, "DrugCurveFilter"]] = {}
register_clean_tiers("CTRPv2", CTRPV2_CLEAN_MIN_RESPONDERS)


def load_ctrpv2_clean(path_data: str = "data", measure: str = "LN_IC50_curvecurator") -> DrugResponseDataset:
    """
    Load CTRPv2_clean: CTRPv2 with only clearly-dead drugs removed (>=15 responders kept).

    Built automatically from the original CTRPv2 download on first use; no new dataset is hosted.

    :param path_data: Path to the location of the CTRPv2 dataset
    :param measure: The name of the column containing the measure to predict, default: LN_IC50_curvecurator
    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs
    """
    return _load_filtered_dataset(path_data, measure, "CTRPv2_clean")


def load_ctrpv2_cleaner(path_data: str = "data", measure: str = "LN_IC50_curvecurator") -> DrugResponseDataset:
    """
    Load CTRPv2_cleaner: CTRPv2 with low-activity drugs removed (>=30 responders kept).

    Built automatically from the original CTRPv2 download on first use; no new dataset is hosted.

    :param path_data: Path to the location of the CTRPv2 dataset
    :param measure: The name of the column containing the measure to predict, default: LN_IC50_curvecurator
    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs
    """
    return _load_filtered_dataset(path_data, measure, "CTRPv2_cleaner")


def load_ctrpv2_cleanest(path_data: str = "data", measure: str = "LN_IC50_curvecurator") -> DrugResponseDataset:
    """
    Load CTRPv2_cleanest: CTRPv2 with low-activity drugs removed (>=50 responders kept).

    Built automatically from the original CTRPv2 download on first use; no new dataset is hosted.

    :param path_data: Path to the location of the CTRPv2 dataset
    :param measure: The name of the column containing the measure to predict, default: LN_IC50_curvecurator
    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs
    """
    return _load_filtered_dataset(path_data, measure, "CTRPv2_cleanest")


def load_beataml2(
    path_data: str = "data",
    measure: str = "LN_IC50_curvecurator",
) -> DrugResponseDataset:
    """
    Loads the BeatAML2 dataset.

    :param path_data: Path to the dataset.
    :param measure: The name of the column containing the measure to predict, default: LN_IC50_curvecurator

    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs.
    """
    return _load_zenodo_dataset(path_data=path_data, measure=measure, file_name="BeatAML2.csv", dataset_name="BeatAML2")


def load_pdx_bruna(
    path_data: str = "data",
    measure: str = "LN_IC50_curvecurator",
) -> DrugResponseDataset:
    """
    Loads the PDX_Bruna dataset.

    :param path_data: Path to the dataset.
    :param measure: The name of the column containing the measure to predict, default: LN_IC50_curvecurator

    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs.
    """
    return _load_zenodo_dataset(
        path_data=path_data, measure=measure, file_name="PDX_Bruna.csv", dataset_name="PDX_Bruna"
    )


def load_custom(
    path_data: str | Path, dataset_name: str = "custom", measure: str = "response", tissue_column: str | None = None
) -> DrugResponseDataset:
    """
    Load custom dataset.

    :param path_data: Path to location of custom dataset
    :param dataset_name: Name of the dataset.
    :param measure: The name of the column containing the measure to predict, default = "response"
    :param tissue_column: The name of the column containing the tissue type. If None, no tissue information is loaded.

    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs
    """
    return DrugResponseDataset.from_csv(
        input_file=path_data, dataset_name=dataset_name, measure=measure, tissue_column=tissue_column
    )


# Used in pipeline
AVAILABLE_DATASETS: dict[str, Callable] = {
    "GDSC1": load_gdsc1,
    "GDSC2": load_gdsc2,
    "CCLE": load_ccle,
    "TOYv1": load_toyv1,
    "TOYv2": load_toyv2,
    "CTRPv1": load_ctrpv1,
    "CTRPv2": load_ctrpv2,
    "CTRPv2_clean": load_ctrpv2_clean,
    "CTRPv2_cleaner": load_ctrpv2_cleaner,
    "CTRPv2_cleanest": load_ctrpv2_cleanest,
    "BeatAML2": load_beataml2,
    "PDX_Bruna": load_pdx_bruna,
}


def load_dataset(
    dataset_name: str,
    path_data: str = "data",
    measure: str = "response",
    curve_curator: bool = False,
    cores: int = 1,
    tissue_column: str | None = None,
    normalize: bool = False,
    clean_min_responders: int | None = None,
    clean_min_responder_frac: float | None = None,
) -> DrugResponseDataset:
    """
    Load a dataset based on the dataset name.

    :param dataset_name: The name of the dataset to load. Can be one of ('GDSC1', 'GDSC2', 'CCLE', 'TOYv1', or 'TOYv2')
        to download provided datasets, or any other name to allow for custom datasets.
    :param path_data: The parent path in which custom or downloaded datasets should be located, or in which raw
        viability data is to be found for fitting with CurveCurator (see param curve_curator for details).
        The location of the datasets are resolved by <path_data>/<dataset_name>/<dataset_name>.csv.
    :param measure: The name of the column containing the measure to predict, default = "response".
        If curve_curator is True, this measure is appended with "_curvecurator", e.g. "response_curvecurator" to
        distinguish between measures provided by the original source of a dataset, or the measures fit by
        CurveCurator.
    :param curve_curator: If True, the measure is appended with "_curvecurator".
        If a custom dataset_name was provided, this will invoke the fitting procedure of raw viability data,
        which is expected to exist at <path_data>/<dataset_name>/<dataset_name>_raw.csv. The fitted dataset will
        be stored in the same folder, in a file called <dataset_name>.csv
    :param cores: Number of cores to use for CurveCurator fitting. Only used when curve_curator is True, default = 1
    :param tissue_column: The name of the column containing the tissue type. If None, no tissue information is loaded.
        This is only used when loading a custom dataset. Default = None.
    :param normalize: Whether to normalize the response values to [0, 1] for curvecurator. Default = False.
        Only used for custom datasets when curve_curator is True.
    :param clean_min_responders: If set, derive a drug-cleaned variant of ``dataset_name`` on the fly, keeping
        only drugs with at least this many reproducible (curve-curated) responder curves. Works for any
        curve-curated base (built-in or custom); the variant is materialised as ``<dataset_name>_clean_min<N>``
        with the base's feature files shared (see :class:`DrugCurveFilter`). Requires curve-curated data.
    :param clean_min_responder_frac: Fraction-based alternative to ``clean_min_responders``: keep only drugs
        whose share of significant responder curves is at least this fraction (in ``(0, 1]``). Materialised as
        ``<dataset_name>_clean_frac<F>``. Set at most one of the two clean_* arguments.
    :return: A DrugResponseDataset containing response, cell line IDs, drug IDs, and dataset name.
    :raises FileNotFoundError: If the custom dataset or raw viability data could not be found at the given path.
    """
    if curve_curator:
        measure += "_curvecurator"
        input_file = Path(path_data).resolve() / dataset_name / f"{dataset_name}_raw.csv"
    else:
        input_file = Path(path_data).resolve() / dataset_name / f"{dataset_name}.csv"

    if clean_min_responders is not None or clean_min_responder_frac is not None:
        drug_filter = DrugCurveFilter(min_responders=clean_min_responders, min_responder_frac=clean_min_responder_frac)
        if clean_min_responders is not None:
            derived_name = f"{dataset_name}_clean_min{clean_min_responders}"
        else:
            derived_name = f"{dataset_name}_clean_frac{clean_min_responder_frac}"
        DERIVED_DATASETS.setdefault(derived_name, (dataset_name, drug_filter))
        return _load_filtered_dataset(path_data, measure, derived_name)

    if dataset_name in AVAILABLE_DATASETS:
        return AVAILABLE_DATASETS[dataset_name](path_data, measure=measure)

    if input_file.is_file():
        if curve_curator:
            fit_curves(
                input_file=str(input_file),
                output_dir=str(input_file.parent),
                dataset_name=dataset_name,
                cores=cores,
                normalize=normalize,
            )
        return load_custom(
            path_data=Path(path_data) / dataset_name / f"{dataset_name}.csv",
            dataset_name=dataset_name,
            measure=measure,
            tissue_column=tissue_column,
        )
    raise FileNotFoundError(f"Custom dataset does not exist at given path: {input_file}")
