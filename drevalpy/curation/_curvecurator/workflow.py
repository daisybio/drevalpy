"""Full in-memory CurveCurator curation workflow."""

from __future__ import annotations

import pandas as pd

from drevalpy.curation._curvecurator.combine import combine
from drevalpy.curation._curvecurator.curvecurator import curvecurator_many
from drevalpy.curation._curvecurator.split import split
from drevalpy.curation._curvecurator.types import CurationFitResult, CurationSplitResult


def curate(
    raw_df: pd.DataFrame,
    *,
    dataset_name: str,
    input_filename: str,
    cores: int = 1,
    normalize: bool = False,
    device: str = "auto",
    chunk_size: int = 1_000,
    gpu_min_curves: int = 1_000,
    gpu_chunk_size: int = 50_000,
    gpu_available: bool = False,
) -> pd.DataFrame:
    """Run split, CurveCurator, and combine in one in-memory workflow.

    :param raw_df: Raw viability table with dose, response, sample, and drug columns.
    :param dataset_name: Dataset name used in metadata and combine output.
    :param input_filename: Source filename recorded in work-item metadata.
    :param cores: Maximum CPU worker threads for CurveCurator execution.
    :param normalize: Whether CurveCurator should normalize responses.
    :param device: Requested PyTorch device string.
    :param chunk_size: Maximum curves per CPU chunk.
    :param gpu_min_curves: Minimum curves before ``auto`` may select an accelerator.
    :param gpu_chunk_size: Maximum curves per accelerator chunk.
    :param gpu_available: Whether GPU resources are available for accelerator chunking.
    :returns: Combined curated dataset table.
    """
    split_result = split(
        raw_df,
        dataset_name=dataset_name,
        input_filename=input_filename,
        cores=cores,
        normalize=normalize,
        device=device,
        chunk_size=chunk_size,
        gpu_min_curves=gpu_min_curves,
        gpu_chunk_size=gpu_chunk_size,
        gpu_available=gpu_available,
    )
    fit_results = curvecurator_many(
        split_result.work_items,
        cores=cores,
        device=device,
        gpu_min_curves=gpu_min_curves,
        gpu_chunk_size=gpu_chunk_size,
    )
    return combine(fit_results, dataset_name=dataset_name)


__all__ = ["CurationFitResult", "CurationSplitResult", "curate"]
