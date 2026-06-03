"""Compatibility wrappers for path-based CurveCurator execution."""

from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from drevalpy.curation._curvecurator.curvecurator import (
    _fit_with_fallback,
    _fit_work_item_on_disk,
    finalize_config,
)
from drevalpy.curation._curvecurator.device import effective_device
from drevalpy.curation._curvecurator.types import CurationWorkItem


@dataclass(frozen=True)
class CurveCuratorWorkItem:
    """Legacy path-based CurveCurator fit job."""

    chunk_dir: Path
    config: dict
    n_curves: int

    def to_curation_work_item(self) -> CurationWorkItem:
        input_path = self.chunk_dir / "curvecurator_input.tsv"
        input_table = pd.read_csv(input_path, sep="\t") if input_path.is_file() else pd.DataFrame()
        meta = self.config.get("Meta", {})
        dataset_name = meta.get("description", meta.get("id", "dataset"))
        return CurationWorkItem(
            work_id=self.chunk_dir.name,
            dataset_name=dataset_name,
            group_key=self.chunk_dir.name,
            chunk_index=None,
            input_table=input_table,
            config=self.config,
            n_curves=self.n_curves,
            input_filename=meta.get("id", "input.csv"),
        )


def _run_one_work_item(
    item: CurveCuratorWorkItem,
    *,
    device: str,
    gpu_min_curves: int,
    gpu_chunk_size: int,
    mad: bool,
) -> Path:
    curves = _fit_work_item_on_disk(
        item.to_curation_work_item(),
        item.chunk_dir,
        device=device,
        gpu_min_curves=gpu_min_curves,
        gpu_chunk_size=gpu_chunk_size,
        mad=mad,
    )
    curves_path = item.chunk_dir / "curves.tsv"
    curves.to_csv(curves_path, sep="\t", index=False)
    return curves_path


def run_curvecurator_work_items(
    work_items: list[CurveCuratorWorkItem],
    *,
    cores: int,
    device: str = "auto",
    gpu_min_curves: int = 1_000,
    gpu_chunk_size: int = 50_000,
    mad: bool = True,
) -> None:
    """Execute legacy path-based work items."""
    if not work_items:
        return

    resolved = [(item, effective_device(device, item.n_curves, gpu_min_curves)) for item in work_items]
    n_gpu = sum(1 for _, dev in resolved if dev != "cpu")
    n_cpu = len(resolved) - n_gpu
    max_workers = max((1 if n_gpu else 0) + min(cores, max(n_cpu, 1)), 1)
    gpu_sem = threading.Semaphore(1)
    errors: list[str] = []

    def _run(item: CurveCuratorWorkItem, eff: str) -> None:
        if eff != "cpu":
            with gpu_sem:
                _run_one_work_item(
                    item,
                    device=eff,
                    gpu_min_curves=0,
                    gpu_chunk_size=gpu_chunk_size,
                    mad=mad,
                )
        else:
            _run_one_work_item(
                item,
                device="cpu",
                gpu_min_curves=gpu_min_curves,
                gpu_chunk_size=gpu_chunk_size,
                mad=mad,
            )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run, item, eff): item.chunk_dir for item, eff in resolved}
        for future in futures:
            chunk_dir = futures[future]
            try:
                future.result()
            except Exception as exc:
                errors.append(f"{chunk_dir.name}: {exc}")

    if errors:
        raise RuntimeError(f"{len(errors)} CurveCurator fit(s) failed:\n" + "\n---\n".join(errors))


def split_group_into_chunks(
    curve_df: pd.DataFrame,
    group_dir: Path,
    *,
    effective_chunk: int,
) -> list[tuple[pd.DataFrame, Path]]:
    """Split a group by unique (sample, drug) pairs when larger than *effective_chunk*."""
    n_curves = curve_df[["sample", "drug"]].drop_duplicates().shape[0]
    if n_curves <= effective_chunk:
        return [(curve_df, group_dir)]

    pairs = curve_df[["sample", "drug"]].drop_duplicates().sort_values(["sample", "drug"])
    chunks: list[tuple[pd.DataFrame, Path]] = []
    n_chunks = math.ceil(n_curves / effective_chunk)
    for i in range(n_chunks):
        chunk_start = i * effective_chunk
        chunk_stop = (i + 1) * effective_chunk
        chunk_pairs = pairs.iloc[chunk_start:chunk_stop]
        chunk_df = curve_df.merge(chunk_pairs, on=["sample", "drug"], how="inner")
        chunk_dir = group_dir / f"chunk_{i}"
        chunks.append((chunk_df, chunk_dir))
    return chunks


__all__ = [
    "CurveCuratorWorkItem",
    "finalize_config",
    "_fit_with_fallback",
    "_run_one_work_item",
    "run_curvecurator_work_items",
    "split_group_into_chunks",
]
