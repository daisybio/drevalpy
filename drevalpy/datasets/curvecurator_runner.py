"""In-process CurveCurator execution via the fork's Python API."""

from __future__ import annotations

import gc
import importlib
import math
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from drevalpy.datasets.curvecurator_device import effective_device


@dataclass(frozen=True)
class CurveCuratorWorkItem:
    """One CurveCurator fit job: config dict, output directory, and curve count."""

    chunk_dir: Path
    config: dict
    n_curves: int


def finalize_config(config: dict, chunk_dir: Path) -> dict:
    """Absolutize Paths and inject ``__file__`` for ``run_pipeline_api``.

    :param config: CurveCurator configuration dictionary.
    :param chunk_dir: Directory used as the base for relative input and output paths.
    :returns: Configuration dictionary with absolute paths.
    """
    chunk_dir = chunk_dir.resolve()
    finalized = {**config, "Paths": dict(config["Paths"])}
    config_path = chunk_dir / "config.toml"
    finalized["__file__"] = {"Path": str(config_path)}
    for key, value in finalized["Paths"].items():
        path = Path(value)
        if not path.is_absolute():
            path = chunk_dir / value
        finalized["Paths"][key] = str(path.resolve())
    return finalized


def _run_pipeline_api(config: dict, *, mad: bool, device: str, gpu_chunk_size: int) -> pd.DataFrame:
    run_pipeline_api = importlib.import_module("curve_curator").run_pipeline_api
    return run_pipeline_api(config, mad=mad, device=device, gpu_chunk_size=gpu_chunk_size)


def _cuda_empty_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception as exc:
        warnings.warn(f"Could not empty accelerator cache: {exc}", RuntimeWarning, stacklevel=2)


def _try_single_device(
    config: dict,
    dev: str,
    gpu_chunk_size: int,
    chunk_name: str,
    *,
    mad: bool,
) -> tuple[pd.DataFrame | None, bool]:
    try:
        return _run_pipeline_api(config, mad=mad, device=dev, gpu_chunk_size=gpu_chunk_size), False
    except SystemExit:
        warnings.warn(
            f"Skipping {chunk_name}: CurveCurator called exit() (all curves filtered out)",
            RuntimeWarning,
            stacklevel=2,
        )
        return pd.DataFrame(), False
    except ValueError as exc:
        if "zero-size array" not in str(exc):
            raise
        warnings.warn(
            f"Skipping {chunk_name}: all curves filtered by CurveCurator",
            RuntimeWarning,
            stacklevel=2,
        )
        return pd.DataFrame(), False
    except Exception as exc:
        if "out of memory" in str(exc).lower() and dev != "cpu":
            return None, True
        raise
    finally:
        if dev != "cpu":
            _cuda_empty_cache()


def _fit_with_fallback(
    config: dict,
    device: str,
    gpu_chunk_size: int,
    chunk_name: str,
    *,
    mad: bool,
) -> pd.DataFrame:
    devices_to_try = [device] if device == "cpu" else [device, "cpu"]
    for dev in devices_to_try:
        result, oom = _try_single_device(config, dev, gpu_chunk_size, chunk_name, mad=mad)
        if not oom:
            if result is None:
                continue
            if dev != device:
                warnings.warn(
                    f"OOM on {device} for {chunk_name} — retried on CPU",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return result
    raise RuntimeError(f"OOM on all devices for {chunk_name}")


def _run_one_work_item(
    item: CurveCuratorWorkItem,
    *,
    device: str,
    gpu_min_curves: int,
    gpu_chunk_size: int,
    mad: bool,
) -> Path:
    eff = effective_device(device, item.n_curves, gpu_min_curves)
    config = finalize_config(item.config, item.chunk_dir)
    fitted = _fit_with_fallback(
        config,
        eff,
        gpu_chunk_size,
        item.chunk_dir.name,
        mad=mad,
    )
    curves_path = item.chunk_dir / "curves.tsv"
    fitted.to_csv(curves_path, sep="\t", index=False)
    del fitted
    gc.collect()
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
    """Execute all work items; write ``curves.tsv`` under each chunk directory.

    :param work_items: CurveCurator jobs to execute.
    :param cores: Maximum CPU worker threads.
    :param device: Requested PyTorch device string.
    :param gpu_min_curves: Minimum curves before ``auto`` may select an accelerator.
    :param gpu_chunk_size: Maximum curves per accelerator chunk.
    :param mad: Whether CurveCurator should use median absolute deviation filtering.
    :raises RuntimeError: If one or more CurveCurator jobs fail.
    """
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
        for future in as_completed(futures):
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
    """Split a group by unique (sample, drug) pairs when larger than *effective_chunk*.

    :param curve_df: Raw curve rows for one compatible dose-range group.
    :param group_dir: Directory for the group or its chunks.
    :param effective_chunk: Maximum curves per returned chunk.
    :returns: Chunked data frames paired with their output directories.
    """
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
