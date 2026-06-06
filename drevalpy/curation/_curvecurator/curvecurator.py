"""In-process CurveCurator execution for one in-memory work item."""

from __future__ import annotations

import importlib
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from drevalpy.curation._curvecurator.device import effective_device
from drevalpy.curation._curvecurator.types import CurationFitResult, CurationWorkItem


def finalize_config(config: dict, *, config_path: Path | None = None) -> dict:
    """Prepare a CurveCurator config dict for ``run_pipeline_api``.

    :param config: CurveCurator configuration dictionary.
    :param config_path: Optional serialized config path recorded in ``__file__``.
    :returns: Configuration dictionary ready for in-memory execution.
    """
    finalized = {**config, "Paths": dict(config["Paths"])}
    resolved_config_path = (config_path or Path("config.json")).expanduser().resolve()
    finalized["__file__"] = {"Path": str(resolved_config_path)}
    for key, value in list(finalized["Paths"].items()):
        path = Path(value)
        if key == "input_file" or not path.is_absolute():
            finalized["Paths"][key] = str((resolved_config_path.parent / path.name).resolve())
    return finalized


def _run_pipeline_api(
    config: dict,
    *,
    input_table: pd.DataFrame,
    mad: bool,
    device: str,
    gpu_chunk_size: int,
) -> pd.DataFrame:
    run_pipeline_api = importlib.import_module("curve_curator").run_pipeline_api
    return run_pipeline_api(
        config,
        input_table,
        mad=mad,
        device=device,
        gpu_chunk_size=gpu_chunk_size,
    )


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
    input_table: pd.DataFrame,
    dev: str,
    gpu_chunk_size: int,
    chunk_name: str,
    *,
    mad: bool,
) -> tuple[pd.DataFrame | None, bool]:
    try:
        return (
            _run_pipeline_api(
                config,
                input_table=input_table,
                mad=mad,
                device=dev,
                gpu_chunk_size=gpu_chunk_size,
            ),
            False,
        )
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
    input_table: pd.DataFrame,
    device: str,
    gpu_chunk_size: int,
    chunk_name: str,
    *,
    mad: bool,
) -> pd.DataFrame:
    devices_to_try = [device] if device == "cpu" else [device, "cpu"]
    for dev in devices_to_try:
        result, oom = _try_single_device(
            config,
            input_table,
            dev,
            gpu_chunk_size,
            chunk_name,
            mad=mad,
        )
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


def _fit_work_item(
    work_item: CurationWorkItem,
    *,
    device: str,
    gpu_min_curves: int,
    gpu_chunk_size: int,
    mad: bool,
    config_path: Path | None = None,
) -> pd.DataFrame:
    routed_device = work_item.config.get("Routing", {}).get("device", device)
    eff = effective_device(routed_device, work_item.n_curves, gpu_min_curves)
    if routed_device != "cpu" and eff == "cpu":
        warnings.warn(
            f"GPU-routed job {work_item.work_id} resolved to CPU on this node — running on CPU",
            RuntimeWarning,
            stacklevel=2,
        )
    if config_path is None:
        with tempfile.TemporaryDirectory(prefix="drevalpy_curvecurator_") as tmp_dir:
            tmp_config_path = Path(tmp_dir) / f"{work_item.work_id}_config.toml"
            config = finalize_config(work_item.config, config_path=tmp_config_path)
            return _fit_with_fallback(
                config,
                work_item.input_table,
                eff,
                gpu_chunk_size,
                work_item.work_id,
                mad=mad,
            )

    config = finalize_config(work_item.config, config_path=config_path)
    return _fit_with_fallback(
        config,
        work_item.input_table,
        eff,
        gpu_chunk_size,
        work_item.work_id,
        mad=mad,
    )


def _fit_work_item_on_disk(
    work_item: CurationWorkItem,
    chunk_dir: Path,
    *,
    device: str,
    gpu_min_curves: int,
    gpu_chunk_size: int,
    mad: bool,
    config_path: Path | None = None,
) -> pd.DataFrame:
    """Compatibility alias that ignores *chunk_dir* and fits in memory."""
    _ = chunk_dir
    return _fit_work_item(
        work_item,
        device=device,
        gpu_min_curves=gpu_min_curves,
        gpu_chunk_size=gpu_chunk_size,
        mad=mad,
        config_path=config_path,
    )


def curvecurator(
    work_item: CurationWorkItem,
    *,
    device: str = "auto",
    gpu_min_curves: int = 1_000,
    gpu_chunk_size: int = 50_000,
    mad: bool = True,
) -> CurationFitResult:
    """Run CurveCurator for exactly one in-memory work item.

    :param work_item: Prepared CurveCurator work item.
    :param device: Requested PyTorch device string.
    :param gpu_min_curves: Minimum curves before ``auto`` may select an accelerator.
    :param gpu_chunk_size: Maximum curves per accelerator chunk.
    :param mad: Whether CurveCurator should use median absolute deviation filtering.
    :returns: In-memory fitted curves for the work item.
    """
    curves = _fit_work_item(
        work_item,
        device=device,
        gpu_min_curves=gpu_min_curves,
        gpu_chunk_size=gpu_chunk_size,
        mad=mad,
    )
    return CurationFitResult(work_id=work_item.work_id, curves=curves, work_item=work_item)


def curvecurator_many(
    work_items: tuple[CurationWorkItem, ...] | list[CurationWorkItem],
    *,
    cores: int = 1,
    device: str = "auto",
    gpu_min_curves: int = 1_000,
    gpu_chunk_size: int = 50_000,
    mad: bool = True,
) -> list[CurationFitResult]:
    """Run CurveCurator for many work items with bounded concurrency.

    :param work_items: Prepared work items.
    :param cores: Maximum CPU worker threads.
    :param device: Requested PyTorch device string.
    :param gpu_min_curves: Minimum curves before ``auto`` may select an accelerator.
    :param gpu_chunk_size: Maximum curves per accelerator chunk.
    :param mad: Whether CurveCurator should use median absolute deviation filtering.
    :returns: In-memory fitted curves for each work item.
    :raises RuntimeError: If one or more CurveCurator jobs fail.
    """
    if not work_items:
        return []

    resolved = [(item, effective_device(device, item.n_curves, gpu_min_curves)) for item in work_items]
    n_gpu = sum(1 for _, dev in resolved if dev != "cpu")
    n_cpu = len(resolved) - n_gpu
    max_workers = max((1 if n_gpu else 0) + min(cores, max(n_cpu, 1)), 1)
    errors: list[str] = []
    results: list[CurationFitResult] = []

    def _run(item: CurationWorkItem) -> CurationFitResult:
        return curvecurator(
            item,
            device=device,
            gpu_min_curves=gpu_min_curves,
            gpu_chunk_size=gpu_chunk_size,
            mad=mad,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run, item): item.work_id for item, _ in resolved}
        for future in futures:
            work_id = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append(f"{work_id}: {exc}")

    if errors:
        raise RuntimeError(f"{len(errors)} CurveCurator fit(s) failed:\n" + "\n---\n".join(errors))

    return results
