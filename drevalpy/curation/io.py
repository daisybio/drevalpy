"""Disk transport helpers for CurveCurator curation CLI workflows."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from drevalpy.curation.types import CurationFitResult, CurationSplitResult, CurationWorkItem

CONFIG_SUFFIX = ".json"
INPUT_SUFFIX = "_input.parquet"
CURVES_SUFFIX = ".parquet"


def job_config_path(output_dir: Path, job_id: str) -> Path:
    """Return the config path for one curation job."""
    return output_dir / f"{job_id}{CONFIG_SUFFIX}"


def job_input_path(output_dir: Path, job_id: str) -> Path:
    """Return the input-table path for one curation job."""
    return output_dir / f"{job_id}{INPUT_SUFFIX}"


def job_curves_path(output_dir: Path, job_id: str) -> Path:
    """Return the fitted-curves path for one curation job."""
    return output_dir / f"{job_id}{CURVES_SUFFIX}"


def job_id_from_curves_path(curves_path: Path) -> str:
    """Extract the job id from a fitted-curves filename."""
    name = curves_path.name
    if name.endswith(INPUT_SUFFIX):
        raise ValueError(
            f"Expected a fitted curves file ending with {CURVES_SUFFIX!r}, got prepared input {curves_path}."
        )
    if not name.endswith(CURVES_SUFFIX):
        raise ValueError(f"Expected a fitted curves file ending with {CURVES_SUFFIX!r}, got {curves_path}.")
    return name[: -len(CURVES_SUFFIX)]


def list_curve_files(output_dir: Path) -> list[Path]:
    """List fitted curve parquet files in a split output directory."""
    return sorted(path for path in output_dir.glob(f"*{CURVES_SUFFIX}") if not path.name.endswith(INPUT_SUFFIX))


def read_work_item(
    config_path: str | Path,
    *,
    input_path: str | Path | None = None,
    job_id: str | None = None,
) -> CurationWorkItem:
    """Load one CurveCurator work item from serialized config and input files."""
    config_file = Path(config_path).expanduser().resolve()
    config = json.loads(config_file.read_text(encoding="utf-8"))
    meta = config.get("Meta", {})
    routing = config.get("Routing", {})
    resolved_job_id = job_id or config_file.name[: -len(CONFIG_SUFFIX)]
    resolved_input = (
        Path(input_path).expanduser().resolve()
        if input_path is not None
        else config_file.parent / f"{resolved_job_id}{INPUT_SUFFIX}"
    )
    return CurationWorkItem(
        work_id=resolved_job_id,
        group_key=str(meta.get("condition", "")),
        chunk_index=None,
        input_table=pd.read_parquet(resolved_input),
        config=config,
        n_curves=int(routing.get("n_curves", 0)),
        input_filename=str(meta.get("id", "")),
    )


def write_split_artifacts(split_result: CurationSplitResult, output_dir: str | Path) -> Path:
    """Write split work items to a flat directory.

    :param split_result: Prepared in-memory split result.
    :param output_dir: Directory for serialized work items.
    :returns: Resolved output directory path.
    """
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    for work_item in split_result.work_items:
        job_id = work_item.work_id
        config_path = job_config_path(root, job_id)
        input_path = job_input_path(root, job_id)
        config_path.write_text(json.dumps(work_item.config, indent=2), encoding="utf-8")
        work_item.input_table.to_parquet(input_path, index=False)

    return root


def write_fit_curves(curves: pd.DataFrame, output_path: str | Path) -> Path:
    """Write fitted CurveCurator curves to disk."""
    resolved = Path(output_path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    curves.to_parquet(resolved, index=False)
    return resolved


def read_fit_results_from_paths(curve_paths: list[Path] | tuple[Path, ...]) -> list[CurationFitResult]:
    """Load fitted results from explicit curve parquet paths."""
    fit_results: list[CurationFitResult] = []
    for curves_path in sorted(Path(path).expanduser().resolve() for path in curve_paths):
        if not curves_path.is_file():
            raise FileNotFoundError(f"Missing fitted curves file: {curves_path}.")
        job_id = job_id_from_curves_path(curves_path)
        config_path = job_config_path(curves_path.parent, job_id)
        fit_results.append(
            CurationFitResult(
                work_id=job_id,
                curves=pd.read_parquet(curves_path),
                work_item=read_work_item(config_path, job_id=job_id),
            )
        )
    return fit_results


def resolve_curve_paths(paths: list[Path]) -> list[Path]:
    """Expand curve file arguments, accepting directories of fitted curve files."""
    resolved: list[Path] = []
    for path in paths:
        resolved_path = path.expanduser().resolve()
        if resolved_path.is_dir():
            curve_files = list_curve_files(resolved_path)
            if not curve_files:
                raise FileNotFoundError(f"No fitted curve files found in {resolved_path}.")
            resolved.extend(curve_files)
            continue
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Missing fitted curves file: {resolved_path}.")
        resolved.append(resolved_path)
    if not resolved:
        raise ValueError("At least one fitted curve file is required.")
    return resolved
