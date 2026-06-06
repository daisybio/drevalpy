"""Serialize curation objects for CLI and Nextflow transport."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from drevalpy.curation._curvecurator.types import CurationFitResult, CurationSplitResult, CurationWorkItem

MANIFEST_FILENAME = "curation_manifest.json"
CONFIG_SUFFIX = "_config.json"
INPUT_SUFFIX = "_input.parquet"
CURVES_SUFFIX = "_curves.parquet"


def job_config_path(output_dir: Path, job_id: str) -> Path:
    """Return the config JSON path for one curation job."""
    return output_dir / f"{job_id}{CONFIG_SUFFIX}"


def job_input_path(output_dir: Path, job_id: str) -> Path:
    """Return the input parquet path for one curation job."""
    return output_dir / f"{job_id}{INPUT_SUFFIX}"


def job_curves_path(output_dir: Path, job_id: str) -> Path:
    """Return the fitted curves parquet path for one curation job."""
    return output_dir / f"{job_id}{CURVES_SUFFIX}"


def job_id_from_config_path(config_path: Path) -> str:
    """Extract a job id from a ``<job_id>_config.json`` path."""
    stem = config_path.name[: -len(CONFIG_SUFFIX)]
    if not stem:
        raise ValueError(f"Invalid curation config path: {config_path}")
    return stem


def _serialize_job_config(work_item: CurationWorkItem) -> dict[str, Any]:
    return work_item.config


def _work_item_from_config(
    config: dict[str, Any],
    input_table: pd.DataFrame,
    *,
    work_id: str,
) -> CurationWorkItem:
    meta = config["Meta"]
    n_curves = config["Routing"]["n_curves"]
    condition = meta.get("condition", work_id)
    return CurationWorkItem(
        work_id=work_id,
        dataset_name=meta.get("description", ""),
        group_key=condition,
        chunk_index=None,
        input_table=input_table,
        config=config,
        n_curves=n_curves,
        input_filename=meta.get("id", ""),
    )


def _deserialize_job_config(payload: dict[str, Any], input_table: pd.DataFrame, *, work_id: str) -> CurationWorkItem:
    return _work_item_from_config(payload, input_table, work_id=work_id)


def write_split_artifacts(
    split_result: CurationSplitResult,
    output_dir: str | Path,
) -> Path:
    """Write split work items and manifest to a flat directory.

    :param split_result: Prepared in-memory split result.
    :param output_dir: Root directory for serialized work items.
    :returns: Path to the written manifest file.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_entries: list[dict[str, Any]] = []

    for work_item in split_result.work_items:
        job_id = work_item.work_id
        config_path = job_config_path(root, job_id)
        input_path = job_input_path(root, job_id)
        curves_path = job_curves_path(root, job_id)

        config_path.write_text(
            json.dumps(_serialize_job_config(work_item), indent=2),
            encoding="utf-8",
        )
        work_item.input_table.to_parquet(input_path, index=False)

        manifest_entries.append(
            {
                "job_id": job_id,
                "config_file": config_path.name,
                "input_file": input_path.name,
                "expected_curves_file": curves_path.name,
                "dataset_name": work_item.dataset_name,
                "group_key": work_item.group_key,
                "chunk_index": work_item.chunk_index,
                "n_curves": work_item.n_curves,
            }
        )

    manifest_path = root / MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_name": split_result.dataset_name,
                "input_filename": split_result.input_filename,
                "work_items": manifest_entries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def read_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Load a curation manifest from disk."""
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def read_work_item(
    config_path: str | Path,
    *,
    input_path: str | Path | None = None,
    job_id: str | None = None,
) -> CurationWorkItem:
    """Deserialize one prepared work item from config JSON and input parquet."""
    config_file = Path(config_path).expanduser().resolve()
    config = json.loads(config_file.read_text(encoding="utf-8"))
    resolved_job_id = job_id or job_id_from_config_path(config_file)
    resolved_input = (
        Path(input_path).expanduser().resolve()
        if input_path is not None
        else job_input_path(config_file.parent, resolved_job_id)
    )
    input_table = pd.read_parquet(resolved_input)
    return _deserialize_job_config(config, input_table, work_id=resolved_job_id)


def write_fit_artifact(
    fit_result: CurationFitResult,
    output_dir: str | Path,
) -> Path:
    """Serialize one fitted result to a flat directory."""
    directory = Path(output_dir)
    curves_path = job_curves_path(directory, fit_result.work_id)
    return write_fit_curves(fit_result.curves, curves_path)


def write_fit_curves(curves: pd.DataFrame, output_path: str | Path) -> Path:
    """Serialize fitted CurveCurator curves to parquet."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    curves.to_parquet(destination, index=False)
    return destination


def read_fit_results_from_manifest(manifest_path: str | Path) -> list[CurationFitResult]:
    """Load fitted results listed in a curation manifest."""
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = read_manifest(manifest_file)
    output_dir = manifest_file.parent
    fit_results: list[CurationFitResult] = []

    for entry in manifest["work_items"]:
        job_id = entry["job_id"]
        curves_path = output_dir / entry["expected_curves_file"]
        if not curves_path.is_file():
            raise FileNotFoundError(f"Missing fitted curves file: {curves_path}")
        config_path = output_dir / entry["config_file"]
        fit_results.append(
            CurationFitResult(
                work_id=job_id,
                curves=pd.read_parquet(curves_path),
                work_item=read_work_item(config_path, job_id=job_id),
            )
        )

    return fit_results
