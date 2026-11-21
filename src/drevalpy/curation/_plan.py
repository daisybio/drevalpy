"""A curation split into steps that can run in one process or as separate jobs.

:func:`plan` numbers the series of a well table and splits the pooled fit (and, with replicate
fits, the per-series fit) into chunks; :meth:`Chunk.fit` fits one chunk; :func:`assemble`
turns the fitted chunks into the curve AnnData. :func:`drevalpy.curation.curate` runs the three
steps in one process. A workflow manager runs them as separate jobs through ``drevalpy curate
plan``, ``fit-chunk`` and ``collect``, with the files :func:`write_plan`, :func:`write_chunk`
and :func:`write_fitted` produce in between.

A chunk file carries everything its fit needs (its curves, its group's description and the fit
options) and nothing else, so equal chunks are equal files: a workflow manager that caches tasks
by input content refits only the chunks whose curves or options changed.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import pandas as pd

from drevalpy.curation._fit import STATUS_COLUMN, FitOptions, GroupInfo, Status, chunk_config, collect, fit_chunk, split
from drevalpy.log import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import anndata
    from upath import UPath

logger = get_logger(__name__)

#: Parquet schema metadata key holding a chunk's description.
_CHUNK_KEY = b"drevalpy.curation.chunk"
_PLAN_FILE = "plan.json"
_SERIES_FILE = "series.parquet"
_CHUNK_DIR = "chunks"
#: Version of the files this module writes; :func:`read_plan` and :func:`read_chunk` reject others.
FORMAT_VERSION = 1


class Fit(StrEnum):
    """Which fit a chunk belongs to."""

    POOLED = "pooled"
    REPLICATE = "replicate"


@dataclass(frozen=True)
class Chunk:
    """One unit of fitting work: up to ``max_chunk_size`` curves of one dose-range group.

    :param name: Unique name within its plan, such as ``pooled-00003``.
    :param fit: The fit the chunk belongs to.
    :param group: Index of its dose-range group within that fit.
    :param info: Its group's description.
    :param options: How its curves are fitted.
    :param frame: Its curves, in CurveCurator's wide format.
    """

    name: str
    fit: Fit
    group: int
    info: GroupInfo
    options: FitOptions
    frame: pd.DataFrame

    def run(self) -> pd.DataFrame:
        """Fit the chunk's curves.

        :returns: The fitted curves, empty if CurveCurator dropped every curve.
        """
        return fit_chunk((self.frame, chunk_config(self.info, self.options)))


@dataclass(frozen=True)
class Plan:
    """The work of one curation, before any curve is fitted.

    :param options: How every curve is fitted.
    :param replicate_fits: Whether every series is also fitted on its own.
    :param series: The series table of :func:`drevalpy.curation._series.number_series`.
    :param infos: The dose-range groups of each fit.
    :param chunks: Every chunk of both fits.
    """

    options: FitOptions
    replicate_fits: bool
    series: pd.DataFrame
    infos: dict[Fit, list[GroupInfo]]
    chunks: list[Chunk]


def plan(
    df: pd.DataFrame, *, replicate_fits: bool, min_doses: int, metadata: Sequence[str], options: FitOptions
) -> Plan:
    """Split the curation of a validated well table into chunks, after numbering its series.

    :param df: Well-level table, already validated by :func:`drevalpy.curation.curate`.
    :param replicate_fits: Whether to also fit every series on its own.
    :param min_doses: Minimum number of distinct doses for a series to be fitted.
    :param metadata: Columns of *df* to keep per series.
    :param options: How every curve is fitted.
    :returns: The plan.
    """
    from drevalpy.curation._replicate_fits import tag_replicates
    from drevalpy.curation._series import number_series

    numbered = number_series(df, metadata=metadata, min_doses=min_doses)
    tables = {Fit.POOLED: numbered.viability}
    if replicate_fits:
        tables[Fit.REPLICATE] = tag_replicates(numbered.viability)
    infos: dict[Fit, list[GroupInfo]] = {}
    chunks: list[Chunk] = []
    for fit, viability in tables.items():
        frames, infos[fit] = split(viability, options)
        chunks.extend(
            Chunk(f"{fit.value}-{i:05d}", fit, group, infos[fit][group], options, frame)
            for i, (frame, group) in enumerate(frames)
        )
    logger.info("Planned %d chunks for %d series", len(chunks), len(numbered.series))
    return Plan(options, replicate_fits, numbered.series, infos, chunks)


def run_chunks(chunks: Sequence[Chunk], max_workers: int) -> dict[str, pd.DataFrame]:
    """Fit every chunk, in *max_workers* local processes.

    :param chunks: The chunks.
    :param max_workers: Number of processes; 1 fits in this process.
    :returns: Each chunk's fit, by chunk name.
    """
    if max_workers <= 1 or len(chunks) <= 1:
        return {chunk.name: chunk.run() for chunk in chunks}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        return dict(zip((chunk.name for chunk in chunks), pool.map(Chunk.run, chunks), strict=True))


def assemble(plan: Plan, fitted: Mapping[str, pd.DataFrame]) -> anndata.AnnData:
    """Build the curve AnnData from the fitted chunks of *plan*.

    :param plan: The plan.
    :param fitted: Every chunk's fit, by chunk name.
    :returns: See :func:`drevalpy.curation.curate`.
    :raises KeyError: If a chunk's fit is missing.
    """
    import anndata

    from drevalpy.curation._anndata import build_anndata

    missing = [chunk.name for chunk in plan.chunks if chunk.name not in fitted]
    if missing:
        raise KeyError(f"{len(missing)} chunk(s) have no fit: {missing[:5]}")
    results = {fit: _collect(plan, fit, fitted) for fit in plan.infos}
    pooled = results[Fit.POOLED]
    fitted_pooled = pooled[pooled[STATUS_COLUMN] == Status.FITTED.value].drop(columns=STATUS_COLUMN)
    adata = build_anndata(fitted_pooled) if not fitted_pooled.empty else anndata.AnnData()
    if plan.replicate_fits:
        from drevalpy.curation._replicate_fits import replicate_table, untag_replicates
        from drevalpy.data._replicate_fits import KEY

        adata.uns[KEY] = replicate_table(untag_replicates(results[Fit.REPLICATE]), pooled, plan.series)
    return adata


def _collect(plan: Plan, fit: Fit, fitted: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Collect the chunks of one fit."""
    chunks = [chunk for chunk in plan.chunks if chunk.fit == fit]
    return collect(
        [(chunk.frame, chunk.group) for chunk in chunks],
        [fitted[chunk.name] for chunk in chunks],
        plan.infos[fit],
        plan.options,
    )


def write_plan(plan: Plan, directory: UPath) -> None:
    """Write *plan* to *directory*: ``plan.json``, ``series.parquet`` and ``chunks/<name>.parquet``.

    :param plan: The plan.
    :param directory: The directory, created if missing.
    """
    (directory / _CHUNK_DIR).mkdir(parents=True, exist_ok=True)
    description = {
        "format": FORMAT_VERSION,
        "options": asdict(plan.options),
        "replicate_fits": plan.replicate_fits,
        "infos": {fit.value: infos for fit, infos in plan.infos.items()},
        "chunks": [chunk.name for chunk in plan.chunks],
    }
    (directory / _PLAN_FILE).write_text(json.dumps(description, indent=2, sort_keys=True))
    plan.series.to_parquet(directory / _SERIES_FILE, index=False)
    for chunk in plan.chunks:
        write_chunk(chunk, directory / _CHUNK_DIR / f"{chunk.name}.parquet")


def read_plan(directory: UPath) -> Plan:
    """Read a plan written by :func:`write_plan`.

    :param directory: The plan directory.
    :returns: The plan.
    """
    description = json.loads((directory / _PLAN_FILE).read_text())
    _check_format(description, directory / _PLAN_FILE)
    chunks = [read_chunk(directory / _CHUNK_DIR / f"{name}.parquet") for name in description["chunks"]]
    return Plan(
        options=FitOptions(**description["options"]),
        replicate_fits=bool(description["replicate_fits"]),
        series=pd.read_parquet(directory / _SERIES_FILE),
        infos={Fit(fit): infos for fit, infos in description["infos"].items()},
        chunks=chunks,
    )


def write_chunk(chunk: Chunk, path: UPath) -> None:
    """Write one chunk as Parquet, its description in the schema metadata.

    :param chunk: The chunk.
    :param path: The file.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(chunk.frame, preserve_index=False)
    description = {
        "format": FORMAT_VERSION,
        "name": chunk.name,
        "fit": chunk.fit.value,
        "group": chunk.group,
        "info": chunk.info,
        "options": asdict(chunk.options),
    }
    metadata = {**(table.schema.metadata or {}), _CHUNK_KEY: json.dumps(description, sort_keys=True).encode()}
    with path.open("wb") as handle:
        pq.write_table(table.replace_schema_metadata(metadata), handle)


def read_chunk(path: UPath) -> Chunk:
    """Read a chunk written by :func:`write_chunk`.

    :param path: The file.
    :returns: The chunk.
    :raises ValueError: If the file is not a chunk of this format.
    """
    import pyarrow.parquet as pq

    with path.open("rb") as handle:
        table = pq.read_table(handle)
    raw = (table.schema.metadata or {}).get(_CHUNK_KEY)
    if raw is None:
        raise ValueError(f"{path} is not a curation chunk: its Parquet metadata has no {_CHUNK_KEY.decode()!r}")
    description = json.loads(raw)
    _check_format(description, path)
    return Chunk(
        name=description["name"],
        fit=Fit(description["fit"]),
        group=int(description["group"]),
        info=description["info"],
        options=FitOptions(**description["options"]),
        frame=table.to_pandas(),
    )


def write_fitted(fitted: pd.DataFrame, path: UPath) -> None:
    """Write one chunk's fit as Parquet.

    :param fitted: :meth:`Chunk.run`'s result.
    :param path: The file.
    """
    fitted = fitted.rename(columns=str)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        fitted.to_parquet(handle, index=False)


def read_fitted(path: UPath) -> pd.DataFrame:
    """Read a chunk's fit written by :func:`write_fitted`.

    :param path: The file.
    :returns: The fit.
    """
    with path.open("rb") as handle:
        return pd.read_parquet(handle)


def _check_format(description: dict, path: UPath) -> None:
    """Reject a file written in another format version."""
    if description.get("format") != FORMAT_VERSION:
        raise ValueError(
            f"{path} has format {description.get('format')!r}, but this drevalpy reads format {FORMAT_VERSION}; "
            "plan and fit it with the same drevalpy version."
        )
