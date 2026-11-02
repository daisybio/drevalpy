"""``drevalpy describe`` command."""

from __future__ import annotations

import csv
import sys
from contextlib import redirect_stdout
from typing import TYPE_CHECKING, Annotated, TextIO

import typer

from ._options import (
    DrugFeaturizerOption,
    ModelsArgument,
    PredictorOption,
    SampleFeaturizerOption,
    resolve_model_specs,
)

if TYPE_CHECKING:
    from drevalpy.models import ModelSpec

#: TSV keys and table headers, in column order.
COLUMNS: dict[str, str] = {
    "model": "Model",
    "predictor": "Predictor",
    "sample_featurizer": "Sample featurizer",
    "drug_featurizer": "Drug featurizer",
    "accelerator": "Accelerator",
    "early_stopping": "Early stopping",
    "licenses": "Licenses",
}


def describe_row(spec: ModelSpec) -> dict[str, str]:
    """Describe one model as one row keyed by the :data:`COLUMNS` keys.

    :param spec: The model to describe.
    :returns: The row, with ``""`` for a featurizer slot the model leaves empty.
    """
    from drevalpy.models.config._parse import format_featurizer_recipe

    config = spec.config
    sample, drug = config.sample_featurizer, config.drug_featurizer
    if spec.requires_early_stopping:
        early_stopping = "required"
    elif spec.supports_early_stopping:
        early_stopping = "optional"
    else:
        early_stopping = "none"
    return {
        "model": spec.name,
        "predictor": config.predictor.name,
        "sample_featurizer": "" if sample is None else format_featurizer_recipe(sample),
        "drug_featurizer": "" if drug is None else format_featurizer_recipe(drug),
        "accelerator": spec.accelerator.value,
        "early_stopping": early_stopping,
        "licenses": ", ".join(spec.licenses),
    }


def _write_tsv(rows: list[dict[str, str]], stream: TextIO) -> None:
    """Write *rows* as TSV with a header row of the column keys.

    :param rows: Rows as returned by :func:`describe_row`.
    :param stream: Text stream to write to.
    """
    writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
    writer.writerow(COLUMNS)
    writer.writerows([row[key] for key in COLUMNS] for row in rows)


def _render_table(rows: list[dict[str, str]]) -> None:
    """Print *rows* as a rich table on stdout.

    :param rows: Rows as returned by :func:`describe_row`.
    """
    from drevalpy.cli.catalog._render import EMPTY_CELL, render_rows

    render_rows(
        [[row[key] or EMPTY_CELL for key in COLUMNS] for row in rows],
        columns=list(COLUMNS.values()),
        title="Models",
        empty_hint="No models to describe.",
    )


def describe_cmd(
    models: ModelsArgument = None,
    sample_featurizer: SampleFeaturizerOption = None,
    drug_featurizer: DrugFeaturizerOption = None,
    predictor: PredictorOption = None,
    tsv: Annotated[
        str | None, typer.Option("--tsv", help="Also write the rows as TSV to FILE ('-' for stdout).", metavar="FILE")
    ] = None,
) -> None:
    """Describe models: their components, whether they use an accelerator, early stopping and licenses.

    The ``--tsv`` output is meant for pipelines such as Nextflow, to route the models that use an accelerator to
    GPU nodes.
    """
    from upath import UPath

    specs = resolve_model_specs(models or [], sample_featurizer or [], drug_featurizer or [], predictor or [])
    rows = [describe_row(spec) for spec in specs]
    if tsv == "-":
        # Keep stdout clean TSV; rich's console writes to whatever sys.stdout is at print time.
        with redirect_stdout(sys.stderr):
            _render_table(rows)
        _write_tsv(rows, sys.stdout)
        return
    _render_table(rows)
    if tsv is not None:
        with UPath(tsv).open("w", encoding="utf-8", newline="") as stream:
            _write_tsv(rows, stream)
        typer.echo(f"Wrote {len(rows)} model(s) to {tsv}")
