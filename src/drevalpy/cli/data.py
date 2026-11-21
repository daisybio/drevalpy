"""``drevalpy data`` command group: ``concat``, ``load``, ``split`` and ``synthetic``."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

import typer
from upath import UPath

from drevalpy.cli._options import (
    Measure,
    MinAbsFoldChange,
    MinRelevanceScore,
    NoQualityFilter,
    NSplits,
    ValidationRatio,
    quality_from_cli,
)
from drevalpy.data._synthetic import DATASET_NAME, MAX_SYNTHETIC_DRUGS, N_DRUGS, N_SAMPLES, N_TISSUES, SEED

data_app = typer.Typer(
    name="data",
    help="Data management commands.",
    no_args_is_help=True,
)


class Join(StrEnum):
    """Values of ``--join``: how views that disagree on feature names are combined."""

    EXACT = "exact"
    INNER = "inner"


def concat_datasets(
    datasets: Annotated[
        list[str],
        typer.Argument(help="Two or more registered dataset names or .h5mu paths, one per study."),
    ],
    output: Annotated[str, typer.Argument(help="Output .h5mu file path.")],
    join: Annotated[
        Join,
        typer.Option(
            "--join",
            case_sensitive=False,
            help="'exact' needs identical feature names in every shared view; 'inner' keeps the shared features.",
        ),
    ] = Join.EXACT,
    name: Annotated[
        str | None,
        typer.Option("--name", help="Name of the result. Defaults to the dataset names joined by '+'."),
    ] = None,
) -> None:
    """Concatenate datasets of different studies into one dataset.

    Each row becomes ``<study>::<sample_id>``, ``obs`` gains ``study`` and
    ``source_sample_id``, views are stacked by rows and drug columns are united.
    Run any split mode on the result, such as ``LStO`` to train on one study and
    test on another.
    """
    from drevalpy.data import concat, load

    combined = concat([load(item) for item in datasets], join=join.value, name=name)
    path = UPath(output)
    combined.save(str(path))
    typer.echo(f"Wrote {combined.name} to {path}")


def load_dataset(
    name: Annotated[str, typer.Argument(help="Registered dataset name or path to a .h5mu file.")],
    output: Annotated[str, typer.Argument(help="Output .h5mu file path.")],
) -> None:
    """Load a dataset and write it to an output file.

    Resolves the dataset by name (downloading if needed) and writes it as .h5mu.
    """
    from drevalpy.data import load

    path = UPath(output)
    dataset = load(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save(path)
    typer.echo(f"Wrote {dataset.name} to {path}")


def split_dataset(
    dataset: Annotated[str, typer.Argument(help="Registered dataset name or path to a .h5mu file.")],
    output_dir: Annotated[str, typer.Argument(help="Output directory for split .npz files.")],
    mode: Annotated[str, typer.Option("--mode", "-m", help="Split mode: LPO, LSO, LDO, LTO or LStO.")] = "LPO",
    n_splits: NSplits = 5,
    validation_ratio: ValidationRatio = 0.1,
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = 42,
    measure: Measure = None,
    no_quality_filter: NoQualityFilter = False,
    min_relevance_score: MinRelevanceScore = None,
    min_abs_fold_change: MinAbsFoldChange = None,
) -> None:
    """Split a dataset into cross-validation folds.

    Writes one .npz file per fold to the output directory. Pairs whose fitted curve fails the
    curve-quality rule are left out of every fold, and so are pairs without the response measure;
    the rule and the measure are recorded in the fold metadata.
    """
    from rich.progress import Progress

    from drevalpy.data import load, split

    quality = quality_from_cli(no_quality_filter, min_relevance_score, min_abs_fold_change)
    ds = load(dataset)
    folds = split(
        ds,
        mode=mode,
        n_splits=n_splits,
        validation_ratio=validation_ratio,
        random_state=seed,
        quality=quality,
        measure=measure,
    )
    out = UPath(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with Progress() as progress:
        task = progress.add_task("Writing folds", total=len(folds))
        for i, fold in enumerate(folds):
            fold_path = out / f"fold_{i}.npz"
            fold.save(str(fold_path))
            progress.advance(task)

    typer.echo(f"Wrote {len(folds)} folds to {out}")


def synthetic_dataset(
    output: Annotated[str, typer.Option("--output", "-o", help="Output .h5mu file path.")],
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = SEED,
    n_samples: Annotated[int, typer.Option("--n-samples", help="Number of samples.")] = N_SAMPLES,
    n_drugs: Annotated[
        int, typer.Option("--n-drugs", help=f"Number of drugs, at most {MAX_SYNTHETIC_DRUGS}.")
    ] = N_DRUGS,
    n_tissues: Annotated[int, typer.Option("--n-tissues", help="Number of distinct tissues.")] = N_TISSUES,
    name: Annotated[
        str, typer.Option("--name", help="Dataset name; give each file its own to concatenate them.")
    ] = DATASET_NAME,
) -> None:
    """Write a small simulated dataset to an .h5mu file.

    The responses mimic a real screen: drugs differ strongly in average potency,
    and hidden sample factors drive both the responses and the gene expression
    features. The file needs no download, so it suits tutorials and offline
    experiments.
    """
    from drevalpy.data import synthetic

    dataset = synthetic(name=name, seed=seed, n_samples=n_samples, n_drugs=n_drugs, n_tissues=n_tissues)

    path = UPath(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save(str(path))
    typer.echo(f"Wrote {dataset.name} to {path}")


data_app.command("concat")(concat_datasets)
data_app.command("load")(load_dataset)
data_app.command("split")(split_dataset)
data_app.command("synthetic")(synthetic_dataset)
