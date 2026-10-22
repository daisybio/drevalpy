"""``drevalpy data split`` command."""

from __future__ import annotations

from typing import Annotated

import typer
from upath import UPath

from drevalpy.cli._quality import MinAbsFoldChange, MinRelevanceScore, NoQualityFilter, quality_from_cli


def split_dataset(
    dataset: Annotated[str, typer.Argument(help="Registered dataset name or path to a .h5mu file.")],
    output_dir: Annotated[str, typer.Argument(help="Output directory for split .npz files.")],
    mode: Annotated[str, typer.Option("--mode", "-m", help="Split mode: LPO, LSO, LDO, LTO or LStO.")] = "LPO",
    n_splits: Annotated[int, typer.Option("--n-splits", "-n", help="Number of CV folds.")] = 5,
    validation_ratio: Annotated[
        float, typer.Option("--validation-ratio", help="Fraction of training data for validation.")
    ] = 0.1,
    random_state: Annotated[int, typer.Option("--random-state", help="Random seed.")] = 42,
    no_quality_filter: NoQualityFilter = False,
    min_relevance_score: MinRelevanceScore = None,
    min_abs_fold_change: MinAbsFoldChange = None,
) -> None:
    """Split a dataset into cross-validation folds.

    Writes one .npz file per fold to the output directory. Pairs whose fitted curve fails the
    curve-quality rule are left out of every fold; the rule is recorded in the fold metadata.
    """
    from rich.progress import Progress

    from drevalpy.data import load, split

    quality = quality_from_cli(no_quality_filter, min_relevance_score, min_abs_fold_change)
    ds = load(dataset)
    out = UPath(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    folds = split(
        ds, mode=mode, n_splits=n_splits, validation_ratio=validation_ratio, random_state=random_state, quality=quality
    )

    with Progress() as progress:
        task = progress.add_task("Writing folds", total=len(folds))
        for i, fold in enumerate(folds):
            fold_path = out / f"fold_{i}.npz"
            fold.save(str(fold_path))
            progress.advance(task)

    typer.echo(f"Wrote {len(folds)} folds to {out}")
