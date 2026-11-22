"""``drevalpy data`` command group: ``concat``, ``load``, ``randomization``, ``split`` and ``synthetic``."""

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
from drevalpy.types.enums import RandomizationMode, RandomizationType

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
    robustness_repeats: Annotated[
        int, typer.Option("--robustness-repeats", min=0, help="Robustness repeats per fold (0 = none).")
    ] = 0,
    measure: Measure = None,
    no_quality_filter: NoQualityFilter = False,
    min_relevance_score: MinRelevanceScore = None,
    min_abs_fold_change: MinAbsFoldChange = None,
) -> None:
    """Split a dataset into cross-validation folds.

    Writes one .npz file per fold to the output directory, plus ``fold_<i>_repeat_<r>.npz``
    files when ``--robustness-repeats`` is positive. Pairs whose fitted curve fails the
    curve-quality rule are left out of every fold, and so are pairs without the response measure;
    the rule and the measure are recorded in the fold metadata.
    """
    from rich.progress import Progress

    from drevalpy.data import load, split
    from drevalpy.execution._robustness import robustness

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

    n_robustness = 0
    with Progress() as progress:
        task = progress.add_task("Writing folds", total=len(folds))
        for i, fold in enumerate(folds):
            fold.save(str(out / f"fold_{i}.npz"))
            if robustness_repeats > 0:
                for repeat, variant in enumerate(robustness(fold, robustness_repeats)):
                    variant.save(str(out / f"fold_{i}_repeat_{repeat}.npz"))
                    n_robustness += 1
            progress.advance(task)

    if n_robustness:
        typer.echo(f"Wrote {len(folds)} folds and {n_robustness} robustness folds to {out}")
    else:
        typer.echo(f"Wrote {len(folds)} folds to {out}")


def randomization_dataset(
    model: Annotated[str, typer.Argument(help="Model name (e.g. ElasticNet, SimpleNeuralNetwork).")],
    dataset: Annotated[str, typer.Argument(help="Registered dataset name or path to a .h5mu file.")],
    output_dir: Annotated[str, typer.Argument(help="Output directory for randomized .h5mu files.")],
    modes: Annotated[
        list[RandomizationMode] | None,
        typer.Option("--mode", "-m", case_sensitive=False, help="Randomization mode(s): SVRC, SVCC, SVRD, SVCD."),
    ] = None,
    randomization_type: Annotated[
        RandomizationType,
        typer.Option("--randomization-type", "-t", case_sensitive=False, help="How to replace the view."),
    ] = RandomizationType.PERMUTATION,
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = 42,
) -> None:
    """Generate randomized datasets for feature importance testing.

    Produces copies of the dataset with views shuffled according to the
    specified randomization modes, based on the model's configured views.
    """
    from rich.progress import Progress

    from drevalpy.data import load
    from drevalpy.execution._randomization import randomization
    from drevalpy.models import ModelSpec

    effective_modes = [mode.value for mode in modes] if modes else [RandomizationMode.SVRC.value]
    model_spec = ModelSpec.from_name(model)
    ds = load(dataset)
    randomized = randomization(
        model_spec, ds, effective_modes, randomization_type=randomization_type, random_state=seed
    )

    out = UPath(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with Progress() as progress:
        task = progress.add_task("Writing randomized datasets", total=len(randomized))
        for rds in randomized:
            mode_tag, view_tag = rds.randomization or ("unknown", "0")
            out_path = out / f"{mode_tag}__{view_tag}.h5mu"
            rds.save(str(out_path))
            progress.advance(task)

    typer.echo(f"Wrote {len(randomized)} randomized datasets to {out}")


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
data_app.command("randomization")(randomization_dataset)
data_app.command("split")(split_dataset)
data_app.command("synthetic")(synthetic_dataset)
