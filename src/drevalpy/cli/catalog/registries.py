"""``drevalpy list`` commands over the registries and the zoo.

The five component registries expose the same surface (``list``, ``get``,
``metadata``, ``table``), so each command over them is a two-line wrapper around
:func:`_show`. Zoo presets and datasets are not component registries; their
commands build rows from :mod:`drevalpy.models.zoo` and the dataset registry and
render them through the same helpers.
"""

from __future__ import annotations

from typing import Annotated

import typer

from drevalpy.errors import UnknownNameError

#: Optional entry name. Given, the command prints that entry's metadata instead
#: of the whole registry table.
EntryName = Annotated[
    str | None,
    typer.Argument(
        metavar="[NAME]",
        help="Show the metadata of a single entry instead of the whole table.",
    ),
]


def _show(registry_name: str, title: str, name: str | None) -> None:
    """Render one registry, or the metadata of a single entry in it.

    :param registry_name: Attribute of :mod:`drevalpy.registry` holding the registry module, e.g. ``"predictor"``.
    :param title: Plural heading for the table, e.g. ``"Predictors"``.
    :param name: Entry to describe, or ``None`` to render the whole table. An unregistered name fails with the
        registry's own unknown-name exception, whose message lists the registered names.
    """
    from drevalpy import registry

    from ._render import render_frame, render_mapping

    module = getattr(registry, registry_name)
    if name is None:
        render_frame(
            module.table(),
            title=title,
            empty_hint=f"No {title.lower()} are registered.",
        )
        return
    render_mapping(module.metadata(name), title=name)


def list_predictors(name: EntryName = None) -> None:
    """List registered predictors."""
    _show("predictor", "Predictors", name)


def list_sample_featurizers(name: EntryName = None) -> None:
    """List registered sample featurizers."""
    _show("sample_featurizer", "Sample featurizers", name)


def list_drug_featurizers(name: EntryName = None) -> None:
    """List registered drug featurizers."""
    _show("drug_featurizer", "Drug featurizers", name)


def list_splitters(name: EntryName = None) -> None:
    """List registered split modes."""
    _show("splitter", "Splitters", name)


def list_visualizations(name: EntryName = None) -> None:
    """List registered visualizations."""
    _show("visualization", "Visualizations", name)


def _unknown_entry(kind: str, name: str, valid: list[str]) -> UnknownNameError:
    """Build the error for an unregistered name, mirroring the registries' own message.

    :param kind: Singular noun for the entry, e.g. ``"zoo preset"``.
    :param name: The name the user asked for.
    :param valid: Every name that is registered.
    :returns: The exception for the caller to raise.
    """
    return UnknownNameError(f"Unknown {kind} {name!r}. Available: {', '.join(valid)}")


def list_zoo_presets(name: EntryName = None) -> None:
    """List zoo presets, the named model configurations accepted by ``run``."""
    from drevalpy.models.zoo import get_zoo_config, list_zoo_names

    from ._render import format_value, render_mapping, render_rows

    names = list_zoo_names()
    if name is None:
        builtin = set(list_zoo_names(include_external=False))
        rows = []
        for preset in names:
            config = get_zoo_config(preset)
            rows.append(
                [
                    preset,
                    format_value(config.model_id),
                    config.scope.value,
                    config.prediction_mode.value,
                    "built-in" if preset in builtin else "external",
                ]
            )
        render_rows(
            rows,
            columns=["Name", "Recipe", "Scope", "Prediction mode", "Origin"],
            title="Zoo presets",
            empty_hint="No zoo presets are registered.",
        )
        return
    if name not in names:
        raise _unknown_entry("zoo preset", name, names)
    config = get_zoo_config(name)
    render_mapping(
        {
            "Recipe": config.model_id,
            "Scope": config.scope.value,
            "Prediction mode": config.prediction_mode.value,
            "Sample featurizer": config.sample_featurizer.name if config.sample_featurizer else None,
            "Drug featurizer": config.drug_featurizer.name if config.drug_featurizer else None,
            "Predictor": config.predictor.name,
        },
        title=name,
    )


def list_datasets(name: EntryName = None) -> None:
    """List registered datasets, built-in and user-registered."""
    from drevalpy.registry.dataset import dataset_registry

    from ._render import render_frame, render_mapping

    frame = dataset_registry.to_dataframe()
    if name is None:
        render_frame(frame, title="Datasets", empty_hint="No datasets are registered.")
        return
    rows = frame[frame["Name"] == name]
    if rows.empty:
        raise _unknown_entry("dataset", name, [str(value) for value in frame["Name"]])
    row = rows.iloc[0].to_dict()
    source = dataset_registry.sources.get(str(row["Source"]))
    if source is not None:
        row["URL"] = source.url
    render_mapping(row, title=name)
