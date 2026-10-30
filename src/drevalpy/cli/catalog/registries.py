"""``drevalpy list`` commands over the registries and the zoo.

The five component registries are :class:`~drevalpy.registry._registry.Registry` instances,
so each command over them is a short wrapper around :func:`_show`. Zoo presets and datasets
have commands of their own that build rows from :mod:`drevalpy.models.zoo` and the dataset
registry and render them through the same helpers.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Annotated, Any

import typer

from drevalpy.errors import UnknownNameError

if TYPE_CHECKING:
    from drevalpy.registry._registry import Registry

#: Optional entry name. Given, the command prints that entry's metadata instead
#: of the whole registry table.
EntryName = Annotated[
    str | None,
    typer.Argument(
        metavar="[NAME]",
        help="Show the metadata of a single entry instead of the whole table.",
    ),
]


def _show(
    registry: Registry[Any],
    title: str,
    name: str | None,
    details: Callable[[str], Mapping[str, Any]] | None = None,
) -> None:
    """Render one registry, or the metadata of a single entry in it.

    :param registry: Registry to render.
    :param title: Plural heading for the table, e.g. ``"Predictors"``.
    :param name: Entry to describe, or ``None`` to render the whole table. An unregistered name fails with the
        registry's own unknown-name exception, whose message lists the registered names.
    :param details: Builds the facts shown for one entry; defaults to the entry's registry metadata.
    """
    from ._render import render_frame, render_mapping

    if name is None:
        render_frame(
            registry.to_dataframe(),
            title=title,
            empty_hint=f"No {title.lower()} are registered.",
        )
        return
    render_mapping(dict((details or registry.metadata)(name)), title=name)


def _component_details(display_name: str, registry: Registry[type[Any]]) -> Callable[[str], Mapping[str, Any]]:
    """Return the catalog facts of a predictor or featurizer, including its class and reference.

    :param display_name: Catalog name of the registry, e.g. ``"predictors"``.
    :param registry: Registry holding the component classes.
    :returns: Function from an entry name to its catalog facts.
    """
    from drevalpy.registry.components._metadata import featurizer_component_metadata, predictor_component_metadata

    build = predictor_component_metadata if display_name == "predictors" else featurizer_component_metadata
    return lambda name: build(display_name, name, registry.get(name))


def list_predictors(name: EntryName = None) -> None:
    """List registered predictors."""
    from drevalpy.registry.predictor import predictor_registry

    _show(predictor_registry, "Predictors", name, _component_details("predictors", predictor_registry))


def _featurizers_for_side(side: str) -> Registry[type[Any]]:
    """Return the registered featurizers that support *side*, as a registry of their own.

    :param side: ``"sample"`` or ``"drug"``.
    :returns: Registry holding only those featurizers, with their metadata.
    """
    from drevalpy.registry._registry import Registry
    from drevalpy.registry.featurizer import featurizers

    subset: Registry[type[Any]] = Registry(f"{side} featurizer")
    for entry, cls in featurizers.items():
        if side in cls.sides:
            subset.register(entry, cls, metadata=featurizers.metadata(entry))
    return subset


def list_sample_featurizers(name: EntryName = None) -> None:
    """List registered featurizers usable on the sample side."""
    registry = _featurizers_for_side("sample")
    _show(registry, "Sample featurizers", name, _component_details("sample_featurizers", registry))


def list_drug_featurizers(name: EntryName = None) -> None:
    """List registered featurizers usable on the drug side."""
    registry = _featurizers_for_side("drug")
    _show(registry, "Drug featurizers", name, _component_details("drug_featurizers", registry))


def list_splitters(name: EntryName = None) -> None:
    """List registered split modes; an alias shows the mode it resolves to."""
    from drevalpy.registry.splitter import canonical, splitter_registry

    def details(entry: str) -> dict[str, Any]:
        mode = canonical(entry)
        return {"Mode": mode, **splitter_registry.metadata(mode)}

    _show(splitter_registry, "Splitters", name, details)


def list_visualizations(name: EntryName = None) -> None:
    """List registered visualizations."""
    from drevalpy.registry.visualization import visualization_registry

    _show(visualization_registry, "Visualizations", name)


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
