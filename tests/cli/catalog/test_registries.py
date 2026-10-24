"""Tests for :mod:`drevalpy.cli.catalog.registries`, run against the real registries (names only, no row counts)."""

from __future__ import annotations

import pytest

from drevalpy.cli.main import app
from tests.cli._helpers import HELP_ENV, make_runner, plain

runner = make_runner()

#: ``(command, registry attribute, a name that registry always contains)``.
REGISTRY_COMMANDS = [
    pytest.param("predictors", "predictor", "naiveMean", id="predictors"),
    pytest.param("sample-featurizers", "sample_featurizer", "identity", id="sample-featurizers"),
    pytest.param("drug-featurizers", "drug_featurizer", "fingerprints", id="drug-featurizers"),
    pytest.param("splitters", "splitter", "LPO", id="splitters"),
    pytest.param("visualizations", "visualization", "heatmap", id="visualizations"),
]


def invoke(*argv: str) -> str:
    return plain(runner.invoke(app, ["list", *argv], env=HELP_ENV).output)


@pytest.mark.parametrize(("command", "attribute", "entry"), REGISTRY_COMMANDS)
def test_table_lists_entries_headers_and_count(command: str, attribute: str, entry: str) -> None:
    from drevalpy import registry

    result = runner.invoke(app, ["list", command], env=HELP_ENV)
    printed = plain(result.output)
    registry_ = getattr(registry, attribute)

    assert result.exit_code == 0, result.output
    assert entry in printed
    assert all(str(column) in printed for column in registry_.table().columns)
    assert f"{len(registry_.list())} entries" in printed


@pytest.mark.parametrize(("command", "attribute", "entry"), REGISTRY_COMMANDS)
def test_single_entry_prints_its_metadata(command: str, attribute: str, entry: str) -> None:
    from drevalpy import registry

    result = runner.invoke(app, ["list", command, entry], env=HELP_ENV)
    metadata = getattr(registry, attribute).metadata(entry)

    assert result.exit_code == 0, result.output
    assert all(field in plain(result.output) for field in metadata)


def test_single_predictor_prints_its_description_but_not_the_whole_table() -> None:
    from drevalpy import registry

    printed = invoke("predictors", "naiveMean")

    assert registry.predictor.metadata("naiveMean")["description"] in printed
    assert "elasticNet" not in printed


def test_splitter_aliases_are_shown_and_resolved() -> None:
    printed = invoke("splitters")
    row = next(line for line in printed.splitlines() if "LSO" in line)
    result = runner.invoke(app, ["list", "splitters", "LCO"], env=HELP_ENV)

    assert "Aliases" in printed
    assert "LCO" in row
    assert result.exit_code == 0, result.output
    assert "LSO" in plain(result.output)


@pytest.mark.parametrize(("command", "attribute", "entry"), REGISTRY_COMMANDS)
def test_unknown_entry_is_an_error_line_naming_the_valid_entries(command: str, attribute: str, entry: str) -> None:
    result = runner.invoke(app, ["list", command, "definitely-not-registered"], env=HELP_ENV)
    printed = plain(result.output)

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert printed.startswith("Error: ")
    assert "definitely-not-registered" in printed
    assert entry in printed


def test_an_empty_registry_prints_a_hint_and_still_exits_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas as pd

    from drevalpy import registry

    monkeypatch.setattr(registry.predictor, "table", lambda: pd.DataFrame(columns=["Name", "Description", "Tags"]))
    result = runner.invoke(app, ["list", "predictors"], env=HELP_ENV)

    assert result.exit_code == 0
    assert "No predictors are registered." in plain(result.output)


def test_zoo_presets_table_lists_every_yaml_stem_with_recipe_and_scope() -> None:
    from pathlib import Path

    import drevalpy.models.zoo as zoo
    from drevalpy.models.zoo import list_zoo_names

    stems = [path.stem for path in Path(zoo.__file__).parent.glob("*.yaml")]
    printed = invoke("zoo-presets")
    row = next(line for line in printed.splitlines() if "SingleDrugElasticNet" in line)

    assert stems
    assert all(stem in printed for stem in stems)
    assert all(header in printed for header in ("Name", "Recipe", "Scope", "Prediction mode", "Origin"))
    assert f"{len(list_zoo_names())} entries" in printed
    assert "single_drug" in row
    assert "singleDrugElasticNet" in row


def test_external_zoo_presets_are_marked() -> None:
    from drevalpy.models.zoo import clear_external_zoo, get_zoo_config, register_external_zoo_entry

    register_external_zoo_entry("MyExternalPreset", get_zoo_config("ElasticNet"))
    try:
        row = next(line for line in invoke("zoo-presets").splitlines() if "MyExternalPreset" in line)
    finally:
        clear_external_zoo()

    assert "external" in row


def test_single_zoo_preset_prints_its_fields_and_unknown_one_fails() -> None:
    printed = invoke("zoo-presets", "ElasticNet")
    missing = runner.invoke(app, ["list", "zoo-presets", "definitely-not-registered"], env=HELP_ENV)

    assert all(field in printed for field in ("Recipe", "Scope", "Sample featurizer", "Predictor"))
    assert "scaledGeneExpression:fingerprints:elasticNet" in printed
    assert "RandomForest" not in printed
    assert missing.exit_code == 1
    assert plain(missing.output).startswith("Error: Unknown zoo preset 'definitely-not-registered'")
    assert "ElasticNet" in plain(missing.output)


def test_datasets_table_and_single_entry() -> None:
    from drevalpy.registry.dataset import dataset_registry

    frame = dataset_registry.to_dataframe()
    table = invoke("datasets")
    single = invoke("datasets", "TOYv1")

    assert "TOYv1" in table
    assert all(str(column) in table for column in frame.columns)
    assert f"{len(frame)} entries" in table
    assert all(field in single for field in ("Source", "File", "Origin", "URL"))
    assert dataset_registry.sources[dataset_registry.datasets["TOYv1"].source].url in single.replace("\n", "")
    assert "GDSC1" not in single


def test_unknown_dataset_fails_and_an_empty_registry_prints_a_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas as pd

    from drevalpy.registry.dataset import dataset_registry

    missing = runner.invoke(app, ["list", "datasets", "definitely-not-registered"], env=HELP_ENV)
    assert missing.exit_code == 1
    assert plain(missing.output).startswith("Error: Unknown dataset 'definitely-not-registered'")
    assert "TOYv1" in plain(missing.output)

    monkeypatch.setattr(dataset_registry, "to_dataframe", lambda: pd.DataFrame(columns=["Name"]))
    assert "No datasets are registered." in invoke("datasets")
