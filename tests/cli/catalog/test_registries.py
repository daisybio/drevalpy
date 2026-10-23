"""Tests for :mod:`drevalpy.cli.catalog.registries`, the per-registry, zoo and dataset commands.

The commands run against the real registries: they are populated on
``import drevalpy.registry`` and reading them has no side effects, so a stub
would only pin the stub. What each test asserts is therefore a name that is
registered by construction (a naive baseline, an ``LPO`` split mode), never a
row count.
"""

from __future__ import annotations

import pytest

from drevalpy.cli.catalog import list_app
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

COMMAND_NAMES = [param.values[0] for param in REGISTRY_COMMANDS]


def invoke(*argv: str) -> str:
    """Run ``drevalpy list`` and return its plain-text output.

    Args:
        *argv: Arguments after ``list``.

    Returns:
        Output with escape codes stripped.
    """
    return plain(runner.invoke(app, ["list", *argv], env=HELP_ENV).output)


class TestGroupWiring:
    """The five registry commands reach the root app.

    The group's own surface (names, help, usage errors) is pinned in
    ``test_init.py``; what matters here is that ``drevalpy list`` is reachable
    from the root app at all.
    """

    def test_root_help_advertises_the_group(self) -> None:
        result = runner.invoke(app, ["--help"], env=HELP_ENV)

        assert "list" in plain(result.output)

    @pytest.mark.parametrize("command", COMMAND_NAMES, ids=COMMAND_NAMES)
    def test_each_command_is_reachable(self, command: str) -> None:
        assert list_app.registered_commands, "the group registers its commands at import time"
        assert command in invoke("--help")


class TestTables:
    """Each command renders its registry's ``table()``."""

    @pytest.mark.parametrize(("command", "attribute", "entry"), REGISTRY_COMMANDS)
    def test_exits_cleanly(self, command: str, attribute: str, entry: str) -> None:
        result = runner.invoke(app, ["list", command], env=HELP_ENV)

        assert result.exit_code == 0, result.output

    @pytest.mark.parametrize(("command", "attribute", "entry"), REGISTRY_COMMANDS)
    def test_lists_a_known_entry(self, command: str, attribute: str, entry: str) -> None:
        assert entry in invoke(command)

    @pytest.mark.parametrize(("command", "attribute", "entry"), REGISTRY_COMMANDS)
    def test_prints_the_registry_column_headers(self, command: str, attribute: str, entry: str) -> None:
        from drevalpy import registry

        columns = [str(column) for column in getattr(registry, attribute).table().columns]
        printed = invoke(command)

        assert all(column in printed for column in columns)

    @pytest.mark.parametrize(("command", "attribute", "entry"), REGISTRY_COMMANDS)
    def test_reports_the_entry_count(self, command: str, attribute: str, entry: str) -> None:
        from drevalpy import registry

        expected = len(getattr(registry, attribute).list())

        assert f"{expected} entries" in invoke(command)


class TestSingleEntry:
    """A positional name switches to that entry's ``metadata()``."""

    @pytest.mark.parametrize(("command", "attribute", "entry"), REGISTRY_COMMANDS)
    def test_exits_cleanly(self, command: str, attribute: str, entry: str) -> None:
        result = runner.invoke(app, ["list", command, entry], env=HELP_ENV)

        assert result.exit_code == 0, result.output

    @pytest.mark.parametrize(("command", "attribute", "entry"), REGISTRY_COMMANDS)
    def test_prints_every_metadata_field(self, command: str, attribute: str, entry: str) -> None:
        from drevalpy import registry

        metadata = getattr(registry, attribute).metadata(entry)
        printed = invoke(command, entry)

        assert all(field in printed for field in metadata)

    def test_prints_the_description(self) -> None:
        from drevalpy import registry

        description = registry.predictor.metadata("naiveMean")["description"]

        assert description in invoke("predictors", "naiveMean")

    def test_does_not_print_the_whole_table(self) -> None:
        """Asking for one predictor must not dump the other twenty-odd."""
        printed = invoke("predictors", "naiveMean")

        assert "elasticNet" not in printed


class TestSplitterAliases:
    """The splitter listing shows an alias next to its mode and resolves it."""

    def test_the_table_has_an_aliases_column(self) -> None:
        assert "Aliases" in invoke("splitters")

    def test_the_table_shows_lco_next_to_lso(self) -> None:
        row = next(line for line in invoke("splitters").splitlines() if "LSO" in line)

        assert "LCO" in row

    def test_the_alias_describes_the_canonical_mode(self) -> None:
        result = runner.invoke(app, ["list", "splitters", "LCO"], env=HELP_ENV)

        assert result.exit_code == 0, result.output
        assert "LSO" in plain(result.output)


class TestUnknownEntry:
    """An unregistered name fails as an ``Error:`` line carrying the registry's own message."""

    @pytest.mark.parametrize(("command", "attribute", "entry"), REGISTRY_COMMANDS)
    def test_fails_with_an_error_line_instead_of_a_traceback(self, command: str, attribute: str, entry: str) -> None:
        result = runner.invoke(app, ["list", command, "definitely-not-registered"], env=HELP_ENV)

        assert result.exit_code == 1
        assert isinstance(result.exception, SystemExit)
        assert plain(result.output).startswith("Error: ")

    @pytest.mark.parametrize(("command", "attribute", "entry"), REGISTRY_COMMANDS)
    def test_names_the_missing_entry_and_lists_the_valid_ones(self, command: str, attribute: str, entry: str) -> None:
        printed = invoke(command, "definitely-not-registered")

        assert "definitely-not-registered" in printed
        assert entry in printed


class TestEmptyRegistry:
    """With nothing registered, the table is replaced by an explanation."""

    def test_prints_a_hint_instead_of_an_empty_table(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pandas as pd

        from drevalpy import registry

        monkeypatch.setattr(
            registry.predictor,
            "table",
            lambda: pd.DataFrame(columns=["Name", "Description", "Tags"]),
        )

        assert "No predictors are registered." in invoke("predictors")

    def test_still_exits_cleanly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An empty registry is a diagnosis, not a CLI failure."""
        import pandas as pd

        from drevalpy import registry

        monkeypatch.setattr(registry.predictor, "table", lambda: pd.DataFrame(columns=["Name"]))
        result = runner.invoke(app, ["list", "predictors"], env=HELP_ENV)

        assert result.exit_code == 0


class TestZooPresets:
    """``list zoo-presets`` shows every preset of :mod:`drevalpy.models.zoo`."""

    def test_exits_cleanly(self) -> None:
        assert runner.invoke(app, ["list", "zoo-presets"], env=HELP_ENV).exit_code == 0

    def test_shows_every_built_in_yaml_stem(self) -> None:
        from pathlib import Path

        import drevalpy.models.zoo as zoo

        stems = [path.stem for path in Path(zoo.__file__).parent.glob("*.yaml")]
        printed = invoke("zoo-presets")

        assert stems
        assert all(stem in printed for stem in stems)

    def test_prints_column_headers_and_count(self) -> None:
        from drevalpy.models.zoo import list_zoo_names

        printed = invoke("zoo-presets")

        assert all(header in printed for header in ("Name", "Recipe", "Scope", "Prediction mode", "Origin"))
        assert f"{len(list_zoo_names())} entries" in printed

    def test_shows_a_recipe_and_scope(self) -> None:
        row = next(line for line in invoke("zoo-presets").splitlines() if "SingleDrugElasticNet" in line)

        assert "single_drug" in row
        assert "singleDrugElasticNet" in row

    def test_marks_external_presets(self) -> None:
        from drevalpy.models.zoo import clear_external_zoo, get_zoo_config, register_external_zoo_entry

        register_external_zoo_entry("MyExternalPreset", get_zoo_config("ElasticNet"))
        try:
            row = next(line for line in invoke("zoo-presets").splitlines() if "MyExternalPreset" in line)
        finally:
            clear_external_zoo()

        assert "external" in row

    def test_single_entry_prints_its_fields(self) -> None:
        printed = invoke("zoo-presets", "ElasticNet")

        assert all(field in printed for field in ("Recipe", "Scope", "Sample featurizer", "Predictor"))
        assert "scaledGeneExpression:fingerprints:elasticNet" in printed

    def test_single_entry_does_not_print_the_whole_table(self) -> None:
        assert "RandomForest" not in invoke("zoo-presets", "ElasticNet")

    def test_unknown_entry_fails_and_lists_valid_names(self) -> None:
        result = runner.invoke(app, ["list", "zoo-presets", "definitely-not-registered"], env=HELP_ENV)

        assert result.exit_code == 1
        assert plain(result.output).startswith("Error: Unknown zoo preset 'definitely-not-registered'")
        assert "ElasticNet" in plain(result.output)


class TestDatasets:
    """``list datasets`` renders the dataset registry."""

    def test_exits_cleanly(self) -> None:
        assert runner.invoke(app, ["list", "datasets"], env=HELP_ENV).exit_code == 0

    def test_lists_the_toy_dataset(self) -> None:
        assert "TOYv1" in invoke("datasets")

    def test_prints_the_registry_column_headers_and_count(self) -> None:
        from drevalpy.registry.dataset import dataset_registry

        frame = dataset_registry.to_dataframe()
        printed = invoke("datasets")

        assert all(str(column) in printed for column in frame.columns)
        assert f"{len(frame)} entries" in printed

    def test_single_entry_prints_its_fields_and_source_url(self) -> None:
        from drevalpy.registry.dataset import dataset_registry

        printed = invoke("datasets", "TOYv1")

        assert all(field in printed for field in ("Source", "File", "Origin", "URL"))
        assert dataset_registry.sources["orakl"].url in printed.replace("\n", "")

    def test_single_entry_does_not_print_the_whole_table(self) -> None:
        assert "GDSC1" not in invoke("datasets", "TOYv1")

    def test_unknown_entry_fails_and_lists_valid_names(self) -> None:
        result = runner.invoke(app, ["list", "datasets", "definitely-not-registered"], env=HELP_ENV)

        assert result.exit_code == 1
        assert plain(result.output).startswith("Error: Unknown dataset 'definitely-not-registered'")
        assert "TOYv1" in plain(result.output)

    def test_empty_registry_prints_a_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pandas as pd

        from drevalpy.registry.dataset import dataset_registry

        monkeypatch.setattr(dataset_registry, "to_dataframe", lambda: pd.DataFrame(columns=["Name"]))

        assert "No datasets are registered." in invoke("datasets")
