"""Tests for the ``drevalpy list`` command group: registry tables, plugins and the table renderer."""

from __future__ import annotations

import importlib
import importlib.metadata
from enum import Enum
from typing import Any

import pandas as pd
import pytest
from rich.text import Text

from drevalpy.cli.catalog import _render, list_app
from drevalpy.cli.catalog import plugins as plugins_module
from drevalpy.cli.catalog import registries as registries_module
from drevalpy.cli.main import app
from tests.cli._helpers import HELP_ENV, GroupSurface, make_runner, plain

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


TRACEBACK = 'Traceback (most recent call last):\n  File "x.py", line 1\n    boom\nRuntimeError: boom\n'


SKIPPED = "drevalpy.components.predictors.thing"


class StubEntryPoint:
    """Minimal stand-in for :class:`importlib.metadata.EntryPoint`."""

    def __init__(self, name: str, value: str) -> None:
        """Record the declared name and target."""
        self.name = name
        self.value = value


@pytest.fixture()
def declared(monkeypatch: pytest.MonkeyPatch) -> list[StubEntryPoint]:
    entries: list[StubEntryPoint] = []
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda **_: entries)
    return entries


@pytest.fixture()
def ledgers(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, str]]:
    from drevalpy import registry

    state: dict[str, dict[str, str]] = {"loaded": {}, "failed": {}, "skipped": {}}
    monkeypatch.setattr(registry, "get_loaded_plugins", lambda: dict(state["loaded"]))
    monkeypatch.setattr(registry, "get_failed_plugins", lambda: dict(state["failed"]))
    monkeypatch.setattr(registry, "get_skipped_builtin_modules", lambda: dict(state["skipped"]))
    return state


def invoke_plugins(*argv: str) -> Any:
    return runner.invoke(app, ["list", "plugins", *argv], env=HELP_ENV)


def text(*argv: str) -> str:
    return plain(invoke_plugins(*argv).output)


def test_help_documents_the_options() -> None:
    helped = text("--help")
    assert "--traceback" in helped
    assert "--strict" in helped


def test_no_plugins_says_so_and_prints_no_table(declared: list[StubEntryPoint], ledgers: dict[str, Any]) -> None:
    assert invoke_plugins().exit_code == 0
    printed = text()
    assert "No packages declare a drevalpy.plugins entry point" in printed
    assert "installed into this interpreter" in printed
    assert "Status" not in printed


def test_a_loaded_plugin_is_listed_with_its_target(declared: list[StubEntryPoint], ledgers: dict[str, Any]) -> None:
    declared.append(StubEntryPoint("my_plugin", "my_plugin.register:setup"))
    ledgers["loaded"]["my_plugin"] = "my_plugin.register:setup"

    printed = text()

    assert "my_plugin" in printed
    assert plugins_module.STATUS_LOADED in printed
    assert "my_plugin.register:setup" in printed
    assert "failed to load" not in printed
    assert invoke_plugins("--strict").exit_code == 0


def test_a_failed_plugin_is_named_with_its_reason(declared: list[StubEntryPoint], ledgers: dict[str, Any]) -> None:
    declared.append(StubEntryPoint("broken_plugin", "broken_plugin:setup"))
    ledgers["failed"]["broken_plugin"] = TRACEBACK

    printed = text()

    assert plugins_module.STATUS_FAILED in printed
    assert "1 plugin(s) failed to load" in printed
    assert "RuntimeError: boom" in printed
    assert "Traceback (most recent call last)" not in printed
    assert "--traceback" in printed
    assert invoke_plugins().exit_code == 0
    assert invoke_plugins("--strict").exit_code == 1
    assert "RuntimeError: boom" in text("--strict")


def test_traceback_option_prints_the_stack_and_drops_the_advert(
    declared: list[StubEntryPoint], ledgers: dict[str, Any]
) -> None:
    ledgers["failed"]["broken_plugin"] = TRACEBACK

    printed = text("--traceback")

    assert "Traceback (most recent call last)" in printed
    assert "Re-run with --traceback" not in printed


def test_a_declared_plugin_in_neither_ledger_is_not_loaded_and_not_a_failure(
    declared: list[StubEntryPoint], ledgers: dict[str, Any]
) -> None:
    declared.append(StubEntryPoint("unseen", "unseen:setup"))

    assert plugins_module.STATUS_NOT_LOADED in text()
    assert "failed to load" not in text()
    assert invoke_plugins("--strict").exit_code == 0


@pytest.mark.parametrize("ledger", ["loaded", "failed"])
def test_a_ledger_entry_missing_from_metadata_is_still_listed(
    declared: list[StubEntryPoint], ledgers: dict[str, Any], ledger: str
) -> None:
    ledgers[ledger]["ghost"] = TRACEBACK

    assert "ghost" in text()


def test_rows_and_failures_are_sorted_by_name(declared: list[StubEntryPoint], ledgers: dict[str, Any]) -> None:
    for name in ("zeta", "alpha", "mu"):
        declared.append(StubEntryPoint(name, f"{name}:setup"))
        ledgers["loaded"][name] = f"{name}:setup"
    printed = text()
    assert printed.index("alpha") < printed.index("mu") < printed.index("zeta")

    ledgers["loaded"].clear()
    for name in ("zeta", "alpha"):
        ledgers["failed"][name] = f"ValueError: {name} exploded"
    printed = text()
    assert printed.index("alpha exploded") < printed.index("zeta exploded")


def test_skipped_builtins_are_reported_but_are_not_plugin_failures(
    declared: list[StubEntryPoint], ledgers: dict[str, Any]
) -> None:
    assert "skipped" not in text()

    ledgers["skipped"][SKIPPED] = TRACEBACK
    printed = text()

    assert SKIPPED in printed
    assert "1 built-in module(s) were skipped" in printed
    assert "RuntimeError: boom" in printed
    assert "failed to load" not in printed
    assert invoke_plugins("--strict").exit_code == 0


@pytest.mark.parametrize(
    ("recorded", "expected"),
    [
        (TRACEBACK, "RuntimeError: boom"),
        ("ValueError: nope\n\n\n", "ValueError: nope"),
        ("ImportError: no module", "ImportError: no module"),
        ("", "unknown error"),
        ("   \n  \n", "unknown error"),
    ],
)
def test_failure_reason_keeps_the_last_line(recorded: str, expected: str) -> None:
    assert plugins_module.failure_reason(recorded) == expected


def test_a_registry_import_failure_exits_one_and_explains(monkeypatch: pytest.MonkeyPatch) -> None:
    real = importlib.import_module

    def fake(name: str, package: str | None = None) -> Any:
        if name == "drevalpy.registry":
            raise RuntimeError("plugin blew up during discovery")
        return real(name, package)

    monkeypatch.setattr(importlib, "import_module", fake)

    result = invoke_plugins()

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    printed = plain(result.output)
    assert "Importing drevalpy.registry failed while loading plugins." in printed
    assert "plugin blew up during discovery" in printed


@pytest.fixture(autouse=True)
def wide_console(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLUMNS", "300")
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("TERM", "dumb")


class Flavour(Enum):
    VANILLA = 1


def output(capsys: pytest.CaptureFixture[str]) -> str:
    return plain(capsys.readouterr().out)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "-"),
        ("", "-"),
        ("   ", "-"),
        ("  spaced  ", "spaced"),
        (True, "yes"),
        (False, "no"),
        (0, "0"),
        (Flavour.VANILLA, "VANILLA"),
        (frozenset({"b", "a"}), "a, b"),
        (frozenset(), "-"),
        ({"solo"}, "solo"),
        (["first", "second"], "first, second"),
        ((), "-"),
        (frozenset({Flavour.VANILLA}), "VANILLA"),
    ],
)
def test_format_value(value: object, expected: str) -> None:
    assert _render.format_value(value) == expected


def test_render_frame_prints_title_headers_rows_cells_and_count(capsys: pytest.CaptureFixture[str]) -> None:
    frame = pd.DataFrame(
        {
            "Name": ["alpha", "beta"],
            "Description": ["First one", "Second one"],
            "Tags": [frozenset({"baseline"}), frozenset()],
        }
    )

    _render.render_frame(frame, title="Things", empty_hint="nothing")
    printed = output(capsys)

    assert all(text in printed for text in ("Things", "Name", "Description", "Tags", "alpha", "beta", "baseline"))
    assert _render.EMPTY_CELL in printed
    assert "2 entries" in printed

    _render.render_frame(frame.head(1), title="Things", empty_hint="nothing")
    assert "1 entry" in output(capsys)


def test_render_frame_on_an_empty_frame_prints_only_the_hint(capsys: pytest.CaptureFixture[str]) -> None:
    _render.render_frame(pd.DataFrame(columns=["Name"]), title="Things", empty_hint="Nothing registered.")
    printed = output(capsys)

    assert "Nothing registered." in printed
    assert "Things" not in printed


def test_render_rows(capsys: pytest.CaptureFixture[str]) -> None:
    _render.render_rows(
        [["plugin", Text("loaded", style="green")], ["alpha", "features [bold] and more"]],
        columns=["Plugin", "Status"],
        title="Plugins",
        empty_hint="none",
        show_count=False,
    )
    printed = output(capsys)

    assert "loaded" in printed
    assert "[bold]" in printed
    assert "entries" not in printed

    _render.render_rows([], columns=["Field"], title="Thing", empty_hint="Nothing here.")
    assert "Nothing here." in output(capsys)


def test_render_mapping_prints_pairs_without_a_count_and_hints_when_empty(capsys: pytest.CaptureFixture[str]) -> None:
    _render.render_mapping({"name": "alpha", "description": "Does a thing"}, title="alpha")
    printed = output(capsys)

    assert all(text in printed for text in ("alpha", "description", "Does a thing"))
    assert "entries" not in printed

    _render.render_mapping({}, title="alpha")
    assert "No metadata recorded for alpha." in output(capsys)


def test_render_empty_and_console(capsys: pytest.CaptureFixture[str]) -> None:
    _render.render_empty("Nothing at all.")

    assert "Nothing at all." in output(capsys)
    assert _render.console() is not _render.console()


CALLBACKS = {
    "predictors": registries_module.list_predictors,
    "sample-featurizers": registries_module.list_sample_featurizers,
    "drug-featurizers": registries_module.list_drug_featurizers,
    "splitters": registries_module.list_splitters,
    "visualizations": registries_module.list_visualizations,
    "zoo-presets": registries_module.list_zoo_presets,
    "datasets": registries_module.list_datasets,
    "plugins": plugins_module.list_plugins,
}


class TestGroup(GroupSurface):
    group = list_app
    path = "list"
    callbacks = CALLBACKS

    @pytest.mark.parametrize("command", CALLBACKS, ids=CALLBACKS)
    def test_dash_h_reaches_each_command(self, command: str) -> None:
        result = runner.invoke(app, ["list", command, "-h"], env=HELP_ENV)

        assert result.exit_code == 0
        assert "Usage" in plain(result.output)
