"""Tests for :mod:`drevalpy.cli.catalog.registries`, the five per-registry commands.

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
    pytest.param("cell-line-featurizers", "cell_line_featurizer", "identity", id="cell-line-featurizers"),
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


class TestUnknownEntry:
    """An unregistered name fails with the registry's own message."""

    @pytest.mark.parametrize(("command", "attribute", "entry"), REGISTRY_COMMANDS)
    def test_exit_code_is_one(self, command: str, attribute: str, entry: str) -> None:
        result = runner.invoke(app, ["list", command, "definitely-not-registered"], env=HELP_ENV)

        assert result.exit_code == 1

    @pytest.mark.parametrize(("command", "attribute", "entry"), REGISTRY_COMMANDS)
    def test_names_the_missing_entry(self, command: str, attribute: str, entry: str) -> None:
        assert "definitely-not-registered" in invoke(command, "definitely-not-registered")

    @pytest.mark.parametrize(("command", "attribute", "entry"), REGISTRY_COMMANDS)
    def test_message_lists_the_valid_entries(self, command: str, attribute: str, entry: str) -> None:
        """The registry's ValueError enumerates what *is* registered; keep that."""
        assert entry in invoke(command, "definitely-not-registered")

    @pytest.mark.parametrize(("command", "attribute", "entry"), REGISTRY_COMMANDS)
    def test_does_not_raise_out_of_the_command(self, command: str, attribute: str, entry: str) -> None:
        result = runner.invoke(app, ["list", command, "definitely-not-registered"], env=HELP_ENV)

        assert isinstance(result.exception, SystemExit)


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
