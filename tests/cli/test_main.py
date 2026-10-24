"""Tests for the root app, its callback and entry point, and the :mod:`drevalpy.cli` surface."""

from __future__ import annotations

import logging
import os

import pytest
import typer
from rich.logging import RichHandler
from upath import UPath

import drevalpy
import drevalpy.cli as cli_pkg
from drevalpy.cli import main as cli_main_module
from drevalpy.cli.main import _DrEvalGroup, app, cli_main
from drevalpy.errors import DrEvalError, InputNotFoundError, InvalidInputError, UnknownNameError
from tests.cli._helpers import HELP_ENV, Recorder, make_runner, plain

runner = make_runner()


COMMANDS = ("run", "single", "train", "predict", "aggregate", "curate", "report", "data", "experiments", "list")


class TestHelp:
    def test_no_arguments_prints_help_and_exits_nonzero(self) -> None:
        result = runner.invoke(app, [], env=HELP_ENV)

        assert "Usage" in plain(result.output)
        assert result.exit_code != 0

    def test_help_lists_every_command_and_dash_h_is_an_alias(self) -> None:
        result = runner.invoke(app, ["-h"], env=HELP_ENV)

        assert result.exit_code == 0
        assert "Usage" in plain(result.output)
        assert all(command in plain(result.output) for command in COMMANDS)
        assert "--version" in plain(result.output)

    @pytest.mark.parametrize("command", COMMANDS)
    def test_dash_h_propagates_to_subcommands(self, command: str) -> None:
        result = runner.invoke(app, [command, "-h"], env=HELP_ENV)

        assert result.exit_code == 0
        assert "Usage" in plain(result.output)

    def test_unknown_command_is_a_usage_error(self) -> None:
        assert runner.invoke(app, ["not-a-command"], env=HELP_ENV).exit_code == 2


MISSING_ARGUMENTS = [
    ["train"],
    ["train", "ElasticNet"],
    ["train", "ElasticNet", "-d", "toy.h5mu"],
    ["train", "ElasticNet", "-o", "model"],
    ["single"],
    ["single", "ElasticNet"],
    ["single", "ElasticNet", "ds.h5mu", "fold.npz"],
    ["predict"],
    ["predict", "model"],
    ["predict", "model", "-d", "toy.h5mu"],
    ["predict", "model", "-o", "out.npz"],
    ["curate"],
    ["curate", "in.csv"],
    ["data", "load"],
    ["data", "load", "TOYv1"],
    ["data", "split"],
    ["data", "split", "TOYv1"],
    ["experiments", "randomization"],
    ["experiments", "randomization", "ElasticNet"],
    ["experiments", "randomization", "ElasticNet", "TOYv1"],
    ["experiments", "robustness"],
    ["experiments", "robustness", "splits"],
]


@pytest.mark.parametrize("argv", MISSING_ARGUMENTS, ids=" ".join)
def test_a_command_missing_a_required_argument_is_a_usage_error(argv: list[str]) -> None:
    assert runner.invoke(app, argv, env=HELP_ENV).exit_code == 2


class TestVersion:
    def test_prints_the_package_version_without_a_subcommand_or_extensions(self, monkeypatch) -> None:
        loader = Recorder()
        monkeypatch.setattr("drevalpy.registry.load_extensions", loader)

        for args in (["--version"], ["--version", "run"]):
            result = runner.invoke(app, args, env=HELP_ENV)
            assert result.exit_code == 0
            assert result.output.strip() == f"drevalpy {drevalpy.__version__}"
        assert loader.call_count == 0


class TestExtensionLoading:
    """Every invocation appends a subcommand before ``-h``: the root help option short-circuits the callback."""

    @pytest.fixture()
    def loader(self, monkeypatch: pytest.MonkeyPatch) -> Recorder:
        recorder = Recorder()
        monkeypatch.delenv("DREVALPY_EXTENSIONS_DIR", raising=False)
        monkeypatch.setattr("drevalpy.registry._extensions.load_extension_dir", recorder)
        return recorder

    @staticmethod
    def _dirs(tmp_path: UPath, *names: str) -> list[UPath]:
        paths = [tmp_path / name for name in names]
        for path in paths:
            path.mkdir()
        return paths

    def test_extension_dirs_are_forwarded_in_order(self, loader: Recorder, tmp_path: UPath) -> None:
        first, second = self._dirs(tmp_path, "one", "two")

        result = runner.invoke(app, ["-e", str(first), "--extensions-dir", str(second), "run", "-h"], env=HELP_ENV)

        assert result.exit_code == 0
        assert [call[0][0] for call in loader.calls] == [str(first), str(second)]

    def test_root_help_short_circuits_before_the_callback(self, loader: Recorder, tmp_path: UPath) -> None:
        (ext_dir,) = self._dirs(tmp_path, "ext")

        result = runner.invoke(app, ["-e", str(ext_dir), "-h"], env=HELP_ENV)

        assert result.exit_code == 0
        assert loader.call_count == 0

    def test_env_dirs_are_loaded_before_the_option(
        self, loader: Recorder, monkeypatch: pytest.MonkeyPatch, tmp_path: UPath
    ) -> None:
        env_dirs = [tmp_path / "env_one", tmp_path / "env_two"]
        monkeypatch.setenv("DREVALPY_EXTENSIONS_DIR", os.pathsep.join(str(d) for d in env_dirs))

        result = runner.invoke(app, ["-e", "from_option", "run", "-h"], env=HELP_ENV)

        assert result.exit_code == 0
        assert [call[0][0] for call in loader.calls] == [str(env_dirs[0]), str(env_dirs[1]), "from_option"]

    @pytest.mark.parametrize("value", [None, ""])
    def test_an_unset_or_empty_env_var_triggers_no_load(
        self, loader: Recorder, monkeypatch: pytest.MonkeyPatch, value: str | None
    ) -> None:
        if value is not None:
            monkeypatch.setenv("DREVALPY_EXTENSIONS_DIR", value)

        result = runner.invoke(app, ["run", "-h"], env=HELP_ENV)

        assert result.exit_code == 0
        assert loader.call_count == 0

    def test_load_extensions_runs_bare_and_then_with_the_flags(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: UPath
    ) -> None:
        recorder = Recorder()
        monkeypatch.setattr("drevalpy.registry.load_extensions", recorder)

        runner.invoke(app, ["-e", str(tmp_path), "-e", "other", "run", "-h"], env=HELP_ENV)
        assert recorder.calls == [((), {}), ((), {"directories": [str(tmp_path), "other"]})]
        recorder.calls.clear()
        runner.invoke(app, ["run", "-h"], env=HELP_ENV)
        assert recorder.calls == [((), {})]

    def test_the_real_loader_accepts_directories_and_rejects_the_rest(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: UPath
    ) -> None:
        monkeypatch.delenv("DREVALPY_EXTENSIONS_DIR", raising=False)
        (empty,) = self._dirs(tmp_path, "empty")
        not_a_dir = tmp_path / "plain.py"
        not_a_dir.write_text("")

        assert runner.invoke(app, ["-e", str(empty), "run", "-h"], env=HELP_ENV).exit_code == 0
        monkeypatch.setenv("DREVALPY_EXTENSIONS_DIR", str(empty))
        assert runner.invoke(app, ["-e", str(empty), "run", "-h"], env=HELP_ENV).exit_code == 0
        monkeypatch.delenv("DREVALPY_EXTENSIONS_DIR")
        result = runner.invoke(app, ["-e", str(not_a_dir), "run", "-h"], env=HELP_ENV)
        assert result.exit_code == 1
        assert "Error: Extension directory not found" in plain(result.output)

    def test_a_missing_env_var_directory_fails_the_command(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: UPath
    ) -> None:
        monkeypatch.setenv("DREVALPY_EXTENSIONS_DIR", str(tmp_path / "missing"))

        result = runner.invoke(app, ["run", "-h"], env=HELP_ENV)

        assert result.exit_code == 1
        assert f"Error: Extension directory not found: {tmp_path / 'missing'}" in plain(result.output)
        assert "Traceback" not in result.output


def _failing_app(error: BaseException) -> typer.Typer:
    failing = typer.Typer(cls=_DrEvalGroup)

    @failing.callback()
    def group() -> None:
        pass

    @failing.command()
    def boom() -> None:
        raise error

    return failing


class TestErrorHandling:
    """The root group prints a drevalpy input error as ``Error: <message>`` and exits 1."""

    @pytest.mark.parametrize(
        "error",
        [DrEvalError, UnknownNameError, InputNotFoundError, InvalidInputError, FileNotFoundError],
        ids=lambda cls: cls.__name__,
    )
    def test_input_errors_become_one_line_without_quotes(self, error: type[Exception]) -> None:
        result = runner.invoke(_failing_app(error("bad input")), ["boom"])

        assert result.exit_code == 1
        assert plain(result.output).strip() == "Error: bad input"

    def test_a_multi_line_message_is_kept(self) -> None:
        result = runner.invoke(_failing_app(DrEvalError("2 run(s) failed:\n  a\n  b")), ["boom"])

        assert plain(result.output).strip() == "Error: 2 run(s) failed:\n  a\n  b"

    def test_other_exceptions_propagate_with_the_exception_attached(self) -> None:
        result = runner.invoke(_failing_app(RuntimeError("bug")), ["boom"])

        assert result.exit_code == 1
        assert isinstance(result.exception, RuntimeError)

    def test_an_unknown_model_is_reported_by_the_real_app(self) -> None:
        result = runner.invoke(app, ["run", "NoSuchModel", "-d", "x"], env=HELP_ENV)

        assert result.exit_code == 1
        assert plain(result.output).startswith("Error: 'NoSuchModel' is not a zoo name")
        assert "Traceback" not in result.output


class TestLogging:
    def test_callback_installs_the_rich_handler(self) -> None:
        logger = logging.getLogger("drevalpy")
        logger.handlers[:] = [h for h in logger.handlers if not isinstance(h, RichHandler)]

        result = runner.invoke(app, ["run", "-h"], env=HELP_ENV)

        assert result.exit_code == 0
        assert any(isinstance(h, RichHandler) for h in logger.handlers)

    @pytest.mark.parametrize(
        ("flags", "level"),
        [([], logging.INFO), (["-v"], logging.DEBUG), (["--verbose"], logging.DEBUG), (["-q"], logging.WARNING)],
    )
    def test_verbosity_flags_set_the_log_level(self, flags: list[str], level: int) -> None:
        result = runner.invoke(app, [*flags, "run", "-h"], env=HELP_ENV)

        assert result.exit_code == 0
        assert logging.getLogger("drevalpy").level == level

    def test_verbose_and_quiet_together_are_a_usage_error(self) -> None:
        result = runner.invoke(app, ["-v", "-q", "run", "-h"], env=HELP_ENV)

        assert result.exit_code == 2
        assert "mutually exclusive" in plain(result.output)


class TestCliMain:
    def test_keyboard_interrupt_maps_to_exit_code_130_without_chaining(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def interrupt() -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr("drevalpy.cli.main.app", interrupt)

        with pytest.raises(SystemExit) as exc_info:
            cli_main()

        assert exc_info.value.code == 130
        assert exc_info.value.__cause__ is None
        assert "Interrupted." in capsys.readouterr().err

    def test_clean_exit_is_propagated_untouched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("drevalpy.cli.main.app", Recorder())

        assert cli_main() is None


def test_exports_the_app_from_main() -> None:
    assert cli_pkg.app is cli_main_module.app


def test_exports_the_console_script_entry_point() -> None:
    assert cli_pkg.cli_main is cli_main_module.cli_main


def test_all_lists_only_the_two_public_names() -> None:
    assert cli_pkg.__all__ == ["app", "cli_main"]
