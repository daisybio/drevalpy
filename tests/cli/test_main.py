"""Tests for :mod:`drevalpy.cli.main`: the root app, its callback and entry point."""

from __future__ import annotations

import logging
import os

import pytest
import typer
from rich.logging import RichHandler
from upath import UPath

import drevalpy
from drevalpy.cli.main import _DrEvalGroup, app, cli_main
from drevalpy.errors import DrEvalError, InputNotFoundError, InvalidInputError, UnknownNameError
from tests.cli._helpers import HELP_ENV, Recorder, make_runner, plain

runner = make_runner()

EXPECTED_COMMANDS = (
    "run",
    "single",
    "train",
    "predict",
    "aggregate",
    "curate",
    "report",
    "data",
    "experiments",
    "list",
)


class TestHelp:
    """The root app is help-first: bare invocation and ``-h`` both print usage."""

    def test_no_arguments_prints_help_instead_of_erroring(self) -> None:
        result = runner.invoke(app, [], env=HELP_ENV)

        assert "Usage" in plain(result.output)

    def test_no_arguments_exits_nonzero(self) -> None:
        result = runner.invoke(app, [], env=HELP_ENV)

        assert result.exit_code != 0

    def test_dash_h_is_accepted_as_a_help_alias(self) -> None:
        result = runner.invoke(app, ["-h"], env=HELP_ENV)

        assert result.exit_code == 0
        assert "Usage" in plain(result.output)

    @pytest.mark.parametrize("command", EXPECTED_COMMANDS, ids=EXPECTED_COMMANDS)
    def test_help_lists_every_registered_command(self, command: str) -> None:
        result = runner.invoke(app, ["--help"], env=HELP_ENV)

        assert command in plain(result.output)

    @pytest.mark.parametrize("command", EXPECTED_COMMANDS, ids=EXPECTED_COMMANDS)
    def test_dash_h_propagates_to_subcommands(self, command: str) -> None:
        result = runner.invoke(app, [command, "-h"], env=HELP_ENV)

        assert result.exit_code == 0
        assert "Usage" in plain(result.output)

    def test_unknown_command_is_a_usage_error(self) -> None:
        result = runner.invoke(app, ["not-a-command"], env=HELP_ENV)

        assert result.exit_code == 2


class TestVersion:
    """``--version`` is an eager root option that prints ``drevalpy <version>``."""

    def test_prints_the_package_version(self) -> None:
        result = runner.invoke(app, ["--version"], env=HELP_ENV)

        assert result.exit_code == 0
        assert result.output.strip() == f"drevalpy {drevalpy.__version__}"

    def test_does_not_need_a_subcommand_or_load_extensions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loader = Recorder()
        monkeypatch.setattr("drevalpy.registry.load_extensions", loader)

        result = runner.invoke(app, ["--version"], env=HELP_ENV)

        assert result.exit_code == 0
        assert loader.call_count == 0

    def test_wins_over_a_subcommand(self) -> None:
        result = runner.invoke(app, ["--version", "run"], env=HELP_ENV)

        assert result.exit_code == 0
        assert result.output.strip() == f"drevalpy {drevalpy.__version__}"

    def test_is_listed_in_the_help(self) -> None:
        assert "--version" in plain(runner.invoke(app, ["--help"], env=HELP_ENV).output)


class TestRegistration:
    """The commands wired up in ``main`` are the ones click knows about."""

    def test_registered_command_names(self) -> None:
        registered = {command.name for command in app.registered_commands}

        assert registered == {"run", "single", "train", "predict", "aggregate", "curate", "report"}

    def test_registered_group_names(self) -> None:
        registered = {group.name for group in app.registered_groups}

        assert registered == {"data", "experiments", "list"}


class TestExtensionLoading:
    """``main_callback`` loads ``DREVALPY_EXTENSIONS_DIR`` first, then the ``-e`` directories.

    The directory loader is patched in :mod:`drevalpy.registry._extensions`, the module
    ``load_extensions`` resolves it from.

    Every invocation here appends a subcommand before ``-h``: click's root-level
    eager help option short-circuits before the group callback runs, so
    ``["-e", d, "-h"]`` would never reach the loader at all.
    """

    def test_extensions_dir_option_is_forwarded(self, monkeypatch: pytest.MonkeyPatch, tmp_path: UPath) -> None:
        loader = Recorder()
        monkeypatch.delenv("DREVALPY_EXTENSIONS_DIR", raising=False)
        monkeypatch.setattr("drevalpy.registry._extensions.load_extension_dir", loader)
        ext_dir = tmp_path / "ext"
        ext_dir.mkdir()

        result = runner.invoke(app, ["--extensions-dir", str(ext_dir), "run", "-h"], env=HELP_ENV)

        assert result.exit_code == 0
        assert loader.args == (str(ext_dir),)

    def test_root_help_short_circuits_before_the_callback(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: UPath
    ) -> None:
        loader = Recorder()
        monkeypatch.delenv("DREVALPY_EXTENSIONS_DIR", raising=False)
        monkeypatch.setattr("drevalpy.registry._extensions.load_extension_dir", loader)
        ext_dir = tmp_path / "ext"
        ext_dir.mkdir()

        result = runner.invoke(app, ["-e", str(ext_dir), "-h"], env=HELP_ENV)

        assert result.exit_code == 0
        assert loader.call_count == 0

    def test_short_option_accepts_repeats_in_order(self, monkeypatch: pytest.MonkeyPatch, tmp_path: UPath) -> None:
        loader = Recorder()
        monkeypatch.delenv("DREVALPY_EXTENSIONS_DIR", raising=False)
        monkeypatch.setattr("drevalpy.registry._extensions.load_extension_dir", loader)
        first = tmp_path / "one"
        second = tmp_path / "two"
        for directory in (first, second):
            directory.mkdir()

        result = runner.invoke(app, ["-e", str(first), "-e", str(second), "run", "-h"], env=HELP_ENV)

        assert result.exit_code == 0
        assert [call[0][0] for call in loader.calls] == [str(first), str(second)]

    def test_env_var_is_loaded_before_the_option(self, monkeypatch: pytest.MonkeyPatch, tmp_path: UPath) -> None:
        loader = Recorder()
        env_dir = tmp_path / "from_env"
        opt_dir = tmp_path / "from_option"
        for directory in (env_dir, opt_dir):
            directory.mkdir()
        monkeypatch.setenv("DREVALPY_EXTENSIONS_DIR", str(env_dir))
        monkeypatch.setattr("drevalpy.registry._extensions.load_extension_dir", loader)

        result = runner.invoke(app, ["-e", str(opt_dir), "run", "-h"], env=HELP_ENV)

        assert result.exit_code == 0
        assert [call[0][0] for call in loader.calls] == [str(env_dir), str(opt_dir)]

    def test_callback_calls_load_extensions_bare_and_then_with_the_flags(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: UPath
    ) -> None:
        loader = Recorder()
        monkeypatch.setattr("drevalpy.registry.load_extensions", loader)

        result = runner.invoke(app, ["-e", str(tmp_path), "-e", "other", "run", "-h"], env=HELP_ENV)

        assert result.exit_code == 0
        assert loader.calls == [((), {}), ((), {"directories": [str(tmp_path), "other"]})]

    def test_callback_without_flags_calls_load_extensions_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loader = Recorder()
        monkeypatch.setattr("drevalpy.registry.load_extensions", loader)

        result = runner.invoke(app, ["run", "-h"], env=HELP_ENV)

        assert result.exit_code == 0
        assert loader.calls == [((), {})]

    def test_env_var_with_several_directories_loads_each_before_the_option(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: UPath
    ) -> None:
        loader = Recorder()
        env_dirs = [tmp_path / "env_one", tmp_path / "env_two"]
        monkeypatch.setenv("DREVALPY_EXTENSIONS_DIR", os.pathsep.join(str(d) for d in env_dirs))
        monkeypatch.setattr("drevalpy.registry._extensions.load_extension_dir", loader)

        result = runner.invoke(app, ["-e", "from_option", "run", "-h"], env=HELP_ENV)

        assert result.exit_code == 0
        assert [call[0][0] for call in loader.calls] == [str(env_dirs[0]), str(env_dirs[1]), "from_option"]

    def test_env_var_directory_missing_fails_the_command(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: UPath
    ) -> None:
        monkeypatch.setenv("DREVALPY_EXTENSIONS_DIR", str(tmp_path / "missing"))

        result = runner.invoke(app, ["run", "-h"], env=HELP_ENV)

        assert result.exit_code == 1
        assert f"Error: Extension directory not found: {tmp_path / 'missing'}" in plain(result.output)

    def test_same_directory_in_env_and_option_loads_without_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: UPath
    ) -> None:
        monkeypatch.setenv("DREVALPY_EXTENSIONS_DIR", str(tmp_path))

        result = runner.invoke(app, ["-e", str(tmp_path), "run", "-h"], env=HELP_ENV)

        assert result.exit_code == 0

    def test_unset_env_var_triggers_no_load(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loader = Recorder()
        monkeypatch.delenv("DREVALPY_EXTENSIONS_DIR", raising=False)
        monkeypatch.setattr("drevalpy.registry._extensions.load_extension_dir", loader)

        result = runner.invoke(app, ["run", "-h"], env=HELP_ENV)

        assert result.exit_code == 0
        assert loader.call_count == 0

    def test_empty_env_var_triggers_no_load(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loader = Recorder()
        monkeypatch.setenv("DREVALPY_EXTENSIONS_DIR", "")
        monkeypatch.setattr("drevalpy.registry._extensions.load_extension_dir", loader)

        result = runner.invoke(app, ["run", "-h"], env=HELP_ENV)

        assert result.exit_code == 0
        assert loader.call_count == 0

    def test_real_loader_accepts_an_empty_directory(self, monkeypatch: pytest.MonkeyPatch, tmp_path: UPath) -> None:
        """An empty dir exercises the real loader without mutating any registry."""
        monkeypatch.delenv("DREVALPY_EXTENSIONS_DIR", raising=False)
        ext_dir = tmp_path / "empty"
        ext_dir.mkdir()

        result = runner.invoke(app, ["-e", str(ext_dir), "run", "-h"], env=HELP_ENV)

        assert result.exit_code == 0

    def test_real_loader_rejects_a_path_that_is_not_a_directory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: UPath
    ) -> None:
        monkeypatch.delenv("DREVALPY_EXTENSIONS_DIR", raising=False)
        not_a_dir = tmp_path / "plain.py"
        not_a_dir.write_text("")

        result = runner.invoke(app, ["-e", str(not_a_dir), "run", "-h"], env=HELP_ENV)

        assert result.exit_code == 1
        assert "Error: Extension directory not found" in plain(result.output)


def _failing_app(error: BaseException) -> typer.Typer:
    """Build an app on the root group class whose only command raises *error*."""
    failing = typer.Typer(cls=_DrEvalGroup)

    @failing.callback()
    def group() -> None:
        """Group callback, so ``boom`` stays a subcommand."""

    @failing.command()
    def boom() -> None:
        """Raise the error under test."""
        raise error

    return failing


class TestErrorHandling:
    """The root group prints a drevalpy input error as ``Error: <message>`` and exits 1."""

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(DrEvalError("bad input"), id="DrEvalError"),
            pytest.param(UnknownNameError("bad input"), id="UnknownNameError"),
            pytest.param(InputNotFoundError("bad input"), id="InputNotFoundError"),
            pytest.param(InvalidInputError("bad input"), id="InvalidInputError"),
            pytest.param(FileNotFoundError("bad input"), id="FileNotFoundError"),
        ],
    )
    def test_input_errors_become_one_line(self, error: Exception) -> None:
        result = runner.invoke(_failing_app(error), ["boom"])

        assert result.exit_code == 1
        assert plain(result.output).strip() == "Error: bad input"

    def test_a_keyerror_subclass_prints_without_quotes(self) -> None:
        result = runner.invoke(_failing_app(UnknownNameError("no such thing")), ["boom"])

        assert plain(result.output).strip() == "Error: no such thing"

    def test_a_multi_line_message_is_kept(self) -> None:
        result = runner.invoke(_failing_app(DrEvalError("2 run(s) failed:\n  a\n  b")), ["boom"])

        assert plain(result.output).strip() == "Error: 2 run(s) failed:\n  a\n  b"

    def test_other_exceptions_propagate_with_the_exception_attached(self) -> None:
        result = runner.invoke(_failing_app(RuntimeError("bug")), ["boom"])

        assert result.exit_code == 1
        assert isinstance(result.exception, RuntimeError)

    def test_a_missing_extension_directory_is_reported_by_the_real_app(self, tmp_path: UPath) -> None:
        result = runner.invoke(app, ["-e", str(tmp_path / "absent"), "list", "predictors"], env=HELP_ENV)

        assert result.exit_code == 1
        assert "Error: Extension directory not found" in plain(result.output)
        assert "Traceback" not in result.output

    def test_an_unknown_model_is_reported_by_the_real_app(self) -> None:
        result = runner.invoke(app, ["run", "NoSuchModel", "-d", "x"], env=HELP_ENV)

        assert result.exit_code == 1
        assert plain(result.output).startswith("Error: 'NoSuchModel' is not a zoo name")
        assert "Traceback" not in result.output


class TestLogging:
    """The callback, not the import, installs the Rich handler."""

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

    def test_quiet_flag_long_form(self) -> None:
        result = runner.invoke(app, ["--quiet", "run", "-h"], env=HELP_ENV)

        assert result.exit_code == 0
        assert logging.getLogger("drevalpy").level == logging.WARNING

    def test_verbose_and_quiet_together_are_a_usage_error(self) -> None:
        result = runner.invoke(app, ["-v", "-q", "run", "-h"], env=HELP_ENV)

        assert result.exit_code == 2
        assert "mutually exclusive" in plain(result.output)


class TestCliMain:
    """``cli_main`` is the console-script wrapper around ``app()``."""

    def test_keyboard_interrupt_maps_to_exit_code_130(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def interrupt() -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr("drevalpy.cli.main.app", interrupt)

        with pytest.raises(SystemExit) as exc_info:
            cli_main()

        assert exc_info.value.code == 130
        assert "Interrupted." in capsys.readouterr().err

    def test_keyboard_interrupt_does_not_chain_the_original_exception(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def interrupt() -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr("drevalpy.cli.main.app", interrupt)

        with pytest.raises(SystemExit) as exc_info:
            cli_main()
        capsys.readouterr()

        assert exc_info.value.__cause__ is None

    def test_clean_exit_is_propagated_untouched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("drevalpy.cli.main.app", Recorder())

        assert cli_main() is None
