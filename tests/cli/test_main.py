"""Tests for :mod:`drevalpy.cli.main`: the root app, its callback and entry point."""

from __future__ import annotations

import pytest
from upath import UPath

from drevalpy.cli.main import app, cli_main
from tests.cli._helpers import HELP_ENV, Recorder, make_runner, plain

runner = make_runner()

EXPECTED_COMMANDS = ("run", "single", "aggregate", "curate", "report", "data", "experiments")


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


class TestRegistration:
    """The commands wired up in ``main`` are the ones click knows about."""

    def test_registered_command_names(self) -> None:
        registered = {command.name for command in app.registered_commands}

        assert registered == {"run", "single", "aggregate", "curate", "report"}

    def test_registered_group_names(self) -> None:
        registered = {group.name for group in app.registered_groups}

        assert registered == {"data", "experiments"}


class TestExtensionLoading:
    """``main_callback`` forwards extension directories to the registry loader.

    Every invocation here appends a subcommand before ``-h``: click's root-level
    eager help option short-circuits before the group callback runs, so
    ``["-e", d, "-h"]`` would never reach the loader at all.
    """

    def test_extensions_dir_option_is_forwarded(self, monkeypatch: pytest.MonkeyPatch, tmp_path: UPath) -> None:
        loader = Recorder()
        monkeypatch.delenv("DREVALPY_EXTENSIONS_DIR", raising=False)
        monkeypatch.setattr("drevalpy.registry.load_extension_dir", loader)
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
        monkeypatch.setattr("drevalpy.registry.load_extension_dir", loader)
        ext_dir = tmp_path / "ext"
        ext_dir.mkdir()

        result = runner.invoke(app, ["-e", str(ext_dir), "-h"], env=HELP_ENV)

        assert result.exit_code == 0
        assert loader.call_count == 0

    def test_short_option_accepts_repeats_in_order(self, monkeypatch: pytest.MonkeyPatch, tmp_path: UPath) -> None:
        loader = Recorder()
        monkeypatch.delenv("DREVALPY_EXTENSIONS_DIR", raising=False)
        monkeypatch.setattr("drevalpy.registry.load_extension_dir", loader)
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
        monkeypatch.setattr("drevalpy.registry.load_extension_dir", loader)

        result = runner.invoke(app, ["-e", str(opt_dir), "run", "-h"], env=HELP_ENV)

        assert result.exit_code == 0
        assert [call[0][0] for call in loader.calls] == [str(env_dir), str(opt_dir)]

    def test_unset_env_var_triggers_no_load(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loader = Recorder()
        monkeypatch.delenv("DREVALPY_EXTENSIONS_DIR", raising=False)
        monkeypatch.setattr("drevalpy.registry.load_extension_dir", loader)

        result = runner.invoke(app, ["run", "-h"], env=HELP_ENV)

        assert result.exit_code == 0
        assert loader.call_count == 0

    def test_empty_env_var_triggers_no_load(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loader = Recorder()
        monkeypatch.setenv("DREVALPY_EXTENSIONS_DIR", "")
        monkeypatch.setattr("drevalpy.registry.load_extension_dir", loader)

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

        assert isinstance(result.exception, FileNotFoundError)


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
