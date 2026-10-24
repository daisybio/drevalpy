"""Cases for ``docs/_snippets.py``: block extraction, markers, failure reporting and namespaces."""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[2] / "docs"

if str(DOCS) not in sys.path:
    sys.path.insert(0, str(DOCS))

from _snippets import SnippetError, collect, run_page  # noqa: E402


def _page(tmp_path: Path, text: str, name: str = "page.rst") -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")
    return path


def _run(page: Path, tmp_path: Path, *, allow_network: bool = False) -> None:
    run_page(page, tmp_path / "work", allow_network=allow_network)


def _failing_line(page: Path, error: SnippetError) -> str:
    return page.read_text().splitlines()[error.line - 1].strip()


def test_collect_finds_blocks_in_tabs_and_ignores_other_directives(tmp_path: Path) -> None:
    page = _page(
        tmp_path,
        """
        .. note::

           Not a code block.

        .. tab-set::

           .. tab-item:: CLI

              .. code-block:: bash

                 drevalpy list splitters

           .. tab-item:: Python

              .. code-block:: python
                 :caption: A caption

                 x = 1
                 y = x + 1

        .. code-block:: yaml

           key: value
        """,
    )
    cli, python, data = collect(page)
    assert (cli.language, cli.code, cli.indent) == ("bash", "drevalpy list splitters", 6)
    assert (python.language, python.code) == ("python", "x = 1\ny = x + 1")
    assert (data.language, data.code) == ("yaml", "key: value")
    lines = page.read_text().splitlines()
    assert lines[cli.line - 1] == "         drevalpy list splitters"
    assert lines[python.directive_line - 1].strip() == ".. code-block:: python"


def test_a_directive_shown_inside_another_block_is_not_a_snippet(tmp_path: Path) -> None:
    page = _page(
        tmp_path,
        """
        .. code-block:: rst

           .. snippet: skip

           .. code-block:: python

              raise RuntimeError("only an example")
        """,
    )
    assert [s.language for s in collect(page)] == ["rst"]
    _run(page, tmp_path)


def test_markers_are_read_from_the_comment_before_the_block(tmp_path: Path) -> None:
    page = _page(
        tmp_path,
        """
        .. snippet: skip
        .. code-block:: bash

           pip install drevalpy

        .. tab-set::

           .. tab-item:: Python

              .. snippet: network

              .. code-block:: python

                 import drevalpy

        .. code-block:: python

           unmarked = True
        """,
    )
    assert [s.marker for s in collect(page)] == ["skip", "network", "run"]


def test_marker_errors_name_the_line(tmp_path: Path) -> None:
    unknown = _page(tmp_path, "Intro.\n\n.. snippet: skipp\n.. code-block:: python\n\n   pass\n")
    with pytest.raises(SnippetError) as caught:
        collect(unknown)
    assert (caught.value.page, caught.value.line) == (unknown, 3)
    assert "skipp" in caught.value.output
    dangling = _page(tmp_path, ".. snippet: skip\n\nA paragraph, not a code block.\n", "dangling.rst")
    with pytest.raises(SnippetError, match="must be followed by a code-block"):
        collect(dangling)


MARKED_PAGE = """
    .. snippet: skip
    .. code-block:: python

       open("skipped", "w").close()

    .. snippet: network

    .. code-block:: python

       open("networked", "w").close()

    .. code-block:: python

       open("plain", "w").close()
    """


@pytest.mark.parametrize(("allow_network", "expected"), [(False, ["plain"]), (True, ["networked", "plain"])])
def test_skip_and_network_markers_when_running(tmp_path: Path, allow_network: bool, expected: list[str]) -> None:
    _run(_page(tmp_path, MARKED_PAGE), tmp_path, allow_network=allow_network)
    assert sorted(os.listdir(tmp_path / "work")) == expected


@pytest.mark.parametrize(("platform", "expected"), [("linux", ["posix"]), ("win32", [])])
def test_posix_blocks_do_not_run_on_windows(tmp_path: Path, monkeypatch, platform: str, expected: list[str]) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    _run(_page(tmp_path, '.. snippet: posix\n.. code-block:: python\n\n   open("posix", "w").close()\n'), tmp_path)
    assert os.listdir(tmp_path / "work") == expected


def test_a_skipped_block_may_contain_anything(tmp_path: Path) -> None:
    _run(_page(tmp_path, ".. snippet: skip\n.. code-block:: bash\n\n   pip install drevalpy | tee log\n"), tmp_path)


def test_a_failing_block_reports_page_line_and_traceback(tmp_path: Path) -> None:
    page = _page(
        tmp_path,
        """
        .. code-block:: python

           value = 1

        .. code-block:: python

           print("before the failure")
           raise ValueError("boom")
        """,
    )
    before = Path.cwd()
    with pytest.raises(SnippetError) as caught:
        _run(page, tmp_path)
    error = caught.value
    assert error.page == page
    assert _failing_line(page, error) == 'raise ValueError("boom")'
    assert "ValueError: boom" in error.output
    assert "before the failure" in error.output
    assert f"{page}:{error.line}" in str(error)
    assert Path.cwd() == before


@pytest.mark.parametrize(
    ("body", "line"),
    [("fine = 1\nnot valid python", "not valid python"), ("raise SystemExit(3)", "raise SystemExit(3)")],
)
def test_syntax_errors_and_sys_exit_are_failures(tmp_path: Path, body: str, line: str) -> None:
    page = _page(tmp_path, ".. code-block:: python\n\n" + textwrap.indent(body, "   ") + "\n")
    with pytest.raises(SnippetError) as caught:
        _run(page, tmp_path)
    assert _failing_line(page, caught.value) == line


def test_blocks_run_in_a_working_directory_that_is_restored(tmp_path: Path) -> None:
    before = Path.cwd()
    _run(_page(tmp_path, ".. code-block:: python\n\n   open('made.txt', 'w').write('x')\n"), tmp_path)
    assert (tmp_path / "work" / "made.txt").read_text() == "x"
    assert Path.cwd() == before


def test_the_namespace_is_shared_within_a_page_and_fresh_for_every_page(tmp_path: Path) -> None:
    page = _page(
        tmp_path,
        """
        .. code-block:: python

           import json
           total = 40
           leaked = 1

        .. code-block:: python

           total += 2
           assert json.dumps(total) == "42"
        """,
    )
    _run(page, tmp_path)
    reading = _page(tmp_path, ".. code-block:: python\n\n   print(leaked)\n", "reading.rst")
    with pytest.raises(SnippetError, match="NameError"):
        _run(reading, tmp_path)


def test_a_class_defined_in_a_block_can_be_pickled_and_the_module_is_dropped_afterwards(tmp_path: Path) -> None:
    page = _page(
        tmp_path,
        """
        .. code-block:: python

           import pickle

           class Thing:
               pass

           assert isinstance(pickle.loads(pickle.dumps(Thing())), Thing)
        """,
    )
    before = set(sys.modules)
    _run(page, tmp_path)
    assert not {name for name in set(sys.modules) - before if name.startswith("_docs_snippets_")}


def test_a_non_drevalpy_bash_line_is_rejected_before_anything_runs(tmp_path: Path) -> None:
    page = _page(
        tmp_path,
        """
        .. code-block:: python

           open("ran", "w").close()

        .. code-block:: bash

           drevalpy list splitters
           pip install drevalpy
        """,
    )
    with pytest.raises(SnippetError) as caught:
        _run(page, tmp_path)
    assert _failing_line(page, caught.value) == "pip install drevalpy"
    assert not (tmp_path / "work" / "ran").exists()


@pytest.mark.parametrize(
    "line",
    [
        "drevalpy list splitters | head",
        "drevalpy list splitters > out.txt",
        "drevalpy list splitters && drevalpy list predictors",
        "drevalpy list splitters; ls",
        "drevalpy run --dataset $DATASET",
        "$ drevalpy list splitters",
    ],
)
def test_shell_syntax_is_rejected(tmp_path: Path, line: str) -> None:
    with pytest.raises(SnippetError):
        _run(_page(tmp_path, f".. code-block:: bash\n\n   {line}\n"), tmp_path)


@pytest.mark.parametrize("language", ["sh", "shell", "zsh", "python3", "py"])
def test_a_misnamed_language_is_rejected_instead_of_silently_skipped(tmp_path: Path, language: str) -> None:
    with pytest.raises(SnippetError, match="code-block::"):
        _run(_page(tmp_path, f".. code-block:: {language}\n\n   x = 1\n"), tmp_path)


@pytest.mark.slow
class TestBashCommands:
    def test_commands_run_through_the_cli_with_comments_and_continuations(self, tmp_path: Path) -> None:
        page = _page(
            tmp_path,
            """
            .. code-block:: bash

               # list the split modes
               drevalpy list splitters

               drevalpy \\
                   list \\
                   predictors
            """,
        )
        _run(page, tmp_path)

    def test_a_failing_command_reports_its_line_and_the_cli_output(self, tmp_path: Path) -> None:
        page = _page(tmp_path, ".. code-block:: bash\n\n   drevalpy list splitters\n   drevalpy list no-such-listing\n")
        with pytest.raises(SnippetError) as caught:
            _run(page, tmp_path)
        assert _failing_line(page, caught.value) == "drevalpy list no-such-listing"
        assert "exit code 2" in caught.value.output

    def test_quoted_arguments_reach_the_cli_as_one_argument(self, tmp_path: Path) -> None:
        page = _page(tmp_path, '.. code-block:: bash\n\n   drevalpy run --models "a b" --nonexistent-flag\n')
        with pytest.raises(SnippetError, match="nonexistent-flag"):
            _run(page, tmp_path)
