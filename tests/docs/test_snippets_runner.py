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


class TestCollect:
    def test_finds_a_block_inside_a_tab_item(self, tmp_path: Path) -> None:
        page = _page(
            tmp_path,
            """
            Title
            =====

            .. tab-set::
               :sync-group: interface

               .. tab-item:: CLI
                  :sync: cli

                  .. code-block:: bash

                     drevalpy list splitters

               .. tab-item:: Python
                  :sync: python

                  .. code-block:: python
                     :caption: A caption

                     x = 1
                     y = x + 1
            """,
        )
        cli, python = collect(page)
        assert (cli.language, cli.code, cli.indent) == ("bash", "drevalpy list splitters", 6)
        assert (python.language, python.code, python.indent) == ("python", "x = 1\ny = x + 1", 6)
        assert page.read_text().splitlines()[cli.line - 1] == "         drevalpy list splitters"
        assert page.read_text().splitlines()[python.line - 1] == "         x = 1"
        assert page.read_text().splitlines()[python.directive_line - 1].strip() == ".. code-block:: python"

    def test_returns_every_language_and_ignores_other_directives(self, tmp_path: Path) -> None:
        page = _page(
            tmp_path,
            """
            .. note::

               Not a code block.

            .. code-block:: yaml

               key: value

            .. code-block:: text

               output

            .. code-block:: python

               z = 1

            Closing paragraph.
            """,
        )
        assert [(s.language, s.code) for s in collect(page)] == [
            ("yaml", "key: value"),
            ("text", "output"),
            ("python", "z = 1"),
        ]

    def test_a_directive_shown_inside_another_block_is_not_a_snippet(self, tmp_path: Path) -> None:
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

    def test_markers_are_read_from_the_comment_before_the_block(self, tmp_path: Path) -> None:
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

    def test_an_unknown_marker_is_an_error_naming_the_line(self, tmp_path: Path) -> None:
        page = _page(
            tmp_path,
            """
            Intro.

            .. snippet: skipp
            .. code-block:: python

               pass
            """,
        )
        with pytest.raises(SnippetError) as caught:
            collect(page)
        assert (caught.value.page, caught.value.line) == (page, 3)
        assert "skipp" in caught.value.output

    def test_a_marker_without_a_block_is_an_error(self, tmp_path: Path) -> None:
        page = _page(
            tmp_path,
            """
            .. snippet: skip

            A paragraph, not a code block.
            """,
        )
        with pytest.raises(SnippetError, match="must be followed by a code-block"):
            collect(page)


class TestMarkersWhenRunning:
    PAGE = """
        .. snippet: skip
        .. code-block:: python

           open("skipped", "w").close()

        .. snippet: network

        .. code-block:: python

           open("networked", "w").close()

        .. code-block:: python

           open("plain", "w").close()
        """

    def test_skip_and_network_blocks_do_not_run_offline(self, tmp_path: Path) -> None:
        _run(_page(tmp_path, self.PAGE), tmp_path)
        assert sorted(os.listdir(tmp_path / "work")) == ["plain"]

    def test_network_blocks_run_when_allowed_but_skip_blocks_never_do(self, tmp_path: Path) -> None:
        _run(_page(tmp_path, self.PAGE), tmp_path, allow_network=True)
        assert sorted(os.listdir(tmp_path / "work")) == ["networked", "plain"]

    def test_a_skipped_block_may_contain_anything(self, tmp_path: Path) -> None:
        page = _page(
            tmp_path,
            """
            .. snippet: skip
            .. code-block:: bash

               pip install drevalpy | tee log

            .. snippet: skip
            .. code-block:: shell

               echo hello
            """,
        )
        _run(page, tmp_path)


class TestPythonBlocks:
    def test_a_failing_block_reports_page_line_and_traceback(self, tmp_path: Path) -> None:
        page = _page(
            tmp_path,
            """
            Intro.

            .. code-block:: python

               value = 1

            More prose.

            .. code-block:: python

               print("before the failure")
               other = 2
               raise ValueError("boom")
            """,
        )
        with pytest.raises(SnippetError) as caught:
            _run(page, tmp_path)
        error = caught.value
        assert error.page == page
        assert page.read_text().splitlines()[error.line - 1].strip() == 'raise ValueError("boom")'
        assert "ValueError: boom" in error.output
        assert "before the failure" in error.output
        assert f"{page}:{error.line}" in str(error)

    def test_a_syntax_error_reports_its_line(self, tmp_path: Path) -> None:
        page = _page(
            tmp_path,
            """
            .. code-block:: python

               fine = 1
               not valid python
            """,
        )
        with pytest.raises(SnippetError) as caught:
            _run(page, tmp_path)
        assert page.read_text().splitlines()[caught.value.line - 1].strip() == "not valid python"

    def test_sys_exit_in_a_block_is_a_failure(self, tmp_path: Path) -> None:
        page = _page(tmp_path, ".. code-block:: python\n\n   raise SystemExit(3)\n")
        with pytest.raises(SnippetError):
            _run(page, tmp_path)

    def test_blocks_run_in_a_working_directory_that_is_restored(self, tmp_path: Path) -> None:
        before = Path.cwd()
        page = _page(tmp_path, ".. code-block:: python\n\n   open('made.txt', 'w').write('x')\n")
        _run(page, tmp_path)
        assert (tmp_path / "work" / "made.txt").read_text() == "x"
        assert Path.cwd() == before

    def test_the_directory_is_restored_after_a_failure(self, tmp_path: Path) -> None:
        before = Path.cwd()
        page = _page(tmp_path, ".. code-block:: python\n\n   raise RuntimeError\n")
        with pytest.raises(SnippetError):
            _run(page, tmp_path)
        assert Path.cwd() == before


class TestNamespaces:
    def test_the_namespace_is_shared_within_a_page(self, tmp_path: Path) -> None:
        page = _page(
            tmp_path,
            """
            .. code-block:: python

               import json
               total = 40

            .. tab-set::

               .. tab-item:: Python

                  .. code-block:: python

                     total += 2
                     assert json.dumps(total) == "42"
            """,
        )
        _run(page, tmp_path)

    def test_the_namespace_is_fresh_for_every_page(self, tmp_path: Path) -> None:
        defining = _page(tmp_path, ".. code-block:: python\n\n   leaked = 1\n", "defining.rst")
        reading = _page(tmp_path, ".. code-block:: python\n\n   print(leaked)\n", "reading.rst")
        _run(defining, tmp_path)
        with pytest.raises(SnippetError, match="NameError"):
            _run(reading, tmp_path)

    def test_a_class_defined_in_a_block_can_be_pickled_and_the_module_is_dropped_afterwards(
        self, tmp_path: Path
    ) -> None:
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


class TestBashBlocks:
    def test_a_line_that_is_not_a_drevalpy_command_is_rejected(self, tmp_path: Path) -> None:
        page = _page(
            tmp_path,
            """
            .. code-block:: bash

               drevalpy list splitters
               pip install drevalpy
            """,
        )
        with pytest.raises(SnippetError) as caught:
            _run(page, tmp_path)
        assert page.read_text().splitlines()[caught.value.line - 1].strip() == "pip install drevalpy"
        assert "drevalpy" in caught.value.output

    def test_nothing_runs_when_a_later_block_is_rejected(self, tmp_path: Path) -> None:
        page = _page(
            tmp_path,
            """
            .. code-block:: python

               open("ran", "w").close()

            .. code-block:: bash

               ls
            """,
        )
        with pytest.raises(SnippetError):
            _run(page, tmp_path)
        assert not (tmp_path / "work" / "ran").exists()

    @pytest.mark.parametrize(
        "line",
        [
            "drevalpy list splitters | head",
            "drevalpy list splitters > out.txt",
            "drevalpy list splitters && drevalpy list predictors",
            "drevalpy list splitters; ls",
            "drevalpy run --dataset $DATASET",
            "drevalpy run --dataset $(pwd)/toy.h5mu",
            "$ drevalpy list splitters",
        ],
    )
    def test_shell_syntax_is_rejected(self, tmp_path: Path, line: str) -> None:
        page = _page(tmp_path, f".. code-block:: bash\n\n   {line}\n")
        with pytest.raises(SnippetError):
            _run(page, tmp_path)

    @pytest.mark.parametrize("language", ["sh", "shell", "zsh", "python3", "py"])
    def test_a_misnamed_language_is_rejected_instead_of_silently_skipped(self, tmp_path: Path, language: str) -> None:
        page = _page(tmp_path, f".. code-block:: {language}\n\n   x = 1\n")
        with pytest.raises(SnippetError, match="code-block::"):
            _run(page, tmp_path)


class TestBashCommands:
    """Extended tier: the first command imports the whole CLI, which costs more than 0.2s."""

    pytestmark = pytest.mark.slow

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
        page = _page(
            tmp_path,
            """
            .. code-block:: bash

               drevalpy list splitters
               drevalpy list no-such-listing
            """,
        )
        with pytest.raises(SnippetError) as caught:
            _run(page, tmp_path)
        assert page.read_text().splitlines()[caught.value.line - 1].strip() == "drevalpy list no-such-listing"
        assert "exit code 2" in caught.value.output
        assert "drevalpy list no-such-listing" in caught.value.output

    def test_quoted_arguments_reach_the_cli_as_one_argument(self, tmp_path: Path) -> None:
        page = _page(tmp_path, '.. code-block:: bash\n\n   drevalpy run --models "a b" --nonexistent-flag\n')
        with pytest.raises(SnippetError, match="nonexistent-flag"):
            _run(page, tmp_path)
