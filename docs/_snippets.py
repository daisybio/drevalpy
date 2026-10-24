"""Find and run the code blocks of tutorial and how-to pages.

A page is plain reStructuredText, so the runner reads its ``code-block``
directives at any indent (inside a ``tab-item``, a ``note``, ...) instead of
asking each test to re-implement the extraction. ``python`` blocks of one page
share a namespace and run in page order; ``bash`` blocks are lists of
``drevalpy ...`` commands run in-process through Typer's ``CliRunner``. Both run
inside a caller-supplied working directory, so a block that writes
``toy.h5mu`` finds it again in the next block.

A block opts out with an RST comment on the line before it (blank lines
between the comment and the block are fine)::

    .. snippet: skip

    .. code-block:: bash

       pip install drevalpy

``skip`` means the block is never run. ``network`` means it runs only when
``allow_network`` is true, for blocks that download built-in datasets.
``posix`` means it is not run on Windows, for blocks that write Plotly PNGs
(Kaleido 0.2.1 hangs there).
"""

from __future__ import annotations

import contextlib
import io
import itertools
import re
import shlex
import sys
import textwrap
import traceback
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

Marker = Literal["run", "skip", "network", "posix"]

#: Languages the runner executes. Any other ``code-block`` (``yaml``, ``text``, ...) is output or data.
RUNNABLE_LANGUAGES = frozenset({"python", "bash"})

#: Spellings that look runnable but are not, mapped to the spelling that is.
MISNAMED_LANGUAGES = {"sh": "bash", "shell": "bash", "zsh": "bash", "python3": "python", "py": "python"}

_MARKER_VALUES = ("skip", "network", "posix")
_DIRECTIVE = re.compile(r"^(?P<indent> *)\.\. (?:code-block|sourcecode|code):: *(?P<language>\S*) *$")
_MARKER = re.compile(r"^(?P<indent> *)\.\. snippet: *(?P<value>\S*) *$")
_OPTION = re.compile(r"^ +:[\w-]+:")
#: Tokens a plain ``drevalpy`` command line never contains; a shell would act on them, the runner would not.
_SHELL_OPERATORS = frozenset("();<>|&")
_PAGE_COUNTER = itertools.count()


class SnippetError(Exception):
    """A snippet that cannot be collected, or that failed when run.

    Attributes:
        page: The page the snippet is on.
        line: 1-based line in *page* of the offending statement or command.
        output: Captured output followed by the traceback or CLI error text.
    """

    def __init__(self, page: Path, line: int, output: str) -> None:
        """Store where the snippet failed and what it printed.

        Args:
            page: The page the snippet is on.
            line: 1-based line in *page* of the offending statement or command.
            output: Captured output followed by the traceback or CLI error text.
        """
        super().__init__(f"{page}:{line}\n{output}")
        self.page = page
        self.line = line
        self.output = output


@dataclass(frozen=True)
class Snippet:
    """One ``code-block`` of a page.

    Attributes:
        page: The page the block is on.
        directive_line: 1-based line of the ``.. code-block::`` directive.
        line: 1-based line of the block's first code line.
        indent: Column of the directive, i.e. how deeply it is nested.
        language: Lower-cased language argument of the directive, ``""`` when absent.
        code: The block body with its indentation removed.
        marker: ``"run"`` for an unmarked block, otherwise the marker comment's value.
    """

    page: Path
    directive_line: int
    line: int
    indent: int
    language: str
    code: str
    marker: Marker


def collect(page: Path) -> list[Snippet]:
    """List every code block of *page* in page order.

    Blocks nested inside another code block (an ``rst`` example that shows the
    directive) are part of that block's text, not snippets of their own.

    Args:
        page: Path of an ``.rst`` file.

    Returns:
        One :class:`Snippet` per block, whatever its language.

    Raises:
        SnippetError: If a ``.. snippet:`` comment has an unknown value or is not
            directly followed by a code block.
    """
    page = Path(page)
    lines = page.read_text(encoding="utf-8").splitlines()
    snippets: list[Snippet] = []
    pending: tuple[int, Marker] | None = None
    index = 0
    while index < len(lines):
        marker = _MARKER.match(lines[index])
        directive = _DIRECTIVE.match(lines[index])
        if marker:
            pending = _read_marker(page, lines, index, marker)
            index += 1
        elif directive:
            marked = pending[1] if pending and pending[0] == index else "run"
            snippet, index = _read_block(page, lines, index, directive, marked)
            snippets.append(snippet)
        else:
            index += 1
    return snippets


def run_page(page: Path, workdir: Path, *, allow_network: bool) -> None:
    """Run the python and bash blocks of *page* in page order.

    Every command is validated before anything runs, so a stray shell line fails
    in milliseconds instead of after the page's training steps.

    Args:
        page: Path of an ``.rst`` file.
        workdir: Directory the blocks run in; created if missing and entered for the whole run.
        allow_network: Whether ``network`` blocks run (they are skipped otherwise).

    Raises:
        SnippetError: If a block cannot be collected, a bash line is not a plain
            ``drevalpy`` command, or a block raises or exits non-zero.
    """
    page = Path(page)
    plan = _plan(page, allow_network=allow_network)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    namespace = types.ModuleType(f"_docs_snippets_{next(_PAGE_COUNTER)}_{page.stem}")
    namespace.__file__ = str(page)
    sys.modules[namespace.__name__] = namespace
    try:
        with contextlib.chdir(workdir):
            for snippet, commands in plan:
                if commands is None:
                    _run_python(snippet, namespace)
                else:
                    for line, argv in commands:
                        _run_command(snippet.page, line, argv)
    finally:
        sys.modules.pop(namespace.__name__, None)


def _read_marker(page: Path, lines: list[str], index: int, match: re.Match[str]) -> tuple[int, Marker]:
    value = match.group("value")
    if value not in _MARKER_VALUES:
        raise SnippetError(page, index + 1, f"unknown snippet marker {value!r}; use one of {_MARKER_VALUES}")
    following = next((i for i in range(index + 1, len(lines)) if lines[i].strip()), len(lines))
    block = _DIRECTIVE.match(lines[following]) if following < len(lines) else None
    if block is None or block.group("indent") != match.group("indent"):
        raise SnippetError(
            page, index + 1, f"'.. snippet: {value}' must be followed by a code-block at the same indent"
        )
    return following, cast("Marker", value)


def _read_block(
    page: Path, lines: list[str], index: int, directive: re.Match[str], marker: Marker
) -> tuple[Snippet, int]:
    indent = len(directive.group("indent"))
    cursor = index + 1
    while cursor < len(lines) and _OPTION.match(lines[cursor]) and _indent_of(lines[cursor]) > indent:
        cursor += 1
    body_start = cursor
    while cursor < len(lines) and (not lines[cursor].strip() or _indent_of(lines[cursor]) > indent):
        cursor += 1
    body = lines[body_start:cursor]
    filled = [offset for offset, text in enumerate(body) if text.strip()]
    first, last = (filled[0], filled[-1]) if filled else (0, -1)
    snippet = Snippet(
        page=page,
        directive_line=index + 1,
        line=body_start + first + 1,
        indent=indent,
        language=directive.group("language").lower(),
        code=textwrap.dedent("\n".join(body[first : last + 1])),
        marker=marker,
    )
    return snippet, cursor


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _plan(page: Path, *, allow_network: bool) -> list[tuple[Snippet, list[tuple[int, list[str]]] | None]]:
    """Select the blocks to run; a bash block carries its parsed commands, a python block ``None``."""
    plan: list[tuple[Snippet, list[tuple[int, list[str]]] | None]] = []
    for snippet in collect(page):
        if snippet.marker == "skip" or (snippet.marker == "network" and not allow_network):
            continue
        if snippet.marker == "posix" and sys.platform == "win32":
            continue
        if snippet.language in MISNAMED_LANGUAGES:
            correct = MISNAMED_LANGUAGES[snippet.language]
            raise SnippetError(
                page, snippet.directive_line, f"write 'code-block:: {correct}', not {snippet.language!r}"
            )
        if snippet.language == "python":
            plan.append((snippet, None))
        elif snippet.language == "bash":
            plan.append((snippet, _commands(snippet)))
    return plan


def _commands(snippet: Snippet) -> list[tuple[int, list[str]]]:
    """Split a bash block into ``(line, argv)`` pairs, rejecting anything but plain ``drevalpy`` commands."""
    commands: list[tuple[int, list[str]]] = []
    for line, text in _logical_lines(snippet):
        try:
            lexer = shlex.shlex(text, posix=True, punctuation_chars=True)
            lexer.whitespace_split = True
            argv = list(lexer)
        except ValueError as error:
            raise SnippetError(snippet.page, line, f"cannot split {text!r}: {error}") from error
        if not argv:
            continue
        if argv[0] != "drevalpy":
            raise SnippetError(snippet.page, line, f"bash lines must start with 'drevalpy', got {text!r}")
        if any(token and set(token) <= _SHELL_OPERATORS or "$" in token or "`" in token for token in argv):
            raise SnippetError(
                snippet.page, line, f"{text!r} uses shell syntax (pipes, redirects, expansion); only plain commands run"
            )
        commands.append((line, argv))
    return commands


def _logical_lines(snippet: Snippet) -> list[tuple[int, str]]:
    """Join backslash continuations and drop blank and comment lines; each result keeps its first line number."""
    joined: list[tuple[int, str]] = []
    carry: tuple[int, str] | None = None
    for offset, raw in enumerate(snippet.code.splitlines()):
        number, text = carry if carry else (snippet.line + offset, "")
        text += raw.strip()
        if text.endswith("\\"):
            carry = (number, text[:-1].rstrip() + " ")
            continue
        carry = None
        if text and not text.startswith("#"):
            joined.append((number, text))
    if carry:
        joined.append((carry[0], carry[1].strip()))
    return joined


def _run_python(snippet: Snippet, namespace: types.ModuleType) -> None:
    filename = str(snippet.page)
    captured = io.StringIO()
    try:
        # Padding with newlines makes every line number in a traceback a line of the page.
        code = compile("\n" * (snippet.line - 1) + snippet.code, filename, "exec")
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            exec(code, namespace.__dict__)  # noqa: S102 - trusted repository docs
    except (Exception, SystemExit) as error:
        line = _failing_line(error, filename) or snippet.line
        text = captured.getvalue() + "".join(traceback.format_exception(error))
        raise SnippetError(snippet.page, line, text) from error


def _failing_line(error: BaseException, filename: str) -> int | None:
    if isinstance(error, SyntaxError) and error.filename == filename:
        return error.lineno
    line = None
    frame = error.__traceback__
    while frame is not None:
        if frame.tb_frame.f_code.co_filename == filename:
            line = frame.tb_lineno
        frame = frame.tb_next
    return line


def _run_command(page: Path, line: int, argv: list[str]) -> None:
    from typer.testing import CliRunner

    from drevalpy.cli.main import app

    result = CliRunner().invoke(app, argv[1:])
    if result.exit_code == 0:
        return
    detail = result.output
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        detail += "".join(traceback.format_exception(result.exception))
    raise SnippetError(page, line, f"$ {shlex.join(argv)}\nexit code {result.exit_code}\n{detail}")
