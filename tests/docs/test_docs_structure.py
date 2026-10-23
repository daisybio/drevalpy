"""Policy checks for the Diátaxis documentation tree.

The rules are plain functions over a docs root, so the tests below run each one
twice: against the real ``docs/`` tree, and against tiny trees written to
``tmp_path`` that break the rule on purpose. The second half is what keeps a
guard from passing only because the pages it should police are stubs.

"Built" pages are the ``.rst`` files Sphinx reads: everything under ``docs/``
that ``exclude_patterns`` in ``docs/conf.py`` does not match.
"""

from __future__ import annotations

import ast
import json
import posixpath
import re
import sys
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from tests._trusted_subprocess import run_trusted_python

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs"
ZOO_DIR = REPO_ROOT / "src" / "drevalpy" / "models" / "zoo"

if str(DOCS) not in sys.path:
    sys.path.insert(0, str(DOCS))

from _snippets import MISNAMED_LANGUAGES, RUNNABLE_LANGUAGES, collect  # noqa: E402

#: Top-level directories a built page may live in (besides the root ``index.rst``).
SECTIONS = ("tutorials", "how-to", "reference", "explanation", "project")

#: Directories Sphinx never reads pages from, whatever ``exclude_patterns`` says.
NON_SOURCE_DIRS = frozenset({"_autosummary", "_build", "_static", "_templates"})

#: Sections whose pages declare an ``.. interface:`` comment and are checked against it.
INTERFACE_SECTIONS = ("tutorials", "how-to")
INTERFACES = ("both", "cli", "python")

#: First words a how-to title may start with: the task, as a verb.
HOWTO_VERBS = frozenset(
    {
        "Add",
        "Build",
        "Choose",
        "Combine",
        "Compare",
        "Configure",
        "Create",
        "Evaluate",
        "Install",
        "List",
        "Load",
        "Make",
        "Migrate",
        "Port",
        "Publish",
        "Register",
        "Run",
        "Save",
        "Test",
        "Train",
        "Tune",
        "Use",
        "Write",
    }
)

#: Every language tag that a reader would take for a runnable ``python`` or ``bash`` block.
CODE_LANGUAGES = RUNNABLE_LANGUAGES | frozenset(MISNAMED_LANGUAGES)

#: Generated files each hand-written reference page must pull in with ``.. include::``.
GENERATED_INCLUDES = {
    "reference/cli.rst": ("_generated_cli.rst",),
    "reference/components.rst": (
        "_generated_sample_featurizers.rst",
        "_generated_drug_featurizers.rst",
        "_generated_predictors.rst",
    ),
    "reference/model_zoo.rst": ("_generated_model_zoo.rst",),
    "reference/examples.rst": ("_generated_examples.rst",),
}

#: Sub-apps that must exist. Kept explicit so removing one fails, while the
#: docs-coverage and shell-example checks derive the real list from the live app
#: and therefore cannot go stale when a sub-app is added.
REQUIRED_CLI_GROUPS = {"data", "experiments", "list"}

_HOWTO_TITLE_RULE = r"=\-~^\"#*+`.'"
_TITLE = re.compile(rf"^(?P<title>\S[^\n]*)\n[{_HOWTO_TITLE_RULE}]{{3,}} *$", re.MULTILINE)
_INTERFACE = re.compile(r"^\.\. interface:[ \t]*(\S*)[ \t]*$", re.MULTILINE)
_DIRECTIVE = re.compile(r"^(?P<indent> *)\.\. (?P<name>[\w:-]+)::(?P<argument>.*)$")
_OPTION = re.compile(r"^ +:(?P<key>[\w-]+):[ \t]*(?P<value>.*)$")
_TOCTREE = re.compile(r"^(?P<indent> *)\.\. toctree::\s*$")
_DREVALPY_LINE = re.compile(r"^\s*drevalpy\b", re.MULTILINE)


def _cli_command_groups() -> tuple[str, ...]:
    """Return the names of the CLI's sub-apps, read from the live Typer app.

    Derived rather than hand-listed: an allowlist here silently stops matching
    when a new sub-app lands, which is exactly what happened to the ``list``
    group. Imported lazily so collecting this module does not pull in the CLI.

    :returns: Every registered sub-app name, e.g. ``("data", "experiments", "list")``.
    """
    from drevalpy.cli.main import app

    return tuple(sorted(group.name for group in app.registered_groups if group.name))


# --------------------------------------------------------------------------- #
# The rules                                                                   #
# --------------------------------------------------------------------------- #


def _matches(pattern: str, path: str) -> bool:
    """Match *path* against a Sphinx pattern: ``*`` and ``?`` stay inside one path segment, ``**`` crosses them."""
    regex = re.escape(pattern)
    regex = regex.replace(r"\*\*", "\0").replace(r"\*", "[^/]*").replace(r"\?", "[^/]").replace("\0", ".*")
    return re.fullmatch(regex, path) is not None


def exclude_patterns(docs: Path) -> list[str]:
    """Read the ``exclude_patterns`` list literal out of ``docs/conf.py`` without running it.

    Running ``conf.py`` regenerates the reference files, which a policy test must not do.

    :param docs: The docs root.
    :returns: The patterns, as Sphinx would see them.
    """
    tree = ast.parse((docs / "conf.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "exclude_patterns" for t in node.targets):
            return list(ast.literal_eval(node.value))
    msg = "conf.py has no literal 'exclude_patterns' assignment"
    raise AssertionError(msg)


def built_pages(docs: Path, patterns: Iterable[str]) -> list[Path]:
    """List the ``.rst`` pages Sphinx would build, sorted.

    A pattern excludes a file or a directory and everything below it.

    :param docs: The docs root.
    :param patterns: ``exclude_patterns`` from ``conf.py``.
    :returns: Paths of the built pages.
    """
    patterns = list(patterns)
    pages = []
    for path in sorted(docs.rglob("*.rst")):
        parts = path.relative_to(docs).as_posix().split("/")
        if NON_SOURCE_DIRS & set(parts[:-1]):
            continue
        prefixes = ("/".join(parts[:end]) for end in range(1, len(parts) + 1))
        if any(_matches(pattern, prefix) for prefix in prefixes for pattern in patterns):
            continue
        pages.append(path)
    return pages


def misplaced_pages(docs: Path, pages: Iterable[Path]) -> list[str]:
    """Return the pages outside the four modes and Project, other than the root ``index.rst``."""
    misplaced = []
    for page in pages:
        parts = page.relative_to(docs).parts
        if parts == ("index.rst",):
            continue
        if len(parts) < 2 or parts[0] not in SECTIONS:
            misplaced.append("/".join(parts))
    return misplaced


def _resolve_entry(own_dir: str, entry: str, docnames: set[str], *, is_glob: bool) -> set[str]:
    linked = re.fullmatch(r".*<(.+)>", entry)
    target = (linked.group(1) if linked else entry).strip()
    if "://" in target or target == "self":
        return set()
    resolved = posixpath.normpath(target.lstrip("/") if target.startswith("/") else posixpath.join(own_dir, target))
    if is_glob:
        return {name for name in docnames if _matches(resolved, name)}
    return {resolved}


def unreachable_pages(docs: Path, pages: Iterable[Path]) -> list[str]:
    """Return the docnames that no chain of toctrees from ``index`` reaches."""
    by_name = {page.relative_to(docs).as_posix().removesuffix(".rst"): page for page in pages}
    names = set(by_name)
    seen, queue = {"index"}, deque(["index"])
    while queue:
        name = queue.popleft()
        if name not in by_name:
            continue
        for target in toctree_targets(by_name[name], docs, names):
            if target not in seen:
                seen.add(target)
                queue.append(target)
    return sorted(names - seen)


def toctree_targets(page: Path, docs: Path, docnames: set[str]) -> set[str]:
    """Return the docnames *page*'s toctrees point at.

    Understands ``Title <target>`` entries, absolute (``/x``) and relative
    targets, and ``:glob:`` trees. External links and ``self`` are dropped.

    :param page: The page to read.
    :param docs: The docs root, which relative targets resolve against.
    :param docnames: Every built docname, for resolving ``:glob:`` entries.
    :returns: Docnames, which may not exist (Sphinx's own build reports those).
    """
    directory = page.parent.relative_to(docs).as_posix()
    directory = "" if directory == "." else directory
    lines = page.read_text(encoding="utf-8").splitlines()
    return _targets_from_lines(lines, directory, docnames)


def _targets_from_lines(lines: list[str], directory: str, docnames: set[str]) -> set[str]:
    targets: set[str] = set()
    for start, line in enumerate(lines):
        match = _TOCTREE.match(line)
        if not match:
            continue
        indent = len(match.group("indent"))
        entries: list[str] = []
        is_glob = False
        for follower in lines[start + 1 :]:
            if not follower.strip():
                continue
            if len(follower) - len(follower.lstrip(" ")) <= indent:
                break
            text = follower.strip()
            if text.startswith(":"):
                is_glob = is_glob or text.startswith(":glob:")
            else:
                entries.append(text)
        for entry in entries:
            targets |= _resolve_entry(directory, entry, docnames, is_glob=is_glob)
    return targets


def page_title(page: Path) -> str | None:
    """Return the page's first section title, or ``None`` when it has none."""
    match = _TITLE.search(page.read_text(encoding="utf-8"))
    return match.group("title").strip() if match else None


def howto_title_violations(docs: Path, pages: Iterable[Path]) -> list[str]:
    """Return the how-to pages whose title does not start with a verb of :data:`HOWTO_VERBS`."""
    violations = []
    for page in pages:
        rel = page.relative_to(docs)
        if rel.parts[0] != "how-to" or rel.name == "index.rst":
            continue
        title = page_title(page)
        first_word = re.match(r"[A-Za-z]+", title or "")
        if first_word is None or first_word[0] not in HOWTO_VERBS:
            violations.append(f"{rel.as_posix()}: title {title!r} does not start with a verb")
    return violations


def code_snippets(page: Path) -> list:
    """Return the python and bash blocks of *page*, whatever marker they carry."""
    return [snippet for snippet in collect(page) if snippet.language in CODE_LANGUAGES]


def explanation_violations(docs: Path, pages: Iterable[Path]) -> list[str]:
    """Return the ``python`` or ``bash`` blocks found on explanation pages."""
    return [
        f"{page.relative_to(docs).as_posix()}:{snippet.line}: {snippet.language} block on an explanation page"
        for page in pages
        if page.relative_to(docs).parts[0] == "explanation"
        for snippet in code_snippets(page)
    ]


def page_interface(page: Path) -> str:
    """Return the page's ``.. interface:`` value, ``"both"`` when it declares none."""
    match = _INTERFACE.search(page.read_text(encoding="utf-8"))
    return match.group(1) if match else "both"


@dataclass(frozen=True)
class _Container:
    name: str
    index: int
    indent: int


def _enclosing(lines: list[str], index: int, indent: int) -> list[_Container]:
    """Return the directives that contain the line at *index*, innermost first."""
    chain = []
    for cursor in range(index - 1, -1, -1):
        line = lines[cursor]
        if not line.strip() or len(line) - len(line.lstrip(" ")) >= indent:
            continue
        match = _DIRECTIVE.match(line)
        if match:
            indent = len(match.group("indent"))
            chain.append(_Container(match.group("name"), cursor, indent))
    return chain


def _options(lines: list[str], index: int) -> dict[str, str]:
    options = {}
    for line in lines[index + 1 :]:
        match = _OPTION.match(line)
        if not match:
            break
        options[match.group("key")] = match.group("value").strip()
    return options


def _tab_syncs(lines: list[str], tab_set: _Container) -> set[str]:
    """Return the ``:sync:`` values of the tab-set's own tab-items (not those of nested tab-sets)."""
    items = []
    for cursor in range(tab_set.index + 1, len(lines)):
        line = lines[cursor]
        if line.strip() and len(line) - len(line.lstrip(" ")) <= tab_set.indent:
            break
        match = _DIRECTIVE.match(line)
        if match and match.group("name") == "tab-item":
            items.append((len(match.group("indent")), _options(lines, cursor).get("sync", "")))
    shallowest = min((indent for indent, _ in items), default=0)
    return {sync for indent, sync in items if indent == shallowest}


def _tab_problem(lines: list[str], directive_line: int, indent: int) -> str | None:
    chain = _enclosing(lines, directive_line - 1, indent)
    tab_set = next((c for c in chain if c.name == "tab-set"), None)
    if tab_set is None or not any(c.name == "tab-item" for c in chain[: chain.index(tab_set)]):
        return "not inside a tab-item of a tab-set"
    if _options(lines, tab_set.index).get("sync-group") != "interface":
        return "its tab-set lacks ':sync-group: interface'"
    missing = {"cli", "python"} - _tab_syncs(lines, tab_set)
    if missing:
        return f"its tab-set lacks the tab(s) with ':sync: {', '.join(sorted(missing))}'"
    return None


def interface_violations(page: Path) -> list[str]:
    """Check *page* against its declared ``.. interface:``.

    - ``both``: every python or bash block sits in a ``tab-set`` with ``:sync-group: interface``
      that has a ``:sync: cli`` and a ``:sync: python`` tab.
    - ``python``: no bash block runs ``drevalpy``.
    - ``cli``: no python block.

    :param page: A tutorial or how-to page.
    :returns: One message per violation, each starting ``path:line``.
    """
    interface = page_interface(page)
    if interface not in INTERFACES:
        return [f"{page}: unknown '.. interface: {interface}'; use one of {INTERFACES}"]
    lines = page.read_text(encoding="utf-8").splitlines()
    violations = []
    for snippet in code_snippets(page):
        where = f"{page}:{snippet.directive_line}: {snippet.language} block"
        if interface == "both":
            problem = _tab_problem(lines, snippet.directive_line, snippet.indent)
            if problem:
                violations.append(f"{where} on an 'interface: both' page: {problem}")
        elif interface == "python" and snippet.language != "python" and _DREVALPY_LINE.search(snippet.code):
            violations.append(f"{where} runs 'drevalpy' on an 'interface: python' page")
        elif interface == "cli" and snippet.language == "python":
            violations.append(f"{where} on an 'interface: cli' page")
    return violations


def stale_label_violations(docs: Path, pages: Iterable[Path]) -> list[str]:
    """Return built pages other than ``project/news.rst`` that name 1.5.1 or the "modularity release"."""
    stale = []
    for page in pages:
        rel = page.relative_to(docs).as_posix()
        text = page.read_text(encoding="utf-8")
        if rel != "project/news.rst" and ("1.5.1" in text or "modularity release" in text.lower()):
            stale.append(rel)
    return stale


def missing_includes(docs: Path) -> list[str]:
    """Return ``page: file`` pairs where a reference page does not include its generated file."""
    missing = []
    for rel, includes in GENERATED_INCLUDES.items():
        text = (docs / rel).read_text(encoding="utf-8")
        missing += [f"{rel}: {name}" for name in includes if f".. include:: {name}" not in text]
    return missing


# --------------------------------------------------------------------------- #
# The real tree                                                               #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def pages() -> list[Path]:
    return built_pages(DOCS, exclude_patterns(DOCS))


def test_the_built_page_list_reads_conf_py(pages: list[Path]) -> None:
    names = {page.relative_to(DOCS).as_posix() for page in pages}
    assert "index.rst" in names
    assert {"tutorials/index.rst", "how-to/index.rst", "reference/index.rst", "explanation/index.rst"} <= names
    assert not {name for name in names if "_generated_" in name}, "generated files are includes, not pages"


def test_every_built_page_lives_in_a_section(pages: list[Path]) -> None:
    assert not misplaced_pages(DOCS, pages)


def test_every_built_page_is_reachable_from_a_toctree(pages: list[Path]) -> None:
    assert not unreachable_pages(DOCS, pages)


def test_explanation_pages_have_no_code_blocks(pages: list[Path]) -> None:
    assert not explanation_violations(DOCS, pages)


def test_how_to_titles_start_with_a_verb(pages: list[Path]) -> None:
    assert not howto_title_violations(DOCS, pages)


def test_tutorial_and_how_to_pages_follow_their_interface(pages: list[Path]) -> None:
    violations = [
        violation
        for page in pages
        if page.relative_to(DOCS).parts[0] in INTERFACE_SECTIONS
        for violation in interface_violations(page)
    ]
    assert not violations, "\n".join(violations)


def test_reference_pages_include_their_generated_files() -> None:
    assert not missing_includes(DOCS)


def test_compatibility_wording_avoids_stale_version_labels(pages: list[Path]) -> None:
    assert not stale_label_violations(DOCS, pages)


def test_zoo_presets_documented_in_model_zoo() -> None:
    from _model_zoo import generate_model_zoo_rst

    zoo_names = {path.stem for path in ZOO_DIR.glob("*.yaml")}
    catalog = (DOCS / "reference" / "model_zoo.rst").read_text(encoding="utf-8")
    assert "_generated_model_zoo.rst" in catalog
    generated = generate_model_zoo_rst()
    missing = sorted(name for name in zoo_names if name not in generated)
    assert not missing, f"Zoo presets missing from generated model zoo catalog: {missing}"
    assert "``scaledGeneExpression:singleDrugElasticNet``" in generated
    assert "``scaledGeneExpression:singleDrugRandomForest``" in generated
    assert "``molirOmics:molir``" in generated
    assert "``superfeltrOmics:superfeltr``" in generated


@pytest.mark.slow
def test_component_catalog_is_registry_driven_and_synchronized() -> None:
    """Extended tier: regenerates the catalog in a fresh interpreter (~2.2s).

    The child process is the point - the generator must produce the committed RST
    from a clean registry, not from whatever this session has registered.
    """
    from drevalpy.registry import _builtins as rb
    from drevalpy.registry.drug_featurizer import drug_featurizer_registry
    from drevalpy.registry.predictor import predictor_registry
    from drevalpy.registry.sample_featurizer import sample_featurizer_registry

    catalog = (DOCS / "reference" / "components.rst").read_text(encoding="utf-8")
    for include in (
        "_generated_sample_featurizers.rst",
        "_generated_drug_featurizers.rst",
        "_generated_predictors.rst",
    ):
        assert f".. include:: {include}" in catalog
    assert ".. list-table::" not in catalog
    assert "Example recipe" not in catalog

    script = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(DOCS)!r})\n"
        "from _component_catalog import generate_component_catalog_rsts\n"
        "sys.stdout.write(json.dumps(generate_component_catalog_rsts(), sort_keys=True))\n"
    )
    completed = run_trusted_python(script, cwd=str(REPO_ROOT))
    assert completed.returncode == 0, completed.stderr
    generated = json.loads(completed.stdout)

    rb.register_builtin_components()
    expected_featurizers = {
        "sample": {
            row["name"]: (
                row["output_format"],
                "\u2705" if row["precompute"] else "\u274c",
                " ".join(row["description"].split()),
            )
            for row in sample_featurizer_registry.list_metadata()
            if row["name"] in rb.BUILTIN_SAMPLE_FEATURIZER_NAMES
        },
        "drug": {
            row["name"]: (
                row["output_format"],
                "\u2705" if row["precompute"] else "\u274c",
                " ".join(row["description"].split()),
            )
            for row in drug_featurizer_registry.list_metadata()
            if row["name"] in rb.BUILTIN_DRUG_FEATURIZER_NAMES
        },
    }
    expected_predictors = {
        row["name"]: (
            row["input_interface"].replace("_", "-").capitalize(),
            " ".join(row["description"].split()),
        )
        for row in predictor_registry.list_metadata()
        if row["name"] in rb.BUILTIN_PREDICTOR_NAMES
    }
    for registry_name, expected in expected_featurizers.items():
        matches = re.findall(
            r"^   \* - ``([^`]+)``\n     - ``([^`]+)``\n     - ([^\n]*)\n     - ([^\n]+)$",
            generated[registry_name],
            flags=re.MULTILINE,
        )
        observed = {
            name: (output_format, precompute, description) for name, output_format, precompute, description in matches
        }
        assert observed == expected

    predictor_matches = re.findall(
        r"^   \* - ``([^`]+)``\n     - ([^\n]+)\n     - ([^\n]+)$",
        generated["predictor"],
        flags=re.MULTILINE,
    )
    observed_predictors = {name: (interface, description) for name, interface, description in predictor_matches}
    assert len(expected_featurizers["sample"]) == len(rb.BUILTIN_SAMPLE_FEATURIZER_NAMES) == 17
    assert len(expected_featurizers["drug"]) == len(rb.BUILTIN_DRUG_FEATURIZER_NAMES) == 10
    assert len(expected_predictors) == len(rb.BUILTIN_PREDICTOR_NAMES) == 28
    assert observed_predictors == expected_predictors


def test_cli_reference_documents_all_subcommands() -> None:
    from _cli_click import generate_cli_reference_rst
    from typer.main import get_command

    from drevalpy.cli.main import app

    reference = (DOCS / "reference" / "cli.rst").read_text(encoding="utf-8")
    assert "_generated_cli.rst" in reference

    generated = generate_cli_reference_rst()
    missing_in_docs = [cmd for cmd in _cli_command_groups() if f"drevalpy {cmd}" not in generated]
    assert not missing_in_docs, f"Generated CLI reference missing commands: {missing_in_docs}"

    click_app = get_command(app)
    names = set(getattr(click_app, "commands", {}))
    # A floor, not a mirror of the app: deriving both sides would make this
    # vacuous. It catches a group being dropped, while the docs check above
    # catches one being added without documentation.
    missing = sorted(REQUIRED_CLI_GROUPS - names)
    assert not missing, f"Typer app missing expected commands: {missing}"


def test_zoo_yaml_files_are_valid() -> None:
    for path in sorted(ZOO_DIR.glob("*.yaml")):
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert loaded is not None, path.name


@pytest.mark.parametrize(
    ("path", "forbidden"),
    [
        (DOCS / "reference" / "python_api.rst", "Backward compatibility"),
        (DOCS / "reference" / "cli.rst", "Backward compatibility"),
    ],
)
def test_generated_reference_pages_omit_compatibility(path: Path, forbidden: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert forbidden not in text


# --------------------------------------------------------------------------- #
# The rules against trees that break them                                     #
# --------------------------------------------------------------------------- #


def _tree(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _toctree(*entries: str, options: str = "") -> str:
    body = "\n".join(f"   {entry}" for entry in entries)
    return f".. toctree::\n{options}\n{body}\n"


BOTH_TABS = """\
Title
=====

.. interface: both

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         drevalpy list splitters

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         print(1)
"""


class TestBuiltPages:
    def test_a_pattern_excludes_files_directories_and_globs(self, tmp_path: Path) -> None:
        docs = _tree(
            tmp_path,
            {
                "index.rst": "x",
                "old/page.rst": "x",
                "old/deeper/page.rst": "x",
                "project/v2_news.rst": "x",
                "project/news.rst": "x",
                "reference/_generated_cli.rst": "x",
                "_build/html/x.rst": "x",
                "_templates/t.rst": "x",
            },
        )
        built = built_pages(docs, ["old", "project/v2_*", "reference/_generated_*.rst"])
        assert [p.relative_to(docs).as_posix() for p in built] == ["index.rst", "project/news.rst"]

    def test_exclude_patterns_are_read_from_conf_py_without_running_it(self, tmp_path: Path) -> None:
        (tmp_path / "conf.py").write_text(
            'raise SystemExit("must not run")\nexclude_patterns = ["a", "b/*"]\n', encoding="utf-8"
        )
        assert exclude_patterns(tmp_path) == ["a", "b/*"]


class TestSections:
    def test_pages_outside_the_sections_are_flagged(self, tmp_path: Path) -> None:
        docs = _tree(
            tmp_path,
            {
                "index.rst": "x",
                "readme.rst": "x",
                "stray/page.rst": "x",
                "tutorials/index.rst": "x",
                "project/news.rst": "x",
            },
        )
        found = misplaced_pages(docs, built_pages(docs, []))
        assert found == ["readme.rst", "stray/page.rst"]


class TestToctreeReachability:
    def test_relative_absolute_titled_and_glob_entries_are_followed(self, tmp_path: Path) -> None:
        docs = _tree(
            tmp_path,
            {
                "index.rst": _toctree("tutorials/index", "Guides <how-to/index>", "/reference/index", "https://x.org"),
                "tutorials/index.rst": _toctree("first", "self"),
                "tutorials/first.rst": "x",
                "how-to/index.rst": _toctree("*", options="   :glob:\n"),
                "how-to/a.rst": "x",
                "how-to/extend/index.rst": "x",
                "reference/index.rst": _toctree("../explanation/index"),
                "explanation/index.rst": "x",
            },
        )
        assert unreachable_pages(docs, built_pages(docs, [])) == ["how-to/extend/index"]

    def test_a_page_missing_from_every_toctree_is_flagged(self, tmp_path: Path) -> None:
        docs = _tree(
            tmp_path,
            {
                "index.rst": _toctree("tutorials/index"),
                "tutorials/index.rst": _toctree("first"),
                "tutorials/first.rst": "x",
                "tutorials/forgotten.rst": "x",
            },
        )
        assert unreachable_pages(docs, built_pages(docs, [])) == ["tutorials/forgotten"]


class TestHowToTitles:
    def test_titles_must_start_with_a_verb(self, tmp_path: Path) -> None:
        docs = _tree(
            tmp_path,
            {
                "how-to/index.rst": "How-to guides\n=============\n",
                "how-to/good.rst": "Run an experiment\n=================\n",
                "how-to/overline.rst": "=================\nTune a model\n=================\n",
                "how-to/noun.rst": "Hyperparameter tuning\n=====================\n",
                "how-to/extend/gerund.rst": ".. interface: python\n\nWriting a featurizer\n====================\n",
                "explanation/why.rst": "Why it works\n============\n",
            },
        )
        found = howto_title_violations(docs, built_pages(docs, []))
        assert [v.split(":")[0] for v in found] == ["how-to/extend/gerund.rst", "how-to/noun.rst"]


class TestExplanationPages:
    def test_python_and_bash_blocks_are_flagged_but_data_blocks_are_not(self, tmp_path: Path) -> None:
        docs = _tree(
            tmp_path,
            {
                "explanation/ok.rst": "T\n=\n\n.. code-block:: yaml\n\n   a: 1\n\n.. code-block:: text\n\n   out\n",
                "explanation/py.rst": "T\n=\n\n.. code-block:: python\n\n   x = 1\n",
                "explanation/sh.rst": "T\n=\n\n.. note::\n\n   .. code-block:: shell\n\n      ls\n",
                "how-to/fine.rst": "T\n=\n\n.. code-block:: python\n\n   x = 1\n",
            },
        )
        found = explanation_violations(docs, built_pages(docs, []))
        assert [v.split(":")[0] for v in found] == ["explanation/py.rst", "explanation/sh.rst"]


class TestInterfaceRules:
    def _violations(self, tmp_path: Path, text: str) -> list[str]:
        page = _tree(tmp_path, {"how-to/page.rst": text}) / "how-to" / "page.rst"
        return interface_violations(page)

    def test_a_both_page_with_a_complete_tab_set_passes(self, tmp_path: Path) -> None:
        assert not self._violations(tmp_path, BOTH_TABS)

    def test_a_both_page_without_an_interface_comment_defaults_to_both(self, tmp_path: Path) -> None:
        text = "T\n=\n\n.. code-block:: python\n\n   x = 1\n"
        assert len(self._violations(tmp_path, text)) == 1

    def test_a_bare_block_is_flagged(self, tmp_path: Path) -> None:
        text = BOTH_TABS + "\n.. code-block:: bash\n\n   drevalpy list splitters\n"
        found = self._violations(tmp_path, text)
        assert len(found) == 1
        assert "not inside a tab-item" in found[0]

    def test_a_tab_set_without_the_sync_group_is_flagged(self, tmp_path: Path) -> None:
        found = self._violations(tmp_path, BOTH_TABS.replace("   :sync-group: interface\n", ""))
        assert len(found) == 2
        assert all("sync-group" in message for message in found)

    def test_a_tab_set_missing_a_tab_is_flagged(self, tmp_path: Path) -> None:
        found = self._violations(tmp_path, BOTH_TABS.replace(":sync: python", ":sync: other"))
        assert len(found) == 2
        assert all("python" in message for message in found)

    def test_a_block_in_a_tab_nested_inside_a_note_still_counts_as_in_the_tab_set(self, tmp_path: Path) -> None:
        text = BOTH_TABS.replace(
            "      .. code-block:: python\n\n         print(1)\n",
            "      .. note::\n\n         .. code-block:: python\n\n            print(1)\n",
        )
        assert not self._violations(tmp_path, text)

    def test_blocks_of_other_languages_are_not_checked(self, tmp_path: Path) -> None:
        text = BOTH_TABS + "\n.. code-block:: text\n\n   output\n\n.. code-block:: yaml\n\n   a: 1\n"
        assert not self._violations(tmp_path, text)

    def test_a_python_page_may_not_run_drevalpy(self, tmp_path: Path) -> None:
        text = (
            "T\n=\n\n.. interface: python\n\n"
            ".. code-block:: bash\n\n   pip install drevalpy\n\n"
            ".. code-block:: bash\n\n   drevalpy list splitters\n\n"
            ".. code-block:: python\n\n   x = 1\n"
        )
        found = self._violations(tmp_path, text)
        assert len(found) == 1
        assert "runs 'drevalpy'" in found[0]

    def test_a_cli_page_may_not_have_python_blocks(self, tmp_path: Path) -> None:
        text = (
            "T\n=\n\n.. interface: cli\n\n"
            ".. code-block:: bash\n\n   pip install drevalpy\n\n"
            ".. code-block:: python\n\n   x = 1\n"
        )
        found = self._violations(tmp_path, text)
        assert len(found) == 1
        assert "python block" in found[0]

    def test_an_unknown_interface_is_flagged(self, tmp_path: Path) -> None:
        found = self._violations(tmp_path, "T\n=\n\n.. interface: bolth\n")
        assert len(found) == 1
        assert "bolth" in found[0]


class TestStaleLabelsAndIncludes:
    def test_only_the_news_page_may_name_old_releases(self, tmp_path: Path) -> None:
        docs = _tree(
            tmp_path,
            {
                "project/news.rst": "1.5.1 and the Modularity Release",
                "project/other.rst": "since 1.5.1",
                "how-to/x.rst": "the modularity release",
                "how-to/y.rst": "fine",
            },
        )
        assert stale_label_violations(docs, built_pages(docs, [])) == ["how-to/x.rst", "project/other.rst"]

    def test_a_reference_page_that_drops_its_include_is_flagged(self, tmp_path: Path) -> None:
        files = {rel: "".join(f".. include:: {n}\n" for n in names) for rel, names in GENERATED_INCLUDES.items()}
        assert not missing_includes(_tree(tmp_path, files))
        files["reference/components.rst"] = ".. include:: _generated_predictors.rst\n"
        assert missing_includes(_tree(tmp_path, files)) == [
            "reference/components.rst: _generated_sample_featurizers.rst",
            "reference/components.rst: _generated_drug_featurizers.rst",
        ]
