"""Policy checks for the Diátaxis documentation tree, run on the real tree and on broken toy trees."""

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

SECTIONS = ("tutorials", "how-to", "reference", "explanation", "project")
NON_SOURCE_DIRS = frozenset({"_autosummary", "_build", "_static", "_templates"})
INTERFACE_SECTIONS = ("tutorials", "how-to")
INTERFACES = ("both", "cli", "python")
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
CODE_LANGUAGES = RUNNABLE_LANGUAGES | frozenset(MISNAMED_LANGUAGES)
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
# A floor, not a mirror of the live app: it catches a group being dropped.
REQUIRED_CLI_GROUPS = {"data", "experiments", "list"}

_TITLE = re.compile(r"^(?P<title>\S[^\n]*)\n[=\-~^\"#*+`.']{3,} *$", re.MULTILINE)
_INTERFACE = re.compile(r"^\.\. interface:[ \t]*(\S*)[ \t]*$", re.MULTILINE)
_DIRECTIVE = re.compile(r"^(?P<indent> *)\.\. (?P<name>[\w:-]+)::(?P<argument>.*)$")
_OPTION = re.compile(r"^ +:(?P<key>[\w-]+):[ \t]*(?P<value>.*)$")
_TOCTREE = re.compile(r"^(?P<indent> *)\.\. toctree::\s*$")
_DREVALPY_LINE = re.compile(r"^\s*drevalpy\b", re.MULTILINE)


def _matches(pattern: str, path: str) -> bool:
    """Match *path* against a Sphinx pattern: ``*`` and ``?`` stay inside one segment, ``**`` crosses them."""
    regex = re.escape(pattern)
    regex = regex.replace(r"\*\*", "\0").replace(r"\*", "[^/]*").replace(r"\?", "[^/]").replace("\0", ".*")
    return re.fullmatch(regex, path) is not None


def exclude_patterns(docs: Path) -> list[str]:
    """Read ``exclude_patterns`` out of ``docs/conf.py`` without running it."""
    tree = ast.parse((docs / "conf.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "exclude_patterns" for t in node.targets):
            return list(ast.literal_eval(node.value))
    msg = "conf.py has no literal 'exclude_patterns' assignment"
    raise AssertionError(msg)


def built_pages(docs: Path, patterns: Iterable[str]) -> list[Path]:
    """List the ``.rst`` pages Sphinx would build, sorted."""
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
    """Return the pages outside the sections, other than the root ``index.rst``."""
    misplaced = []
    for page in pages:
        parts = page.relative_to(docs).parts
        if parts != ("index.rst",) and (len(parts) < 2 or parts[0] not in SECTIONS):
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


def toctree_targets(page: Path, docs: Path, docnames: set[str]) -> set[str]:
    """Return the docnames *page*'s toctrees point at (``Title <target>``, absolute, relative, ``:glob:``)."""
    directory = page.parent.relative_to(docs).as_posix()
    directory = "" if directory == "." else directory
    return _targets_from_lines(page.read_text(encoding="utf-8").splitlines(), directory, docnames)


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


def page_title(page: Path) -> str | None:
    """Return the page's first section title, or ``None``."""
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
    """Return the ``:sync:`` values of the tab-set's own tab-items (not nested tab-sets')."""
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
    """Check *page* against its declared ``.. interface:`` (both / python / cli)."""
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


@pytest.fixture(scope="module")
def pages() -> list[Path]:
    return built_pages(DOCS, exclude_patterns(DOCS))


def test_the_built_page_list_reads_conf_py(pages: list[Path]) -> None:
    names = {page.relative_to(DOCS).as_posix() for page in pages}
    assert {"index.rst", "tutorials/index.rst", "how-to/index.rst", "reference/index.rst"} <= names
    assert not {name for name in names if "_generated_" in name}


def test_the_real_tree_follows_the_rules(pages: list[Path]) -> None:
    interface_pages = [page for page in pages if page.relative_to(DOCS).parts[0] in INTERFACE_SECTIONS]
    assert not misplaced_pages(DOCS, pages)
    assert not unreachable_pages(DOCS, pages)
    assert not explanation_violations(DOCS, pages)
    assert not howto_title_violations(DOCS, pages)
    assert not [v for page in interface_pages for v in interface_violations(page)]
    assert not missing_includes(DOCS)
    assert not stale_label_violations(DOCS, pages)


def test_zoo_presets_documented_in_model_zoo() -> None:
    from _model_zoo import generate_model_zoo_rst

    generated = generate_model_zoo_rst()
    missing = sorted(path.stem for path in ZOO_DIR.glob("*.yaml") if path.stem not in generated)
    assert not missing
    assert "``scaledGeneExpression:singleDrugElasticNet``" in generated
    assert "``molirOmics:molir``" in generated


@pytest.mark.slow
def test_component_catalog_is_registry_driven_and_synchronized() -> None:
    """Regenerate the catalog in a fresh interpreter: it must come from a clean registry."""
    from drevalpy.registry import _builtins as rb
    from drevalpy.registry.drug_featurizer import drug_featurizer_registry
    from drevalpy.registry.predictor import predictor_registry
    from drevalpy.registry.sample_featurizer import sample_featurizer_registry

    catalog = (DOCS / "reference" / "components.rst").read_text(encoding="utf-8")
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

    def featurizer_rows(registry, builtin):
        return {
            row["name"]: (
                row["output_format"],
                "\u2705" if row["precompute"] else "\u274c",
                " ".join(row["description"].split()),
            )
            for row in registry.list_metadata()
            if row["name"] in builtin
        }

    expected = {
        "sample": featurizer_rows(sample_featurizer_registry, rb.BUILTIN_SAMPLE_FEATURIZER_NAMES),
        "drug": featurizer_rows(drug_featurizer_registry, rb.BUILTIN_DRUG_FEATURIZER_NAMES),
    }
    for registry_name, rows in expected.items():
        matches = re.findall(
            r"^   \* - ``([^`]+)``\n     - ``([^`]+)``\n     - ([^\n]*)\n     - ([^\n]+)$",
            generated[registry_name],
            flags=re.MULTILINE,
        )
        assert {name: (fmt, pre, desc) for name, fmt, pre, desc in matches} == rows

    expected_predictors = {
        row["name"]: (row["input_interface"].replace("_", "-").capitalize(), " ".join(row["description"].split()))
        for row in predictor_registry.list_metadata()
        if row["name"] in rb.BUILTIN_PREDICTOR_NAMES
    }
    predictor_matches = re.findall(
        r"^   \* - ``([^`]+)``\n     - ([^\n]+)\n     - ([^\n]+)$", generated["predictor"], flags=re.MULTILINE
    )
    assert {name: (interface, desc) for name, interface, desc in predictor_matches} == expected_predictors


def test_cli_reference_documents_all_subcommands() -> None:
    from _cli_click import generate_cli_reference_rst
    from typer.main import get_command

    from drevalpy.cli.main import app

    generated = generate_cli_reference_rst()
    groups = sorted(group.name for group in app.registered_groups if group.name)
    assert not [cmd for cmd in groups if f"drevalpy {cmd}" not in generated]
    assert REQUIRED_CLI_GROUPS <= set(getattr(get_command(app), "commands", {}))


def test_zoo_yaml_files_are_valid() -> None:
    for path in sorted(ZOO_DIR.glob("*.yaml")):
        assert yaml.safe_load(path.read_text(encoding="utf-8")) is not None, path.name


@pytest.mark.parametrize("name", ["python_api.rst", "cli.rst"])
def test_generated_reference_pages_omit_compatibility(name: str) -> None:
    assert "Backward compatibility" not in (DOCS / "reference" / name).read_text(encoding="utf-8")


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


def test_the_page_list_honours_exclude_patterns_and_never_runs_conf_py(tmp_path: Path) -> None:
    docs = _tree(
        tmp_path,
        {
            "conf.py": 'raise SystemExit("must not run")\nexclude_patterns = ["old", "project/v2_*"]\n',
            "index.rst": "x",
            "old/deeper/page.rst": "x",
            "project/v2_news.rst": "x",
            "project/news.rst": "x",
            "readme.rst": "x",
            "reference/_generated_cli.rst": "x",
            "_build/html/x.rst": "x",
        },
    )
    built = built_pages(docs, [*exclude_patterns(docs), "reference/_generated_*.rst"])
    assert [p.relative_to(docs).as_posix() for p in built] == ["index.rst", "project/news.rst", "readme.rst"]
    assert misplaced_pages(docs, built) == ["readme.rst"]


def test_toctree_reachability_follows_every_entry_form_and_flags_orphans(tmp_path: Path) -> None:
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


def test_title_explanation_and_stale_label_rules_flag_broken_pages(tmp_path: Path) -> None:
    docs = _tree(
        tmp_path,
        {
            "how-to/index.rst": "How-to guides\n=============\n",
            "how-to/good.rst": "Run an experiment\n=================\n",
            "how-to/overline.rst": "=================\nTune a model\n=================\n",
            "how-to/noun.rst": "Hyperparameter tuning\n=====================\n",
            "how-to/x.rst": "the modularity release",
            "explanation/ok.rst": "T\n=\n\n.. code-block:: yaml\n\n   a: 1\n",
            "explanation/py.rst": "T\n=\n\n.. code-block:: python\n\n   x = 1\n",
            "explanation/sh.rst": "T\n=\n\n.. note::\n\n   .. code-block:: shell\n\n      ls\n",
            "project/news.rst": "1.5.1 and the Modularity Release",
            "project/other.rst": "since 1.5.1",
        },
    )
    built = built_pages(docs, [])
    assert [v.split(":")[0] for v in howto_title_violations(docs, built)] == ["how-to/noun.rst", "how-to/x.rst"]
    assert [v.split(":")[0] for v in explanation_violations(docs, built)] == [
        "explanation/py.rst",
        "explanation/sh.rst",
    ]
    assert stale_label_violations(docs, built) == ["how-to/x.rst", "project/other.rst"]


def test_a_reference_page_that_drops_its_include_is_flagged(tmp_path: Path) -> None:
    files = {rel: "".join(f".. include:: {n}\n" for n in names) for rel, names in GENERATED_INCLUDES.items()}
    assert not missing_includes(_tree(tmp_path, files))
    files["reference/components.rst"] = ".. include:: _generated_predictors.rst\n"
    assert missing_includes(_tree(tmp_path, files)) == [
        "reference/components.rst: _generated_sample_featurizers.rst",
        "reference/components.rst: _generated_drug_featurizers.rst",
    ]


NESTED_NOTE = BOTH_TABS.replace(
    "      .. code-block:: python\n\n         print(1)\n",
    "      .. note::\n\n         .. code-block:: python\n\n            print(1)\n",
)
PYTHON_PAGE = (
    "T\n=\n\n.. interface: python\n\n.. code-block:: bash\n\n   pip install drevalpy\n\n"
    ".. code-block:: bash\n\n   drevalpy list splitters\n\n.. code-block:: python\n\n   x = 1\n"
)
CLI_PAGE = (
    "T\n=\n\n.. interface: cli\n\n.. code-block:: bash\n\n   pip install drevalpy\n\n"
    ".. code-block:: python\n\n   x = 1\n"
)


@pytest.mark.parametrize(
    ("text", "count", "needle"),
    [
        (BOTH_TABS, 0, ""),
        (NESTED_NOTE, 0, ""),
        (BOTH_TABS + "\n.. code-block:: text\n\n   output\n", 0, ""),
        ("T\n=\n\n.. code-block:: python\n\n   x = 1\n", 1, "not inside a tab-item"),
        (BOTH_TABS + "\n.. code-block:: bash\n\n   drevalpy list splitters\n", 1, "not inside a tab-item"),
        (BOTH_TABS.replace("   :sync-group: interface\n", ""), 2, "sync-group"),
        (BOTH_TABS.replace(":sync: python", ":sync: other"), 2, "python"),
        (PYTHON_PAGE, 1, "runs 'drevalpy'"),
        (CLI_PAGE, 1, "python block"),
        ("T\n=\n\n.. interface: bolth\n", 1, "bolth"),
    ],
    ids=[
        "complete",
        "nested-in-note",
        "other-language",
        "default-both",
        "bare-block",
        "no-sync-group",
        "missing-tab",
        "python-page",
        "cli-page",
        "unknown",
    ],
)
def test_interface_rules(tmp_path: Path, text: str, count: int, needle: str) -> None:
    page = _tree(tmp_path, {"how-to/page.rst": text}) / "how-to" / "page.rst"
    found = interface_violations(page)
    assert len(found) == count
    assert all(needle in message for message in found)
