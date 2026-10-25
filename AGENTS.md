# Contributor and agent notes

Before running tests, prek, coverage or the docs build, read the "How to test the
project" and "How to submit changes" sections of `docs/project/contributing.rst`: the
commands, test tiers and markers are written there once. This file holds only the
conventions that are not.

Package code is in `src/drevalpy/`; `drevalpy/...` paths below are relative to `src/`.

## Design notes

Before changing a module, read its module docstring: the design rules of each area (where a
concern goes, why a module is `_`-prefixed, how to add a metric) are written there, beside the
code they govern.

## Code standards

The maintainer reviews every change against these rules; check the whole diff against them
before pushing, not only the files you touched last.

- **Use the library.** A formula or algorithm that numpy, pandas, scipy, scikit-learn or
  networkx provides is a call to that function, not a reimplementation: metrics are
  `scipy.stats` / `sklearn.metrics` calls, a standard error is `DataFrame.sem()`, a
  degree-preserving rewire is `networkx.double_edge_swap`. drevalpy owns only what is its
  own, such as the undefined-score rule in `evaluation/_metrics.score_pairs`.
- **One way to do each thing.** One entry point per task; parallel lists, aliases,
  compatibility wrappers and duck-typed variants of the same function are findings. A rule
  lives in one function that every caller goes through (`views.find_view` for every view
  lookup, `feature_format.check_format` for every declared format).
- **Derive, don't store twice.** A value that follows from another is computed from it
  (`HPOConfig.direction` reads the metric). A saved file holds each value once, and v2
  formats carry no legacy fallbacks.
- **Enums for closed sets.** A fixed set of choices is a `StrEnum` (`Metric`,
  `RandomizationType`), so errors and the CLI list the valid options.
- **Layers point down.** `types/` imports nothing from `registry`, `models`,
  `visualization` or `components` (except `components/contracts/`); code that needs those lives in the higher package (precompute in
  `components/featurizers/precompute.py`, plot requirements in `visualization/`). A barrel
  `__init__.py` re-exports only the names user code imports through it; everything else is
  imported from its defining module.
- **Small files.** Prefer several focused modules over one large one; split a file that
  mixes concerns, and mirror the split in `tests/`.
- **Nothing dead.** Unused code, stale re-exports, leftover test scaffolding, tests of
  deleted code and stale wording are findings.
- **Docs in one place.** Each fact has one home, preferably the docstring of the module it
  governs (`_`-prefixed modules for internal design notes, since public module docstrings
  render in the API reference). This file stays the only `AGENTS.md` and holds only what
  the code and `docs/project/contributing.rst` do not already say.
- **PR descriptions** say what to read closely, what to skim and what is risky to merge,
  without numbers that go stale on every edit (line or file counts).

## Names

Name things in code, tests and docs with the terms in `GLOSSARY.md`; when a concept has no
term yet, add one there rather than inventing a synonym. Write "sample" for the biological
source and "pair" for one sample-drug measurement. `tests/test_terminology_policy.py` fails
on any "cell line" under `src/` or `tests/` that its hand-written `ALLOWLIST` does not cover
with a reason, and on allowlist entries that no longer match; allowlist only text about data
that really are cell lines.

## Platforms

drevalpy must work on Windows: CI runs the tests on `windows-latest` as well as Linux. Use
`UPath` from `universal_pathlib` for every path, so remote filesystems (S3, GCS) work too:

```python
from upath import UPath
```

Typer cannot take `UPath` in a CLI parameter annotation: take `str` and convert inside the
function body. Open files in a `with` block (`UPath.open`), so Windows can delete or replace
them afterwards, and read and write text with `encoding="utf-8"`.

Support Python 3.11 to 3.13. On 3.11, `"x" in SomeStrEnum` raises `TypeError`: test
membership against the values.

## Tests

- Keep the suite serial, in file order: no `pytest-xdist`, `--dist loadfile`/`loadscope`
  or random ordering. The component registries are process-global and the
  `BUILTIN_*_NAMES` sets in `drevalpy/registry/_builtins.py` are lazy singletons that
  cache on first access.
- A test that calls `load_extensions` or a `register_*` decorator evicts on teardown with
  `restore_component_registries()` / `isolated_component_registries()` from
  `tests/registry/_helpers.py`, plus `clear_external_zoo()` when it loads an external zoo.
  `register_builtin_components()` only adds, so it is no teardown.
- Shared test code is a `_`-prefixed module beside the tests it serves (`tests/_import_shims.py`,
  `tests/models/synthetic_fixtures.py`, the `_helpers.py` files); look for one before writing a
  fixture out again. A `test_init.py` pins its package surface with a class from
  `tests/_barrel_surface.py`, whose docstring says how to record the origins.
- Mark a test of 0.2s or more `slow` with `pytestmark` at file or class level, and drop
  the mark when an optimisation brings it back under 0.2s.
- `tests/` follows the tree of `src/drevalpy/`, one file per feature area: put a new test
  in the file of its area, and start a file only for a new area, aiming for a few hundred
  lines. Fixtures shared by a directory go in its `conftest.py`.
- Test directories are namespace packages (`--import-mode=importlib`,
  `pythonpath = ["."]`), so they get no `__init__.py`. The exceptions are
  `tests/__init__.py`, which keeps `tests` from colliding with a third-party `tests`
  package, and module packages with content of their own, such as `tests/synthetic/`.
- Cross-package checks live at the `tests/` root (`test_import_cost_policy.py`, the
  terminology policy) and in `tests/docs/`.

## Import cost

`tests/test_import_cost_policy.py` asserts that none of `FORBIDDEN_STARTUP_IMPORTS`
(torch, sklearn, scipy, pandas, matplotlib, ...) reach `sys.modules` on `import drevalpy`.
When it fails, move the module-scope import into the function that needs it, with
`if TYPE_CHECKING:` for annotation-only uses; the list itself stays as it is. Two cases a
function-local import cannot fix:

- A **base class** from a forbidden library has to exist when the `class` statement runs.
  Drop the base where it earns nothing, or move the class into its own private module, as
  `components/featurizers/sample/_proteomics_transformer.py` does, and import it inside the
  functions that use it.
- A module-scope **side effect** that must run before the library is imported anywhere
  stays eager: `xgboost_pred.py` calls `_set_xgboost_thread_defaults()` at module scope to
  keep OpenMP from crashing.

The same file guards `DEFERRED_TRAINING_SYMBOLS`: a symbol moved out of its module's
import-time path must still resolve from its new module.

## Coverage

The `tests` job of `.github/workflows/run_tests.yml` fails below
`[tool.coverage.report].fail_under`. Meet the floor with tests; the floor and
`[tool.coverage.run].source = ["drevalpy"]` stay as they are, since `source` holds
never-imported modules in the denominator at 0%.

## Docstrings

Use Sphinx field lists (`:param x:`, `:returns:`, `:raises:`), checked by pydoclint
(`[tool.pydoclint]` in `pyproject.toml`), with types in the annotations and every field
saying something specific. Document `__init__` parameters in the `__init__` docstring.
Write `:raises X:` only for an exception the function body raises itself, and mention
exceptions from callees in prose. Write variadic parameters as `:param \**kwargs:` inside
an `r"""` docstring, so both pydoclint and Sphinx accept them.

## Docs

The docs follow Diátaxis (`tutorials/`, `how-to/`, `reference/`, `explanation/`,
`project/`); the rules are in the "How to write documentation" section of
`docs/project/contributing.rst`, and `tests/docs/test_docs_structure.py` enforces them.
`tests/docs/test_snippets.py` runs every code block on tutorial and how-to pages, so check
a changed page with `uv run pytest -m "not network" tests/docs` as well as the Sphinx build.
