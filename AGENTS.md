# Agent notes

## Commands

```bash
uv run prek run --all-files      # all hooks: ruff, whitespace/format, complexipy, fast tests
uv run prek install              # run hooks on commit
uv run pytest                    # fast tier (addopts default: -m "not slow and not network")
uv run pytest -m slow            # extended tier
uv run pytest -m "not network"   # fast + extended, i.e. what CI runs
uv run --group docs sphinx-build -W docs docs/_build
```

Coverage is enforced in CI, not by the hooks. Reproduce it yourself before pushing
a change that adds or moves shipped code:

```bash
uv run pytest -m "not network" --cov --cov-report=json --cov-report=term-missing
uv run python tools/coverage_gate.py
```

## Tests

- Markers: a test is `slow` if it costs >= 0.2s; `network` means it downloads
  remote artifacts and cannot run on a machine without internet access.
  A command-line `-m` **replaces** the one in `addopts` rather than adding to it.
- Apply `slow` with `pytestmark` at file or class level, not per test, and drop
  the marker again when an optimisation brings the test back under 0.2s.
- Do **not** add `pytest-xdist`; it measured slower than serial. `--dist loadfile`/`loadscope` and random ordering are also unsafe: the component
  registries are process-global and the `BUILTIN_*_NAMES` sets in
  `drevalpy/registry/_builtins.py` are lazy singletons that cache on first access.
- Any test that calls `load_extensions` or a `register_*` decorator must evict on
  teardown via `restore_component_registries()` / `isolated_component_registries()`
  from `tests/registry/_helpers.py`, plus `clear_external_zoo()` when an external
  zoo is loaded. `register_builtin_components()` is **not** a teardown - it only adds.

## Test layout

`tests/` mirrors `drevalpy/`, one test file per module, enforced by
`tests/test_module_mirror_policy.py`. Rules in order of precedence:

1. **Public module.** `drevalpy/a/b/c.py` -> `tests/a/b/test_c.py`.
2. **Package surface.** A package's `__init__.py` re-exports go in `test_init.py`
   in the mirrored directory.
3. **Private module.** `_foo.py` is covered through the public entry point that
   exposes it and needs no dedicated file.
4. **All-private package.** Mirror the private modules with the leading
   underscore stripped: `_block_specs.py` -> `test_block_specs.py`.

- Merge several small test files covering one source module into the single
  mirrored file; split one test file spanning several source modules along module
  lines.
- No `__init__.py` in test directories - `--import-mode=importlib` with
  `pythonpath = ["."]` makes them namespace packages. Add one only for a genuine
  module package with content of its own, such as `tests/synthetic/`.
- Never create stub test files or directories holding only an `__init__.py` to
  satisfy the guard.
- Files that legitimately mirror no single module: `tests/docs/`, the
  cross-package policy guards at the `tests/` root, and
  `tests/test_import_cost_policy.py`. `EXEMPT_MODULES` in the mirror policy is a
  last resort.

## Import cost

`tests/test_import_cost_policy.py` asserts that none of
`FORBIDDEN_STARTUP_IMPORTS` (torch, sklearn, pandas, matplotlib, ...) reach
`sys.modules` on `import drevalpy`. When it fails, move the module-scope import
into the method that needs it, using `if TYPE_CHECKING:` for annotation-only
uses. Never delete an entry from the list to make the failure go away.

Two cases a function-local import cannot fix:

- A **base class** from a forbidden library has to exist when the `class`
  statement executes. Drop the base where it earns nothing, or move the class
  into its own private module and re-export it lazily, as
  `components/featurizers/cell_line/_proteomics_transformer.py` does.
- A module-scope **side effect** that must run before the library is imported
  anywhere stays eager - `xgboost_pred.py` calls
  `_set_xgboost_thread_defaults()` at module scope to keep OpenMP from crashing.

The same file guards `DEFERRED_TRAINING_SYMBOLS` and `LAZY_RE_EXPORTS`: a moved
symbol must keep resolving through the module-level `__getattr__`, and a name
that does not exist must still raise `AttributeError`.

## Coverage gate

Two floors, both in the `tests` job of `.github/workflows/run_tests.yml`:
`[tool.coverage.report].fail_under` for the aggregate, and
`tools/coverage_gate.py` per module against
`[tool.drevalpy.coverage_gate].min_file_coverage`. Keep
`[tool.coverage.run].source = ["drevalpy"]` - it holds never-imported modules in
the denominator at 0%.

An entry in the `exemptions` table is debt, not a policy decision, and carries a
comment saying why the module cannot reach the floor.

- Work from the "exemptions that can be lowered or deleted" list the gate prints.
- Delete an entry once the module reaches `min_file_coverage`; if it improves but
  not that far, lower the recorded floor to the newly measured value.
- Never raise a floor to make a regression pass, and never lower `fail_under` to
  make a change fit. Raise `min_file_coverage` when the table empties.

## Path handling

Use `UPath` from `universal_pathlib` instead of `pathlib.Path` throughout, so
remote filesystems (S3, GCS) work transparently:

```python
from upath import UPath
```

Typer does not support `UPath` in CLI parameter annotations - take `str` and
convert inside the function body.
