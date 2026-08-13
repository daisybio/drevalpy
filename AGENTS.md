# Agent notes

## Running prek / committing

Hooks are managed by prek (a fast, drop-in replacement for pre-commit). Run all hooks:

```bash
uv run prek run --all-files
```

Install the git hook so it runs automatically on commit:

```bash
uv run prek install
```

The hooks are ruff, the whitespace/format checks, `complexipy` and the **fast
test tier**. They do not measure coverage; see "Coverage gate" below for what CI
adds on top. All 10 hooks against the whole tree take **22.3s**, of which the
fast-tier pytest hook is ~14.4s. Before the suite work that produced the tiers
below, the same hook set was 98.6s warm and 281.5s cold, because pytest ran the
whole suite under coverage.

## Running tests

The suite is tiered. `[tool.pytest.ini_options].addopts` in `pyproject.toml`
defaults to `-m "not slow and not network"`, so the bare command is the **fast
tier**:

```bash
uv run pytest
```

That is 3844 passed plus 7 skipped in 14.4s warm, with 73 deselected - 98.1% of
the suite. It is also exactly what the `prek` pytest hook runs, so a green hook
means a green fast tier and nothing more.

The two deselected groups:

- **`slow`** - the extended tier: tests that spawn a fresh interpreter, fit
  curves, or train models. Membership is a measurement, not a taste: **a test is
  `slow` if it costs >=0.2s**. Apply the marker with `pytestmark` at file or
  class level rather than decorating individual tests, and drop the marker again
  when an optimisation brings a test back under the threshold. 65 tests, 26.8s.
- **`network`** - needs credentialed artifact downloads (pretrained weights,
  remote annotations). The bucket refuses anonymous reads, so these cannot run
  on an unauthenticated machine at all. 8 tests.

Run the extended tier on its own, or both tiers together, before you push
anything that touches training, curation, the CLI or the registry:

```bash
uv run pytest -m slow            # extended tier only
uv run pytest -m "not network"   # fast + extended, i.e. what CI runs
```

A command-line `-m` **replaces** the one in `addopts` rather than adding to it,
which is why `-m "not network"` selects the whole suite (3909 passed, 7 skipped,
8 deselected) and `-m network` selects the 8 network tests.

CI runs `-m "not network"` with coverage in `.github/workflows/run_tests.yml`;
the network tests run weekly in `.github/workflows/run_network_tests.yml`, which
skips itself with a notice when the artifact credentials are not configured
rather than failing.

### Parallelism and test order

Two things were measured rather than assumed, so neither needs re-litigating:

- **`pytest-xdist` is deliberately not a dependency, because it does not pay.**
  Its measured ceiling was ~11% off the wall clock of the full-coverage CI shape
  at `-n 4`; `-n 8` and `-n auto` were break-even. On the fast tier every worker
  count was **9-70% slower** than serial. That is the expected outcome now that
  `import drevalpy` costs 0.21s instead of 3.59s: parallelism was only ever
  hiding the per-process import cost, and each worker still pays it. The
  measurement box is an Apple M5 (4 performance + 6 efficiency cores), which is
  why only `-n 4` moved at all - a 4-core CI runner would likely be break-even
  or worse.
- **`--dist loadfile`/`loadscope` and random ordering are not safe today.** The
  component registries are process-global, and the `BUILTIN_*_NAMES` sets in
  `drevalpy/registry/_builtins.py` are lazy singletons that cache on first
  access. A test that registers a name and does not evict it therefore changes
  what a _later_ test sees: whichever test resolves those sets first counts the
  leak as a built-in, and that surfaces as
  `tests/docs/test_docs_structure.py::test_component_catalog_is_registry_driven_and_synchronized`
  reporting extra "built-in" featurizers. The leaks were invisible under the
  default alphabetical order only because `tests/docs/` collects before
  `tests/models/` and `tests/registry/`.

  That audit has since been done across the whole tree, and the three files that
  leaked - `tests/models/config/test_spec.py`,
  `tests/models/config/test_validation.py`,
  `tests/models/config/test_block_specs.py` - plus
  `tests/registry/test_extensions.py` now all evict on teardown via
  `restore_component_registries()` / `isolated_component_registries()` in
  `tests/registry/_helpers.py` (and `clear_external_zoo()` where an external zoo
  is loaded). Note that `register_builtin_components()` alone is **not** a
  teardown: it only adds, so it leaves the test's own names behind - two of those
  files failed exactly that way. Before reaching for any distribution mode or
  `-p randomly`, re-audit any test that calls `load_extensions` or a `register_*`
  decorator, and run the suspect files against `tests/docs/` in both orders.

## Building docs

```bash
uv run --group docs sphinx-build -W docs docs/_build
```

## Test layout: one test file per public module

The `tests/` tree mirrors the `drevalpy/` tree. For a public source module
`drevalpy/a/b/c.py` there is exactly one test file `tests/a/b/test_c.py`.

Rules, in order of precedence:

1. **Public module.** `drevalpy/a/b/c.py` -> `tests/a/b/test_c.py`.
2. **Package surface.** A package's `__init__.py` re-exports are tested in
   `test_init.py` in the mirrored directory (precedent:
   `drevalpy/models/config/__init__.py` -> `tests/models/config/test_init.py`).
3. **Private module.** `_foo.py` is normally covered through the public entry
   point that exposes it and needs no dedicated file.
4. **All-private package.** Where a package has no public submodules, rule 3
   would collapse the whole subtree into a single `test_init.py`. Instead,
   mirror the private modules with the **leading underscore stripped**:
   `_block_specs.py` -> `test_block_specs.py`. This follows the existing
   precedent in `tests/models/config/` and is how `drevalpy/registry/` is
   covered (e.g. `drevalpy/registry/predictor/_registry.py` ->
   `tests/registry/predictor/test_registry.py`), since everything there is
   private apart from the subpackage `__init__.py` files.

Where several small test files would cover one source module, merge them into
the single mirrored file rather than inventing suffixes. Where one test file
spans several source modules, split it along module lines, or place it with the
module it predominantly exercises and say why in the module docstring.

### Documented exceptions

These do not mirror a single source module and are expected to stay where they are:

- `tests/docs/` - tests the generators under `docs/`, which are not part of the
  shipped package.
- The cross-package policy guards at the `tests/` root, which scan the whole
  package tree rather than mirroring one module: `test_boundary.py`,
  `test_featurizer_block_policy.py`, `test_module_mirror_policy.py`,
  `test_layering_policy.py`, `test_architecture_policy.py`.
- `tests/test_import_cost_policy.py` - guards the cost of `import drevalpy`
  itself rather than mirroring one module. `drevalpy/registry/__init__.py`
  registers builtins at import time, so every registered `predictor.py` is on
  the critical path of every CLI invocation. The test spawns a fresh interpreter,
  imports `drevalpy`, and asserts that none of the 16 modules in
  `FORBIDDEN_STARTUP_IMPORTS` reached `sys.modules` - `torch`,
  `pytorch_lightning`, `torch_geometric`, `sklearn`, `scipy`, `pandas`,
  `xgboost`, `lightgbm`, `mudata`, `anndata`, `matplotlib`, `seaborn`,
  `scikit_posthocs`, `IPython`, `wandb`, `optuna`. Deferring them all took
  `import drevalpy` from 3.59s to **0.21s**, which is why the guard exists.

  The usual remedy when it fails is to move the offending module-scope import
  into the method that needs it (with `if TYPE_CHECKING:` for annotation-only
  uses); do not delete the entry from `FORBIDDEN_STARTUP_IMPORTS` to make the
  failure go away. Two shapes the remedy does not cover, both found the hard way:

  - A class whose **base class** comes from a forbidden library cannot use a
    function-local import at all - the base has to exist when the `class`
    statement executes. Either drop the base where it earns nothing
    (`DataLoader` accepts any object with `__getitem__`/`__len__`) or move the
    class into its own private module and re-export it lazily, as
    `components/featurizers/cell_line/_proteomics_transformer.py` does.
  - A module-scope **side effect** that has to run before the library is
    imported anywhere must stay eager. `xgboost_pred.py` still calls
    `_set_xgboost_thread_defaults()` at module scope: deferring it let a test's
    own `importorskip("xgboost")` import xgboost _after_ torch's OpenMP runtime
    had loaded, and the suite died with `SIGSEGV`.

  Because a deferred import can no longer fail at registration time - where
  `_import_modules` in `drevalpy/registry/_builtins.py` catches it and reports
  via `get_skipped_builtin_modules()` - the same file also asserts registration
  skipped nothing and that every symbol in `DEFERRED_TRAINING_SYMBOLS` still
  resolves. `LAZY_RE_EXPORTS` covers the other half: where deferring an import
  meant moving a symbol, the historical import path must keep resolving through
  the module-level `__getattr__`, and must still raise `AttributeError` for a
  name that does not exist, so a rename cannot silently turn into `None`.

### No `__init__.py` in test directories

Do **not** add `__init__.py` to a new test directory. `[tool.pytest.ini_options]`
sets `--import-mode=importlib` with `pythonpath = ["."]`, so test directories
resolve as namespace packages, and test modules that share a basename across
directories do not collide.

Add an `__init__.py` only when the directory is a genuine module package with
content of its own - `tests/synthetic/__init__.py` re-exports the synthetic
dataset builders, so it needs one. Shared helpers are plain modules
(`tests/models/synthetic_fixtures.py`,
`tests/components/predictors/_helpers.py`) and are imported by their full dotted
path without any marker file.

### The mirror is enforced

`tests/test_module_mirror_policy.py` **fails** the run when a source module has no
mirrored test file. It walks every non-`__init__` module under `drevalpy/` and
accepts either spelling for a private module (`_foo.py` is satisfied by
`test__foo.py` or by the house-style `test_foo.py`), so rules 3 and 4 above are
both honoured.

```bash
uv run pytest tests/test_module_mirror_policy.py -q
```

One module is exempt, listed in `EXEMPT_MODULES` in that file with its reason:
`components/featurizers/cell_line/gene_lists/_make_gene_lists.py`, a maintenance
script that regenerates the packaged gene-list CSVs, is imported by no shipped
code path and is also excluded from coverage measurement. Adding to that list is a
last resort - keep it as short as it is.

Do **not** create stub test files, or directories holding nothing but an
`__init__.py`, to satisfy the guard. An empty mirror manufactures the appearance
of coverage, and the per-module coverage floor below will catch it anyway.

## Coverage gate

**Coverage is enforced in CI, not locally.** The floors only mean anything on the
full suite, so measuring them on commit means paying for the extended tier plus
coverage instrumentation on every commit - that combination was a 98.6s warm,
281.5s cold pre-commit hook. The hook now runs the fast tier without `--cov`
(~14.4s), and the gate lives in the `tests` job of
`.github/workflows/run_tests.yml`, which runs the full `-m "not network"` suite.

So the old promise that coverage is "not bypassable with `--no-verify`" no longer
holds: a green commit hook says nothing about coverage. Run the commands below
yourself before pushing a change that adds or moves shipped code.

Two levels are enforced, both in that CI job:

1. **Aggregate.** `[tool.coverage.report].fail_under` in `pyproject.toml`. Note
   `[tool.coverage.run].source = ["drevalpy"]` is load-bearing: it keeps
   never-imported modules in the denominator at 0% instead of letting them vanish
   from the report, and drops `tests/` from the numerator.
2. **Per module.** `tools/coverage_gate.py`, because `coverage.py` can only fail
   on the aggregate, which lets one untested module hide behind a well-tested
   package. It reads `coverage.json` (not `.coverage`), so it is unit-testable -
   see `tests/tools/test_coverage_gate.py`.

Reproduce what CI does (currently 91.13% aggregate against `fail_under = 89`,
over 276 modules, ~50s warm for the suite plus the gate):

```bash
uv run pytest -m "not network" --cov --cov-report=json --cov-report=term-missing
uv run python tools/coverage_gate.py
```

Every module must reach `[tool.drevalpy.coverage_gate].min_file_coverage` unless
it appears in the `exemptions` table, which maps a module path to its own lower
floor.

**An exemption is debt, not a policy decision.** Each one carries a comment
saying why the module cannot reach the floor; if you cannot write that sentence
honestly, write tests instead. To retire one:

- The gate prints an "exemptions that can be lowered or deleted" list on every
  run, naming exempted modules that now clear the global floor or that sit at
  least three points above their recorded floor. Work from that list.
- Once a module reaches `min_file_coverage`, **delete** its entry rather than
  raising it.
- If it improves but not that far, lower the recorded floor to the newly measured
  value. Never raise a floor to make a regression pass.
- When the table empties out, raise `min_file_coverage` itself and re-measure.

Both numbers ratchet in one direction. Raise `fail_under` when the measured total
moves up, leaving a point or two of headroom and no more; never lower it to make a
change fit.

The network-gated artifact featurizers (`chemberta`, `smilesvec`, `molgnet`,
`bionic`) are deliberately **not** exempted. Their download paths carry
`@pytest.mark.network` and are deselected in the measured run, but their offline
logic is tested directly and clears the floor regardless.

## Path handling

Use `universal_pathlib` (`UPath`) instead of `pathlib.Path` throughout the codebase. This enables transparent access to remote filesystems (S3, GCS, etc.). Import as:

```python
from upath import UPath
```

Note: Typer does not support `UPath` in CLI parameter annotations. Use `str` for path parameters and convert to `UPath` inside the function body.
