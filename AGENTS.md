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

## Running tests

```bash
uv run pytest
```

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

### Coverage backlog

The mirroring is not complete yet. As of this writing **76 of 148 public
non-`__init__` modules have no mirrored test file** (down from 104 of 147).
Regenerate the live list rather than reading a copy - `tests/test_module_mirror_policy.py`
already computes it and warns per module:

```bash
uv run pytest tests/test_module_mirror_policy.py -q -rw
```

That guard is intentionally **warn-only**. Note it walks every module including
private ones and expects the underscore to be preserved (`test__foo.py`), so it
over-reports relative to rules 3 and 4 above.

Priority areas, largest and most user-facing first:

| Area                                                  | Missing / total |
| ----------------------------------------------------- | --------------- |
| `drevalpy/cli/` (+ `cli/data/`, `cli/experiments/`)   | 10 / 10         |
| `drevalpy/visualization/` (+ `plots/`)                | 10 / 10         |
| `drevalpy/types/data/` (+ `batch/`, `dataset_utils/`) | 11 / 15         |
| `drevalpy/components/featurizers/drug/`               | 7 / 11          |
| `drevalpy/components/predictors/literature/`          | 7 / 18          |
| `drevalpy/components/featurizers/cell_line/`          | 5 / 18          |
| `drevalpy/data/splitters/`                            | 4 / 4           |
| `drevalpy/types/results/`                             | 4 / 4           |

`drevalpy/cli/` is the single largest gap: there is no `tests/cli/` directory at
all and none of the 10 CLI modules are tested, while the CLI is the primary user
entry point. Other named gaps worth closing early are
`components/predictors/lightgbm_pred.py`, `components/predictors/sklearn_tabular.py`,
`components/predictors/state_errors.py` and `models/tuning/config.py`.

Do **not** create stub test files, or directories holding nothing but an
`__init__.py`, to close the numbers - an empty mirror manufactures the appearance
of coverage. Once the backlog is closed, tighten
`tests/test_module_mirror_policy.py` to the convention above and make it fail
rather than warn; enabling that while the backlog is open would just red CI.

## Path handling

Use `universal_pathlib` (`UPath`) instead of `pathlib.Path` throughout the codebase. This enables transparent access to remote filesystems (S3, GCS, etc.). Import as:

```python
from upath import UPath
```

Note: Typer does not support `UPath` in CLI parameter annotations. Use `str` for path parameters and convert to `UPath` inside the function body.
