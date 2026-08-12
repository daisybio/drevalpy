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

CI and the `prek` hook both run `-m "not network"` (network-marked tests need
credentialed artifact downloads) and produce a coverage report; see the coverage
gate section below for the exact invocation.

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

Coverage is enforced at two levels, both wired into `prek` and CI so neither is
bypassable with `--no-verify`:

1. **Aggregate.** `[tool.coverage.report].fail_under` in `pyproject.toml`. Note
   `[tool.coverage.run].source = ["drevalpy"]` is load-bearing: it keeps
   never-imported modules in the denominator at 0% instead of letting them vanish
   from the report, and drops `tests/` from the numerator.
2. **Per module.** `tools/coverage_gate.py`, because `coverage.py` can only fail
   on the aggregate, which lets one untested module hide behind a well-tested
   package. It reads `coverage.json` (not `.coverage`), so it is unit-testable -
   see `tests/tools/test_coverage_gate.py`.

Reproduce what the hooks do:

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
