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
uv run python tools/size_gate.py
```

`tools/size_gate.py` caps AST statements per module against
`[tool.drevalpy.size_gate].max_module_statements`, which is what keeps the mixin
splits in `models/mixins/` and `components/featurizers/` from growing back. It runs
as a prek hook too, and needs no test run. Its exemptions follow the coverage
gate's ethos: a recorded ceiling is a _measured_ count, lowered as a module shrinks
and never raised to let a regression through.

Repowise needs **both** coverage reports, which is the opposite of what a single
`.coverage` ingest suggests - measured against repowise 0.42.0:

```bash
uv run pytest -m "not network" --cov --cov-context=test   # writes .coverage + coverage.json
repowise coverage add .coverage                           # per-test map ONLY
uv run coverage lcov -o coverage.lcov
repowise coverage add coverage.lcov && rm coverage.lcov   # per-file line coverage
repowise coverage status                                  # expect both sections
```

`repowise coverage add .coverage` reports only `Built the test-to-code map: N test->file record(s)` and leaves `repowise coverage status`'s line-coverage figure
untouched at whatever a previous lcov ingest left - it does **not** carry per-file
coverage, despite `--help` presenting `.coverage` as the richer input. So the lcov
round-trip is not a fallback; it is required for the coverage-derived health
markers. The two ingests are independent and both persist in `.repowise/`.

`--cov-context=test` is what records the per-test map (`coverage run --contexts=test`, the command repowise's `--help` prints, is not a valid
coverage.py option - measured against 7.15.2). The map is what
`repowise impacted-tests` and the `missing_tests` / `tests_to_run` fields of
`repowise risk` read; both are verified working here.

`repowise coverage add` does **not** read coverage.py's JSON report (`--format`
takes `lcov|cobertura|clover|repowise-json`, and `repowise-json` is Repowise's
own schema).

Without an ingest Repowise guesses from filenames, and its `has_test_file` probe
does not recognise `test_init.py`, so every re-export barrel is reported as an
untested hotspot. One artifact survives **both** ingest paths: lcov emits no `DA:`
lines for a zero-statement file, and `.coverage` does not fix it, so the eleven
one-line `__init__.py` files under `components/featurizers/{cell_line,drug}`,
`components/featurizers/shared` and the seven literature model packages read 0%
rather than 100% and each takes a full `-2.00` `coverage_gradient` penalty, 22
points in total. Ignore those; the other `__init__.py` files carry genuine partial
coverage and are real debt.

### Per-PR risk check

```bash
repowise risk --target <path> --changed-file <path> [--changed-file <path> ...]
```

Note the singular, repeatable `--changed-file`. The response leads with a
directive naming `will_break`, `missing_cochanges`, `missing_tests` and
`tests_to_run`; the last two are populated only once the test-to-code map above is
ingested. `repowise impacted-tests <revspec>` answers the narrower "which tests
cover these changed lines" - it takes a **git revspec**, not a path, and a
single-commit argument diffs that commit against its parent.

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
   in the mirrored directory. The guard does not check `__init__.py` at all, so
   write one only where the surface is worth pinning.
3. **Private module.** `_foo.py` needs a mirror of its own. The guard accepts
   either `test__foo.py` or the underscore-stripped `test_foo.py`, and the
   stripped form is the house style - see `tests/models/config/` and
   `tests/registry/`. An all-private package therefore mirrors as a directory of
   stripped names: `_block_specs.py` -> `test_block_specs.py`. What the file
   _exercises_ may still be the public entry point that exposes the module; what
   it may never be is a stub written to silence the guard.

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
  last resort - one entry today, carrying a comment for why the module is not
  library code, and `test_exempt_modules_still_exist` fails once it outlives its
  module.
- Shared test code is a `_`-prefixed module beside the tests it serves -
  `tests/_barrel_surface.py`, `tests/_trusted_subprocess.py`,
  `tests/registry/_helpers.py` - imported as `from tests._x import y`. The
  underscore keeps it out of collection, and the mirror policy walks `drevalpy/`
  only, so no mirror is demanded for it. Reach for the existing ones before
  writing a fixture out again:
  - `tests/_import_shims.py::block_imports(monkeypatch, *prefixes)` - fails an
    optional third-party import so the guidance message it raises can be pinned.
    Patches `builtins.__import__`, because the dependency is imported _inside_
    the method under test, by which point a `sys.modules` edit is too late.
  - `tests/models/synthetic_fixtures.py::synthetic_mudataset` - the one builder
    behind every synthetic `Dataset`: two cell lines, two drugs, and whichever
    cell-line views the caller asks for via `extra_views`. Add a view to
    `_VIEW_SPECS` rather than assembling AnnData in a test file.
  - `tests/components/predictors/literature/_helpers.py::two_by_two_batch` - the
    twelve-keyword `ModelInputBatch.from_response` every literature predictor
    test needs; a caller passes only the feature blocks it consumes.
  - `tests/components/featurizers/cell_line/_helpers.py::assert_uses_precomputed_variant`
    - the `fetch`-hit assertion shared by every dense cell-line featurizer.
  - `tests/models/config/_stubs.py` - throwaway featurizer/predictor
    registrations whose only real content is their contract. Callers must be
    under `isolated_component_registries`.
- A `test_init.py` pins its package surface by subclassing
  `ReExportSurface` / `DeclaredSurface` / `SingletonFacadeSurface` from
  `tests/_barrel_surface.py` and recording `origins` (`name -> defining module`)
  in the file itself. Keep that table hand-written: deriving it from `__all__`
  makes `test_all_matches_the_recorded_surface` unfalsifiable, and record each
  origin against a module the barrel does _not_ import the name from.

## Featurizers

- `DenseViewFeaturizer`, `register_for_sides` and `FeaturizerStorageMixin` are
  **public**: re-exported from `drevalpy/plugin/__init__.py` and pinned by
  `tests/plugin/test_init.py`, so renaming them is a breaking change even though
  two of them live in `_`-prefixed modules.
- A side-agnostic featurizer is written **once**, in
  `drevalpy/components/featurizers/shared/`, and bound to both entity sides by
  `register_for_sides` from `featurizers/_side_binding.py`, which derives,
  registers and namespace-injects one subclass per side. Never hand-write a
  second per-side copy - that duplication is exactly what `shared/` removed.
- `side` is a `ClassVar` stamped onto the class by the **registry**
  (`drevalpy/registry/featurizer/_base.py`), not by the base class, and
  `list_stored_variants` is a `classmethod` reading `cls.side`. That is why each
  side needs its own generated subclass instead of one class registered twice.
- Registration is by **directory scan**:
  `drevalpy/registry/_builtins.py::_discover_modules` imports every `*.py` in a
  component directory except `base`, `__init__` and names starting with `_`.
  So a shared-but-unregistered base module must be `_`-prefixed to stay out of
  the scan, and `shared/` has its own `_shared_featurizer_modules()` wired into
  `register_native_components()`.
- Single-view dense featurizers subclass `DenseViewFeaturizer`
  (`featurizers/_dense_view.py`) and override only their distinct transform step.
  `cell_line/base.py` and `drug/base.py` host the per-side
  `DenseView{CellLine,Drug}Featurizer` bases.
- `Featurizer` in `base.py` keeps only the abstract hooks and the public wrappers
  that guard them; the split-out concerns live in mixins beside it:
  `FeaturizerStorageMixin` in `featurizers/storage.py`, `NanToleranceMixin` in
  `featurizers/_nan_tolerance.py` (`_detect_valid`,
  `_warn_if_above_threshold`, `_expand_blocks_with_nan`, `nan_threshold`) and
  `FeaturizerDeclarationsMixin` in `featurizers/_declarations.py` (the
  `__init_subclass__` contract normalization, `resolve_input_views`,
  `output_block_specs_for_config`, and the `contract` / `precompute` /
  `requires_view` / `entity_id_only` / `input_views` / `source_views` ClassVars).
- The public `fit` / `transform` / `transform_blocks` wrappers stay in `base.py`
  beside the abstract hooks they wrap. Moving them out with the NaN policy would
  split the documented subclass contract across two files and measured as a
  _rise_ in `base.py`'s LCOM4. `HPOStrategy` stays too - the plugin barrel
  re-exports it from `base.py`.
- `DenseViewFeaturizer._restore_dense_state` in `featurizers/_dense_view.py` is
  the shared `set_state` path: it restores `view` / `output_dim` / `fitted`, the
  three fields `__init__` owns, leaving each subclass only its own fitted object.
  Used by `normalized_proteomics.py`, `scaled_gene_expression.py`, `pca.py` and
  `landmark.py`. `pharmaformer_gene_expression.py` is deliberately excluded: its
  `self._is_fitted = bool(state.get("fitted"))` resets to `False` on an absent
  key, where the shared path leaves the flag alone, so folding it in would change
  behaviour.

## DRPModel mixins

`models/drp_model.py` holds config and identity only - `model_config`,
`_from_resolved_config`, `_apply_model_config`, the five properties,
`log_hyperparameters`. Behaviour hangs off mixins in `models/mixins/`, so add new
behaviour to the mixin that owns that concern rather than back into the base:

- `_training.py` - `DRPTrainingMixin`, owning `train` / `predict`. It drives
  `_stack` and `_empty_training`, which `drp_model.py` only declares in
  `_init_runtime_fields`; the one other writer is `_persistence.py`, resetting
  `_empty_training` on load.
- `_train_args.py` - `resolve_train_args`, returning a frozen `TrainCallArgs`. It
  touches no instance state, which is why it was its own LCOM4 island and is a
  function rather than a mixin method. `TrainCallArgs.is_dataset_form` /
  `.is_feature_source_form` name the input form once; they replaced a bare
  6-tuple that forced `train` to re-derive it with `isinstance` checks.
- `_feature_matrix.py` - `DRPFeatureMatrixMixin`, owning
  `get_concatenated_features` / `get_feature_matrices` - the hand-rolled-model
  path that the component stack replaced.

## Hyperparameter keys

The key grammar - slot constants, prefix builders, key parsers - has one home in
`drevalpy/models/_hp_key_grammar.py`, which must stay a dependency-free leaf: it
imports nothing from `drevalpy` at module scope, and that is what keeps
`models/config` and `models/tuning` decoupled.

`TunableComponentMixin` in `components/contracts/hyperparameter_space.py` carries
`get_hyperparameter_space` / `get_default_hyperparameters` / `get_state` /
`set_state` for **both** component kinds; `Featurizer` and `Predictor` each mix it
in. It lives beside `validate_hyperparameter_space`, which it calls and which both
component packages already imported - any home inside `featurizers/` or
`predictors/` would have inverted a dependency between the two siblings. Override
`get_hyperparameter_space` to declare what is tunable and `get_state` / `set_state`
together when there is fitted state to round-trip; `get_default_hyperparameters` is
not an override point. `Predictor.is_fitted` stays on `Predictor` - it is
predictor-only.

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

Three floors, all in the `tests` job of `.github/workflows/run_tests.yml`:
`[tool.coverage.report].fail_under` for the aggregate, `tools/coverage_gate.py` per
module against `[tool.drevalpy.coverage_gate].min_file_coverage`, and
`tools/size_gate.py` per module against
`[tool.drevalpy.size_gate].max_module_statements`. Keep
`[tool.coverage.run].source = ["drevalpy"]` - it holds never-imported modules in
the denominator at 0%.

An entry in either `exemptions` table is debt, not a policy decision, and carries a
comment saying why the module cannot reach the floor.

- Work from the "exemptions that can be lowered or deleted" list each gate prints.
- Delete an entry once the module reaches `min_file_coverage`; if it improves but
  not that far, lower the recorded floor to the newly measured value.
- Never raise a floor to make a regression pass, and never lower `fail_under` to
  make a change fit. Raise `min_file_coverage` when the table empties.
- The same rules run the size gate in the other direction: lower a recorded
  ceiling as a module shrinks, never raise it to admit a regression.

## Repowise false positives

Checked directly, all artifacts of how the code is wired rather than debt:

- Both "Break Cycle" plans from `repowise health --refactoring-targets`. In
  `models` the only back edge is the `TYPE_CHECKING`-guarded import at
  `drevalpy/models/mixins/_persistence_io.py:21-22`, which is already the
  correct pattern; in `visualization/plots` none of `heatmap.py`, `violin.py`,
  `cross_study_table.py` imports `plots/__init__.py` - it is barrel attribution.
- `repowise dead-code`'s `unused_export` hits for `LassoPredictor`,
  `SVRPredictor`, `GradientBoostingPredictor` and `KNNPredictor` (reported at
  100% confidence) and the `unreachable_file` hit for
  `components/featurizers/cell_line/gene_lists/_make_gene_lists.py`. No static
  importer exists because `registry/_builtins.py::_discover_modules` registers by
  directory scan. `docs/conf.py` is a Sphinx entry point, not dead either.
- `coverage_gradient` on `sparsego/*`, `dipk/predictor.py`, `data/datasets/*` and
  `models/mixins/_hyperparameters.py` - the same debt already recorded with
  reasons in `[tool.drevalpy.coverage_gate.exemptions]`, counted a second time.
  Work it through the gate's exemption list, not the health score.

`hotspot_health` is a local review aid, not a number to gate on. One tree measures
three ways: **10.0** on the `fetch-depth: 1` checkout `actions/checkout` does by
default (every commit-history marker silently vanishes), **6.13** on a full clone
with no coverage ingested, **6.25** in a working tree with coverage ingested. 68% of
total finding impact comes from markers derived from commit history
(`co_change_scatter`, `hidden_coupling`, `churn_risk`, `prior_defect`,
`change_entropy`, `function_hotspot`, `code_age_volatility`, `knowledge_loss`), so
the score also drifts as unrelated commits land and cannot be moved by editing the
working tree. `tools/size_gate.py` is the CI ratchet instead. Use
`repowise health --trend` to review, and read a drop as a prompt to look rather
than as a failure.

## Path handling

Use `UPath` from `universal_pathlib` instead of `pathlib.Path` throughout, so
remote filesystems (S3, GCS) work transparently:

```python
from upath import UPath
```

Typer does not support `UPath` in CLI parameter annotations - take `str` and
convert inside the function body.
