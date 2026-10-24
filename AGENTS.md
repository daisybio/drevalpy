# Contributor and agent notes

Commands, test tiers and markers, coverage reproduction, prek and the Sphinx build
are documented once, in the "How to test the project" and "How to submit changes"
sections of `docs/project/contributing.rst`. This file holds only the conventions
that are not there.

The package sits under `src/`; `drevalpy/...` paths in this file are relative to
`src/`.

## Docs

The docs follow Diátaxis (`tutorials/`, `how-to/`, `reference/`, `explanation/`,
`project/`); the rules are in the "How to write documentation" section of
`docs/project/contributing.rst`, and `tests/docs/test_docs_structure.py` enforces them.
Use the terms in `GLOSSARY.md`. `tests/docs/test_snippets.py` runs every code block on
tutorial and how-to pages, so check a changed page with
`uv run pytest -m "not network" tests/docs` as well as the Sphinx build.

## Tests

- Markers (`slow`, `network`) and the rule that a command-line `-m` replaces the one
  in `addopts` are in `contributing.rst`. Apply `slow` with `pytestmark` at file or
  class level, not per test, and drop it again when an optimisation brings the test
  back under 0.2s.
- Do **not** add `pytest-xdist` or enable `--dist loadfile`/`loadscope` or random
  ordering: the component registries are process-global and the `BUILTIN_*_NAMES`
  sets in `drevalpy/registry/_builtins.py` are lazy singletons that cache on first
  access.
- Any test that calls `load_extensions` or a `register_*` decorator must evict on
  teardown via `restore_component_registries()` / `isolated_component_registries()`
  from `tests/registry/_helpers.py`, plus `clear_external_zoo()` when an external
  zoo is loaded. `register_builtin_components()` is **not** a teardown - it only adds.
- Terminology: write "sample" for the biological source and "pair" for one
  sample-drug measurement (`GLOSSARY.md`). `tests/test_terminology_policy.py`
  fails on any "cell line" under `src/` or `tests/` that its hand-written
  `ALLOWLIST` does not cover with a reason, and on allowlist entries that no
  longer match. Allowlist only text about data that really are cell lines.

## Test layout

`tests/` follows the directory tree of `src/drevalpy/`, but files are grouped by
feature area, not one per module: `tests/components/featurizers/sample/` holds a
handful of themed files such as `test_dense_views.py` and `test_gene_lists.py`.
CI enforces total coverage (see "Coverage" below).

- Put a new test in the existing file for its feature area; start a new file only
  for a new area. Aim for files of a few hundred lines, not one per module and not
  one per package once the package spans unrelated concerns.
- Shared fixtures for a directory go in its `conftest.py`, not in each file.
- Fold tests that differ only in their inputs into one parametrized test.
- No `__init__.py` in test directories - `--import-mode=importlib` with
  `pythonpath = ["."]` makes them namespace packages. Add one only for a genuine
  module package with content of its own, such as `tests/synthetic/`.
- Cross-package checks live at the `tests/` root (`test_import_cost_policy.py`,
  the terminology policy) and in `tests/docs/`.
- Shared test code is a `_`-prefixed module beside the tests it serves -
  `tests/_barrel_surface.py`, `tests/_trusted_subprocess.py`,
  `tests/registry/_helpers.py` - imported as `from tests._x import y`. The
  underscore keeps it out of collection. Reach for the existing ones before
  writing a fixture out again:
  - `tests/_import_shims.py::block_imports(monkeypatch, *prefixes)` - fails an
    optional third-party import so the guidance message it raises can be pinned.
    Patches `builtins.__import__`, because the dependency is imported _inside_
    the method under test, by which point a `sys.modules` edit is too late.
  - `tests/models/synthetic_fixtures.py::synthetic_mudataset` - the one builder
    behind every synthetic `Dataset`: two samples, two drugs, and whichever
    sample views the caller asks for via `extra_views`. Add a view to
    `_VIEW_SPECS` rather than assembling AnnData in a test file.
  - `tests/components/predictors/literature/_helpers.py::two_by_two_batch` - the
    twelve-keyword `ModelInputBatch.from_response` every literature predictor
    test needs; a caller passes only the feature blocks it consumes.
  - `tests/components/featurizers/sample/_helpers.py::assert_uses_precomputed_variant`
    - the `fetch`-hit assertion shared by every dense sample featurizer.
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
  `sample/base.py` and `drug/base.py` host the per-side
  `DenseView{Sample,Drug}Featurizer` bases.
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

## DRPModel structure

`models/drp_model.py` holds `DRPModel`, the public runtime. It owns
an immutable `ModelSpec` and exposes `spec` / `name` / `config` /
`hyperparameters` / `sample_views` / `drug_views`. `train` / `predict` /
`save` / `load` are thin: `train` and `predict` delegate to a private
`models/_pipeline._ModelPipeline`, which owns all fitted state and a
`PipelineState` (`UNTRAINED` / `EMPTY` / `FITTED`); `save` / `load` call
`models/mixins/_persistence_io.py`. Add new behaviour to the module that owns that
concern rather than back into `DRPModel`:

- `spec.py` - frozen `ModelSpec(name, config)`: `from_name`, `resolve`,
  `default_hyperparameters`, `hyperparameter_space`, early-stopping flags. It holds
  no fitted state and is reused across folds and tuning trials.
- `_components.py` - `build_components`, constructing fresh featurizers and the
  predictor from a `ResolvedModelConfig`. No fitted state lives here.
- `_feature_preparation.py` - `transform_entities` (fit on training entities, then
  transform unique entities once) and `usable_responses` (drop pairs a
  component's missing-input policy rejects).
- `_target_transform.py` - `validate_fit_scopes`, `fit_target_transform` and
  `apply_target_transform` for the optional response transformer.
- `_pipeline.py` - `_ModelPipeline`: `reset`, `fit`, `predict`,
  `component_state` / `restore_component_state`.
- `experiment/_tracking.py` - Weights & Biases tracking (`tracking_run`, `log_summary`,
  `log_metrics`), outside `DRPModel`; `wandb` is imported lazily.
- `mixins/_persistence_io.py` - versioned single-ZIP checkpoint I/O
  (`save_model`, `load_model`, `load_model_payload`). `DRPModel` is imported only
  under `TYPE_CHECKING` there.
- `factory.py` - resolves zoo/spec names to `ModelConfig` objects
  (`model_config_for_name`, `reject_unknown_spec`, `zoo_config`).

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
  `components/featurizers/sample/_proteomics_transformer.py` does.
- A module-scope **side effect** that must run before the library is imported
  anywhere stays eager - `xgboost_pred.py` calls
  `_set_xgboost_thread_defaults()` at module scope to keep OpenMP from crashing.

The same file guards `DEFERRED_TRAINING_SYMBOLS` and `LAZY_RE_EXPORTS`: a moved
symbol must keep resolving through the module-level `__getattr__`, and a name
that does not exist must still raise `AttributeError`.

## Coverage

The `tests` job of `.github/workflows/run_tests.yml` fails below
`[tool.coverage.report].fail_under`; `contributing.rst` has the command to reproduce
it. Keep `[tool.coverage.run].source = ["drevalpy"]` - it holds never-imported modules
in the denominator at 0%. Never lower `fail_under` to make a change fit.

## Docstrings

Use Sphinx field lists (`:param x:`, `:returns:`, `:raises:`), checked by pydoclint
(`[tool.pydoclint]` in `pyproject.toml`). Put types in annotations, not in the
docstring, and write no placeholder fields such as `:returns: Result.`.
Document `__init__` parameters in the `__init__` docstring. Use `:raises X:` only
for exceptions the function body raises itself; mention exceptions from callees
in prose. Write variadic parameters as `:param \**kwargs:` inside an `r"""`
docstring, so both pydoclint and Sphinx accept them.

## Path handling

Use `UPath` from `universal_pathlib` instead of `pathlib.Path` throughout, so
remote filesystems (S3, GCS) work transparently:

```python
from upath import UPath
```

Typer does not support `UPath` in CLI parameter annotations - take `str` and
convert inside the function body.
