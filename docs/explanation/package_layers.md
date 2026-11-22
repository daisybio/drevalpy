---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# How the packages fit together

If you open `src/drevalpy/` for the first time, you will see about a dozen subpackages and a few
top-level modules. They are organized into four layers — from user-facing entry points at the top
down to shared data types and utilities at the bottom.

Dependencies generally flow downward: higher layers import from lower ones (skipping layers when
they do not need them), while the foundation stays independent of workflows and UI code. At the
bottom of this page, a live script inspects the actual imports in `src/drevalpy/` on every
documentation build so the dependency table never drifts out of date.

## Four layers, top to bottom

```text
┌─────────────────────────────────────────────────────────────────────┐
│  4. Entry points                                                    │
│     cli/   _experiment, _run, _train, _predict   plugin/   testing/ │
└─────────────────────────────────┬───────────────────────────────────┘
                                  ▼  imports flow downward
┌─────────────────────────────────────────────────────────────────────┐
│  3. Workflows                                                       │
│     curation/   data/   models/   execution/   visualization/       │
└─────────────────────────────────┬───────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. Extensible building blocks                                      │
│     components/   registry/                                         │
└─────────────────────────────────┬───────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  1. Shared foundation                                               │
│     types/   evaluation/   _utils/   errors.py   log.py             │
└─────────────────────────────────────────────────────────────────────┘
```

If you are looking for where a particular piece of code lives, start here:

| If you are working on...                                           | Look in                                                         |
| ------------------------------------------------------------------ | --------------------------------------------------------------- |
| CLI commands or Python entry points (`experiment`, `run`, `train`) | `cli/`, `_experiment.py`, `_run.py`, `_train.py`, `_predict.py` |
| Third-party plugin API or plugin conformance tests                 | `plugin/`, `testing/`                                           |
| Fitting dose-response curves from well tables                      | `curation/`                                                     |
| Loading, splitting, concatenating, or precomputing datasets        | `data/`                                                         |
| Model recipes, specs, hyperparameter tuning, or the model zoo      | `models/`                                                       |
| Fold execution, tracking, randomization, or robustness tests       | `execution/`                                                    |
| Plots or the MultiQC HTML report                                   | `visualization/`                                                |
| Featurizers or predictors (base classes and built-ins)             | `components/`                                                   |
| Component/dataset registries or plugin discovery                   | `registry/`                                                     |
| Shared containers (`Dataset`, `Fold`, `PredictorInput`, results)   | `types/`                                                        |
| Evaluation metrics and per-group scoring                           | `evaluation/`                                                   |

## Inside each layer

### Layer 4: Entry points

These modules are the front doors into DrEvalPy:

- **Top-level runners** (`_experiment.py`, `_run.py`, `_train.py`, `_predict.py`) implement the main
  Python functions re-exported by `drevalpy`: {py:func}`~drevalpy._experiment.experiment`,
  {py:func}`~drevalpy._run.run`, and {py:func}`~drevalpy._train.train`. They wire together
  `data`, `models`, `execution`, `evaluation`, and `visualization`.
- **`cli/`** wraps those runners in a Typer CLI (`cli.app`, `cli.cli_main`), with one module per
  command group (`cli/experiment.py`, `cli/train.py`, and so on).
- **`plugin/`** is a single-module barrel re-exporting the stable API for third-party plugins
  (`from drevalpy.plugin import Featurizer, register_featurizer, ...`). Internal module paths may
  move, while names exported here are the compatibility promise.
- **`testing/`** provides conformance checks that plugin authors can run in their own test suites
  to verify their featurizers and predictors meet DrEvalPy's contracts.

### Layer 3: Workflows

These five packages carry out the main stages of a drug-response workflow, from curve fitting to
reporting:

- **`curation/`** fits dose-response curves from raw well tables using CurveCurator.
  {py:func}`~drevalpy.curation.curate` groups wells into dose series, fits pooled and replicate
  curves, and returns an {py:class}`anndata.AnnData`, importing heavy libraries (`anndata`,
  `pandas`, `curve_curator`) only when called. It depends only on `errors.py` and `log.py` from
  Layer 1. See {doc}`dataset_lifecycle`.
- **`data/`** handles loading, splitting, concatenating, precomputing, and generating synthetic
  datasets: {py:func}`~drevalpy.data.datasets._load.load`,
  {py:func}`~drevalpy.data.splitters._split.split`, {py:func}`~drevalpy.data._concat.concat`,
  {py:func}`~drevalpy.data._synthetic.synthetic`, `data/splitters/builtins.py`, and
  `data/precompute.py`.
- **`models/`** turns names and recipes into runnable models:
  {py:class}`~drevalpy.models.spec.ModelSpec`, {py:class}`~drevalpy.models.drp_model.DRPModel`,
  response transformations (`models/response_transformation/`), hyperparameter tuning
  (`models/tuning/`), and the built-in zoo presets (`models/zoo/`). See {doc}`model_assembly` and
  {doc}`/how-to/train_save_load`.
- **`execution/`** drives fold execution, progress tracking, and stress tests:
  {py:func}`~drevalpy.execution._randomization.randomization` and
  {py:func}`~drevalpy.execution._robustness.robustness`. PyTorch training helpers in `components/`
  also read `execution._tracking.active_run` to log training progress.
- **`visualization/`** builds plots and the MultiQC HTML report:
  {py:class}`~drevalpy.visualization.base.Plot`,
  {py:class}`~drevalpy.visualization.requirements.PlotRequirement`,
  {py:func}`~drevalpy.visualization.report.create_report`, and the concrete plots under
  `visualization/plots/`.

### Layer 2: Extensible building blocks

DrEvalPy is built around pluggable components and registries that discover them:

- **`components/`** holds the base classes for featurizers and predictors
  ({py:class}`~drevalpy.components.featurizers.featurizer.Featurizer`, `PrecomputableFeaturizer`,
  {py:class}`~drevalpy.components.predictors.abstract.base.Predictor`, `MatrixPredictor`,
  `SingleDrugPredictor`, `FeatureFreePredictor`) together with all built-in implementations under
  `featurizers/` and `predictors/`. See {doc}`models_from_components`.
- **`registry/`** manages the registries for predictors, sample and drug featurizers, splitters,
  visualizations, and datasets, plus extension and plugin discovery
  ({py:func}`~drevalpy.registry._extensions.load_extensions`,
  {py:func}`~drevalpy.registry._plugins.discover_plugins`). When you run `import drevalpy`,
  `registry/__init__.py` calls `register_builtins()` and `discover_plugins()` so built-in and
  third-party components are immediately available. See {doc}`registries_and_plugins`.

### Layer 1: Shared foundation

Everything above builds on a small set of shared types, metrics, and utilities:

- **`types/`** defines the shared value types passed across packages:
  {py:class}`~drevalpy.types.Dataset`, {py:class}`~drevalpy.types.Fold`,
  {py:class}`~drevalpy.types.SelectionMask`, the result hierarchy
  ({py:class}`~drevalpy.types.results.ExperimentResult`,
  {py:class}`~drevalpy.types.results.ModelResult`), and the data-flow containers
  ({py:class}`~drevalpy.types.data.response_table.ResponseTable`,
  {py:class}`~drevalpy.types.data.feature_source.FeatureSource`,
  {py:class}`~drevalpy.types.data.feature_block.FeaturizerOutput`,
  {py:class}`~drevalpy.types.data.feature_format.FeatureFormat`,
  {py:class}`~drevalpy.types.data.training_context.TrainingContext`,
  {py:class}`~drevalpy.types.data.predictor_input.PredictorInput`; see {doc}`data_flow`). Its
  `__init__.py` re-exports only `Dataset`, `Fold`, and `SelectionMask`; everything else is imported
  from its defining module.
- **`evaluation/`** owns metrics and scoring: {py:class}`~drevalpy.evaluation._metrics.Metric`,
  {py:func}`~drevalpy.evaluation._scoring.evaluate`, and
  {py:func}`~drevalpy.evaluation._scoring.evaluate_per_group`. Its only internal dependency is the
  `STUDY_SEPARATOR` constant imported from `data/_concat_tables.py` for grouping pairs by study.
  (Because importing a submodule of `data` executes `data/__init__.py`, `types/` indirectly touches
  `data/` and `registry/` at import time through `evaluation/`.)
- **`errors.py`**, **`log.py`**, and **`_utils/`** provide custom exceptions, logging, and small
  shared helpers such as filename sanitizing and accelerator selection.

## Three rules that keep the layers clean

### 1. `types/` never imports higher layers

Because almost every package depends on `types/`, it never imports from `registry`, `models`,
`visualization`, or `components`. Any helper that needs a higher package lives in that higher
package instead — for example, precomputing views lives in `data/precompute.py` and plot
requirements live in `visualization/`. `types/` does import `evaluation` so result objects can
compute metrics.

### 2. Mutual dependencies stay small and explicit

Two package pairs genuinely import each other at runtime:

- **`components` ↔ `registry`**: `registry/_builtins.py` imports every built-in component module
  to register it, and those modules apply `@register_predictor` and `@register_featurizer`
  decorators imported from `registry`. If `components.featurizers.shared.concat` happens to be
  imported mid-initialization of `models.config`, `register_builtins` retries once so startup still
  succeeds.
- **`models` ↔ `execution`**: hyperparameter tuning in `models/tuning/hpo.py` uses
  `execution._tracking` to record trials. The shared fit path in `execution` calls the tuner
  and uses `models` to train the final model.

One other pair shows up when imports are aggregated at the top-level package boundary, even though
no individual module participates in a cycle:

- **`_utils` ↔ `types`**: `_utils/accelerator.py` imports the `Accelerator` enum from `types/`,
  while `types/results/` imports `_utils/filenames.py`.

### 3. Keep `import drevalpy` fast

Importing `drevalpy` should feel instant — running `drevalpy --help` or loading a dataset should
not pay the startup cost of PyTorch, scikit-learn, SciPy, pandas, or Matplotlib.

`tests/test_import_cost_policy.py` enforces this by spawning a fresh Python interpreter, importing
`drevalpy`, and asserting that none of the libraries in `FORBIDDEN_STARTUP_IMPORTS` entered
`sys.modules`. Whenever a module loaded at startup needs one of those libraries, import it inside
the function that uses it (and guard annotation-only imports with `if TYPE_CHECKING:`).

Two cases use a different pattern:

- **Subclassing a heavy library**: If a class must inherit from a forbidden library — for example,
  `GroupMeanCenterer` inherits from scikit-learn's `TransformerMixin` — place it in its own private
  module (`models/response_transformation/_group_mean.py`) and import that module lazily when
  needed.
- **Required startup side effects**: If a module must run a lightweight hook before a heavy library
  is ever imported, that hook stays eager. For instance, `components/predictors/xgboost_pred.py`
  calls `_set_xgboost_thread_defaults()` at import time to avoid OpenMP crashes on macOS.

The same test also checks `DEFERRED_TRAINING_SYMBOLS`, ensuring that symbols moved off the startup
path (such as PyTorch predictor classes) still resolve cleanly when explicitly imported. See
{doc}`/project/contributing` and `tests/test_import_cost_policy.py` for the full rules.

## Verified against `src/drevalpy/`

Every time the documentation builds, the hidden script below walks all `.py` files under
`src/drevalpy/`, extracts their module-level runtime imports, checks the `types/` boundary rule,
and prints the live package-level dependency table.

```{code-cell} ipython3
:tags: [hide-input]

import ast
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import drevalpy

src_root = Path(drevalpy.__file__).parent


class Import(NamedTuple):
    """One internal import statement."""

    source: str
    target: str
    scope: str  # "runtime", "TYPE_CHECKING", or "function-local"


def node_name(module_path: str) -> str:
    """Return the top-level package or module name for a dotted drevalpy path."""
    if not module_path.startswith("drevalpy."):
        return module_path
    parts = module_path.split(".")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return parts[1] if len(parts) > 1 else "drevalpy"


def resolve_relative_import(source_module: str, level: int, target: str | None) -> str:
    """Resolve a relative import to an absolute module path."""
    parts = source_module.split(".")
    is_package_init = parts[-1] == "__init__"

    if is_package_init:
        parts = parts[:-1]
        if level > 1:
            parts = parts[: -(level - 1)]
    else:
        parts = parts[:-level] if level <= len(parts) else parts[:1]

    if target:
        return ".".join([*parts, target])
    return ".".join(parts)


def walk_imports(root: Path) -> list[Import]:
    """Walk all .py files and extract drevalpy.* imports by scope."""
    imports = []
    for py_file in root.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        rel_path = py_file.relative_to(root)
        source_module = "drevalpy." + str(rel_path.with_suffix("")).replace("/", ".")

        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except SyntaxError:
            continue

        class ImportVisitor(ast.NodeVisitor):
            def __init__(self):
                self.imports = []
                self.in_function = 0
                self.in_type_checking = 0

            def visit_FunctionDef(self, node):
                self.in_function += 1
                self.generic_visit(node)
                self.in_function -= 1

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_If(self, node):
                is_type_check = (isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING") or (
                    isinstance(node.test, ast.Attribute) and node.test.attr == "TYPE_CHECKING"
                )
                if is_type_check:
                    self.in_type_checking += 1
                    self.generic_visit(node)
                    self.in_type_checking -= 1
                else:
                    self.generic_visit(node)

            def visit_Import(self, node):
                scope = (
                    "TYPE_CHECKING"
                    if self.in_type_checking
                    else ("function-local" if self.in_function else "runtime")
                )
                for alias in node.names:
                    if alias.name.startswith("drevalpy."):
                        self.imports.append((alias.name, scope))

            def visit_ImportFrom(self, node):
                scope = (
                    "TYPE_CHECKING"
                    if self.in_type_checking
                    else ("function-local" if self.in_function else "runtime")
                )
                if node.level == 0:
                    if node.module and node.module.startswith("drevalpy"):
                        self.imports.append((node.module, scope))
                    elif node.module == "drevalpy":
                        for alias in node.names:
                            self.imports.append((f"drevalpy.{alias.name}", scope))
                else:
                    target = resolve_relative_import(source_module, node.level, node.module)
                    if target.startswith("drevalpy."):
                        self.imports.append((target, scope))

        visitor = ImportVisitor()
        visitor.visit(tree)
        for target, scope in visitor.imports:
            imports.append(Import(source_module, target, scope))

    return imports


imports = walk_imports(src_root)
runtime_imports = [imp for imp in imports if imp.scope == "runtime"]

edges = defaultdict(int)
for imp in runtime_imports:
    source = node_name(imp.source)
    target = node_name(imp.target)
    if source != target:
        edges[(source, target)] += 1

by_source = defaultdict(set)
for src, tgt in edges:
    by_source[src].add(tgt)

cycles = []
seen = set()
for src, tgt in edges:
    if (tgt, src) in edges and (src, tgt) not in seen and (tgt, src) not in seen:
        cycles.append((src, tgt))
        seen.add((src, tgt))

types_runtime = [imp for imp in runtime_imports if node_name(imp.source) == "types"]
forbidden = sorted(
    {
        imp.target
        for imp in types_runtime
        if node_name(imp.target) in {"registry", "models", "visualization", "components"}
    }
)

print(
    f"Scanned {len(imports)} internal imports "
    f"({len(runtime_imports)} runtime, "
    f"{sum(1 for imp in imports if imp.scope == 'TYPE_CHECKING')} TYPE_CHECKING, "
    f"{sum(1 for imp in imports if imp.scope == 'function-local')} function-local)"
)
print(f"types/ runtime imports: {', '.join(sorted(by_source['types']))}")
print(f"Forbidden types/ imports: {forbidden or 'none'}")
print("Mutual package dependencies:", ", ".join(f"{s} ↔ {t}" for s, t in sorted(cycles)))
print()
print(f"{'Source':<22} Runtime targets")
print("-" * 78)
for src in sorted(by_source):
    targets = ", ".join(sorted(by_source[src]))
    if targets:
        print(f"{src:<22} {targets}")
```
