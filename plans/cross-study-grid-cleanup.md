# Cross-study by concatenation, component grids, and the cleanup bundle

## 1. Current state and goal

**Today:**

- **One dataset per experiment.** `drevalpy.run(models, dataset, split_mode, ...)`
  (`src/drevalpy/_run.py`) takes one `Dataset`. `ExperimentResult` rejects runs with mixed
  `dataset_name`s. Nothing can merge two MuData files. `CrossStudyTableVisualization`
  looks for `cross-study-*` runs that nothing produces. The docs say v2 has no cross-study
  path (`explanation/evaluation_design.rst:160-170`, `how-to/migrate/*.rst`).
- **Drugs across studies.** Drug columns are PubChem CIDs, but each dataset picks its own
  SMILES in `dreval_curation` `map_drugs`.
  - GDSC strips stereochemistry and CTRP keeps it; only 1 of 75 shared GDSC2/CTRPv2 drugs has
    the same string.
  - Three molecules (sildenafil, XAV-939, pemetrexed) have different CIDs in different studies.
- **Models.** The CLI takes zoo names only (`ModelSpec.from_name` → `zoo_model_config`).
  Recipes (`sample:drug:predictor`) parse only through `models.config.io.from_spec`.
  Config validation (`models/config/validation.py::validate`, about 1 ms) rejects
  incompatible combinations. It ignores the name-only `required_sample_blocks` /
  `required_drug_blocks`, so `scaledGeneExpression:fingerprints:naiveTissueMean` validates.
- **The run loop** (`product(models, folds)` → `randomization()` → `single()`):
  - has no error handling, so one failing fit aborts the experiment;
  - rebuilds the randomized datasets for every fold, although they don't depend on it.
- **Features are aligned by position.** `DenseViewFeaturizer` stores no feature names, and
  `pca` silently zero-fills missing ones. Applying a model to a dataset whose columns are in
  a different order mixes up genes without any error.
- **Smaller gaps:**
  - The curve-quality filter is hard-wired in `_folds.observed_mask`; a dataset without
    the quality layers raises `KeyError` when split.
  - W&B is not reachable from `run`/`single`.
  - ElasticNet's untuned `alpha=1.0` predicts a constant.
  - `tests/synthetic/builders.py` and `drevalpy.synthetic()` are two builders.
  - 8 docstrings say "feature view".

**Goal:**

1. **Cross-study.** `drevalpy.data.concat([GDSC2, CTRPv2])` returns one clean `Dataset`, so
   every existing function works on it unchanged. The new `LStO` split trains on one study
   and tests on another. `summary_table(group_by="study")` reports per study. `run` and the
   CLI accept several datasets and concatenate them. Curated files give each drug ID one
   canonical SMILES.
2. **Grids.** `drevalpy.models.combinations(sample=[...], drug=[...], predictors=[...])` and
   `drevalpy run -s ... -d ... -p ...` evaluate every valid combination and report the
   skipped ones. A failing model no longer aborts the run.
3. **Correctness.** Featurizers check feature names at predict time, and predictors' declared
   inputs are enforced when a model is composed.
4. **Cleanup:**
   - a curve-quality rule object with an opt-out;
   - W&B for trials, final fits and per-epoch curves;
   - ElasticNet default `alpha=0.1`;
   - randomized datasets built once per model;
   - one synthetic builder;
   - the docstring wording.

**Non-goals:**

- Fitted-featurizer reuse across models (decided: no).
- Unifying units across studies. `concat` raises on a unit mismatch; it does not convert
  `log2_cpm` to TPM.
- nf-core/drugresponseeval on v2, and the 2.0.0 release and branch merge.
- Making the bucket public.

## 2. Approach

Cross-study is a **data operation, not an experiment mode**. `concat` produces an ordinary
`Dataset`:

- Each row is one (study, sample), with ID `<study>::<sample_id>`.
- `obs` gains `study` and `source_sample_id`.
- Omics rows are stacked per view.
- Drug columns are the union of CIDs.
- Response layers are the union, NaN-filled.

`split`, `single`, HPO, randomization, `aggregate` and the report then need no changes. The
`LStO` splitter is the only cross-study-specific piece: one fold per study, built like
`LTO` but on `obs["study"]`. Per-study evaluation needs no new results field, because the
study is the prefix of every recorded sample ID.

The grid is a **thin expansion layer on top of recipes**. `ModelSpec.from_name` learns to
accept recipe strings and names them by the full recipe string, which is unique. The
expansion function validates each product with the existing config-time validator and
returns the valid specs plus the skipped ones with their reasons. Enforcing predictors'
declared inputs makes that validator strict enough for grids to be useful.

The run loop changes only in `_run.py`:

- a `try` per run that records `RunFailure`s;
- randomization hoisted out of the fold loop;
- dataset lists concatenated;
- a `CurveQuality` rule passed to `split`;
- `wandb_project` passed through.

Applying the quality rule **before** the splitter keeps the splitter protocol unchanged. The
splitters then only read `isfinite(response)`, so plugin splitters keep working.

Curation is a separate, parallel track. A new shared stage resolves each CID to one
canonical CID (by InChIKey) and one SMILES (PubChem isomeric, then RDKit). The harmonize
stages read it. No curve fit stage depends on it.

**Decisions:**

- **Cross-study = concat + split:** user's design. Every existing function that takes a
  dataset works unchanged.
- **One row per study and sample, always prefixed `<study>::<id>`:**
  - no measurement is lost (33,492 GDSC2/CTRPv2 pairs are measured twice);
  - the study can be read from any `RunResult` sample ID.
- **Mismatched views:** feature names or units differ → `ValueError`. `join="inner"` keeps the
  shared features; a unit mismatch always raises (user: missing feature = error).
- **Drug SMILES:** curation guarantees one SMILES per CID (PubChem isomeric + RDKit). `concat`
  raises on a conflict instead of choosing one.
- **One CID per InChIKey** in curation, using the lowest CID; the other IDs go to
  `native_ids`.
- **Re-upload:** overwrite `gen3/`; bucket versioning keeps the old files.
- **New `LStO` splitter** (user), not a generalized LTO:
  - one fold per study;
  - validation is a random `validation_ratio` share of the training pairs;
  - the test study's samples and drugs may also appear in training.
- **Response layers:** union, NaN-filled.
- **Delete `CrossStudyTableVisualization`.** LStO runs appear in the normal tables.
- **`group_by="study"`** for `summary_table`, `--group-by` and the report, read from the
  `::` prefix. Runs without a prefix give NaN, like other undefined groups.
- **Entry points:** `drevalpy.data.concat`, `drevalpy data concat`, `run(dataset=[...])` and a
  repeatable `run --dataset`. `single` stays path-only.
- **Feature names are checked at predict** in every dense featurizer, missing = error. `pca`
  stops zero-filling. Checkpoint `FORMAT_VERSION` 5; v4 is rejected with a retrain message.
- **Grid interface:**
  - `drevalpy.models.combinations(...)` in Python;
  - on the CLI, `run` takes zoo names or recipe strings as MODEL, plus repeatable
    `--sample-featurizer/-S`, `--drug-featurizer/-D`, `--predictor/-P`. `-d` and `-s`
    already mean dataset and split mode, so the grid flags take capitals.
- **Invalid combinations** are skipped with a report; the grid raises only if none is valid.
- **Grid model names** are the full recipe strings including views.
- **No featurizer reuse** across models.
- **Failures** are recorded and the run continues:
  - `ExperimentResult.failures: tuple[RunFailure, ...]`, saved to `failures.json`;
  - the CLI exits 1 if there were any;
  - if every run fails, `run` raises.
- **Curve quality** is a `CurveQuality` rule object, applied before the splitter.
  - Constructors `CurveQuality.default()` and `.off()`.
  - Python: `split(quality=)` and `run(quality=)`.
  - CLI: `--no-quality-filter`, `--min-relevance-score` and `--min-abs-fold-change` on
    `run` and `data split`.
  - The rule is recorded in fold metadata.
- **W&B:** one run per trial and one final run per fold with test metrics (like 1.x), plus
  per-epoch curves from the Lightning predictors.
- **ElasticNet** default `alpha=0.1`; the HPO space is unchanged.
- **Randomized datasets** are built once per model, not per fold.
- **One synthetic builder.** `synthetic()` gains opt-in extras, and the test builder becomes
  a call into it.
- **Docstring wording** "feature view" → "view", done by the package that owns each file.
- **Curation** is a separate, parallel track.

## 3. Technical design

### Modules

| Module | Interface (small) | Hides | New / changed |
| --- | --- | --- | --- |
| `data/_concat.py` | `concat(datasets: Sequence[Dataset], *, join: Literal["exact","inner"]="exact", name: str \| None=None) -> Dataset` | ID prefixing, `obs` union with `study`/`source_sample_id`, per-view row stacking, feature-name and unit checks, inner join, response union with NaN-filled layers, drug `var` union with a SMILES/metadata conflict check, `uns` merge (keeps `dataset_name` = `name` or `"A+B"`), rejection of randomized inputs | new |
| `data/splitters/lsto.py` | registered splitter `LStO` (protocol unchanged) | reads studies via `sample_column("study")`, one fold per study, random validation pairs, error if fewer than 2 studies | new |
| `data/quality.py` | `CurveQuality` (frozen; `default()`, `off()`, `mask(ds) -> ndarray`, `to_metadata() -> dict`) | the threshold table and the layer lookups; `off()` needs no layers | changed (`curve_quality_mask` kept as a thin wrapper) |
| `data/__init__.py::split` | `split(dataset, mode, n_splits=5, validation_ratio=0.1, random_state=42, *, quality: CurveQuality \| None = None)` | builds a quality-masked `MuDataLike` view, calls the splitter, records `quality` in metadata | changed |
| `data/splitters/_folds.py` | `observed_mask(ds) = isfinite(ds.response_matrix)` | — | changed |
| `types/data/mudatalike.py` + `dataset.py` | `sample_column(column, ids) -> ndarray` (`get_tissue` delegates to it) | missing IDs → NaN, missing column → `KeyError` | changed |
| `models/_combinations.py` | `combinations(*, sample=(), drug=(), predictors) -> Combinations(specs: list[ModelSpec], skipped: list[SkippedCombination(recipe, reason)])` | product, recipe formatting, validation, de-duplication; raises `ValueError` if nothing is valid | new |
| `models/spec.py::ModelSpec.from_name` | accepts a zoo name or a recipe string | recipe parse; name = the recipe string as given | changed |
| `models/config/validation.py` | `validate(config)` (unchanged) | also enforces `required_{side}_blocks` names against emitted blocks for every predictor | changed |
| `types/results/failure.py` | `RunFailure(model_name, fold_index, randomization, robustness_trial, error_type, message)` | JSON round-trip | new |
| `types/results/experiment.py` | `ExperimentResult(run_results, failures=())`; `.failures`; save/load `failures.json` | allows empty `run_results` only when `failures` is non-empty | changed |
| `_run.py::run` | `run(models, dataset: Dataset \| str \| Sequence[Dataset \| str], split_mode, ..., quality=None, wandb_project=None, on_error: Literal["record","raise"]="record")` | concatenation, randomization hoisting, per-run `try` | changed |
| `_single.py::single` | `single(..., wandb_project=None)` | the final-fit W&B run | changed |
| `experiment/_tracking.py` | `tracking_run(project, *, name, config, tags) -> ContextManager`; `log_summary(metrics)` | lazy `wandb` import; no-op when the project is `None` | new |
| `components/predictors/literature/_lightning_training.py` | unchanged | attaches `WandbLogger(experiment=wandb.run)` when a tracking run is active (checked via `sys.modules`, not an import) | changed |
| `components/featurizers/_dense_view.py` (+ `pca`, `landmark`, `scaled_gene_expression`, `normalized_proteomics`, the literature omics featurizers) | unchanged `fit`/`transform` | stores `feature_names` at fit; reindexes by name at transform; `FeatureMismatchError(ValueError)` lists missing names | changed |
| `models/mixins/_persistence_io.py` | `FORMAT_VERSION = 5` | v4 → retrain message | changed |
| `data/_synthetic.py::synthetic` | adds `omics` values for all 5 views, `drug_views=()`, `uns_payloads=()`, `extra_layers=()` | lazy RDKit/subword-nmt imports | changed |
| `_grouped_evaluation.py` | `GROUPINGS = ("drug","sample","study")` | `study` label = prefix before `::`, else NaN | changed |
| `cli/data/concat.py` | `drevalpy data concat DATASET... OUTPUT [--join inner] [--name]` | `load` per argument, `Dataset.save` | new |
| curation `drugs/resolve_structures.py` (stage `resolve_drug_structures`) | `data/03_drugs/structures.parquet`: `pubchem_id → canonical_id, canonical_smiles, inchikey` | PubChem isomeric SMILES, RDKit canonicalization, InChIKey → lowest CID, fallback to the study's SMILES for pseudo-IDs | new |

Deletion test:

- **`_tracking.py` earns its place.** Without it, `wandb` init, tags and finish code would be
  repeated in `hpo.py`, `_single.py` and the Lightning helper, and the import-cost guard would
  need a check in each.
- **`failure.py` stays a separate module.** It is a data type that both `ExperimentResult`
  and the CLI read.
- **`Combinations` stays a type.** It is the return type, so the skipped list can be printed.

### Call trees

**Concatenate and split across studies:**

```text
drevalpy.data.concat([gdsc2, ctrpv2]) -> Dataset                    [new]
├─ _check_inputs(datasets)            # ≥2, unique names, none randomized
├─ _stack_views(datasets, join)       # per view: var_names / uns unit equal, or intersect
├─ _stack_obs(datasets)               # <study>::<id>, obs.study, obs.source_sample_id
├─ _union_response(datasets)          # rows stacked, columns = CID union, layers union NaN
└─ _union_drug_var(datasets)          # same CID → equal canonical_smiles, else ValueError
   -> Dataset(MuData, name="GDSC2+CTRPv2")

drevalpy.split(ds, "LStO", quality=None) -> list[SplitMasks]        [changed]
├─ rule = quality or CurveQuality.default()                         [new]
├─ view = _QualityMasked(ds, rule.mask(ds))   # MuDataLike, failed pairs NaN   [new]
├─ splitter = splitter_registry.get("LStO")
│  └─ lsto(view, n_splits, validation_ratio, random_state)          [new]
│     ├─ studies = view.sample_column("study", view.sample_ids)     [new]
│     ├─ observed = observed_mask(view)          # isfinite only   [changed]
│     └─ per study s: test = rows(s) ∧ observed; train/val = random pairs of the rest
│     -> list[SplitMasks]   (len = n studies; n_splits ignored, recorded in metadata)
│  └─ _wrap_with_validation → validate_folds(..., "LStO")           [changed]
└─ metadata: dataset, split_mode, fold_index, fold_id, quality=rule.to_metadata()   [changed]
```

**`run` with datasets, a grid and failures:**

```text
cli.run(models, --dataset A --dataset B, -S pca -S raw[gene_expression] -P elasticNet, ...)   [changed]
├─ specs = [ModelSpec.from_name(m) for m in models]       # zoo or recipe   [changed]
├─ if grid flags: grid = combinations(sample=S, drug=D or ("fingerprints",), predictors=P)   [new]
│  ├─ print grid.skipped (recipe: reason)
│  └─ specs += grid.specs
├─ quality = CurveQuality.from_cli(no_filter, min_rel, min_fc)     [new]
└─ result = drevalpy.run(specs, [A, B], split_mode, quality=..., wandb_project=...)   [changed]
   ├─ ds = concat([load(A), load(B)]) if several else load(...)    [changed]
   ├─ folds = split(ds, split_mode, random_state=seed, quality=quality)   [changed]
   ├─ for model in specs:
   │  ├─ rand = randomization(model, ds, ...)    # once per model   [changed]
   │  └─ for fold in folds (+ robustness copies):
   │     └─ for run_ds in [ds, *rand]:
   │        try:  single(model, run_ds, fold, ..., wandb_project)    [changed]
   │        except Exception as e: failures.append(RunFailure.from_exception(...))   [new]
   └─ -> ExperimentResult(run_results, failures)   # raises ExperimentFailedError if no runs   [changed]
   result.save(out)  → also failures.json                           [changed]
   print failures; exit 1 if any                                    [new]
```

**`single` with W&B:**

```text
single(spec, ds, masks, ..., wandb_project)                          [changed]
├─ build_execution(...)
└─ _execute(execution, wandb_project)
   ├─ hpam_tune(..., split_index=fold_index, wandb_project, wandb_base_config)   [changed: now passed]
   │  └─ per trial: tracking_run(project, name=f"{model}_split{i}_trial{n}") ...  [changed: uses _tracking]
   ├─ with tracking_run(project, name=f"{model}_split{i}_final", tags=[model, split_mode, dataset, "final"]):   [new]
   │  ├─ DRPModel.train(...)  → Lightning predictors attach WandbLogger(experiment=wandb.run)   [changed]
   │  ├─ predict, score_predictions
   │  └─ log_summary({f"test_{k}": v for k, v in metrics})          [new]
   └─ -> RunResult
```

**Predict with name checks:**

```text
DRPModel.predict(dataset, scope)
└─ _ModelPipeline.predict → transform_entities(featurizer, source, ids)
   └─ DenseViewFeaturizer.transform(source, ids)                     [changed]
      ├─ names = source.get_feature_names(view)
      ├─ missing = self._feature_names - names → FeatureMismatchError if any   [new]
      └─ X = matrix[:, index_of(self._feature_names)]   # reorder by name   [new]
```

**Curation (dreval_curation):**

```text
map_drugs ×7, map_drugs_PRISM ×2                     (unchanged, not rerun)
└─ resolve_drug_structures                           [new stage]
   ├─ union of data/03_drugs/mapping/*.parquet CIDs
   ├─ PubChem `smiles` (isomeric) per CID, cached     → RDKit MolToSmiles
   ├─ InChIKey → lowest CID (canonical_id)
   └─ data/03_drugs/structures.parquet
harmonize_response ×9 / harmonize_per_replicate ×9 / assign_random_replicates ×9   [changed: read structures]
annotate_drugs                                       [changed: canonical ids]
assemble_* ×9, subsample_toy, report                 (rerun)
check_drug_structures                                [new: one SMILES per CID across all 07_assembled files]
fit_curves* ×18                                      (unaffected: no drug deps)
```

### Tests

- **`concat`:**
  - two `synthetic(name=...)` datasets with overlapping sample and drug IDs: the prefixes,
    `obs.study`/`source_sample_id`, the stacked view values, the response layer union with
    NaN, and drug `var` union;
  - errors: different var_names (exact), the inner join keeping the intersection, a unit
    mismatch raising even with inner, a SMILES conflict, a randomized input, fewer than 2
    inputs, duplicate names;
  - the result round-trips through `save`/`load`, and `split` on it with LPO works.
  - Slow: GDSC2 + CTRPv2 from the cache (`network`), 576 shared samples become 1,152 rows.
- **`LStO`:**
  - one fold per study, the test equals all quality-pass pairs of that study, the validation
    share, and the masks disjoint;
  - fewer than 2 studies → `ValueError`; a dataset without `study` → `KeyError` with a hint
    to use `concat`.
  - `validate_folds("LStO")` rejects a study in both train and test.
- **`CurveQuality`:**
  - `default()` equals the old `curve_quality_mask` output;
  - `off()` splits a dataset without quality layers;
  - custom thresholds; the metadata round-trips through `SplitMasks.save`;
  - the CLI flags map onto the rule;
  - a plugin splitter that ignores quality still sees NaN on failed pairs.
- **`combinations`:**
  - valid, skipped with reasons, de-duplication, nothing valid → `ValueError`;
  - names unique and equal to the recipe;
  - `ModelSpec.from_name` accepts a recipe and still accepts zoo names; an unknown token
    raises.
- **`validate` (naive inputs):** `scaledGeneExpression:fingerprints:naiveTissueMean` is
  rejected; every zoo preset still validates (parametrized over `list_zoo()`).
- **`run`:**
  - a stub predictor that raises on one fold → `ExperimentResult.failures` has one entry and
    the other runs are present;
  - `failures.json` round-trips;
  - CLI exit code 1 and the printed list;
  - every run fails → `ExperimentFailedError`; `on_error="raise"` re-raises;
  - `randomization` is called once per model (spy);
  - a dataset list concatenates.
- **Feature names:**
  - a permuted var order gives the same predictions as the original;
  - a missing gene → `FeatureMismatchError`;
  - `pca` no longer zero-fills;
  - a v4 checkpoint → the retrain message; a v5 checkpoint round-trips.
- **W&B:**
  - with a fake `wandb` module in `sys.modules`: one run per trial plus one final run per
    fold with `test_*` summary keys;
  - Lightning predictors attach a logger only when a run is active;
  - `import drevalpy` still imports no `wandb` (import-cost guard).
- **`group_by="study"`:** a concat run gives one row per study; a plain run gives NaN rows
  that are skipped.
- **`synthetic` extras:** each opt-in extra is present and loads through the matching
  featurizer; the test builder's output is equal to the pre-change output (snapshot of
  shapes and names).
- **ElasticNet:** the untuned default on `synthetic()` does not predict a constant.
- **Curation:** a unit test of `resolve_structures` on a fixture (a stereo and a non-stereo
  input for one CID give one SMILES; two CIDs with one InChIKey collapse to the lowest); the
  `check_drug_structures` stage passes on the outputs.

### Risks and open questions

- **Required-block enforcement can reject current zoo presets** whose featurizers emit
  blocks under other names (e.g. SparseGO declares `fingerprints`). Mitigation: P4's
  done-when check validates every zoo preset; a mismatch is fixed in the declaration, not by
  weakening the check.
- **`MuDataLike` gains `sample_column`.** Third-party implementations, if any, break.
  `drevalpy.testing` conformance fixtures need the method. It goes in news.rst as breaking.
- **Plugin splitters that call `curve_quality_mask` themselves** now filter twice. That is
  harmless, since the filter is idempotent, but `quality=off()` won't reach them. Documented
  in the splitter how-to.
- **Memory:** concatenating GDSC2 and CTRPv2 holds both full MuData objects plus the result
  (about 1 GB on disk). It is fine on this machine; the how-to says so.
- **Scale differences across studies** (GDSC2 mean pEC50 6.31, CTRPv2 5.55) dominate LStO
  RMSE. The explanation page says so and points to the correlations and
  `--response-transformation`. Per-study normalization stays out of scope.
- **The PubChem isomeric SMILES lookup** is about 2,500 network requests. The stage caches
  them, and a CID PubChem lacks keeps the study's SMILES (RDKit-canonical), with a warning.
- **A new checkpoint format** invalidates saved v4 models. That is accepted, since v2 is
  unreleased.
- **Re-uploading over `gen3/`** was decided here. The upload itself still waits for a
  go-ahead at the time, after C1's checks.

## 4. Work packages

Run `export PATH=$HOME/.local/bin:$PATH` before `uv`. Subagents use
`amazon-bedrock/eu.anthropic.claude-sonnet-5-5`. Shared checks:

- `CHECK_FAST`: `uv run --frozen pytest -q -p no:cacheprovider <listed tests>`.
- `HOOKS`: `SKIP=pytest uv run --frozen prek run --all-files`. Run `git add -N` on new files
  first.
- Size and coverage gates: `uv run --frozen python tools/size_gate.py`.

Run one heavy job at a time (earlyoom). No package commits; the coordinator commits per
package after review.

### Wave 0: shared interfaces (one package, drevalpy)

#### W0: stubs and small shared types

- **Files owned:**
  - `types/data/mudatalike.py`, `types/data/dataset.py` (`sample_column`, fully implemented;
    `get_tissue` delegates);
  - `data/quality.py` (`CurveQuality` fully implemented, `curve_quality_mask` delegating);
  - `types/results/failure.py` (new, full);
  - `types/results/experiment.py` (`failures` param, `failures.json` save/load, the
    empty-runs rule);
  - stubs that raise `NotImplementedError` with final signatures and docstrings:
    - `data/_concat.py::concat`
    - `models/_combinations.py::combinations` / `Combinations` / `SkippedCombination`
    - `experiment/_tracking.py::tracking_run` / `log_summary`
  - the exports in `data/__init__.py` and `models/__init__.py`;
  - signature-only additions: `split(quality=)`, `run(dataset: ... | Sequence, quality=,
    wandb_project=, on_error=)`, `single(wandb_project=)`, each accepted and passed nowhere
    yet;
  - `Validation` Literal plus the `_execution.py` tuple gaining `"LStO"`, with a validator
    stub;
  - `GLOSSARY.md`: a new term **study** ("one screen's dataset, e.g. GDSC2; in a
    concatenated dataset the `study` column of `obs`").
- **Depends on:** nothing.
- **Interface:** everything in the Modules table marked "interface". Stubs raise
  `NotImplementedError("W1-<package>")`.
- **Done when:**
  - `uv run --frozen pytest -q -m "not network"` is green;
  - the new types have their tests (`tests/types/results/test_failure.py`, the `sample_column`
    and `CurveQuality` tests);
  - `test_import_cost_policy.py` passes;
  - `HOOKS` passes.
- **Notes:** `dataset.py` is at its size-gate ceiling. Put `sample_column` in
  `dataset_utils/` as a mixin or helper, as `_stored_identity.py` did.

### Wave 1: parallel packages (disjoint files; drevalpy P1–P7, curation C1)

#### P1: `concat`

- **Files owned:** `data/_concat.py`, `cli/data/concat.py` (plus registration in
  `cli/data/__init__.py`), `tests/data/test_concat.py`, `tests/cli/data/test_concat.py`.
- **Depends on:** W0.
- **Interface:** `concat(datasets, *, join="exact", name=None) -> Dataset`;
  `drevalpy data concat DATASET... OUTPUT [--join exact|inner] [--name NAME]` (arguments
  resolved with `drevalpy.data.load`).
- **Done when:** the `concat` tests from §3 pass; GDSC2 + CTRPv2 from
  `DREVALPY_CACHE_DIR=/tmp/opencode/r4cache` concatenate (1,854 rows, CID union) and
  `split(..., "LPO")` works on the result; `HOOKS` passes.
- **Notes:**
  - Until C1 lands, the cached files have conflicting SMILES. Test the conflict error with
    them, and the happy path with synthetic data.
  - Never mutate the inputs.
  - `uns` keys other than `dataset_name` must be equal across inputs, or be dropped with a
    warning.

#### P2: `LStO`, `group_by="study"`, delete the cross-study table

- **Files owned:**
  - `data/splitters/lsto.py`, `data/splitters/_validation.py` (LStO validator);
  - `_grouped_evaluation.py`, `types/results/model.py` and `_summary.py` (grouping checks),
    `cli/_summary.py`;
  - `visualization/plots/_group_metrics.py`, `comparison_scatter.py`;
  - delete `visualization/plots/cross_study_table.py`, its registration and its tests;
  - the tests mirroring these files.
- **Depends on:** W0.
- **Interface:** splitter `LStO` (validation `"LStO"`); `GROUPINGS` includes `"study"`.
- **Done when:** the LStO and group_by tests from §3 pass; `drevalpy list splitters` shows
  `LStO`; the report builds on a concat run with no cross-study plot; `HOOKS` passes.
- **Notes:** LStO ignores `n_splits` (one fold per study) and records the real count in
  metadata. Its validator checks that `study` sets are disjoint between train/val and test.

#### P3: quality before the splitter

- **Files owned:** `data/__init__.py::split` body, `data/splitters/_folds.py`, `lpo.py`,
  `lso.py`, `ldo.py`, `lto.py`, `cli/data/split.py`, `cli/_quality.py` (new: shared Typer
  options plus `CurveQuality.from_cli`), and their tests.
- **Depends on:** W0.
- **Interface:** `split(..., quality=None)`; the `_QualityMasked` view (private); CLI options
  `--no-quality-filter`, `--min-relevance-score`, `--min-abs-fold-change`, reused by P7.
- **Done when:**
  - with the default rule, every existing split is mask-identical (compare `fold_id`s on
    `synthetic()` and TOYv1 before/after);
  - `off()` splits a dataset without quality layers;
  - the metadata round-trips;
  - `HOOKS` passes.

#### P4: grid and enforced predictor inputs

- **Files owned:**
  - `models/_combinations.py`, `models/spec.py`, `models/config/validation.py`;
  - the naive predictors' declarations if they need correction (`naive/*.py`);
  - `models/config/model.py` and `registry/featurizer/_validate.py` (docstring wording);
  - the tests mirroring these files.
- **Depends on:** W0.
- **Interface:** `combinations(...) -> Combinations`; `ModelSpec.from_name(zoo_or_recipe)`.
- **Done when:**
  - the combinations and validation tests pass;
  - every `list_zoo()` preset validates;
  - `combinations(sample=["raw[gene_expression]","pca[gene_expression]"],
    drug=["fingerprints"], predictors=["elasticNet","naiveTissueMean"])` returns 2 specs and
    2 skipped;
  - `HOOKS` passes.

#### P5: feature names at predict, checkpoint v5

- **Files owned:**
  - `components/featurizers/_dense_view.py`, `pca.py`, `landmark.py`,
    `scaled_gene_expression.py`, `normalized_proteomics.py`, `pharmaformer_gene_expression.py`,
    `molir_omics`, `superfeltr_omics`;
  - `models/mixins/_persistence_io.py`;
  - the docstring wording in `featurizers/base.py`, `_declarations.py`, `drug/base.py`,
    `sample/base.py`, `types/data/dataset_utils/feature_access.py`;
  - the tests mirroring these files.
- **Depends on:** W0.
- **Interface:** unchanged `fit`/`transform`; `FeatureMismatchError(ValueError)` exported
  from `drevalpy.components.featurizers`; `FORMAT_VERSION = 5`.
- **Done when:** the feature-name tests pass; `tests/models` and `tests/components` are green;
  the size gate passes (lower ceilings where modules shrink); `HOOKS` passes.

#### P6: W&B tracking

- **Files owned:**
  - `experiment/_tracking.py`, `_single.py`, `cli/single.py` (`--wandb-project`);
  - `models/tuning/hpo.py` (use `_tracking`);
  - `components/predictors/literature/_lightning_training.py` and the predictors that bypass
    it for per-epoch metrics (pharmaformer, precily);
  - `models/mixins/_logging.py` (delete it if `_tracking` replaces it: deletion test);
  - the tests mirroring these files.
- **Depends on:** W0.
- **Interface:** `single(..., wandb_project=None)`; `tracking_run`, `log_summary`.
- **Done when:** the W&B tests from §3 pass with a fake `wandb`; the import-cost guard passes;
  `HOOKS` passes.

#### P7: run loop

- **Files owned:**
  - `_run.py`, `cli/run.py`;
  - `experiment/_randomization.py` (only if the signature needs it);
  - `visualization/report.py` (a "Failed runs" table when `failures` is non-empty);
  - `tests/test_run.py`, `tests/cli/test_run.py`, `tests/visualization/test_report.py`.
- **Depends on:** W0. It calls P1/P3/P4/P6 only through W0's signatures, so it can be
  written in parallel. Its end-to-end test runs in Wave 2.
- **Interface:**
  - `run(..., dataset=str|Dataset|Sequence, quality=, wandb_project=, on_error=)`;
  - CLI: repeatable `--dataset`, MODEL accepts recipes, `-S/-D/-P`, the quality options
    from `cli/_quality.py`, `--wandb-project`, exit 1 on failures.
- **Done when:** the run tests from §3 pass (with the stubs replaced by fakes); `HOOKS`
  passes.

#### P8: ElasticNet default and one synthetic builder

- **Files owned:** `components/predictors/sklearn_models.py`, `data/_synthetic.py`,
  `tests/synthetic/builders.py`, `testing/synthetic.py`, and their tests.
- **Depends on:** W0.
- **Interface:** `synthetic(..., drug_views=(), uns_payloads=(), extra_layers=())`; `omics`
  accepts all five views.
- **Done when:**
  - the synthetic and ElasticNet tests pass;
  - `tests/models/test_global_models.py` and `test_single_drug_models.py` are green with
    the test builder delegating;
  - the import-cost guard passes;
  - `HOOKS` passes.

#### C1: one SMILES per drug ID (dreval_curation)

- **Files owned:** in `/data/Software/dreval/dreval_curation`:
  - `src/data_curation/drugs/resolve_structures.py` (new);
  - `response/harmonize_response.py`, `harmonize/collapse.py` if needed;
  - `drugs/annotate_drugs.py`;
  - `dvc.yaml` (the new `resolve_drug_structures` and `check_drug_structures` stages, plus
    deps);
  - tests.
- **Depends on:** nothing.
- **Interface:** `data/03_drugs/structures.parquet` (`pubchem_id, canonical_id,
  canonical_smiles, inchikey`); the assembled `response.var` keeps its columns, with
  `canonical_smiles` now consistent.
- **Done when:**
  - `dvc repro` reruns only the stages listed in the curation call tree (no `fit_curves*`; check with
    `dvc status` beforehand);
  - `check_drug_structures` passes;
  - GDSC2/CTRPv2 shared CIDs have 75/75 identical SMILES;
  - sildenafil, XAV-939 and pemetrexed have one CID each;
  - curation tests and hooks pass.
  - Then **ask before**: `dvc commit`/`dvc push`, and the upload over `gen3/`.
- **Notes:**
  - Cache the PubChem responses under `data/03_drugs/`.
  - PRISM multi-structure strings take the CID's PubChem SMILES.
  - Pseudo-IDs keep the study's SMILES, RDKit-canonical when it parses.

### Wave 2: integration and docs

#### I1: integration check (coordinator)

- **Depends on:** all of Wave 1.
- **Done when:**
  - the full suite passes: `pytest -m "not network" --cov --cov-report=json`, plus
    `coverage_gate.py`, `size_gate.py`, DOCS and HOOKS;
  - an end-to-end run passes:
    `drevalpy run ElasticNet "pca[gene_expression]:fingerprints:ridge" -d GDSC2 -d CTRPv2 -s LStO --no-hpo --group-by study`
    on the cache, giving 2 folds, per-study rows and no failures;
  - with C1 uploaded, the network snippets pass.

#### D1: docs

- **Files owned:**
  - new how-tos `docs/how-to/run_cross_study.rst` and `evaluate_component_grids.rst`;
  - `reference/split_modes.rst` (LStO), `reference/datasets.rst` (concat; recount after C1),
    `reference/metrics.rst` (`group_by="study"`);
  - `explanation/evaluation_design.rst` (cross-study now via concat + LStO; the scale caveat);
  - `how-to/migrate/{python,cli}.rst` (`cross_study_datasets` → concat + LStO; `wandb_project`
    is back);
  - `how-to/choose_models.rst` (recipes on the CLI);
  - the extend-splitter how-to (quality is applied before the splitter);
  - `project/news.rst`, including the breaking items: checkpoint v5, `MuDataLike.sample_column`,
    stricter validation, `CrossStudyTableVisualization` removed, ElasticNet default;
  - `plans/docs-diataxis-followups.md` (remove the items done here).
- **Depends on:** I1.
- **Done when:** DOCS, `tests/docs` and the term sweep pass; every new snippet runs, with the
  cross-study one marked `network`.
