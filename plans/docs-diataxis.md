# Restructure the drevalpy v2 docs by Diátaxis mode

## 1. Current state and goal

**Today:** `docs/index.rst` has five toctrees: Getting started, Concepts, Python guide,
CLI guide and Project. The Python and CLI tracks repeat the same topics:
`python/{datasets,models,experiments,visualization,extensions}.rst` against
`cli/{datasets,models,experiments,visualization,extensions}.rst`. Most pages mix modes.
For example, `python/extensions.rst` (612 lines) is part reference table, part how-to and
part explanation, and `concepts/from_components_to_models.rst` (486 lines) mixes the recipe
grammar with a how-to on search spaces. There is no tutorial: `cli/quickstart.rst` and
`python/quickstart.rst` are one block each, with no expected output.

About 25 statements contradict the code (see the fact list in section 3). They survived
because `tests/docs/test_model_workflows.py` runs Python blocks from only three pages, and
only blocks at column 0.

The four `project/v2_*.rst` pages describe changes between internal v2 development states,
not between 1.6.0 and v2. `docs/project/architecture_audit/` and most of
`docs/project/literature_audit/` are agent work logs.

The built-in datasets in `s3://orakl-open-source-data/drevalpy_h5mu/gen3/` return
`403 Forbidden` without AWS credentials, so a new user cannot load any of them.

**Goal:**

- The docs have four sections: Tutorials, How-to guides, Reference and Explanation, plus
  Project.
- Each how-to shows its task once, with synced **CLI** and **Python** tabs.
- Three tutorials run offline on a synthetic dataset.
- Three how-tos cover migrating from 1.x: one for CLI users, one for Python users and one
  for model authors.
- `GLOSSARY.md` holds one name per concept and is rendered in the reference.
- A docs test runs every snippet on the tutorial and how-to pages, so stale snippets fail
  CI.
- Five small code gaps found during the audit are closed, so that the docs can describe
  one consistent behaviour.
- "Cell line" is renamed to "sample" in code, tests, docs and the two sister repos
  (drevalpy-plugin and dreval_curation), because the datasets also hold organoids, PDX
  and primary patient cells. "Cell line" stays only where data really are cell lines.

**Non-goals:**

- Redirects for old URLs. Old bookmarks will return 404; only the README links are
  updated.
- Making the S3 bucket public.
- A cross-study entry point. The docs will state that v2 has none.
- W&B logging from `run`, which is not wired through today.
- Renaming code parameters to match the glossary, apart from the sample rename in
  Wave R. `split(mode=)` stays, and so does the `experiments` CLI group.
- A fixed obs index name, or a `sample_type` column. See the dataset schema rule in
  section 2.
- Renaming LCO in nf-core/drugresponseeval. The LCO alias keeps it working.
- Merging `tests/synthetic/builders.py` into the new public synthetic dataset. The test
  suite keeps its richer builder.

## 2. Approach

Build the new tree next to the old one. Wave 0 creates every new page as a titled stub,
points `index.rst` at them, and adds the old directories to `exclude_patterns`. That way
writers can read the old files in place while the build stays green. Each later docs
package owns a disjoint set of new pages and copies, fixes or drops content from the old
pages. The last wave deletes the old files and the exclusions.

Code changes go first, in Wave 1. They are small and independent, and the pages written in
Wave 2 describe the behaviour they produce. The snippet runner also lands in Wave 1, so
that every docs package in Wave 2 can prove its own pages run.

**Decisions:**

- Top level is the four modes. CLI and Python are synced tabs inside how-tos and
  tutorials: the user wanted one task shown once.
- Scope is restructure, fix and fill gaps: moving pages without fixing them would only
  relocate about 25 errors.
- Delete `project/v2_migration.rst`, `v2_architecture.rst`, `v2_architecture_migration.rst`
  and `v2_correctness_review.rst`: they are artifacts. Durable content moves into
  explanation and reference pages, and new 1.x → v2 migration guides replace them.
- Write one migration page per audience: CLI users, Python users and model authors.
- Delete `docs/project/architecture_audit/` and everything in `literature_audit/` except
  two things:
  - `differences.md` is rewritten as an explanation page.
  - `manifest.json`, `srmf-reference.mat` and `octave-version.txt` move to
    `tools/literature_reference/`, because the runners there and `test_global_models.py`
    use them.
- No redirects; update README links only.
- Trim README to a pitch, one install line, a working CLI and Python example, and links.
  Fix the `black` badge and the removed `construct_model` example.
- `GLOSSARY.md` at the repo root is the single source. Docs render it with `myst-parser`.
- Theme: `pydata-sphinx-theme` (replaces `sphinx_rtd_theme`). The top navbar holds the
  four Diátaxis sections; the Project pages sit in its "More" dropdown. Pages must not use
  rtd-only CSS classes (`wy-*`, `rst-*`).
  It was written during the interview, and its terms are binding for every page.
- Tutorials use a synthetic dataset written to disk, so they run offline and in tests.
  How-tos may use GDSC2 and the other built-in datasets, and say that loading those needs
  read access to the bucket.
- Three tutorials:
  - benchmark your first models;
  - build and benchmark your own predictor;
  - tune a model and stress-test it.
- Run every Python and CLI block on tutorial and how-to pages. A block opts out with an RST
  comment marker.
- Replace the old layout test with a mode-based guard.
- Python reference adds the `drevalpy` root, `drevalpy.registry` and `drevalpy.models.zoo`.
  It leaves out `drevalpy.cli` and `drevalpy.curation`.
- Fix these code gaps:
  - `drevalpy run --response-transformation`, with default `none` everywhere, which also
    changes `drevalpy single`;
  - `load_extensions()` reads `DREVALPY_EXTENSIONS_DIR` when `directories is None`;
  - the dataset registry raises on a conflicting name, accepts an identical repeat
    silently, and replaces on `override=True`;
  - `drevalpy list zoo-presets` and `drevalpy list datasets`;
  - `drevalpy.data.synthetic()`, also available as `drevalpy.synthetic`, plus
    `drevalpy data synthetic`.
- Rename "cell line" to "sample" before any other work (Wave R), so that every later
  package, and every page, uses the new names:
  - **Glossary rule:** a _sample_ is the biological source (cell line, organoid, PDX,
    primary patient cells). A _pair_ is one sample–drug measurement. A _trial_ is one
    tuning configuration. Prose never uses "sample" for a pair or as a verb.
  - **Mechanical name map:** `cell_line` → `sample` and `CellLine` → `Sample`
    everywhere. For example:
    - `CellLineFeaturizer` → `SampleFeaturizer`, `DenseViewCellLineFeaturizer` →
      `DenseViewSampleFeaturizer`, `CellLineFeatureSource` → `SampleFeatureSource`,
      `register_cell_line_featurizer` → `register_sample_featurizer`;
    - `components/featurizers/cell_line/` → `components/featurizers/sample/` and
      `registry/cell_line_featurizer/` → `registry/sample_featurizer/`, with the tests
      mirrored;
    - the side literal `"cell_line"` → `"sample"`, and the zoo YAML and `ModelConfig`
      key `cell_line_featurizer` → `sample_featurizer`;
    - `Dataset.cell_line_ids` → `sample_ids`, `DRPModel.cell_line_views` →
      `sample_views`, `synthetic(n_cell_lines=)` → `n_samples`;
    - `drevalpy list cell-line-featurizers` → `list sample-featurizers`;
    - `NaiveCellLineMeanPredictor` → `NaiveSampleMeanPredictor`.
    - `Tissue` and the `NaiveTissue*` names stay.
  - **Collisions with "sample":** `hpo_num_samples` / `--hpo-num-samples` →
    `hpo_num_trials` / `--hpo-trials`. Internal names that mean a pair
    (`train_samples`, `es_samples`, `_build_samples`, the triplet
    `*_sample_idx` / `*_sample_indices`, `indices_similar_samples`, `dissimilar_samples`)
    → `*_pair*`. Third-party parameter names (`min_samples_leaf`, `colsample_bytree`,
    Optuna `sampler`) stay verbatim.
  - **No aliases**, except the split mode: LCO becomes LSO (leave-sample-out), and
    `"LCO"` stays accepted as an alias that resolves to LSO. Results always report LSO.
    `docs/_static/img/LCO.png` becomes `LSO.png`.
  - **Stored files:** the checkpoint `FORMAT_VERSION` goes from 3 to 4, and version 3
    is rejected with a message saying to retrain. `RunResult` writes `sample_ids`, and
    reading a file with `cell_line_ids` raises an error that names the version that
    changed. The dataset uns key `cell_line_featurizer_variants` becomes
    `sample_featurizer_variants`.
  - **Dataset schema:** the obs index holds a unique sample ID that suits the dataset:
    the Cellosaurus accession for cell-line screens, the PDX model ID for PDX_Bruna, the
    patient sample ID for BeatAML2. drevalpy never reads the index name. `tissue` is the
    only required obs column, because LTO and the tissue featurizer read it.
    `cellosaurus_id` and `cell_line_name` are optional columns, present only in
    datasets whose samples are cell lines. No `sample_type` column.
  - **Datasets are regenerated** with this schema and uploaded over `gen3/`. The upload
    happens only after the user approves it.
  - **Curation stale stages:** first check whether the `core/compute` change in
    `7ab6470` preserves behaviour. If it does, `dvc commit -f` the stale fit stages with
    the user's approval instead of refitting.

## 3. Technical design

### Page map (the shared interface for every docs package)

Every page below exists as a stub after Wave 0, with exactly this path and title. Links
between pages use `:doc:` with these paths, so writers can link to pages that are not
written yet. "Source" names where the content comes from; "new" means it is written from
the code.

The Interface column on how-to pages is a comment on the page's second line:
`.. interface: both` (the default), `.. interface: python` or `.. interface: cli`.

**Tutorials** (`docs/tutorials/`, owner D1)

| Path                       | Title                                  | Interface | Source                             |
| -------------------------- | -------------------------------------- | --------- | ---------------------------------- |
| `index.rst`                | Tutorials                              | –         | new                                |
| `first_benchmark.rst`      | Benchmark your first models            | both      | new; replaces both quickstarts     |
| `own_predictor.rst`        | Build and benchmark your own predictor | python    | new; uses `docs/examples/toy_*.py` |
| `tune_and_stress_test.rst` | Tune a model and stress-test it        | both      | new                                |

**How-to guides: using DrEvalPy** (`docs/how-to/`, owner D2)

| Path                        | Title                                        | Interface | Source                                                                             |
| --------------------------- | -------------------------------------------- | --------- | ---------------------------------------------------------------------------------- |
| `index.rst`                 | How-to guides                                | –         | new                                                                                |
| `install.rst`               | Install DrEvalPy                             | cli       | `getting_started/installation.rst` (install steps only)                            |
| `load_and_split.rst`        | Load and split a dataset                     | both      | `python/datasets.rst`, `cli/datasets.rst`                                          |
| `bring_your_own_data.rst`   | Use your own dataset                         | both      | `concepts/datasets.rst` 63-107, `drevalpy curate` (new)                            |
| `choose_models.rst`         | Choose models from the zoo or write a recipe | both      | `python/models.rst` 9-33, `cli/models.rst`                                         |
| `run_experiment.rst`        | Run an experiment                            | both      | `python/experiments.rst` 10-95, `cli/experiments.rst` 13-88                        |
| `run_folds_in_parallel.rst` | Run folds in parallel and combine them       | both      | `cli/experiments.rst` 90-138, `python/experiments.rst` 97-140                      |
| `tune_hyperparameters.rst`  | Tune hyperparameters                         | both      | `python/experiments.rst` 142-283, `concepts/from_components_to_models.rst` 358-441 |
| `run_stress_tests.rst`      | Run randomization and robustness tests       | both      | `python/experiments.rst` 285-296, `cli/experiments.rst` 168-181                    |
| `make_report.rst`           | Evaluate results and build a report          | both      | `python/visualization.rst`, `cli/visualization.rst`                                |
| `train_save_load.rst`       | Train, save and load a single model          | python    | `python/models.rst` 35-101                                                         |
| `list_available.rst`        | List what is available                       | both      | new (`drevalpy list *`)                                                            |

**How-to guides: extending DrEvalPy** (`docs/how-to/extend/`, owner D3)

| Path                      | Title                                  | Interface | Source                                                                                 |
| ------------------------- | -------------------------------------- | --------- | -------------------------------------------------------------------------------------- |
| `index.rst`               | Extend DrEvalPy                        | –         | new                                                                                    |
| `write_featurizer.rst`    | Write a featurizer                     | python    | `python/extensions.rst` 53-217                                                         |
| `write_predictor.rst`     | Write a predictor                      | python    | `python/extensions.rst` 219-268                                                        |
| `write_splitter.rst`      | Write a splitter                       | python    | `python/extensions.rst` 270-359                                                        |
| `write_visualization.rst` | Write a visualization                  | python    | `python/extensions.rst` 361-397                                                        |
| `test_components.rst`     | Test your components                   | python    | `python/extensions.rst` 399-452                                                        |
| `load_extensions.rst`     | Load extension files and directories   | both      | `python/extensions.rst` 495-545, 581-612, `cli/extensions.rst`, `cli/models.rst` 41-79 |
| `publish_plugin.rst`      | Publish components as a plugin package | python    | `python/extensions.rst` 546-579, `cli/extensions.rst` 49-64                            |
| `add_dataset_source.rst`  | Register a dataset source              | both      | `python/extensions.rst` 454-483                                                        |

**How-to guides: migrating from 1.x** (`docs/how-to/migrate/`, owner D4)

| Path         | Title                              | Interface | Source                       |
| ------------ | ---------------------------------- | --------- | ---------------------------- |
| `index.rst`  | Migrate from DrEvalPy 1.x          | –         | new                          |
| `cli.rst`    | Migrate command-line runs from 1.x | cli       | new, from `git show v1.6.0:` |
| `python.rst` | Migrate Python scripts from 1.x    | python    | new, from `git show v1.6.0:` |
| `models.rst` | Port a 1.x model to components     | python    | new, from `git show v1.6.0:` |

**Reference** (`docs/reference/`, owner D5 unless noted)

| Path                  | Title                                 | Source                                                                                                            |
| --------------------- | ------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `index.rst`           | Reference                             | new                                                                                                               |
| `cli.rst`             | Command-line reference                | includes `_generated_cli.rst` (generator from P8)                                                                 |
| `python_api.rst`      | Python API reference                  | `python/api/index.rst`, with `drevalpy`, `drevalpy.registry`, `drevalpy.models.zoo` added                         |
| `model_zoo.rst`       | Model zoo                             | `concepts/model_zoo.rst` plus `_generated_model_zoo.rst`                                                          |
| `components.rst`      | Components                            | `concepts/component_catalog.rst` 94-114 plus the three `_generated_*` catalogs                                    |
| `recipes.rst`         | Recipe syntax                         | `concepts/from_components_to_models.rst` 23-345 (grammar only)                                                    |
| `hyperparameters.rst` | Hyperparameter keys and search spaces | `concepts/from_components_to_models.rst` 443-474, YAML format                                                     |
| `datasets.rst`        | Datasets and file format              | `concepts/datasets.rst` 8-61, 79-107, 109-162 (tables and format)                                                 |
| `split_modes.rst`     | Split modes and stress tests          | `concepts/evaluation.rst` 7-79 (facts only)                                                                       |
| `metrics.rst`         | Metrics and response transformations  | `concepts/evaluation.rst` 88-127                                                                                  |
| `environment.rst`     | Environment variables                 | new: the six `DREVALPY_*` variables                                                                               |
| `extension_files.rst` | Extension and zoo file formats        | new: extension dir, zoo YAML (single and mapping), dataset-declaration YAML, `drevalpy.plugins` entry-point group |
| `results.rst`         | Results layout                        | new: what `run`, `single` and `aggregate` write                                                                   |
| `examples.rst`        | Example components                    | includes `_generated_examples.rst`                                                                                |
| `glossary.rst`        | Glossary                              | Wave 0: `.. include:: ../../GLOSSARY.md` with `:parser: myst_parser.sphinx_`                                      |

**Explanation** (`docs/explanation/`, owner D6)

| Path                         | Title                                              | Source                                                                                     |
| ---------------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `index.rst`                  | Explanation                                        | new                                                                                        |
| `evaluation_design.rst`      | Why split modes and baselines matter               | `concepts/evaluation.rst` 7-54, 81-86, 129-134 (fixed)                                     |
| `models_from_components.rst` | How models are built from components               | `concepts/component_catalog.rst` 1-92, `from_components_to_models.rst` 1-21, 347-356       |
| `training_lifecycle.rst`     | How a model trains: scopes and early stopping      | A/E/V and `UNTRAINED`/`EMPTY`/`FITTED` from `project/v2_*.rst`, `python/models.rst` 90-101 |
| `registries_and_plugins.rst` | How registries and plugins fit together            | `concepts/registries.rst` 1-127 (fixed)                                                    |
| `reproducibility.rst`        | What makes a run reproducible                      | `python/experiments.rst` 159-178, `cli/experiments.rst` 151-166                            |
| `datasets_provenance.rst`    | Where the datasets come from                       | `concepts/datasets.rst` 164-185                                                            |
| `literature_models.rst`      | How the literature models differ from their papers | `project/literature_audit/differences.md`, rewritten for users                             |

**Project** (`docs/project/`, owner D7): `contributing.rst`, `citing.rst`, `news.rst`,
`contributors.rst` and `memes.rst` stay. The toctree drops the `v2_*` pages.

`docs/index.rst` (Wave 0) has a three-paragraph landing text, a four-row "where to go"
list and the five toctrees. `docs/readme.rst` is deleted in Wave 3.

### Writing conventions every docs package follows

- Apply the `technical-writing` and `unslop` skills.
- Use the terms in `GLOSSARY.md`. Never use an `_Avoid_` term, except where it names code
  (for example `split(mode=)`).
- Name how-to titles by the task, starting with a verb. Tutorial titles say what the
  reader builds.
- Interface tabs:

  ```rst
  .. tab-set::
     :sync-group: interface

     .. tab-item:: CLI
        :sync: cli

        .. code-block:: bash

           drevalpy run --models ElasticNet --dataset toy.h5mu --split-mode LSO

     .. tab-item:: Python
        :sync: python

        .. code-block:: python

           results = drevalpy.run(...)
  ```

- Snippet markers are RST comments on the line before a code block:
  - `.. snippet: skip` means the block is never run (install commands, shell setup).
  - `.. snippet: network` means the block runs only under `-m network` (built-in datasets).
- CLI blocks use `bash`, and every executed line starts with `drevalpy `. Python blocks on
  one page share one namespace and run in page order, in a fresh temporary working
  directory.
- Reference pages describe only. Explanation pages have no `python` or `bash` blocks.
- When the code has a gap, describe what the code does. For example: "v2 has no
  cross-study entry point."

### Known contradictions to fix while moving content

Each fix belongs to the package that owns the target page.

| Old location                                    | Correct fact (evidence)                                                                                                                                                                                                                                                                                   | Owner  |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| README.md 122-138                               | No `construct_model`. Use `ModelSpec.from_name` + `drevalpy.run(models: list[ModelSpec], ...)` (`_run.py:24`)                                                                                                                                                                                             | D7     |
| README.md 10                                    | Linter is ruff (`.pre-commit-config.yaml`)                                                                                                                                                                                                                                                                | D7     |
| README.md 74, installation.rst 60-72            | Docker tag mismatch (`:main` against `:latest`). Check the published tag in `.github/workflows`                                                                                                                                                                                                           | D2     |
| installation.rst 12, 23                         | Python `>=3.11,<3.14` (`pyproject.toml:8`)                                                                                                                                                                                                                                                                | D2     |
| concepts/datasets.rst 69-72                     | `curate` input columns: `drug, sample, concentration, intensity` (+ optional `replicate`), `curation/_preprocess.py:8` (`cell_line` today; Wave R renames it to `sample`). Output is `.h5ad` (`cli/curate.py:36,52`)                                                                                      | D2     |
| concepts/datasets.rst 73-77                     | No CSV loader. `load` takes a registered name or an existing `.h5mu` path (`data/datasets/_load.py:87-126`)                                                                                                                                                                                               | D2, D5 |
| concepts/datasets.rst 54, 94                    | `response.X` holds pEC50. Modality names are `gene_expression`, `copy_number_variation_gistic`, … (`types/data/modalities.py:27-35`)                                                                                                                                                                      | D5     |
| concepts/datasets.rst 8-49, cli/datasets.rst 26 | `TOYv1` is also registered (`available_datasets.json`)                                                                                                                                                                                                                                                    | D5     |
| concepts/evaluation.rst 50-53                   | `run` never adds `NaiveMeanEffectsPredictor` and treats all models alike (`_run.py:63-75`). Normalizing uses `ExperimentResult.normalize(reference_model=...)`                                                                                                                                            | D5, D6 |
| concepts/evaluation.rst 84                      | A robustness test reshuffles pair order and keeps the model seed (`experiment/_robustness.py:8-27`)                                                                                                                                                                                                       | D5, D6 |
| concepts/evaluation.rst 100                     | The report normalizes only when `reference_model` is given (`visualization/report.py:115-117`)                                                                                                                                                                                                            | D2     |
| concepts/evaluation.rst 129-134                 | v2 has no cross-study entry point; only `CrossStudyTableVisualization` exists                                                                                                                                                                                                                             | D6     |
| component_catalog.rst 112                       | MOLIR needs `copy_number_variation_gistic` and is single-drug (`literature/molir/predictor.py:58-62`)                                                                                                                                                                                                     | D5     |
| from_components_to_models.rst 265               | No `expression`/`cnv` view aliases. Use `gene_expression` etc. (`zoo/SRMF.yaml:1`)                                                                                                                                                                                                                        | D5     |
| model_zoo.rst 10                                | A zoo file holds one preset (name = file stem) or a mapping of name → configuration (`models/zoo/_external_load.py:52-73`)                                                                                                                                                                                | D5     |
| registries.rst 35-38                            | After P5 every registry, including datasets, raises on a conflicting name and accepts `override=True`                                                                                                                                                                                                     | D6     |
| registries.rst 91-97                            | Built-in source is `s3://…`. Only the `s3` and `http` fsspec backends are installed                                                                                                                                                                                                                       | D3, D6 |
| registries.rst 108-111                          | `PlotRequirement` has no "reference model" (`visualization/requirements.py:6-12`)                                                                                                                                                                                                                         | D3     |
| registries.rst 132-137, extensions.rst 584-589  | After P4, `DREVALPY_EXTENSIONS_DIR` works in Python through `load_extensions()`                                                                                                                                                                                                                           | D3, D5 |
| python/models.rst 49-50                         | `model_checkpoint_dir=None` means a temporary directory (`drp_model.py:92,101-105`)                                                                                                                                                                                                                       | D2     |
| python/models.rst                               | `DRPModel.load` unpickles; trust only your own files (`drp_model.py:135-138`)                                                                                                                                                                                                                             | D2     |
| python/extensions.rst 185, 189                  | `FeaturizerResult(entity_ids, blocks, projection)` (`types/data/batch/feature_block.py:70-72`)                                                                                                                                                                                                            | D3     |
| python/extensions.rst 511-515                   | `ModelSpec(name, ModelConfig)`; a dict raises `TypeError` (`models/spec.py:33-34`)                                                                                                                                                                                                                        | D3     |
| python/extensions.rst 145                       | Registry name is `concatFeaturizers`                                                                                                                                                                                                                                                                      | D3     |
| python/extensions.rst 462                       | Import dataset registration from the public surface, not `drevalpy.registry.dataset` internals                                                                                                                                                                                                            | D3     |
| python/extensions.rst 525-530                   | `load_extension_dir` loads `*.py` and `*.yaml` (`registry/_extensions.py:174-201`)                                                                                                                                                                                                                        | D3     |
| python/experiments.rst 48-95, 133-140           | `run` also takes `response_transformation`; after P3 both interfaces default to none                                                                                                                                                                                                                      | D2     |
| python/datasets.rst 45                          | `run` always uses `n_splits=5` (`_run.py:56`)                                                                                                                                                                                                                                                             | D2     |
| python/visualization.rst 47-54                  | Classes are `ComparisonScatterVisualization`, `CriticalDifferenceVisualization`, `CrossStudyTableVisualization`, `HeatmapVisualization`, `RegressionScatterVisualization`, `ViolinVisualization`, `LeaderboardVisualization` (`visualization/plots/*`); none is re-exported from `drevalpy.visualization` | D2     |
| python/visualization.rst 35                     | `RunResult` has `ground_truth`, not `response` (`types/results/run.py:58-59`)                                                                                                                                                                                                                             | D2     |
| python/visualization.rst 64-65                  | `Visualization` has four abstract methods (`visualization/base.py:62-89`)                                                                                                                                                                                                                                 | D3     |
| cli/visualization.rst 64-65                     | Advises seven folds; `run` uses five                                                                                                                                                                                                                                                                      | D2     |
| python/api/index.rst 22, 41-42                  | Registries live in `drevalpy.registry` and tuning in `drevalpy.models.tuning`; no custom response tables                                                                                                                                                                                                  | D5     |
| v2_migration.rst 45-46                          | `FeaturizerResult` fields are `entity_ids, blocks, projection`; `matrix` is a property                                                                                                                                                                                                                    | D6     |
| AGENTS.md 61                                    | Eight literature model packages, not seven                                                                                                                                                                                                                                                                | D7     |
| `DREVALPY_CONFIG_DIR`                           | Undocumented; holds `datasets.json` (`data/_paths.py:11`)                                                                                                                                                                                                                                                 | D5     |

### Modules

| Module                                              | Interface (small)                                                                                                                                                                                                                                                                                                                                                                                                                     | Hides                                                                                                                                                                                                            | New / changed                            |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| `docs/_snippets.py`                                 | `collect(page: Path) -> list[Snippet]`; `run_page(page: Path, workdir: Path, *, allow_network: bool) -> None`, raises `SnippetError(page, line, output)`                                                                                                                                                                                                                                                                              | RST parsing at any indent (inside `tab-item`), marker comments, shared Python namespace, `typer.testing.CliRunner` against `drevalpy.cli.main.app`, `shlex` argv splitting, rejecting non-`drevalpy` shell lines | new                                      |
| `tests/docs/test_snippets.py`                       | pytest parametrized over every page in `tutorials/` and `how-to/`                                                                                                                                                                                                                                                                                                                                                                     | offline and network variants                                                                                                                                                                                     | new (replaces `test_model_workflows.py`) |
| `tests/docs/test_docs_structure.py`                 | pytest                                                                                                                                                                                                                                                                                                                                                                                                                                | mode rules (see Tests)                                                                                                                                                                                           | rewritten                                |
| `docs/_cli_click.py`                                | `generate_cli_reference_rst()`, `write_generated_cli_reference()`                                                                                                                                                                                                                                                                                                                                                                     | recursion into groups (`data`, `experiments`, `list`), true root description                                                                                                                                     | changed                                  |
| `drevalpy/data/_synthetic.py`                       | `synthetic(*, name, n_samples=60, n_drugs, n_tissues, omics, feature_names, n_features, missing_fraction, low_quality_fraction, seed) -> Dataset`, today's `build_synthetic_dataset` signature with the default sample count raised from 24 to 60, plus drug `canonical_smiles` so `fingerprints` works; `n_drugs` above `MAX_SYNTHETIC_DRUGS` (the length of the fixed SMILES list, at least 8) raises `ValueError` naming the limit | the response model below (fixed effect sizes), quality layers, SMILES list                                                                                                                                       | new (moved from `testing/synthetic.py`)  |
| `drevalpy/cli/data/synthetic.py`                    | `drevalpy data synthetic --output PATH [--seed N] [--n-samples N] [--n-drugs N] [--n-tissues N]`; defaults match `synthetic()`                                                                                                                                                                                                                                                                                                        | `synthetic(...).save(path)`; turns the `ValueError` into a CLI usage error                                                                                                                                       | new                                      |
| `drevalpy/cli/catalog/registries.py`                | `drevalpy list zoo-presets`, `drevalpy list datasets`                                                                                                                                                                                                                                                                                                                                                                                 | zoo lookup, `dataset_registry.to_dataframe()`, shared `_render`                                                                                                                                                  | changed                                  |
| `drevalpy/cli/run.py`, `cli/single.py`              | `--response-transformation {none,standard,minmax,robust}` (default `none`); use the values the code accepts                                                                                                                                                                                                                                                                                                                           | mapping `none` → `None`                                                                                                                                                                                          | changed                                  |
| `drevalpy/registry/_extensions.py::load_extensions` | `directories=None` now means "`DREVALPY_EXTENSIONS_DIR` if set"                                                                                                                                                                                                                                                                                                                                                                       | env lookup in one place                                                                                                                                                                                          | changed                                  |
| `drevalpy/registry/dataset`                         | `register_source(name, base_url, storage_options=None, *, override=False)`, `register_dataset(name, source, file, *, override=False)`; raises `ValueError` on a conflicting entry                                                                                                                                                                                                                                                     | equality check against the persisted and built-in entry; identical repeat is a no-op                                                                                                                             | changed                                  |

The deletion test: `docs/_snippets.py` earns its place. Without it, every docs test would
re-implement RST block extraction, which is what `test_model_workflows.py` does today
with a column-0 regex. The new `drevalpy.data.synthetic` replaces
`testing/synthetic.py` rather than wrapping it. `drevalpy.testing.build_synthetic_dataset`
becomes an alias, because plugin tests already import it.

**Response model of `drevalpy.data.synthetic`.** Real screens show a Simpson-like
pattern: drugs differ so much in average potency that pooled scores look high even when
a model ranks samples poorly within each drug. The synthetic data reproduces this so the
tutorials can show it. With `S` samples, `D` drugs, `T` tissues and `K = 3` hidden
factors:

```text
y[s, d] = 2 + drug[d] + tissue[t(s)] + z[s] · a + z[s] · w[d] + noise[s, d]
drug[d]   = 1.5 · Φ⁻¹((d + 0.5) / D), randomly permuted   # spread drug means, not drawn
tissue[t] ~ N(0, 0.3²)
z[s]      ~ N(0, I_K)                                     # hidden sample factors
a         ~ N(0, (0.4/√K)² I_K)                           # sample effect from features
w[d]      ~ N(0, (0.6/√K)² I_K)                           # drug × sample interaction
noise     ~ N(0, 0.5²)
omics X[s] = 6 + z[s] · L + N(0, 0.5²),  L ~ N(0, 1)^(K × n_features)
```

- The effect sizes are fixed module constants, not parameters.
- The drug means are spread by quantiles rather than drawn at random. With only 8 drugs,
  random draws made the pooled score swing between seeds (minimum pooled Pearson 0.63
  against 0.74 with quantiles, over 20 seeds).
- Every omics view is a noisy mixture of the same `z`, so features carry signal and a
  randomization test that permutes a view visibly hurts.
- `response.X` stays in pEC50 units, and the `LN_IC50` layer is derived as today.

The default `n_samples` rises from 24 to 60. A simulation of this model (an LPO-like 25 %
holdout, per-drug ElasticNet on 16 features, 20 seeds) measured:

| `n_samples` | drug-mean pooled r (mean / min) | NaiveMeanEffects per-drug r | ElasticNet per-drug r | ElasticNet beats naive |
| ----------- | ------------------------------- | --------------------------- | --------------------- | ---------------------- |
| 24          | 0.81 / 0.63                     | 0.45                        | 0.56                  | 12 / 20 seeds          |
| 60          | 0.83 / 0.74                     | 0.43                        | 0.66                  | 18 / 20 seeds          |

At 24 samples the ranking is a coin flip, which would teach the wrong lesson. At 60 it is
stable, and the dataset is still small: 60 × 8 pairs, a few hundred kB.

### Call trees

```text
sphinx-build -W docs …
└─ docs/conf.py::_write_generated_references()                       [changed: output paths]
   ├─ _cli_click.write_generated_cli_reference()                     [changed]
   │  └─ writes docs/reference/_generated_cli.rst                    [changed path; nested groups expanded]
   ├─ _component_catalog.write_generated_component_catalogs()
   │  └─ writes docs/reference/_generated_{sample_featurizers,drug_featurizers,predictors}.rst   [changed path]
   ├─ _model_zoo.write_generated_model_zoo()
   │  └─ writes docs/reference/_generated_model_zoo.rst              [changed path]
   └─ _examples.verify_documented_examples()
      └─ writes docs/reference/_generated_examples.rst               [changed path]
reference/glossary.rst
└─ .. include:: ../../GLOSSARY.md  :parser: myst_parser.sphinx_      [new]
```

```text
pytest tests/docs/test_snippets.py                                   [new]
└─ for page in tutorials/**/*.rst, how-to/**/*.rst
   └─ _snippets.run_page(page, tmp_path, allow_network=False)        [new]
      ├─ _snippets.collect(page) -> list[Snippet]                    [new]
      ├─ python block  -> exec(code, namespace)                      shared per page
      ├─ bash block    -> CliRunner().invoke(app, shlex.split(line)[1:]) for each `drevalpy …` line
      └─ failure       -> raise SnippetError(page, line, output)
```

```text
drevalpy data synthetic --output toy.h5mu --seed 0 --n-samples 60 --n-drugs 8 --n-tissues 6   [new]
└─ cli/data/synthetic.py::synthetic_cmd(output, seed, n_samples, n_drugs, n_tissues)          [new]
   └─ drevalpy.data.synthetic(seed=..., n_samples=..., n_drugs=..., n_tissues=...) -> Dataset  [new]
      ├─ n_drugs > MAX_SYNTHETIC_DRUGS -> raise ValueError (CLI: usage error, non-zero exit)
      └─ Dataset.save(output)
```

```text
drevalpy list zoo-presets | datasets                                 [new]
├─ cli/catalog/registries.py::list_zoo_presets()                     [new]
│  └─ models.zoo.list_zoo_names(), get_zoo_config(name) -> rows -> _render
└─ cli/catalog/registries.py::list_datasets()                        [new]
   └─ registry.dataset.dataset_registry.to_dataframe() -> _render
```

```text
drevalpy run … --response-transformation standard                    [changed]
└─ cli/run.py::run_cmd(..., response_transformation="none")          [changed]
   └─ drevalpy.run(..., response_transformation=None | "standard")
drevalpy single …                                                    [changed default: none]
└─ cli/single.py::single_cmd(..., response_transformation="none")
```

```text
drevalpy -e DIR <command>
└─ cli/main.py::main_callback(extensions_dir)                        [changed]
   ├─ registry.load_extensions()                                     [changed]
   │  └─ directories is None -> [os.environ["DREVALPY_EXTENSIONS_DIR"]] if set
   │     └─ load_extension_dir(env_dir)
   └─ if extensions_dir: registry.load_extensions(directories=extensions_dir)
      └─ load_extension_dir(d) for d in extensions_dir                 (env first, then flags, as today)
```

```text
registry.dataset.register_source(name, base_url, storage_options, *, override=False)   [changed]
└─ dataset_registry.register_source(...)                             [changed]
   ├─ config = load_config(); existing = builtin or config entry for name
   ├─ existing == new entry          -> return (no write)
   ├─ existing and not override      -> raise ValueError(f"Source {name!r} already registered")
   └─ else                           -> config.sources[name] = entry; save_config(config)
register_dataset(name, source, file, *, override=False)               same rule, on datasets
```

### Tests

- **Seam: `docs/_snippets.py`**, through `tests/docs/test_snippets.py` (offline, marked
  `slow`, plus a `network` variant) and `tests/docs/test_snippets_runner.py`, which feeds
  small RST strings to `collect` and `run_page`. Cases:
  - a block inside a `tab-item` is found;
  - `skip` and `network` markers are honoured;
  - a failing Python block raises `SnippetError` with page and line;
  - a non-`drevalpy` line in a bash block is rejected;
  - the namespace is shared within a page and fresh across pages.
- **Seam: the page layout**, through `tests/docs/test_docs_structure.py`:
  - every built `.rst` lives under `tutorials/`, `how-to/`, `reference/`, `explanation/`
    or `project/`, or is `index.rst`, and is reachable from a toctree;
  - `explanation/` pages have no `python` or `bash` blocks;
  - every how-to title starts with a verb from `HOWTO_VERBS` in the test file;
  - on `interface: both` pages, every `python` or `bash` block sits in a `tab-set` with
    `:sync-group: interface` and both a `cli` and a `python` tab;
  - `interface: python` pages have no `bash` `drevalpy` lines;
  - `reference/cli.rst`, `components.rst`, `model_zoo.rst` and `examples.rst` include
    their generated files;
  - the existing zoo-preset, component-catalog and CLI-subcommand checks are re-pointed
    to the new paths;
  - `1.5.1` and "modularity release" are forbidden outside `project/news.rst`;
  - the "Backward compatibility … Before 1.6.0" rule is dropped, because migration now
    has its own pages.
- **Seam: `tests/docs/test_examples.py`**: every `docs/examples/*.py` except
  `toy_conformance.py` is literal-included by some page under `docs/`. It is no longer
  tied to `python/extensions.rst`. The `DEAD_REFERENCES` check runs over
  `how-to/extend/`.
- **Seam: `drevalpy.data.synthetic`**, through `tests/data/test_synthetic.py` (moved from
  `tests/testing/test_synthetic.py`). The cases stay the same, plus:
  - drug `canonical_smiles` exist;
  - `n_drugs=MAX_SYNTHETIC_DRUGS` works, and `n_drugs=MAX_SYNTHETIC_DRUGS + 1` raises
    `ValueError` naming the limit;
  - `ModelSpec.from_name("ElasticNet")` trains on the dataset;
  - on the default seed, under LPO with the built-in metrics:
    - `NaiveDrugMeanPredictor` reaches pooled Pearson ≥ 0.7, while its per-drug Pearson
      is undefined or 0, which is the Simpson pattern;
    - `NaiveMeanEffectsPredictor`'s mean per-drug Pearson is at least 0.3 lower than its
      pooled Pearson;
    - `ElasticNet` beats `NaiveMeanEffectsPredictor` on mean per-drug Pearson;
  - the response model is deterministic: the same seed gives byte-identical matrices;
  - `drevalpy.synthetic is drevalpy.data.synthetic`.
  - `tests/testing/test_init.py` records the new origin module.
  - `tests/test_import_cost_policy.py` still passes.
- **Seam: CLI commands**, through mirrored tests under `tests/cli/`:
  - `data synthetic` writes a loadable file, and `--n-samples`, `--n-drugs` and
    `--n-tissues` set the shape of the loaded dataset;
  - `data synthetic --n-drugs` above the limit exits non-zero, with the limit in the
    message;
  - `list zoo-presets` shows every `models/zoo/*.yaml` stem;
  - `list datasets` shows `TOYv1`;
  - `run --response-transformation standard` reaches `drevalpy.run`;
  - `single` defaults to none.
- **Seam: `load_extensions`**:
  - with the env var set and `directories=None`, it loads that directory;
  - with explicit directories, the env var is not read;
  - the CLI loads the env directory, then the flag directories.
  - Use `restore_component_registries()` on teardown, as AGENTS.md requires.
- **Seam: dataset registry**, using a temporary `DREVALPY_CONFIG_DIR`:
  - an identical repeat leaves `datasets.json` untouched;
  - a different entry raises `ValueError`;
  - `override=True` replaces;
  - a conflict with a built-in name raises;
  - loading the same dataset-declaration YAML twice works.

### Risks and open questions

- **The synthetic dataset may be too small or too dirty for zoo presets.** Mitigation: P1
  is done only when `ElasticNet`, `RandomForest` and `NaiveMeanEffectsPredictor` finish an
  LSO `run` on it in under a minute.
- **The new response model changes `build_synthetic_dataset` output** for every existing
  test and for drevalpy-plugin, which uses it with `n_samples=200`. Tests that pin values
  rather than shapes must be re-recorded in P1. Tests that pass a size
  explicitly keep their size. P1 runs the plugin's test suite too.
- **The thresholds come from a simulation, not from drevalpy's own pipeline.** If the
  default seed misses one under the real metrics, P1 may pick another default seed. It
  may not loosen a threshold or change the effect sizes without asking the user.
- **Snippet runtime.** Tutorials train real models. Keep tutorial hyperparameters tiny,
  and mark `test_snippets.py` `slow` so it runs in CI's `-m "not network"` and not in the
  fast tier.
- **`html_context["github_version"] = "development"` in `conf.py`.** The default branch
  is changing. The user decides the new value. Until then D7 leaves it unchanged.
- **`drevalpy curate` writes `.h5ad`, but `load` takes only `.h5mu`.** If no supported
  path leads from curated output to a loadable dataset, D2 documents the steps that exist
  and lists the gap as a follow-up. D2 does not invent a workflow.
- **Changing the `single` default to `none` changes results** for anyone relying on
  `standard`. v2 is unreleased, so `project/news.rst` mentions it and there is no
  deprecation path.
- **The 1.x facts come from git history.** D4 reads `git show v1.6.0:<path>` and checks
  every flag or name it maps. Anything it cannot map is listed as removed.
- **Datasets are not publicly readable.** How-tos say so plainly in the
  `load_and_split.rst` prerequisites. Making the bucket public is the user's decision,
  outside this plan.
- **The rename is one large, entangled change.** R1 touches about 1,040 hits in 135
  `src` files and about 1,780 in `tests`. Splitting it across parallel agents would
  produce merge conflicts on shared modules, so one agent does it, in commits split by
  area. The terminology guard (R1) keeps the old word from creeping back in.
- **The rename makes curation DVC stages stale**, because their code dependencies
  change. A pure rename preserves behaviour, but that has to be shown, not assumed.
  R3 reruns the cheapest stale stage of each kind and compares its outputs with the
  committed ones. Only if they are identical does it ask to `dvc commit -f` the rest.
- **Curation pins drevalpy by git revision.** R3 needs R1 pushed to `origin/v2` to
  bump `uv.lock`. Pushing needs the user's approval.
- **Overwriting `gen3/` changes what every current v2 user downloads.** Released 1.6.0
  reads Zenodo and is not affected. R4 is a manual step the user runs or approves.
- **Follow-up issues D7 records in `plans/docs-diataxis-followups.md`:**
  - LCO → LSO in nf-core/drugresponseeval;
  - a cross-study entry point;
  - W&B through `run`;
  - merging `tests/synthetic/builders.py` into `drevalpy.data.synthetic`;
  - a public bucket;
  - redirects;
  - ElasticNet's default `alpha=1.0` predicts a constant on the synthetic data without
    tuning (deferred by the user; the tutorials tune it);
  - curation fits depend on the OpenBLAS kernel (pEC50 moves by up to about 0.11);
  - the "Install with: pip install drevalpy[report]" hints are removed; check that
    no docs page still mentions a `report` extra.

## 4. Parallelization

The common commands below assume `export PATH=$HOME/.local/bin:$PATH` and the
`drevalpy` repo root as the working directory:

- `DOCS` = `uv run --frozen --group docs sphinx-build -E -W -q docs /tmp/opencode/docs_build`
- `DOCTESTS` = `uv run --frozen pytest -q -p no:cacheprovider -m "not network" tests/docs`
- `HOOKS` = `uv run --frozen prek run --all-files`

The waves run in this order: R → 0 → 1 → 2 → 3.

### Wave R: sample rename (before everything else)

Wave R comes first because every later package writes code or pages with the new names.
R1 runs alone. R2 and R3 run in parallel after R1, and R4 runs after R3.

#### R1: rename in drevalpy

- **Files owned:** all of `src/`, `tests/`, the zoo YAMLs, `pyproject.toml` (the
  coverage-gate and size-gate exemption paths), `AGENTS.md`, `README.md` (names only),
  `docs/_static/img/LCO.png` → `LSO.png`, and the old docs tree. Old pages get only the
  edits that keep `DOCS` green: `automodule`/`autosummary` targets, literal includes,
  `docs/examples/toy_cell_line_featurizer.py` → `toy_sample_featurizer.py`, and
  generator output names. Prose is rewritten later, in Wave 2.
- **Interface:** the name map, collision rules, LCO alias and stored-file rules in
  section 2. Also:
  - the splitter registry gains `aliases: tuple[str, ...] = ()` on `register`.
    `split(mode="LCO")` and `--split-mode LCO` resolve to LSO, and listings show the
    alias next to LSO;
  - `drevalpy/curation/_preprocess.py` requires a `sample` column instead of
    `cell_line`. This is the contract R3 fills;
  - `CELL_LINE_IDENTIFIER` in `data/utils.py` becomes `CELL_LINE_NAME_COLUMN`
    (`"cell_line_name"`), documented as optional. Nothing in drevalpy may require it or
    `cellosaurus_id`. `Dataset` docstrings say "sample IDs" instead of "cellosaurus
    IDs" (`types/data/dataset.py:275,321-360`);
  - `synthetic()` writes an unnamed index of `SYNTH-000`-style IDs plus `tissue`, and no
    `cellosaurus_id` or `cell_line_name`, so the tutorials exercise the minimal
    schema;
  - a test loads a dataset with only `tissue` in obs and runs LSO and LTO splits plus a
    naive baseline on it;
  - a new policy test, `tests/test_terminology_policy.py`, fails on any
    `cell.?line|CellLine` match in `src/` or `tests/` outside an allowlist. The
    allowlist holds the optional columns `cellosaurus_id` and `cell_line_name`, the
    `"LCO"` alias and its tests, the format-version error messages, and dataset
    descriptions of real cell-line screens. Each entry carries a reason.
- **Order inside the package:** one commit each for `registry` + `components`, then
  `models` + zoo, then `data` + `experiment` + `types` + `evaluation`, then `cli`, then
  `visualization`, then the stored formats, then the LSO alias, then tests and the
  policy test. Run the suite after each commit.
- **Depends on:** nothing.
- **Done when:**
  - `uv run --frozen pytest -q -m "not network"` passes;
  - `uv run pytest -m "not network" --cov --cov-report=json`, then
    `uv run python tools/coverage_gate.py` and `uv run python tools/size_gate.py`,
    pass. No floor is raised and no ceiling is lowered for the rename;
  - `HOOKS` and `DOCS` pass;
  - `grep -rnE "cell_line|CellLine|cell-line" src tests --include=*.py --include=*.yaml`
    matches only allowlisted lines;
  - a checkpoint saved before the rename fails to load with the "retrain" message, and
    `drevalpy run ... --split-mode LCO` reports LSO.
- **Context:** section 2 (rename decisions); AGENTS.md "Featurizers", "Test layout",
  "Import cost" and "Coverage gate"; `models/mixins/_persistence_io.py:31-32,147,156,183`;
  `types/results/run.py:139,202`; `components/featurizers/storage.py:13`;
  `data/splitters/lco.py:10`; `registry/splitter/_registry.py:76`.

#### R2: rename in drevalpy-plugin

- **Files owned:** all of `/data/Software/dreval/drevalpy-plugin`.
- **Interface:**
  - `mudataset.cell_line_ids` → `sample_ids` (`tissue_splitter.py:66`) and
    `build_synthetic_dataset(n_cell_lines=)` → `n_samples` (`tests/conftest.py:15`);
  - its own names: `MIN_CELL_LINES` → `MIN_SAMPLES`, the `MIN_*_CELL_LINES` constants,
    and the fold metadata keys `n_*cell_lines*` → `n_*samples*`;
  - README and docstrings.
- **Depends on:** R1 (the plugin uses `../drevalpy` editable).
- **Done when:** the plugin's `uv run pytest` and its hooks pass, and
  `grep -rniE "cell.?line" src tests README.md` finds nothing.

#### R3: rename in dreval_curation, and regenerate datasets

- **Files owned:** all of `/data/Software/dreval/dreval_curation`, including
  `dvc.yaml`, `params.yaml`, `dvc.lock` and `uv.lock`.
- **Interface:**
  - Python identifiers, CLI help, README and AGENTS.md follow the name map;
  - `response/prepare/common.py:19` (`CURATE_COLUMNS`) and `:47`, plus the six
    `CELL_LINE_COLUMN` constants, fill the `sample` column R1 now requires;
  - stage names and data directories (`process_cell_line_meta*`,
    `data/06_cell_lines/`) become `process_sample_meta*` and `data/06_samples/`;
  - the written schema follows section 2:
    - `harmonize_response.py:40` `OBS_INDEX_NAME` and `harmonize_per_replicate.py:66,100`
      name the index `sample_id`, and `assemble_mudata.py:89` indexes by it;
    - CMP and PRISM datasets keep the Cellosaurus accession as the ID and also write the
      `cellosaurus_id` and `cell_line_name` columns;
    - PDX_Bruna and BeatAML2 use their own IDs and write neither column. That means
      `omics/transcriptomics/bruna.py:55`, `beataml.py:75` and the constant-tissue table
      in `cell_lines/meta.py:147-173` stop filling them with subject IDs;
    - `meta.py:38` `REQUIRED_COLUMNS` becomes `["tissue"]`, with the two cell-line
      columns required only on the CMP path.
- **Steps:**
  1. Bump drevalpy in `uv.lock` to the pushed R1 commit. Ask the user to push first.
  2. Settle the stale fit stages left by `7ab6470`. Diff `src/data_curation/core/compute`
     across that commit. Rerun `fit_curves` for the smallest dataset and compare its
     output with the committed one. If they are identical, ask to `dvc commit -f` the
     remaining `fit_curves@*`, `fit_curves_per_replicate@*` and `record_fit_libs`
     stages. If not, stop and report.
  3. Apply the rename, then do the same for every stage made stale by it: rerun the
     cheapest stale stage of each kind, compare, and ask before `dvc commit -f`.
  4. `dvc repro` the assembly stages (`assemble_mudata@*`, `assemble_mudata_prism@*`,
     `assemble_mudata_rna_only@*`, `subsample_toy`).
- **Depends on:** R1.
- **Done when:**
  - curation's `uv run pytest` and hooks pass;
  - `dvc status` is clean;
  - every file in `data/07_assembled/` loads with `drevalpy.load(path)` from R1's
    drevalpy, and `drevalpy run --models NaiveMeanEffectsPredictor --dataset data/07_assembled/TOYv1.h5mu --split-mode LSO` finishes;
  - in `PDX_Bruna.h5mu` and `BeatAML2.h5mu`, `mdata.obs` has `tissue` but no
    `cellosaurus_id` or `cell_line_name`. In every CMP and PRISM file, both columns are
    present and the index is unique;
  - `grep -rniE "cell.?line" src tests` matches only the written keys and
    real-cell-line source descriptions.

#### R4: upload (manual, needs approval)

- Copy `data/07_assembled/{GDSC1,GDSC2,CTRPv1,CTRPv2,BeatAML2,PDX_Bruna,TOYv1}.h5mu` over
  `s3://orakl-open-source-data/drevalpy_h5mu/gen3/`. Update the checksums in
  `src/drevalpy/data/datasets/available_datasets.json` if it records any, and run
  `uv run --frozen pytest -m network tests/data`.
- **Depends on:** R3 and the user's explicit go-ahead. The agent prepares the commands;
  the user runs or approves them.

### Wave 0: interfaces

#### P0: skeleton and build wiring

- **Files owned:**
  - every stub page in the page map, as a title plus its `.. interface:` comment;
  - `docs/index.rst`, `docs/conf.py` and `docs/reference/glossary.rst`;
  - the output-path constants in `docs/_cli_click.py`, `_component_catalog.py`,
    `_model_zoo.py` and `_examples.py` (paths only);
  - the `includes` in `reference/{cli,components,model_zoo,examples}.rst`;
  - `pyproject.toml` (`myst-parser` in the `docs` group) and `uv.lock`;
  - `.gitignore` (generated paths).
- **Interface:**
  - The page map in section 3 becomes real paths.
  - `conf.py` adds `myst_parser` to extensions.
  - `exclude_patterns` gains `concepts`, `python`, `cli`, `getting_started`,
    `project/v2_*`, `project/architecture_audit` and `project/literature_audit`, plus the
    new `reference/_generated_*.rst`.
  - The generators write into `docs/reference/`.
  - The old generated-file exclusions are removed.
  - `tests/docs/test_docs_structure.py` gets only path re-points, so it stays green. Its
    rewrite belongs to P7.
- **Depends on:** R1.
- **Done when:** `DOCS` and `DOCTESTS` pass, and `GLOSSARY.md` renders at
  `/tmp/opencode/docs_build/reference/glossary.html`.
- **Context:** sections 2 and 3 (page map, call tree 1); `docs/conf.py`; the
  `tests/docs/` constants.

### Wave 1 (code, runner and guard, all in parallel)

#### P1: synthetic dataset

- **Files owned:**
  - `src/drevalpy/data/_synthetic.py` (new), `src/drevalpy/data/__init__.py`,
    `src/drevalpy/__init__.py`;
  - `src/drevalpy/testing/synthetic.py` (becomes the re-export) and
    `src/drevalpy/testing/__init__.py`;
  - `src/drevalpy/cli/data/__init__.py` and `src/drevalpy/cli/data/synthetic.py` (new);
  - `tests/data/test_synthetic.py` (moved from `tests/testing/test_synthetic.py`),
    `tests/testing/test_init.py`, `tests/cli/data/test_synthetic.py` and
    `tests/test_import_cost_policy.py` (only if a lazy re-export entry is needed).
- **Interface:** `drevalpy.data.synthetic(...) -> Dataset`, also available as
  `drevalpy.synthetic` and as
  `drevalpy data synthetic --output PATH [--seed N] [--n-samples N] [--n-drugs N] [--n-tissues N]`.
  It includes drug `canonical_smiles` from a fixed list of real SMILES, so the `fingerprints`
  featurizer works. `drevalpy.testing.build_synthetic_dataset` is an alias.
- **Depends on:** Wave 0.
- **Done when:**
  - `uv run --frozen pytest -q -m "not network" tests/data tests/testing tests/cli tests/test_import_cost_policy.py`
    passes, and so does the full `uv run --frozen pytest -q -m "not network"`, because
    the new response model changes every test that uses the builder;
  - drevalpy-plugin's `uv run pytest` passes against the changed builder;
  - `uv run --frozen drevalpy data synthetic --output /tmp/opencode/toy.h5mu && uv run --frozen drevalpy run --models NaiveMeanEffectsPredictor ElasticNet RandomForest --dataset /tmp/opencode/toy.h5mu --split-mode LSO`
    finishes in under 60 s (check the real `run` flag names first);
  - `uv run --frozen drevalpy data synthetic --output /tmp/opencode/big.h5mu --n-samples 40 --n-drugs 8 --n-tissues 4`
    writes a dataset that `drevalpy.load("/tmp/opencode/big.h5mu")` reads back with
    40 samples and 8 drugs;
  - `HOOKS` passes.
- **Context:** section 3 (Modules, call tree 3, Tests); AGENTS.md "Test layout" and
  "Import cost"; `src/drevalpy/testing/synthetic.py`; `tests/synthetic/builders.py`, for
  how it adds SMILES.

#### P2: `drevalpy list zoo-presets` and `list datasets`

- **Files owned:** `src/drevalpy/cli/catalog/__init__.py`,
  `src/drevalpy/cli/catalog/registries.py` and their mirrored tests under
  `tests/cli/catalog/`.
- **Interface:** two new `list` subcommands, rendered through `cli/catalog/_render.py`.
- **Depends on:** Wave 0.
- **Done when:** `uv run --frozen pytest -q -m "not network" tests/cli` and `HOOKS` pass.
- **Context:** section 3 (call tree 4); `src/drevalpy/models/zoo/__init__.py`;
  `registry/dataset/_registry.py::to_dataframe`.

#### P3: response transformation flag

- **Files owned:** `src/drevalpy/cli/run.py`, `src/drevalpy/cli/single.py` and their
  mirrored tests.
- **Interface:** `--response-transformation` on both commands, default `none`, mapped to
  `None`.
- **Depends on:** Wave 0.
- **Done when:** `uv run --frozen pytest -q -m "not network" tests/cli` and `HOOKS` pass.
- **Context:** section 3 (call tree 5); `src/drevalpy/_run.py`; `src/drevalpy/_single.py`;
  `models/_target_transform.py`, for the accepted names.

#### P4: `DREVALPY_EXTENSIONS_DIR` in `load_extensions`

- **Files owned:** `src/drevalpy/registry/_extensions.py`, `src/drevalpy/cli/main.py`,
  `tests/registry/test_extensions.py` and the mirrored test for `cli/main.py`.
- **Interface:** `load_extensions(directories=None)` reads the env var. The CLI calls
  `load_extensions()` and then `load_extensions(directories=flags)`.
- **Depends on:** Wave 0.
- **Done when:**
  `uv run --frozen pytest -q -m "not network" tests/registry tests/cli` and `HOOKS` pass.
- **Context:** section 3 (call tree 6); AGENTS.md "Tests", for registry teardown.

#### P5: dataset registry conflict policy

- **Files owned:** `src/drevalpy/registry/dataset/__init__.py`, `_registry.py` and
  `tests/registry/dataset/*`. The dataset-declaration YAML loader in
  `registry/_extensions.py` belongs to P4 and needs no change, because loading the same
  YAML twice is an identical repeat and is therefore a no-op.
- **Interface:** `override: bool = False` keyword on `register_source` and
  `register_dataset`. A conflict raises `ValueError`, matching the component registries'
  `"… already registered"` messages. An identical repeat is a no-op.
- **Depends on:** Wave 0.
- **Done when:** `uv run --frozen pytest -q -m "not network" tests/registry` and `HOOKS`
  pass.
- **Context:** section 3 (call tree 7, Tests); `registry/predictor/_registry.py:64`.

#### P6: snippet runner

- **Files owned:** `docs/_snippets.py`, `tests/docs/test_snippets.py`,
  `tests/docs/test_snippets_runner.py`, and deletion of `tests/docs/test_model_workflows.py`.
- **Interface:** `collect`, `run_page` and `SnippetError`, as in the Modules table.
  `test_snippets.py` is parametrized over the pages in `tutorials/` and `how-to/` (stubs
  pass trivially).
- **Depends on:** Wave 0.
- **Done when:** `DOCTESTS` passes, and `test_snippets_runner.py` covers every case under
  Tests.
- **Context:** section 3 (writing conventions, Modules, call tree 2, Tests); the current
  `tests/docs/test_model_workflows.py`.

#### P7: layout guard

- **Files owned:** `tests/docs/test_docs_structure.py` and `tests/docs/test_examples.py`.
- **Interface:** the rules under "Seam: the page layout" and "Seam:
  `tests/docs/test_examples.py`". To keep Wave 1 green, `test_examples.py` accepts that
  the literal includes are still on the excluded `python/extensions.rst`, until D3
  re-points them.
- **Depends on:** Wave 0.
- **Done when:** `DOCTESTS` passes against the stub tree.
- **Context:** section 3 (writing conventions, Tests); the current versions of both files.

#### P8: CLI reference generator

- **Files owned:** `docs/_cli_click.py` (logic only; P0 changed the path).
- **Interface:** the generated reference documents every subcommand of `data`,
  `experiments` and `list`. The root block describes the global `-e` option instead of
  "Run the full experiment suite".
- **Depends on:** Wave 0.
- **Done when:**
  - `DOCS` and `DOCTESTS` pass;
  - `grep -c "drevalpy list predictors" docs/reference/_generated_cli.rst` prints at
    least 1;
  - `grep -c "Run the full experiment suite" docs/reference/_generated_cli.rst` prints 0.
- **Context:** section 3 (call tree 1); `src/drevalpy/cli/main.py`.

#### P9: audit folders

- **Files owned:** `docs/project/architecture_audit/` (delete), and
  `docs/project/literature_audit/` (delete all except `differences.md`). Move
  `manifest.json`, `srmf-reference.mat` and `octave-version.txt` to
  `tools/literature_reference/`. Also `tools/literature_reference/README.md`,
  `tests/models/test_global_models.py:241` and `.prettierignore`.
- **Interface:** `test_global_models.py` reads
  `tools/literature_reference/manifest.json`. Relative paths inside the moved manifest are
  updated.
- **Depends on:** Wave 0.
- **Done when:**
  - `uv run --frozen pytest -q -m "not network" tests/models/test_global_models.py` and
    `HOOKS` pass;
  - `grep -rn "architecture_audit\|literature_audit" --exclude-dir=.git --exclude-dir=.venv .`
    lists only `docs/conf.py` (the exclusion) and `differences.md`, which D6 reads.
- **Context:** section 2 (Decisions).

### Wave 2 (pages, all in parallel)

Every docs package follows the writing conventions and fixes the contradictions assigned
to it in section 3. Each one owns only the pages listed for it in the page map.

The shared done-when check for each package:

- `DOCS` passes;
- `DOCTESTS` passes;
- `uv run --frozen pytest -q -p no:cacheprovider -m network tests/docs/test_snippets.py -k <dir>`
  passes on a machine with bucket access;
- the term sweep finds nothing on its pages:
  `grep -rniE "zoo model|test mode|building block|\batom\b|omics view|feature view|primary model|cell.?line|\bLCO\b" <its pages>`.
  Any "cell line" hit must describe data that really are cell lines, and any LCO hit
  must be the alias note on the split-mode reference page.

#### D1: tutorials

- **Files owned:** `docs/tutorials/*.rst`.
- **Interface:**
  - Each tutorial starts with `drevalpy data synthetic` or `drevalpy.synthetic(...)`.
  - Every step shows the output the reader should see.
  - The first tutorial (`first_benchmark`) shows the Simpson pattern on the reader's
    own results: `NaiveDrugMeanPredictor` scores high pooled and near zero per drug,
    and ElasticNet beats `NaiveMeanEffectsPredictor` only on the per-drug metric. It
    links to the explanation page on metrics for why.
  - The tutorials end by reading the report (D1), comparing the reader's predictor with a
    naive baseline (`own_predictor`), and reading HPO and stress-test results
    (`tune_and_stress_test`).
  - `own_predictor.rst` literal-includes `docs/examples/toy_ridge_predictor.py` and
    `toy_sample_featurizer.py`.
- **Depends on:** P0, P1, P3, P6.
- **Done when:** shared check.
- **Context:** sections 2 and 3 (page map, writing conventions); `docs/examples/`;
  `technical-writing` "Tutorial".

#### D2: how-to guides for using DrEvalPy

- **Files owned:** `docs/how-to/*.rst` (not the subdirectories).
- **Interface:** as in the page map. `list_available.rst` uses all eight `drevalpy list`
  subcommands.
- **Depends on:** P0, P1, P2, P3, P6.
- **Done when:** shared check.
- **Context:** section 3 (page map, contradictions marked D2); the source pages named in
  the map.

#### D3: how-to guides for extending DrEvalPy

- **Files owned:** `docs/how-to/extend/*.rst`.
- **Interface:** as in the page map. This package moves every `literalinclude` of
  `docs/examples/*.py` here.
- **Depends on:** P0, P4, P5, P6, P7.
- **Done when:** shared check, plus `uv run --frozen pytest -q -m "not network" tests/docs/test_examples.py`
  passes with the temporary allowance from P7 removed.
- **Context:** section 3 (contradictions marked D3); `python/extensions.rst`;
  `cli/extensions.rst`; `drevalpy/plugin/__init__.py`.

#### D4: migration guides

- **Files owned:** `docs/how-to/migrate/*.rst`.
- **Interface:**
  - `cli.rst` maps every v1.6.0 CLI flag and script (`git show v1.6.0:` the CLI and
    `pyproject.toml` scripts) to v2, or lists it as removed.
  - `python.rst` maps `MODEL_FACTORY`, the model classes and the experiment functions.
  - `models.rst` ports a 1.x `DRPModel` subclass to a featurizer and predictor pair.
  - All three share a table that maps 1.x model names to v2 zoo names, or marks them
    removed (for example EnsembleMF, PaccMann, ElasticNetModel, SVMRegressor). Check each
    name against `models/zoo/`.
  - Each page says whether 1.x results and checkpoints load in v2 (verify in
    `models/mixins/_persistence_io.py`).
  - Each page maps the 1.x cell-line names it touches (`cell_line_ids`, `LCO`,
    `test_mode`, `--hpo-num-samples` if present) to the v2 sample names, and says that
    LCO is still accepted as an alias for LSO.
- **Depends on:** P0, P6.
- **Done when:** shared check, and every v2 name on the pages appears in
  `uv run --frozen drevalpy --help`, in `drevalpy list zoo-presets`, or in
  `src/drevalpy` (grep).
- **Context:** section 2; `git tag` (`v1.6.0`).

#### D5: reference

- **Files owned:** the hand-written `docs/reference/*.rst` pages (not `glossary.rst` and
  not the generated includes).
- **Interface:** as in the page map. `python_api.rst` adds autosummary entries for
  `drevalpy` (`run`, `single`, `load`, `split`, `synthetic`, `randomization`,
  `robustness`), `drevalpy.registry` and `drevalpy.models.zoo`, so that
  `:func:`~drevalpy.run``resolves.`environment.rst`documents all six`DREVALPY\_\*`
  variables.
- **Depends on:** P0, P1, P4, P5.
- **Done when:** shared check, plus every `DREVALPY_` name found by
  `grep -rhoE "DREVALPY_[A-Z_]+" src/drevalpy | sort -u` appears in
  `docs/reference/environment.rst`.
- **Context:** section 3 (contradictions marked D5); the source pages named in the map.

#### D6: explanation

- **Files owned:** `docs/explanation/*.rst`.
- **Interface:**
  - As in the page map; no code blocks.
  - `literature_models.rst` keeps the per-model adaptations from `differences.md` (what
    changed and why). It drops hashes, commits and audit wording.
  - `training_lifecycle.rst` is the only page that explains A/E/V and the pipeline states.
    Other pages link to it.
  - `metrics.rst` explains why pooled scores mislead (drug potency dominates the pooled
    variance, a Simpson-like effect), and why DrEvalPy reports per-drug metrics and
    compares against `NaiveMeanEffectsPredictor`. The first tutorial links here.
- **Depends on:** P0.
- **Done when:** shared check.
- **Context:** section 3 (contradictions marked D6); `project/v2_architecture.rst`;
  `project/v2_migration.rst`; `docs/project/literature_audit/differences.md`.

#### D7: README, project pages, AGENTS.md

- **Files owned:**
  - `README.md`, `docs/project/{contributing,news}.rst`, `AGENTS.md` (the docs commands
    and a new "Docs layout" section on modes, interface tabs and snippet markers, plus the
    "seven" → "eight" fix);
  - `.github/pull_request_template.md`;
  - `plans/docs-diataxis-followups.md`.
- **Interface:**
  - README links point at the new paths.
  - Its Python example uses `ModelSpec.from_name` and `drevalpy.run`.
  - `news.rst` notes the `single` default change and the new commands.
- **Depends on:** P0, P1, P2, P3.
- **Done when:** shared check, plus every `drevalpy.readthedocs.io/en/latest/<path>.html`
  in README maps to an existing `docs/<path>.rst`.
- **Context:** sections 2 and 3 (Risks).

### Wave 3: integration

#### P10: delete the old tree and sweep

- **Files owned:**
  - deletion of `docs/concepts/`, `docs/python/` (except anything P0 moved),
    `docs/cli/`, `docs/getting_started/`, `docs/readme.rst`, `docs/project/v2_*.rst` and
    `docs/project/literature_audit/`;
  - `docs/conf.py` (removing the temporary exclusions);
  - leftover cross-page link fixes in any page.
- **Interface:** none new.
- **Depends on:** all of Wave 2.
- **Done when:**
  - `DOCS` and `HOOKS` pass;
  - `uv run --frozen pytest -q -p no:cacheprovider -m "not network"` passes in full;
  - the term sweep from Wave 2 over `docs/` (excluding generated files) and `README.md`
    finds nothing;
  - `git grep -n "getting_started/\|concepts/\|python/quickstart\|cli/quickstart" -- docs README.md AGENTS.md`
    finds nothing.
- **Context:** this whole plan.
