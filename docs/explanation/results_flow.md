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

# How a run becomes results: from a fold to the report

Once a model is ready to train on a fold (see {doc}`data_flow` and {doc}`training_lifecycle`), a lot
still happens before numbers appear in a report. This page follows that second half of an
experiment: freezing the fold setup, running tuning and final training through one shared function,
saving each run safely to disk, aggregating scores, normalizing against a naive baseline, and
selecting plots for the MultiQC report.

For the exact on-disk file schemas, see {doc}`/reference/results`. For how random seeds are
derived, see {doc}`reproducibility`.

## The trajectory at a glance

```text
drevalpy.run / drevalpy.single / drevalpy.train
  │  RunStore.open writes <out>/metadata.json (run and single)
  │  split → robustness() / randomization() variants (run)
  ▼
Per model, fold, and dataset variant (_single._fit_fold)
  │  build_execution → ExecutionSpec(Fold, RunIdentity, hpo, seed, metadata)
  │    Fold: train (A), optional stopping (E), val (V), test
  │          train_val (A ∪ V) is derived from train and val
  ▼
experiment._fit.fit  ← shared by every entry point
  │  seed_everything(fit_seed(...))
  │  hpam_tune → best hyperparameters + TrialResult list
  │  DRPModel(spec, best).train(final_training, early_stopping=E)
  │  predict(test) → NaN check → evaluate → metrics
  ▼
RunResult (predictions, ground_truth, sample_ids, drug_ids, metrics, trials, ...)
  │  RunStore.save → <out>/<model>/fold-<i>..._<digest>.npz
  ▼
ExperimentResult (runs, failures)
  ├── ModelResult per model (runs grouped by model_name)
  │    select_runs (ordinary vs. stress-test runs)
  │    aggregate_metrics (stored pooled scores)
  │    aggregate_group_metrics (recomputed per-group scores)
  └── normalize(reference) → residuals, raw_metrics, reference_runs
       │
       ▼
visualization.report.create_report
  │  registry.visualization.applicable(experiment) → PlotRequirement → MultiQC
```

## Step 1: Freezing fold decisions in `ExecutionSpec`

Before tuning or training starts, `build_execution` constructs an immutable
{py:class}`~drevalpy.experiment._execution.ExecutionSpec` that captures every decision about the
fold: its {py:class}`~drevalpy.types.data.fold.Fold`, its
{py:class}`~drevalpy.types.results._identity.RunIdentity`, and its tuning settings.

Freezing these choices up front ensures that hyperparameter tuning, the final fit, and any stress
tests (robustness or randomization runs) all work on the exact same scopes and seed. Given the same
`ExecutionSpec`, a fold always behaves the same way.

Each scope in the {py:class}`~drevalpy.types.data.fold.Fold` uses a
{py:class}`~drevalpy.types.data.selection_mask.SelectionMask`, a boolean mask of shape
`(n_samples, n_drugs)`.

- **`train` (A).** Each tuning trial fits on these pairs.
- **`stopping` (E).** For models that support early stopping, `build_execution` cuts a
  seeded holdout from the validation set. For other models, `stopping` is `None`.
- **`val` (V).** Tuning scores each trial on the validation pairs left after any
  early-stopping holdout.
- **`train_val` (A ∪ V).** This property derives the final training scope from
  `train` and `val`. The stopping pairs stay held out for the final fit.
- **`test`.** The final model predicts and is scored on these held-out pairs.

The early-stopping slice is created by
{func}`~drevalpy.data.splitters._folds.early_stopping_split`, which cuts along the same leakage
unit as the split mode, seeded from the validation mask's `pair_seed`. Under LSO, `stopping` holds
whole samples; under LDO, whole drugs; under LStO, whole pairs (because studies are already the
test unit and validation is drawn by pairs).

```{code-cell} ipython3
import warnings

warnings.filterwarnings("ignore")

import drevalpy
from drevalpy.data.splitters._folds import early_stopping_split
from drevalpy.experiment._execution import build_execution
from drevalpy.models.spec import ModelSpec
from drevalpy.registry.splitter import canonical

dataset = drevalpy.synthetic(omics=("gene_expression",), n_samples=50, n_drugs=20)
folds = drevalpy.split(dataset, "LSO", n_splits=3, validation_ratio=0.3)
fold = folds[0]

model_spec = ModelSpec.from_name("NaiveMeanEffectsPredictor")
execution = build_execution(
    model_spec,
    dataset,
    fold,
    hyperparameter_tuning=True,
    hpo_metric="RMSE",
    hpo_num_trials=4,
    hpo_random_state=None,
    seed=42,
)

stopping_scope = execution.fold.stopping
print("Fold:")
print(f"  train:     {len(execution.fold.train)} pairs")
print(f"  stopping:  {len(stopping_scope) if stopping_scope is not None else 'None'}")
print(f"  val:       {len(execution.fold.val)} pairs")
print(f"  train_val: {len(execution.fold.train_val)} pairs")
print(f"  test:      {len(execution.fold.test)} pairs")
print(f"\nIdentity: {execution.identity.model_name}, fold {execution.identity.fold_index}, {execution.identity.split_mode}")

# Show the early-stopping cut explicitly.
stopping, remaining_val = early_stopping_split(dataset, fold, canonical("LSO"))
print("\nEarly-stopping cut on this fold:")
print(f"  E (stopping):   {len(stopping)} pairs")
print(f"  V (remaining):  {len(remaining_val)} pairs")
```

## Step 2: One shared training path (`fit`)

Every entry point that trains a model — `drevalpy.run`, `drevalpy.single`, `drevalpy.train`, and
even the golden-test recorder — funnels through {func}`~drevalpy.experiment._fit.fit`. Keeping one
shared path guarantees that a run produces the exact same `RunResult` no matter which command
launched it.

Inside `fit`, six steps run in order:

1. Seed the global RNG state with `fit_seed(execution.seed, identity)`.
2. Run `hpam_tune` to select the best hyperparameters and collect a list of
   {py:class}`~drevalpy.types.results.run.TrialResult` objects (or return the model's defaults if
   tuning is turned off).
3. Build a {py:class}`~drevalpy.models.drp_model.DRPModel` with those hyperparameters and train it
   on `fold.train_val`, passing `fold.stopping` for early stopping.
4. Predict on `fold.test` and check for non-finite values: if more than 20% of finite ground-truth
   pairs receive a `NaN` or infinite prediction, raise {py:class}`~drevalpy.errors.NaNPredictionError`.
5. Score the predictions against ground truth with {func}`~drevalpy.evaluation.evaluate`.
6. Return the fitted model alongside its {py:class}`~drevalpy.types.results.run.RunResult`.

```{code-cell} ipython3
from drevalpy import single

result = single(
    model_spec,
    dataset,
    fold,
    hyperparameter_tuning=False,
    response_transformation=None,
    hpo_metric="RMSE",
    hpo_num_trials=0,
    hpo_random_state=None,
    seed=42,
    wandb_project=None,
    n_jobs=1,
)

print(f"Run predictions: {len(result.predictions)} pairs")
print(f"Metrics: {', '.join(f'{k}={v:.3f}' for k, v in list(result.metrics.items())[:3])}")
```

## Step 3: Identifying runs with three keys

Every run carries a {py:class}`~drevalpy.types.results._identity.RunIdentity` recording the model
name, dataset name, split mode, fold index or fold ID, and any stress-test settings (randomization
or robustness repeat). From those fields, it derives three keys, each serving a different job:

- **`fold_key`**: identifies one fold across ordinary and stress-test runs
  (`(dataset_name, measure, split_mode, "id" or "index", fold_id or fold_index)`). Used to group
  runs that belong to the same fold.
- **`checkpoint_key`**: adds the model name, randomization, and robustness repeat to `fold_key`.
  Because every field is known _before_ training starts, `RunStore` can check whether a run is
  already saved on disk.
- **`run_key`**: adds execution provenance (a JSON digest of settings including seeds) to
  `checkpoint_key`, so two runs with different seeds or settings never collide in reports.

When a run is saved, `run_filename` combines a human-readable prefix with a 16-character SHA-256
digest of `checkpoint_key` — for example `fold-0_SVRC_gene_expression_repeat-1_<digest>.npz`.
Because the filename depends only on identity, not completion order, saving the same experiment
again always maps to the same file paths.

```{code-cell} ipython3
print(f"model_name:     {result.identity.model_name}")
print(f"fold_index:     {result.identity.fold_index}")
print(f"fold_id:        {result.identity.fold_id!r}")
print(f"checkpoint_key: {result.identity.checkpoint_key}")
print(f"run_key length: {len(result.identity.run_key)} items")
print(f"run_id (JSON):  {result.run_id[:60]}...")
```

## Step 4: Atomic saves and resuming interrupted experiments

When `drevalpy.run` opens an output directory, `RunStore` immediately writes `metadata.json` with
the experiment settings. Each run is then saved as soon as it finishes: `RunStore.save` writes to a
temporary `.partial` file and atomically replaces the final `.npz`, so a killed process never
leaves a corrupt run file behind. Inside the `.npz` are the prediction and ground-truth arrays,
`sample_ids`, `drug_ids`, optional `trial_<i>_predictions`, and a JSON `_metadata` payload.

Saving one file per run is what makes resuming work: if you re-run `drevalpy.run` with the same
settings and output directory, any run already on disk is loaded instead of refitted.

Stress-test runs (randomization and robustness repeats) skip hyperparameter tuning altogether.
Because ordinary folds run first, each stress-test run simply reuses the best hyperparameters found
by the ordinary run on the same fold. If the ordinary run failed, its stress-test runs are recorded
as failures as well.

```{code-cell} ipython3
import tempfile
from upath import UPath

with tempfile.TemporaryDirectory() as tmpdir:
    result_path = UPath(tmpdir) / "run.npz"
    result.save(result_path)
    print(f"Saved {result_path.stat().st_size} bytes")
    print(f"Arrays: predictions {result.predictions.shape}, ground_truth {result.ground_truth.shape}")
    print(f"Pair IDs: sample_ids {result.sample_ids.shape}, drug_ids {result.drug_ids.shape}")
```

## Step 5: Three result levels — Run, Model, Experiment

Results are organized into three nested containers:

- {py:class}`~drevalpy.types.results.run.RunResult`: predictions, ground truth, and pooled scores
  for one model on one fold.
- {py:class}`~drevalpy.types.results.model.ModelResult`: all runs belonging to one model.
- {py:class}`~drevalpy.types.results.ExperimentResult`: all models in an experiment, plus any
  recorded failures.

Having all three levels lets reports and analysis scripts work at the granularity they need. At the
model level, `aggregate_metrics` computes the mean and standard deviation of the pooled scores
stored in each run file. Per-group scores (by drug, sample, tissue, and so on) are not stored on
disk; instead, `aggregate_group_metrics` recomputes them on demand via
`evaluation.evaluate_per_group`.

By default, `ModelResult.select_runs` returns only ordinary runs, keeping randomization and
robustness repeats out of the main summary metrics unless you explicitly ask for them. Built-in
report plots leave stress-test runs untouched on disk for custom analysis.

```{code-cell} ipython3
experiment = drevalpy.types.results.ExperimentResult([result])

print(f"Experiment: {experiment.dataset_name}, {experiment.split_mode}")
print(f"Models: {experiment.model_names}")
print(f"Model 0 runs: {len(experiment.models[0].runs)}")
agg = experiment.models[0].aggregate_metrics
print(f"Model 0 RMSE: mean={agg['RMSE']['mean']:.3f}, std={agg['RMSE']['std']:.3f}")
```

## Step 6: Normalization — measuring what a model adds beyond a baseline

A simple naive baseline like `NaiveMeanEffectsPredictor` already explains a substantial share of
drug-response variance using only average sample and drug effects. To measure what a model adds
_beyond_ that baseline,
{meth}`ExperimentResult.normalize <drevalpy.types.results.ExperimentResult.normalize>` subtracts a
reference model's predictions from both the model's predictions and the ground truth, then
recomputes all metrics on the resulting residuals.

On those residuals, a Pearson correlation near zero means the model adds nothing beyond the
baseline. Error metrics such as RMSE stay unchanged, because subtracting the same reference value
from both `y_pred` and `y_true` leaves their difference untouched (see {doc}`metrics`).

To keep the baseline consistent, the reference model always trains **without a response
transformation** — `drevalpy.run` and `drevalpy.single` enforce this automatically so the
transformation cannot shift the ruler.

`normalize()` returns a new `ExperimentResult` without discarding the original numbers:

- `raw_metrics` and `raw_per_drug_pearson` keep the pre-normalization pooled scores and mean
  per-drug Pearson correlation so the leaderboard can show both.
- `normalized_by` records which reference model was used.
- `reference_runs` keeps lightweight summaries of the reference model's runs (metrics only, arrays
  dropped) so the leaderboard and critical-difference plot can still display it.

```{code-cell} ipython3
with tempfile.TemporaryDirectory() as tmpdir:
    out_dir = UPath(tmpdir) / "results"
    exp = drevalpy.run(
        models=[
            ModelSpec.from_name("NaiveMeanEffectsPredictor"),
            ModelSpec.from_name("NaiveDrugMeanPredictor"),
        ],
        dataset=dataset,
        split_mode="LSO",
        n_splits=2,
        validation_ratio=0.3,
        hyperparameter_tuning=False,
        output_dir=str(out_dir),
        seed=42,
    )

    files = sorted(p.relative_to(out_dir) for p in out_dir.rglob("*") if p.is_file())
    print(f"Files written: {len(files)}")
    print(f"  metadata.json present: {(out_dir / 'metadata.json').exists()}")
    print(f"  Model directories: {[p.name for p in out_dir.iterdir() if p.is_dir()]}")

    run0 = exp.models[1].runs[0]  # NaiveDrugMeanPredictor, fold 0
    print("\nBefore normalization:")
    print(f"  predictions[0]={run0.predictions[0]:.3f}, ground_truth[0]={run0.ground_truth[0]:.3f}")
    print(f"  Pearson={run0.metrics['Pearson']:.3f}, RMSE={run0.metrics['RMSE']:.3f}")
    print(f"  normalized_by={exp.normalized_by}")

    exp_norm = exp.normalize("NaiveMeanEffectsPredictor")
    run0_norm = exp_norm.models[0].runs[0]
    print(f"\nAfter normalize() ({run0_norm.identity.model_name}):")
    print(f"  predictions[0]={run0_norm.predictions[0]:.3f}, ground_truth[0]={run0_norm.ground_truth[0]:.3f}")
    print(f"  Pearson={run0_norm.metrics['Pearson']:.3f}, RMSE={run0_norm.metrics['RMSE']:.3f}")
    print(f"  raw_metrics['Pearson']={run0_norm.raw_metrics['Pearson']:.3f} (before normalization)")
    print(f"  normalized_by={exp_norm.normalized_by}")
    print(f"  reference_runs: {len(exp_norm.reference_runs)} summaries (arrays dropped)")
```

## Step 7: How the report picks its plots

Finally, {func}`~drevalpy.visualization.report.create_report` turns an `ExperimentResult` into a
MultiQC HTML report. Rather than hard-coding a fixed list of plots, it asks
{func}`~drevalpy.registry.visualization.applicable` which registered plot classes match the
experiment.

Each plot declares what it needs using {py:class}`~drevalpy.visualization.requirements.PlotRequirement`
flags — such as `MULTIPLE_MODELS` or `MULTIPLE_FOLDS` — along with a `result_type`
(`ExperimentResult` for one experiment-wide plot, or `ModelResult` for one plot per model). Plots
whose requirements are not met are simply skipped, and the rest return MultiQC sections via
`plot.sections(result)`.

```{code-cell} ipython3
from drevalpy.registry.visualization import applicable
from drevalpy.visualization.requirements import PlotRequirement, requirements_met

for label, exp_result in [("one run", experiment), ("two models, two folds", exp)]:
    met = [req.name for req in PlotRequirement if requirements_met(exp_result, frozenset([req]))]
    print(f"{label}: requirements met = {met}")
    print(f"  applicable plots: {[plot.name for plot in applicable(exp_result)]}")
```
