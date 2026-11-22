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

# How a dataset is made, stored and loaded

Before a model can train, raw well-level viability measurements have to be fitted into
dose-response curves, combined with sample and drug features into an `.h5mu` file, registered so
they can be loaded by name, and filtered for curve quality.

This page follows a dataset through that entire lifecycle — all offline on a small synthetic
screen. See {doc}`datasets_provenance` for why DrEvalPy refits curves and filters by quality,
{doc}`data_flow` for what happens after a dataset is loaded, and {doc}`/how-to/bring_your_own_data`
for a step-by-step guide to packaging your own screen.

## The trajectory at a glance

```text
Well table (CSV / Parquet)
  │  drevalpy.curation.curate(df)
  │  validates columns, numbers dose series, tags replicates
  ▼
_plan.plan
  │  splits into dose-range groups, normalizes group-wide, chunks into work units
  ▼
_plan.run_chunks (local) or Chunk.run per chunk (cluster / workflow manager)
  │  CurveCurator fits each chunk independently
  ▼
_plan.assemble
  │  reassembles groups, applies SAM significance thresholds, builds AnnData
  ▼
AnnData (.h5ad)
  X: pEC50 (n_samples, n_drugs)
  layers: LN_IC50, relevance_score, fold_change, regulation, R2, RMSE, ...
  uns: {"replicate_fits": DataFrame} (optional)

─── Assembling the dataset (see /how-to/bring_your_own_data) ───
Combine the response AnnData with sample views, drug features, and metadata
→ Dataset(mdata, name=...).save(path)

MuData (.h5mu)
  mod["response"]: X, layers, var (canonical_smiles), varm (drug views)
  mod[view]: sample views (gene_expression, proteomics, ...)
  obs: tissue, study, source_sample_id
  uns: drug_graphs, precomputed_views, ...
  │  [optional: data.precompute.precompute_all]
  ▼
Registration
  Built-in: available_datasets.json (name, file, sha256, source)
  User: register_source + register_dataset → config_dir/datasets.json
  ▼
drevalpy.load(name_or_path)
  │  local .h5mu path → Dataset.load(path)
  │  registered name  → cache lookup (SHA-256 verified) → download if missing or corrupt
  ▼
Dataset (MuData wrapper)
  ▼
Splits & quality filtering
  drevalpy.split(dataset, mode) → Folds
  observed_mask(dataset, CurveQuality) → boolean mask of passing pairs
```

## Step 1: Turning well measurements into curves

{py:func}`drevalpy.curation.curate` is the entry point for fitting dose-response curves. You pass
it a long table with one row per well (`sample`, `drug`, `concentration` in µM, and `intensity` as
viability), plus optional `experiment`, `replicate`, and `position` columns when your screen has
replicates. It returns an {py:class}`~anndata.AnnData` of shape `(n_samples, n_drugs)` holding the
fitted curve metrics.

Curation is deliberately **identifier-agnostic**: `sample` and `drug` are treated as opaque labels.
That lets a pipeline run the slow curve-fitting step once on raw screen or vendor identifiers, and
resolve canonical IDs (Cellosaurus accessions, PubChem CIDs, duplicate collapsing) afterward by
simply renaming indices.

```{code-cell} ipython3
:tags: [remove-stderr]

import contextlib
import io
import os
import tempfile
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# Build a small well table: 3 samples × 3 drugs, one dose series each.
# Include one flat curve (drugC, no effect) and one unmeasured pair (S2 × drugC).
rng = np.random.default_rng(42)
concentrations = 10.0 ** np.arange(-3, 4)
ec50 = {"drugA": 0.5, "drugB": 5.0, "drugC": 0.5}
rows = []
for sample in range(3):
    shift = 10 ** rng.normal(0, 0.3)
    for drug, base in ec50.items():
        if sample == 2 and drug == "drugC":
            continue
        for conc in concentrations:
            if drug == "drugC":
                viability = 1.0 + rng.normal(0, 0.02)
            else:
                viability = 1 / (1 + (conc / (base * shift)) ** 1.5) + rng.normal(0, 0.03)
            rows.append((f"S{sample}", drug, conc, max(0, viability)))

wells = pd.DataFrame(rows, columns=["sample", "drug", "concentration", "intensity"])
print(f"Well table: {len(wells)} rows, {wells['sample'].nunique()} samples, {wells['drug'].nunique()} drugs")
print(f"First 3 rows: {list(zip(wells['sample'][:3], wells['drug'][:3], wells['concentration'][:3].round(3)))}")
```

Curating this table returns an {py:class}`~anndata.AnnData` whose `X` holds `pEC50` and whose
`layers` store every other metric: derived summaries (`EC50`, `IC50`, `LN_IC50`, `AUC`),
goodness-of-fit statistics (`R2`, `RMSE`, `p_value`, `log_p_value`, `f_value`, `f_value_sam`),
curve parameters (`slope`, `front`, `back`, `fold_change`) with their standard errors, quality
scores (`relevance_score`, `signal_quality`), and the regulation verdict (`regulation`). Nothing is
filtered out yet; every fitted curve is kept.

```{code-cell} ipython3
:tags: [remove-stderr]

from drevalpy.curation import curate

# Use fit_speed="fast" for the demo; real pipelines use "exhaustive".
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    adata = curate(wells, fit_speed="fast")

print(f"AnnData shape: {adata.shape}")
print(f"X (pEC50) range: [{np.nanmin(adata.X):.2f}, {np.nanmax(adata.X):.2f}]")
print(f"Layers (first 8): {list(adata.layers.keys())[:8]}")
print(f"relevance_score of S0 × drugA: {adata['S0', 'drugA'].layers['relevance_score'][0, 0]:.3f}")
```

## Step 2: Splitting curation into plan, fit, and collect

Fitting tens of thousands of curves on a large screen can take hours. To make that work easy to
parallelize on a cluster with Snakemake or Nextflow, {py:mod}`drevalpy.curation._plan` breaks
curation into three stages:

1. **Plan** ({py:func}`~drevalpy.curation._plan.plan`): checks the columns of the well table,
   numbers dose series, groups wells by dose range, runs any group-wide normalization once (so results never depend on chunk size), and
   splits each group into chunks of at most `max_chunk_size` curves. Each
   {py:class}`~drevalpy.curation._plan.Chunk` is self-contained — identical curves and fit options
   produce identical chunk files, so content-addressed workflow caches only refit chunks that
   actually changed.
2. **Fit** ({py:meth}`Chunk.run <drevalpy.curation._plan.Chunk.run>`): runs CurveCurator on a
   single chunk on one core. Failures are turned into clean rows rather than crashes: a dropped
   curve gets `status = "fit_failed"` with `NaN` metrics, and even a chunk where CurveCurator
   drops every curve (triggering an internal `exit()`) is caught and returned as an empty result.
3. **Collect** ({py:func}`~drevalpy.curation._plan.assemble`): stitches each group's fitted chunks
   back together, applies SAM significance thresholding once across the entire group (again keeping
   results independent of chunking), extracts the final metrics, and pivots everything into an
   {py:class}`~anndata.AnnData`.

Locally, `curate()` runs all three stages in one process. On a cluster, you can drive them
separately with `drevalpy curate plan`, `drevalpy curate fit-chunk`, and `drevalpy curate collect`
({py:mod}`drevalpy.cli.curate`), using {py:func}`~drevalpy.curation._plan.write_plan`,
{py:func}`~drevalpy.curation._plan.write_chunk`, and
{py:func}`~drevalpy.curation._plan.write_fitted` between jobs.

```{code-cell} ipython3
:tags: [remove-stderr]

from drevalpy.curation._fit import FitOptions
from drevalpy.curation._plan import plan

work = plan(wells, options=FitOptions(fit_speed="fast"))
print(f"Planned {len(work.chunks)} chunks for {len(work.series)} series")
print(f"Chunk 0: {len(work.chunks[0].frame)} curves, group {work.chunks[0].group}")
print(f"Group info: {work.infos[work.chunks[0].fit][work.chunks[0].group]}")
```

## Step 3: Assembling the `MuData` file

Curation intentionally stops at a response-only {py:class}`~anndata.AnnData`. Curve fitting is the
expensive one-time step; mapping identifiers, attaching sample omics views, and choosing drug
features are fast and often iterated.

To turn the curation output into a training-ready dataset, you assemble a
{py:class}`~mudata.MuData` containing the `response` modality alongside your sample views (such as
`gene_expression`) and drug features (`canonical_smiles` in `response.var`, matrices in
`response.varm`), wrap it in {py:class}`~drevalpy.types.data.dataset.Dataset`, and save it as an
`.h5mu` file. See {doc}`/how-to/bring_your_own_data` for the full recipe.

```{code-cell} ipython3
:tags: [remove-stderr]

import anndata as ad
import mudata as md

samples = adata.obs_names

# Synthetic gene expression: 3 samples × 10 genes.
gene_expr = ad.AnnData(
    rng.normal(6, 1, (len(samples), 10)).astype(np.float32),
    obs=pd.DataFrame({"tissue": ["Lung", "Blood", "Lung"]}, index=samples),
    var=pd.DataFrame(index=[f"GENE{i}" for i in range(10)]),
)

# Drug metadata: canonical SMILES (used by chemical featurizers).
adata.var["canonical_smiles"] = ["CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "C"]

with md.set_options(pull_on_update=False):
    mdata = md.MuData({"response": adata, "gene_expression": gene_expr})

print(f"MuData: {mdata}")
print(f"response.X shape: {mdata.mod['response'].X.shape}")
print(f"gene_expression.X shape: {mdata.mod['gene_expression'].X.shape}")
print(f"Sample tissues: {list(mdata.mod['gene_expression'].obs['tissue'])}")
```

Wrapping the `MuData` in {py:class}`~drevalpy.types.data.dataset.Dataset` validates its structure
and writes the `.h5mu` archive via {py:meth}`~drevalpy.types.data.dataset.Dataset.save`:

```{code-cell} ipython3
:tags: [remove-stderr]

from upath import UPath

from drevalpy.types.data.dataset import Dataset

dataset = Dataset(mdata, name="TOY_DEMO")

tmpdir = tempfile.TemporaryDirectory()
path = UPath(tmpdir.name) / "toy.h5mu"
dataset.save(path)
print(f"Saved to {path.name}, size {path.stat().st_size} bytes")
```

## Step 4: Precomputing fixed featurizer outputs (optional)

Some featurizers need no fitting — they apply a fixed transformation to a view or compute molecular
descriptors directly from SMILES. {py:func}`drevalpy.data.precompute.precompute` runs such a
featurizer once at its default settings, stores the resulting precomputed view in `response.varm`
(drug side) or as a sample modality, and records the settings under `uns["precomputed_views"]`.

When a model later uses that featurizer with matching settings, it reads the precomputed view
directly; if you change the settings, it computes fresh features on demand.
{py:func}`~drevalpy.data.precompute.precompute_all` does this for every registered precomputable
featurizer supported by the dataset. All shipped datasets include precomputed views, and you can
optionally run `precompute_all` on your own dataset before saving it.

## Step 5: Registering datasets by name

To load a dataset by a short name like `"GDSC1"` instead of a file path, it needs a registry entry.
DrEvalPy checks two registries:

- **Built-in datasets**:
  `available_datasets.json`, shipped in {py:mod}`drevalpy.data.datasets`, lists ten
  datasets (`GDSC1`, `GDSC2`, `CTRPv1`, `CTRPv2`, `CCLE`, `PRISM_19Q4`, `PRISM_20Q2`, `BeatAML2`,
  `PDX_Bruna`, `TOYv1`) with their filenames, SHA-256 checksums, and remote source names.
- **User datasets**:
  {py:func}`register_source <drevalpy.registry.dataset._registry.register_source>` and
  {py:func}`register_dataset <drevalpy.registry.dataset._registry.register_dataset>` write entries
  to `datasets.json` in your config directory ({py:func}`~drevalpy.data._paths.get_config_dir`,
  overridable with `DREVALPY_CONFIG_DIR`). A source holds a base URL and optional
  `storage_options` (such as S3 credentials); a dataset entry names the file within that source and
  an optional `sha256` hash.

See {doc}`registries_and_plugins` for details on configuring custom sources.

```{code-cell} ipython3
:tags: [remove-stderr]

from drevalpy.registry.dataset import dataset_registry

print(f"Built-in datasets ({len(list(dataset_registry.names()))}): {sorted(dataset_registry.names())[:5]} ...")

entry = dataset_registry.datasets["GDSC1"]
source_entry = dataset_registry.sources[entry.source]
print(f"GDSC1 entry: file={entry.file}, sha256={entry.sha256[:16]}..., source={entry.source}")
print(f"Source '{entry.source}': url={source_entry.url[:40]}...")
```

## Step 6: Loading by name or path with a checksum-verified cache

When you call {py:func}`drevalpy.load <drevalpy.data.datasets._load.load>`, DrEvalPy resolves the
argument in three steps:

1. **Direct file path**: if the argument points to an existing `.h5mu` file, load it immediately
   with {py:meth}`Dataset.load <drevalpy.types.data.dataset.Dataset.load>`.
2. **Registered name**: look up the entry and call
   {py:func}`~drevalpy.data.datasets._cache.fetch_registered`, which checks the local cache
   directory ({py:func}`~drevalpy.data._paths.get_default_data_dir`, overridable with
   `DREVALPY_CACHE_DIR`):
   - If the file is already cached, verify its SHA-256 hash (when configured) and load it. If the
     file is corrupt or its hash no longer matches a newer version, delete the cached copy and
     re-download. Environment errors such as permission failures or missing dependencies are
     propagated without deleting the cache
     ({py:func}`~drevalpy.data.datasets._cache._is_corruption_error`).
   - If the file is not cached (or was just removed), download it with
     {py:func}`~drevalpy.data._transfer.download_file`, verifying the hash along the way.
3. **Not found**: if neither matches, raise {py:exc}`~drevalpy.errors.InputNotFoundError`.

Here is that full registration-and-cache round trip on our toy `.h5mu` file, pointed at temporary
config and cache directories so your local setup stays untouched:

```{code-cell} ipython3
:tags: [remove-stderr]

import hashlib
from unittest import mock

import drevalpy
from drevalpy.data._paths import get_default_data_dir
from drevalpy.registry.dataset import register_dataset, register_source

with (
    tempfile.TemporaryDirectory() as temp_config,
    tempfile.TemporaryDirectory() as temp_cache,
    mock.patch.dict(os.environ, {"DREVALPY_CONFIG_DIR": temp_config, "DREVALPY_CACHE_DIR": temp_cache}),
):
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"File sha256: {sha256[:16]}...")

    register_source("local_demo", base_url=str(path.parent))
    register_dataset("toy_demo", source="local_demo", file=path.name)

    # First load: registry lookup → cache miss → copy from local source into cache.
    loaded_by_name = drevalpy.load("toy_demo")
    print(f"Loaded by name: {loaded_by_name.name}, shape {loaded_by_name.response_matrix.shape}")
    print(f"Cached files: {[p.name for p in UPath(get_default_data_dir()).glob('*.h5mu')]}")
```

## Step 7: Filtering curve quality at split time

Published screens store _every_ CurveCurator fit, including flat or noisy curves where no real
dose-response effect occurred. Models should be evaluated on meaningful curves, yet DrEvalPy never
deletes failing pairs from the `.h5mu` file. Instead, {py:class}`drevalpy.data.quality.CurveQuality`
defines thresholds over the response quality layers, and
{py:func}`drevalpy.data.splitters.observed_mask` builds a boolean mask of pairs that both have a
finite response and pass those thresholds. The splitter applies that mask when creating folds.

Keeping quality filtering at split time rather than file-creation time has three practical
benefits:

- **Flexible thresholds per experiment**: one experiment can demand strict curve quality while
  another tests how models handle noisier curves, all from the same `.h5mu` file.
- **Fair comparisons across models**: applying the rule once when splitting guarantees every model
  in an experiment trains and is scored on the exact same pairs.
- **Preserved diagnostics**: keeping quality metrics as response layers lets predictors read them
  (for example `regulation` via
  {py:attr}`~drevalpy.components.predictors.abstract.base.Predictor.response_layers`) and lets you
  inspect the quality of held-out curves.

{py:meth}`CurveQuality.default <drevalpy.data.quality.CurveQuality.default>` matches CurveCurator's
`alpha = 0.05` (`min_relevance_score >= -log10(0.05)`) and `fc_lim = 0.45`
(`min_abs_fold_change >= 0.45`). {py:meth}`CurveQuality.off <drevalpy.data.quality.CurveQuality.off>`
keeps every finite response without reading quality layers — useful for datasets that carry only a
`pEC50` matrix.

```{code-cell} ipython3
:tags: [remove-stderr]

from drevalpy.data.quality import _RULES, CurveQuality
from drevalpy.data.splitters import observed_mask

loaded = Dataset.load(path)
print("Layers CurveQuality reads:", sorted({layer for layer, _, _ in _RULES.values()}))

rule = CurveQuality.default()
mask = rule.mask(loaded)
print(f"\nDefault rule: {rule.to_metadata()}")
print(f"Quality mask passes: {mask.sum()} / {mask.size}")
print(f"Quality mask (S0, S1, S2 × drugA, drugB, drugC):\n{mask}")

# observed_mask combines the quality rule with the finite-response check.
obs_mask = observed_mask(loaded, rule)
print(f"\nObserved mask (finite & quality): {obs_mask.sum()} / {obs_mask.size}")
print(f"Observed mask:\n{obs_mask}")
measured = ~np.isnan(loaded.response_matrix)
print(f"Unmeasured pairs: {(~measured).sum()} | measured pairs failing the rule: {(measured & ~obs_mask).sum()}")

strict = CurveQuality(min_relevance_score=2.0, min_abs_fold_change=0.6)
strict_mask = observed_mask(loaded, strict)
print(f"\nStrict rule ({strict.to_metadata()}) passes: {strict_mask.sum()} / {strict_mask.size}")

tmpdir.cleanup()
```

That brings a dataset from raw well measurements all the way to filtered folds. See {doc}`data_flow`
for what happens next — from the loaded dataset through featurizers and into predictors.
