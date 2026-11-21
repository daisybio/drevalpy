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

# How data flows from an `.h5mu` file to a predictor

Between reading a dataset file from disk and calling a predictor's `_fit` or `_predict`,
DrEvalPy passes data through a small sequence of typed containers. Each step has one job:
separating raw storage from featurization, keeping features indexed per unique entity instead
of duplicated per pair, and preventing leakage across splits. This page traces that path from
disk to predictor on the built-in `TOYv1` dataset; see {doc}`models_from_components` for how
components declare compatibility and {doc}`training_lifecycle` for how folds, tuning and early
stopping run around it.

## The trajectory at a glance

```text
.h5mu file
  │  drevalpy.load(...) / Dataset.load(path)
  ▼
Dataset (MuData)
  ├── response modality   (n_samples × n_drugs)  X, response layers, var (SMILES), varm (drug views)
  ├── sample modalities   (n_samples × n_features)  gene_expression, proteomics, mutations, ...
  ├── mdata.obs           sample metadata (tissue, study, source_sample_id)
  └── mdata.uns           drug_graphs, precomputed_views, auxiliary metadata (e.g. pathways_gmt)
  │
  │  split(dataset, mode) ──► Fold(train, val, test: SelectionMask)
  ▼
DRPModel.train / predict
  ├── ResponseTable       flat (sample_ids, drug_ids, response, layers) for one SelectionMask
  ├── FeatureSource       uniform per-side read-only view over Dataset (Side.SAMPLE / Side.DRUG)
  ├── Featurizer          fit/transform on unique usable entity IDs ──► FeaturizerOutput
  └── PredictorInput          pairs + entity tables (FeaturizerOutput) + integer pair indices
  │
  ▼
Predictor.fit(data) / predict(data)
  ├── MatrixPredictor       data.to_feature_matrix() ──► dense (n_pairs, d_sample + d_drug)
  ├── SingleDrugPredictor   routes pairs per drug, shrinks entity tables per worker
  └── Block / PyTorch       indexes data.sample / data.drug lazily via pair indices
```

## Step 1: Reading the `.h5mu` file into a `Dataset`

`drevalpy.load` (which calls `Dataset.load` on the cached or local `.h5mu` path) reads a
single `.h5mu` file (an HDF5-backed `MuData` archive) and wraps it in
{py:class}`~drevalpy.types.data.dataset.Dataset`. Every modality in `mdata.mod` shares the
`obs` axis of samples:

- `mdata.mod["response"]` is a `(n_samples, n_drugs)` matrix whose `X` holds the default
  measure and whose `layers` hold alternative response measures and curve-fit diagnostics
  (such as `LN_IC50`, `AUC` or `regulation`). Drug metadata (`canonical_smiles`) lives in
  `response.var` and precomputed dense drug views (such as `morgan_fingerprint`) live in
  `response.varm`, because drugs are the `var` axis of `response`.
- Other modalities in `mdata.mod` hold sample views such as `gene_expression`, `proteomics` or
  `methylation`, aligned to `mdata.obs`.
- Auxiliary payloads that cannot be represented as aligned matrices (such as molecular graphs
  under `drug_graphs` or precomputed-view settings under `precomputed_views`) live in
  `mdata.uns`.

`Dataset.with_measure(name)` switches which response layer `dataset.response_matrix` returns
without copying the underlying `MuData`. See {doc}`/reference/datasets` for the full file
layout.

```{code-cell} ipython3
import drevalpy

dataset = drevalpy.load("TOYv1")
response_mod = dataset.mdata.mod["response"]

print(dataset)
print("Modalities:", list(dataset.mdata.mod.keys()))
print("response.layers (first 6):", [k for k in response_mod.layers if k is not None][:6])
print("response.var columns (first 5):", list(response_mod.var.columns)[:5])
print("mdata.obs columns:", list(dataset.mdata.obs.columns))
```

## Step 2: Selecting pairs with `SelectionMask` and `ResponseTable`

A splitter never copies the dataset. Instead, `drevalpy.split` produces a list of
{py:class}`~drevalpy.types.data.fold.Fold` objects, each holding three boolean masks (`train`,
`val`, `test`) of shape `(n_samples, n_drugs)` wrapped as
{py:class}`~drevalpy.types.data.selection_mask.SelectionMask` (plus an optional `pair_seed` that defines
a deterministic pair order for shuffling).

When `DRPModel.train` or `DRPModel.predict` runs on a scope, it extracts the selected positions
into a flat, immutable {py:class}`~drevalpy.types.data.response_table.ResponseTable`:

- `sample_ids`, `drug_ids` and `response` are 1D arrays of length `n_pairs` in the mask's pair
  order. Training passes `observed_only=True` to drop unmeasured (`NaN`) entries; prediction
  keeps every requested position.
- `layers` holds any extra response layers requested by `Predictor.response_layers` (for
  example `regulation` for `SingleDrugMorgoth`), aligned to the same pairs.
- During training, the optional response transformation is fitted on the training
  `ResponseTable` and applied to `response` (and to the early-stopping `ResponseTable` when
  present), leaving `layers` untouched. At prediction time, the inverse transformation restores
  original units.

```{code-cell} ipython3
from drevalpy.types.data.response_table import ResponseTable

fold = drevalpy.split(dataset, "LSO", n_splits=5)[0]
train_table = ResponseTable.from_dataset(
    dataset,
    fold.train,
    observed_only=True,
    layers=("regulation",),
)
val_table = ResponseTable.from_dataset(dataset, fold.val, observed_only=True)

print("SelectionMask shape:", fold.train.shape, "| selected pairs:", len(fold.train))
print("train_table length:", len(train_table), "| val_table length:", len(val_table))
print("First 3 pairs:", list(zip(train_table.sample_ids[:3], train_table.drug_ids[:3], train_table.response[:3].round(2))))
print("Extra layers:", {k: v.shape for k, v in train_table.layers.items()})
```

## Step 3: Reading raw views through `FeatureSource`

Featurizers never inspect `MuData` directly. `DRPModel` wraps the dataset in one
{py:class}`~drevalpy.types.data.feature_source.FeatureSource` per side (`Side.SAMPLE` and, when
a drug featurizer is configured, `Side.DRUG`).

`FeatureSource` hides the storage asymmetry of `MuData` behind one read-only interface:
`source.view(name, ids)` returns a float32 matrix aligned to the requested entity IDs (with
`NaN` rows for unknown entities), reading `mdata.mod` / `response.obsm` on the sample side and
`response.varm` on the drug side. Dedicated accessors expose `feature_names`, `view_settings`,
`smiles`, `graphs`, `tissue` and `uns`. Because both sides expose the same interface, shared
featurizers such as `oneHot`, `constant`, `concatFeaturizers` and `PrecomputableFeaturizer`
work identically on samples and drugs.

```{code-cell} ipython3
import numpy as np

from drevalpy.types.data.feature_source import FeatureSource, Side

sample_source = FeatureSource(dataset, Side.SAMPLE)
drug_source = FeatureSource(dataset, Side.DRUG)

sample_ids = np.unique(train_table.sample_ids)
drug_ids = np.unique(train_table.drug_ids)

expr = sample_source.view("gene_expression", sample_ids[:3])
prot = sample_source.view("proteomics", sample_ids[:3])

print("Sample gene_expression slice:", expr.shape, expr.dtype)
print("Sample proteomics slice:", prot.shape, prot.dtype)
print("Sample tissues:", sample_source.tissue(sample_ids[:3]))
print("Drug SMILES:", drug_source.smiles(drug_ids[:3]))
```

## Step 4: Filtering and featurizing unique entities into `FeaturizerOutput`

Before fitting, `models/_feature_preparation.usable_pairs` asks each featurizer's
`valid_mask(source, unique_ids)` once which entities have usable input (by default, rows that
are not all `NaN`) and drops pairs whose sample or drug is unusable.

`prepare_features` then drives each featurizer on **unique entity IDs** rather than repeated
pairs:

1. During training, `featurizer.fit` receives the unique training entity IDs (plus
   `train_pair_ids` and `es_pair_ids` for featurizers that need pair counts or an
   early-stopping holdout), and `featurizer.transform` runs once on the union of unique
   training and early-stopping entity IDs.
2. During prediction, `valid_mask` filters out unusable entities before `transform`, and
   `prepare_features` pads `NaN` (or `None` for object blocks) back in for them afterwards.
   Featurizers therefore never have to handle all-`NaN` entities themselves.

Each featurizer returns a {py:class}`~drevalpy.types.data.feature_block.FeaturizerOutput` with
two fields:

- `entity_ids`: 1D array of unique entity IDs in row order.
- `blocks`: immutable mapping from block name to
  {py:class}`~drevalpy.types.data.feature_block.FeatureBlock` (carrying `values`, `format`,
  optional `feature_names`, optional `annotations`, and `entity_aligned`).

When a consumer reads `result.matrix`, `FeaturizerOutput` returns the single entity-aligned
numeric block as-is or horizontally concatenates all entity-aligned numeric blocks in insertion
order, skipping non-numeric and global (`entity_aligned=False`) blocks. `ConcatFeaturizer`
relies on this directly: it collects its children's named numeric blocks into one
`FeaturizerOutput`, so block predictors can still look up individual child blocks by name while
matrix predictors receive the horizontal concatenation through `.matrix`.

```{code-cell} ipython3
from drevalpy.components.featurizers.drug.fingerprints import FingerprintsFeaturizer
from drevalpy.components.featurizers.sample.scaled_gene_expression import ScaledGeneExpressionFeaturizer
from drevalpy.components.featurizers.sample.tissue import TissueFeaturizer
from drevalpy.components.featurizers.shared.concat import ConcatFeaturizer

sample_featurizer = ConcatFeaturizer(
    featurizers=[
        ScaledGeneExpressionFeaturizer(gene_list="landmark_genes"),
        TissueFeaturizer(),
    ],
)
drug_featurizer = FingerprintsFeaturizer()

all_sample_ids = np.union1d(train_table.sample_ids, val_table.sample_ids)
all_drug_ids = np.union1d(train_table.drug_ids, val_table.drug_ids)

sample_featurizer.fit(sample_source, sample_ids)
sample_output = sample_featurizer.transform(sample_source, all_sample_ids)

drug_featurizer.fit(drug_source, drug_ids)
drug_output = drug_featurizer.transform(drug_source, all_drug_ids)

print("Unique samples featurized:", len(sample_output.entity_ids), "(vs", len(train_table), "training pairs)")
print(
    "Sample blocks:",
    {name: (block.values.shape, f"entity_aligned={block.entity_aligned}") for name, block in sample_output.blocks.items()},
)
print("Sample .matrix (entity-aligned numeric blocks concatenated):", sample_output.matrix.shape)
print("Drug blocks:", {name: block.values.shape for name, block in drug_output.blocks.items()})
```

## Step 5: Indexing pairs and entities in `PredictorInput`

`DRPModel` combines the `ResponseTable` with the sample and optional drug `FeaturizerOutput`
tables into a {py:class}`~drevalpy.types.data.predictor_input.PredictorInput`:

- **Pair arrays** (length `n_pairs`): `sample_ids`, `drug_ids`, `response` (`None` at
  prediction when unlabelled), and `response_layers`.
- **Entity tables**: `sample` (`FeaturizerOutput`) and `drug` (`FeaturizerOutput | None`),
  holding one row per unique entity.
- **Pair index arrays** (int64, length `n_pairs`): `sample_pair_idx` and `drug_pair_idx`,
  mapping each pair to its row in `sample` and `drug` (derived on construction via
  `FeaturizerOutput.rows_for`).
- **Training context**: optional `early_stopping` (`ResponseTable` whose entities are already
  present in `sample` and `drug`) and `training_context` (checkpoint directory and `n_jobs`).

Keeping entity tables separate from pair indices is the key memory trade-off in DrEvalPy. A
large screen such as GDSC2 has roughly 200,000 pairs across ~1,000 samples and ~250 drugs;
pre-expanding a 17,000-gene expression matrix to pair level would duplicate each sample row
hundreds of times and allocate over 10 GB per fold. Indexing keeps the entity matrix compact
and lets neural predictors gather mini-batches lazily.

```{code-cell} ipython3
from drevalpy.types.data.predictor_input import PredictorInput

train_input = PredictorInput(
    train_table,
    sample=sample_output,
    drug=drug_output,
    early_stopping=val_table,
)

print("pairs:", len(train_input))
print("sample entity table:", train_input.sample.matrix.shape, "| sample_pair_idx:", train_input.sample_pair_idx.shape)
print("drug entity table:  ", train_input.drug.matrix.shape, "| drug_pair_idx:  ", train_input.drug_pair_idx.shape)
print("First 5 sample_pair_idx:", train_input.sample_pair_idx[:5])
assert np.array_equal(
    train_input.sample.entity_ids[train_input.sample_pair_idx],
    train_input.sample_ids,
)
```

## Step 6: How predictors consume `PredictorInput`

The base {py:class}`~drevalpy.components.predictors.abstract.base.Predictor` wraps every
subclass's `_fit(data)` and `_predict(data)`. In `fit`, it validates that `data.response` is
present, drops pairs with `NaN` features or targets (via `data[mask]`), and filters
`data.early_stopping` the same way. In `predict`, it runs `_predict` only on valid pairs and
returns `NaN` for the rest.

Subclasses then consume `data` in one of three patterns:

- **Matrix predictors** (`MatrixPredictor`: scikit-learn models, XGBoost, LightGBM). Call
  `data.to_feature_matrix()` to materialize the dense `(n_pairs, d_sample + d_drug)` design
  matrix and pass `(x, y)` to `_fit_matrix` / `_predict_matrix`.
- **Single-drug predictors** (`SingleDrugPredictor`: `SingleDrugElasticNet`,
  `SingleDrugRandomForest`, `SingleDrugKNN`, `SingleDrugMorgoth`, `MOLIR`, `SuperFELTR`).
  Route pairs by `data.drug_ids`, shrink each drug's `PredictorInput` so its entity tables carry
  only the samples measured for that drug, assign each drug its own checkpoint directory, and
  run `_fit_drug(drug_id, data)` across `data.training_context.n_jobs` worker processes.
- **Block and neural predictors** (`NeuralNetwork`, `DIPK`, `PaccMann`, `PharmaFormer`,
  `DrugGNN`, `SparseGO`, `Precily`, `EnsembleMF`, `SRMF`). Read `data.sample.matrix` or named
  blocks `data.sample.blocks[name].values` / `data.drug.blocks[name].values` together with
  `data.sample_pair_idx` and `data.drug_pair_idx`. PyTorch predictors wrap these arrays in a
  lazy pair `DataLoader` so full pair-level feature matrices are never materialized in memory;
  early-stopping pairs are indexed into the same entity tables via
  `response_rows(data, data.early_stopping)`.

```{code-cell} ipython3
from drevalpy.components.predictors.sklearn_models import ElasticNetPredictor

# Pattern 1: MatrixPredictor materializes the dense pair-level design matrix.
x_train = train_input.to_feature_matrix()
print("Materialized design matrix:", x_train.shape)

# Pattern 3: Block / neural predictors gather a mini-batch lazily by index.
batch = slice(0, 4)
x_expr_batch = train_input.sample.blocks["gene_expression"].values[train_input.sample_pair_idx[batch]]
x_tissue_batch = train_input.sample.blocks["tissue"].values[train_input.sample_pair_idx[batch]]
x_drug_batch = train_input.drug.matrix[train_input.drug_pair_idx[batch]]
print("Lazy mini-batch shapes (expr, tissue, drug):", x_expr_batch.shape, x_tissue_batch.shape, x_drug_batch.shape)

# End-to-end fit and predict on the held-out validation pairs.
val_input = PredictorInput(val_table, sample=sample_output, drug=drug_output)
predictor = ElasticNetPredictor(hyperparameters={"alpha": 0.1, "l1_ratio": 0.5})
predictor.fit(train_input)
preds = predictor.predict(val_input)
print("Validation predictions shape:", preds.shape, "| first 5:", preds[:5].round(3))
```
