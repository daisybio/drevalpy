Create a dataset from your own screen
=====================================

.. interface: both

DrEvalPy loads datasets from ``.h5mu`` files (see :doc:`/reference/datasets`). It has no loader
for CSV or other formats. This page shows how to turn your own dose-response measurements into
such a file: fit the dose-response curves, add the annotations the benchmark needs, and save the
result. The file then works as ``--dataset path.h5mu`` and as ``drevalpy.load("path.h5mu")``.

The example is a tiny simulated screen of 6 samples and 3 drugs, so it runs in seconds. A real
screen has the same shape and takes longer: fitting costs a fraction of a second per curve.

Prepare the dose-response table
-------------------------------

``curate`` reads a long table with one row per well. Four columns are required:

``sample``
   The sample identifier.

``drug``
   The drug identifier.

``concentration``
   The concentration of the drug, in µM. The fitted pEC50 follows from it.

``intensity``
   The measured response at that concentration. The example uses the fraction of viable cells,
   which falls as the dose rises.

The identifiers are labels. They become the sample and drug IDs of the dataset, so each sample
and each drug needs one unique label.

If you measured replicates, three optional columns describe them:

``experiment``
   One independently seeded culture of the sample. Different experiments of a pair are
   biological replicates.

``replicate``
   One dose series within its experiment, such as one plate. The series of one experiment are
   technical replicates of each other. The labels only need to differ within an experiment:
   ``P1`` of experiment ``E1`` and ``P1`` of ``E2`` are two series.

``position``
   The order of the wells. If every dose of a series was measured k ≥ 2 times, ``curate`` sorts
   the wells by position and splits the series into k series. Without ``position``, repeated
   doses are averaged.

For example, these wells hold two experiments of the pair ``S0`` × ``drugA``, the first with
two plates (only the lowest doses are shown):

.. list-table::
   :header-rows: 1

   * - ``sample``
     - ``drug``
     - ``experiment``
     - ``replicate``
     - ``concentration``
     - ``intensity``
   * - S0
     - drugA
     - E1
     - P1
     - 0.001
     - 0.99
   * - S0
     - drugA
     - E1
     - P1
     - 0.01
     - 0.98
   * - S0
     - drugA
     - E1
     - P2
     - 0.001
     - 1.01
   * - S0
     - drugA
     - E2
     - P1
     - 0.001
     - 0.97

The plates ``P1`` and ``P2`` of ``E1`` are technical replicates, and plate ``P1`` of ``E2`` is a
biological replicate of both. The
pooled curve of a pair is fitted to every well of every series. Without ``replicate`` each pair
is one series, and without ``experiment`` all series of a pair belong to one experiment.

A series needs at least ``min_doses`` distinct doses (default 4, ``--min-doses`` on the command
line). A shorter series is left out of the fit, and a pair without any series that long is
missing from the result.

``curate`` raises ``InvalidInputError`` for any other column. To keep a column, such as a plate
barcode or a seeding date, list it with ``metadata=("Seeding_Date",)`` (``--metadata
Seeding_Date``, repeatable); it appears, lowercased, in the replicate-fits table described
below. Otherwise drop it before the fit.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      The command reads a CSV or Parquet file (``.csv``, ``.parquet`` or ``.pq``). The block below
      writes a small example file; skip it if you have your own.

      .. code-block:: python

         import numpy as np
         import pandas as pd

         rng = np.random.default_rng(0)
         concentrations = 10.0 ** np.arange(-3, 4)
         ec50 = {"drugA": 0.5, "drugB": 5.0, "drugC": 50.0}
         rows = []
         for sample in range(6):
             shift = 10 ** rng.normal(0, 0.4)
             for drug, base in ec50.items():
                 for conc in concentrations:
                     viability = 1 / (1 + (conc / (base * shift)) ** 1.5) + rng.normal(0, 0.03)
                     rows.append((drug, f"S{sample}", conc, viability))
         table = pd.DataFrame(rows, columns=["drug", "sample", "concentration", "intensity"])
         table.to_csv("screen.csv", index=False)

   .. tab-item:: Python
      :sync: python

      ``curate`` takes a ``pandas.DataFrame``. The block below builds a small example; replace it
      with ``pandas.read_csv`` or ``pandas.read_parquet`` for your own data.

      .. code-block:: python

         import numpy as np
         import pandas as pd

         rng = np.random.default_rng(0)
         concentrations = 10.0 ** np.arange(-3, 4)
         ec50 = {"drugA": 0.5, "drugB": 5.0, "drugC": 50.0}
         rows = []
         for sample in range(6):
             shift = 10 ** rng.normal(0, 0.4)
             for drug, base in ec50.items():
                 for conc in concentrations:
                     viability = 1 / (1 + (conc / (base * shift)) ** 1.5) + rng.normal(0, 0.03)
                     rows.append((drug, f"S{sample}", conc, viability))
         table = pd.DataFrame(rows, columns=["drug", "sample", "concentration", "intensity"])

The table has 6 samples × 3 drugs × 7 concentrations = 126 rows, which make 18 curves.

Fit the curves
--------------

``curate`` fits a dose-response curve to every sample-drug pair. It returns, for each pair, the
pEC50 and the other curve summaries and quality measures, as one table of samples by drugs.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         drevalpy curate run screen.csv screen.h5ad

      The command logs the progress of the fit and writes the result to ``screen.h5ad``, an
      AnnData file. ``--cores`` sets the number of worker processes (default 4), and
      ``--normalize`` applies a median normalization before the fit.

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         from drevalpy.curation import curate

         response = curate(table)
         print(response)

      The call logs the progress of the fit and returns an ``AnnData`` with the samples as rows
      and the drugs as columns. ``max_workers`` sets the number of worker processes (default
      4) and ``normalize=True`` applies a median normalization before the fit.

      .. code-block:: text

         AnnData object with n_obs × n_vars = 6 × 3
             layers: 'EC50', 'IC50', 'LN_IC50', 'AUC', 'fold_change', 'slope', 'front', 'back', 'R2', 'RMSE', 'p_value', 'log_p_value', 'f_value', 'f_value_sam', 'relevance_score', 'signal_quality', 'regulation', 'pec50_error', 'slope_error', 'front_error', 'back_error', None (.X)

``X`` holds the pEC50 of each pair, which is the response the models predict. The layers hold the
other summaries (``EC50``, ``IC50``, ``AUC``) and the curve-quality measures. The splitters read
the quality layers ``relevance_score`` and ``fold_change``: they drop the pairs whose curve fits
poorly or barely moves (see :doc:`/reference/datasets`). Splitting a dataset without these two
layers raises ``KeyError``. If your dataset has only a pEC50 matrix, switch the rule off with
``quality=CurveQuality.off()`` in ``drevalpy.split`` and ``drevalpy.run``, or ``--no-quality-filter``
on the command line; every pair with a finite response is then kept.

A curve that CurveCurator cannot fit leaves its pair out of the result instead of stopping the
run. To fit a large screen as parallel jobs, for example on a cluster, see :doc:`curate_at_scale`.

Fit each replicate on its own
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The pooled curve hides how well the replicates of a pair agree. With ``replicate_fits=True``
(``--replicate-fits``), ``curate`` also fits every series on its own and stores one row per
series in ``uns["replicate_fits"]``. The example below simulates 2 samples and 2 drugs, each
pair in two experiments of two plates.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      Write the example table:

      .. code-block:: python

         rng = np.random.default_rng(1)
         seeded = {"E1": "2026-03-02", "E2": "2026-03-09"}
         rows = []
         for sample, shift in [("S0", 1.0), ("S1", 8.0)]:
             for drug, base in [("drugA", 0.5), ("drugB", 5.0)]:
                 for experiment in ["E1", "E2"]:
                     culture = shift * 10 ** rng.normal(0, 0.15)
                     for plate in ["P1", "P2"]:
                         for conc in concentrations:
                             viability = 1 / (1 + (conc / (base * culture)) ** 1.5) + rng.normal(0, 0.03)
                             rows.append((sample, drug, experiment, plate, seeded[experiment], conc, viability))
         columns = ["sample", "drug", "experiment", "replicate", "Seeding_Date", "concentration", "intensity"]
         pd.DataFrame(rows, columns=columns).to_csv("replicated.csv", index=False)

      Fit it, keeping the seeding date of each series:

      .. code-block:: bash

         drevalpy curate run replicated.csv replicated.h5ad --replicate-fits --metadata Seeding_Date

      Read the table in Python:

      .. code-block:: python

         import anndata as ad

         fits = ad.read_h5ad("replicated.h5ad").uns["replicate_fits"]
         pair = fits[(fits["sample_id"] == "S0") & (fits["pubchem_id"] == "drugA")].sort_values("replicate")
         print(pair[["experiment", "series_key", "biological_replicate", "technical_replicate",
                     "source", "status", "pEC50", "seeding_date"]].to_string(index=False))

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         rng = np.random.default_rng(1)
         seeded = {"E1": "2026-03-02", "E2": "2026-03-09"}
         rows = []
         for sample, shift in [("S0", 1.0), ("S1", 8.0)]:
             for drug, base in [("drugA", 0.5), ("drugB", 5.0)]:
                 for experiment in ["E1", "E2"]:
                     culture = shift * 10 ** rng.normal(0, 0.15)
                     for plate in ["P1", "P2"]:
                         for conc in concentrations:
                             viability = 1 / (1 + (conc / (base * culture)) ** 1.5) + rng.normal(0, 0.03)
                             rows.append((sample, drug, experiment, plate, seeded[experiment], conc, viability))
         columns = ["sample", "drug", "experiment", "replicate", "Seeding_Date", "concentration", "intensity"]
         replicated = pd.DataFrame(rows, columns=columns)

         curves = curate(replicated, replicate_fits=True, metadata=["Seeding_Date"])
         fits = curves.uns["replicate_fits"]
         pair = fits[(fits["sample_id"] == "S0") & (fits["pubchem_id"] == "drugA")].sort_values("replicate")
         print(pair[["experiment", "series_key", "biological_replicate", "technical_replicate",
                     "source", "status", "pEC50", "seeding_date"]].to_string(index=False))

The rows of the pair ``S0`` × ``drugA``:

.. code-block:: text

   experiment series_key  biological_replicate  technical_replicate    source status    pEC50 seeding_date
           E1         P1                     0                    0 replicate fitted 6.274473   2026-03-02
           E1         P2                     0                    1 replicate fitted 6.235189   2026-03-02
           E2         P1                     1                    0 replicate fitted 6.247917   2026-03-09
           E2         P2                     1                    1 replicate fitted 6.261418   2026-03-09

Keep the rows with ``status == "fitted"``: a series that was too short or that CurveCurator could
not fit has no metrics. A pair with a single series that can be fitted carries its pooled curve,
marked ``source == "pooled"``. Every column is listed under :ref:`reference/datasets:File layout`.

To compare technical with biological variability, take the spread of pEC50 within each
experiment and the spread of the experiment means within each pair:

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      Continue in the same Python session:

      .. code-block:: python

         fitted = fits[fits["status"] == "fitted"]
         per_experiment = fitted.groupby(["sample_id", "pubchem_id", "experiment"], observed=True)["pEC50"]
         technical = per_experiment.std().mean()
         biological = per_experiment.mean().groupby(level=["sample_id", "pubchem_id"], observed=True).std().mean()
         print(f"technical SD: {technical:.3f}, biological SD: {biological:.3f}")

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         fitted = fits[fits["status"] == "fitted"]
         per_experiment = fitted.groupby(["sample_id", "pubchem_id", "experiment"], observed=True)["pEC50"]
         technical = per_experiment.std().mean()
         biological = per_experiment.mean().groupby(level=["sample_id", "pubchem_id"], observed=True).std().mean()
         print(f"technical SD: {technical:.3f}, biological SD: {biological:.3f}")

.. code-block:: text

   technical SD: 0.025, biological SD: 0.084

In this simulation each culture shifts the EC50, so the experiments of a pair differ more than
the plates of one experiment.

Add the annotations and save the dataset
----------------------------------------

The fit knows nothing about your samples and drugs beyond their labels. A dataset needs:

- ``obs["tissue"]``: the tissue of each sample. The ``LTO`` split mode and the tissue-based
  baselines read it.
- ``var["drug_name"]`` and ``var["canonical_smiles"]``: the name and the SMILES of each drug. The
  ``fingerprints`` featurizer, which most zoo presets use for drugs, computes from the SMILES.

Put the response table into a ``MuData`` as the modality ``response``, copy the sample annotations to
its global ``obs``, and save it as a ``Dataset``. No command does these steps; use Python. This
works the same after the CLI and the Python route above.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      Read ``screen.h5ad`` in Python:

      .. code-block:: python

         import anndata as ad
         import mudata as md

         import drevalpy
         from drevalpy.types import Dataset

         response = ad.read_h5ad("screen.h5ad")
         response.obs["tissue"] = ["Lung", "Blood", "Skin"] * 2
         response.var["drug_name"] = response.var_names
         response.var["canonical_smiles"] = [
             "CC(=O)OC1=CC=CC=C1C(=O)O",
             "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
             "CC(=O)NC1=CC=C(C=C1)O",
         ]

         with md.set_options(pull_on_update=False):
             mdata = md.MuData({"response": response})
         mdata.obs = response.obs.copy()
         Dataset(mdata, name="MyScreen").save("my_screen.h5mu")

   .. tab-item:: Python
      :sync: python

      Continue with the ``response`` from ``curate``:

      .. code-block:: python

         import mudata as md

         from drevalpy.types import Dataset

         response.obs["tissue"] = ["Lung", "Blood", "Skin"] * 2
         response.var["drug_name"] = response.var_names
         response.var["canonical_smiles"] = [
             "CC(=O)OC1=CC=CC=C1C(=O)O",
             "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
             "CC(=O)NC1=CC=C(C=C1)O",
         ]

         with md.set_options(pull_on_update=False):
             mdata = md.MuData({"response": response})
         mdata.obs = response.obs.copy()
         Dataset(mdata, name="MyScreen").save("my_screen.h5mu")

Each list holds one value per row of the AnnData, in the order of ``response.obs_names`` and
``response.var_names``. In your own data, fill the columns from your metadata tables. The
``set_options`` call makes ``MuData`` keep the global annotations under their plain names. The
name you give the ``Dataset`` is what results and reports show as the dataset name. The ``.h5mu``
file layout is in :doc:`/reference/datasets`.

Use the dataset
---------------

Load the file by path to check it, and pass the path wherever a command asks for a dataset.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         drevalpy run NaiveDrugMeanPredictor NaiveMeanEffectsPredictor \
             --dataset my_screen.h5mu --split-mode LPO --no-hpo

      The command ends with:

      .. code-block:: text

         ExperimentResult
             Dataset: MyScreen
             Split mode: LPO
             Normalized by: None
             Models: 2
                 NaiveDrugMeanPredictor (5 folds): MSE=0.1570, RMSE=0.3638, MAE=0.3250, R^2=0.5778, Pearson=0.9383, Spearman=0.7967, Kendall=0.7173
                 NaiveMeanEffectsPredictor (5 folds): MSE=0.3129, RMSE=0.5084, MAE=0.4556, R^2=0.2382, Pearson=0.7569, Spearman=0.8400, Kendall=0.8000

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         import drevalpy
         from drevalpy import ModelSpec

         dataset = drevalpy.load("my_screen.h5mu")
         print(dataset)
         models = [ModelSpec.from_name(name) for name in ["NaiveDrugMeanPredictor", "NaiveMeanEffectsPredictor"]]
         results = drevalpy.run(models, dataset, "LPO", hyperparameter_tuning=False)
         print(results)

      .. code-block:: text

         Dataset
             Name: MyScreen
             Samples: 6
             Drugs: 3
             Measured pairs: 18
             Randomization: None
             Modalities:
         ExperimentResult
             Dataset: MyScreen
             Split mode: LPO
             Normalized by: None
             Models: 2
                 NaiveDrugMeanPredictor (5 folds): MSE=0.1570, RMSE=0.3638, MAE=0.3250, R^2=0.5778, Pearson=0.9383, Spearman=0.7967, Kendall=0.7173
                 NaiveMeanEffectsPredictor (5 folds): MSE=0.3129, RMSE=0.5084, MAE=0.4556, R^2=0.2382, Pearson=0.7569, Spearman=0.8400, Kendall=0.8000

``Modalities:`` is empty because the dataset has no sample features yet. The baselines need none.
With 18 pairs the scores are noisy: use a screen of realistic size for results you report.
Models that use features need them, as the next section shows.

Add sample features
-------------------

A model that uses features of the samples, such as gene expression, reads them from a further
modality of the ``MuData``. Add one AnnData per kind of feature, with the same sample labels as
the response, and name the modality after the feature: ``gene_expression``, ``methylation``,
``mutations``, ``proteomics`` or ``copy_number_variation_gistic``. :doc:`/reference/recipes` and
:doc:`/how-to/choose_models` show which featurizer reads which view.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      No command adds features. Run the code of the Python tab in the same session as the Python
      block above: it continues with the ``response`` and ``mdata`` defined there.

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         import anndata as ad
         import numpy as np

         values = np.random.default_rng(0).normal(size=(6, 4)).astype("float32")
         expression = ad.AnnData(
             X=values,
             obs=response.obs[[]].copy(),
             var=pd.DataFrame(index=[f"GENE{i}" for i in range(4)]),
         )
         mdata = md.MuData({"response": response, "gene_expression": expression})
         mdata.obs = response.obs.copy()
         Dataset(mdata, name="MyScreen").save("my_screen_expression.h5mu")
         print(drevalpy.load("my_screen_expression.h5mu"))

      The values are random here; use your measurements, with one row per sample.

      .. code-block:: text

         Dataset
             Name: MyScreen
             Samples: 6
             Drugs: 3
             Measured pairs: 18
             Randomization: None
             Modalities:
                 gene_expression: 6 × 4

Match the gene lists of the presets
-----------------------------------

Most zoo presets keep a shipped gene list instead of the whole view, so that a model reads the
same genes on every dataset. The lists are those of drevalpy 1.6.x:

- ``landmark_genes`` (867 genes) for the sklearn models, ``SimpleNeuralNetwork``,
  ``DrugGNN``, ``PharmaFormer``, ``DIPK``, ``EnsembleMF`` and ``SingleDrugKNN``;
- ``gene_list_paccmann_network_prop`` for ``PaccMann`` and ``tcga_autoencoder_genes_available``
  for ``RandomForestAEMolGNet``;
- ``drug_target_genes_all_drugs`` in the ``mutations`` view for the ``MultiView*`` models;
- the ``*_intersection`` lists of each view for ``MOLIR`` and ``SuperFELTR``.

If your view lacks any gene of the list, the model stops instead of training on a smaller
selection, and the error names the list, the view and the number of missing genes. The
``gene_expression`` view of the example holds four made-up genes, so ``Ridge`` fails:

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. snippet: skip

      .. code-block:: bash

         drevalpy run Ridge --dataset my_screen_expression.h5mu --split-mode LPO --no-hpo

      .. code-block:: text

         Error: view 'gene_expression' lacks 867 of 867 genes in 'landmark_genes' (missing: INSIG1, FOXO3, CDH3, ORC1, ITGAE, ...). Pass gene_list=None to use every gene of the view.

      To keep all genes of the view, compose the model from a recipe: a featurizer in a recipe
      has ``gene_list=None``, so it reads the whole view.

      .. code-block:: bash

         drevalpy run "scaledGeneExpression:fingerprints:ridge" \
             --dataset my_screen_expression.h5mu --split-mode LPO --no-hpo

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         from drevalpy import ModelSpec
         from drevalpy.errors import InvalidInputError
         from drevalpy.models.config import from_recipe

         dataset = drevalpy.load("my_screen_expression.h5mu")
         try:
             drevalpy.run([ModelSpec.from_name("Ridge")], dataset, "LPO", hyperparameter_tuning=False, on_error="raise")
         except InvalidInputError as error:
             print(error)

         ridge = ModelSpec("RawRidge", from_recipe("scaledGeneExpression:fingerprints:ridge"))
         print(drevalpy.run([ridge], dataset, "LPO", hyperparameter_tuning=False))

      The first call prints the error:

      .. code-block:: text

         view 'gene_expression' lacks 867 of 867 genes in 'landmark_genes' (missing: INSIG1, FOXO3, CDH3, ORC1, ITGAE, ...). Pass gene_list=None to use every gene of the view.

      A recipe passes no gene list, so the second call trains on the four genes.

There are two ways to proceed. Add the missing genes to the view, or opt out of the list. A
featurizer that takes ``gene_list`` keeps the whole view when it is ``None`` (``landmarkGenes``
takes ``gene_list_stem`` instead). A recipe string sets no list, as above, and a zoo file in an
extension directory sets the option by name (``gene_list: null``); see :doc:`/reference/recipes`
and :doc:`/reference/datasets`. Opting out changes what the model reads, so its scores are no
longer those of the preset.

``PDX_Bruna`` and ``BeatAML2`` lack some genes of several lists, including ``landmark_genes``, so
the presets raise this error on them unless you opt out; the gene-list section of
:doc:`/reference/datasets` has the details. The built-in screens and ``TOYv1`` hold the genes of
the lists their presets use. The naive baselines need no list.

Share the dataset by name
-------------------------

A path works on your machine. To load the dataset by name, or to share it through a bucket or
a file server, register it; see :doc:`/how-to/extend/add_dataset_source`.
