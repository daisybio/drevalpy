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

``curate`` reads a long table with one row per measurement and the columns:

``drug``
   The drug identifier.

``sample``
   The sample identifier.

``concentration``
   The concentration of the drug, in µM. The fitted pEC50 follows from it.

``intensity``
   The measured response at that concentration. The example uses the fraction of viable cells,
   which falls as the dose rises.

``replicate``
   Optional. Include it if you measured replicates.

The identifiers are labels. They become the sample and drug IDs of the dataset, so each sample
and each drug needs one unique label.

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

         drevalpy curate screen.csv screen.h5ad

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
layers raises ``KeyError``.

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

         md.set_options(pull_on_update=False)
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

         md.set_options(pull_on_update=False)
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

Share the dataset by name
-------------------------

A path works on your machine. To load the dataset by name, or to share it through a bucket or
a file server, register it; see :doc:`/how-to/extend/add_dataset_source`.
