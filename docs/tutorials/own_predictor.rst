Build and benchmark your own predictor
======================================

.. interface: python

In this tutorial you write a sample featurizer and a predictor, register them, and evaluate
the resulting model against a naive baseline. Both components are short, complete and
registered with a decorator. You finish with a result table that puts your model next to
``NaiveMeanEffectsPredictor``, the reference every model has to beat.

You need DrEvalPy installed. :doc:`first_benchmark` explains how to read the result tables used
here. The dataset is simulated, so everything runs offline in a few seconds.

Create a dataset
----------------

.. code-block:: python

   import drevalpy

   dataset = drevalpy.synthetic()

The dataset has 60 samples, 8 drugs and a ``gene_expression`` view of about 11,600 gene features per sample.
It also holds the SMILES string of each drug, which the built-in ``fingerprints`` featurizer
needs.

Write a sample featurizer
-------------------------

A model is assembled from components. A *featurizer* turns one or more views of one side, the
samples or the drugs, into named feature blocks. A *predictor* turns feature blocks into
predictions. :doc:`/explanation/models_from_components` explains the idea.

Your featurizer standardizes the ``gene_expression`` view: it subtracts the mean and divides by
the standard deviation of each feature, both learned on the training samples only.

.. literalinclude:: /examples/toy_sample_featurizer.py
   :language: python
   :caption: toy_sample_featurizer.py

These parts matter:

``input_views``
   Names the view the featurizer reads. Registration is rejected without it.

``_fit``
   Learns one mean and one standard deviation per feature from the training samples.

``_transform_features``
   Standardizes the requested samples with those statistics and returns them as a named
   numeric block. A predictor asks for a block by its name.

``@register_sample_featurizer("toySample", ...)``
   Adds the class to the registry under the name ``toySample``, which is how a recipe refers
   to it. ``contract=FeatureFormat.NUMERIC_MATRIX`` declares that the output is a matrix of
   numbers.

``get_state`` and ``set_state``
   Carry the fitted statistics into a saved model.

:doc:`/how-to/extend/write_featurizer` covers the other hooks and the drug side.

Write a predictor
-----------------

The predictor fits a ridge regression on the feature blocks of each pair.

.. literalinclude:: /examples/toy_ridge_predictor.py
   :language: python
   :caption: toy_ridge_predictor.py

``MatrixPredictor`` hands your code one dense matrix per call, with the sample features and the
drug features of every pair side by side. You implement ``_fit_matrix`` and ``_predict_matrix``.
The rest of the class declares what DrEvalPy needs to know:

``get_hyperparameter_space``
   Declares ``alpha``, the strength of the ridge penalty, with a search range and a default.
   Tuning draws from the range; without tuning the predictor uses the default of 1.0.

``@register_predictor("toyRidge", ...)``
   Registers the predictor. Both contracts are ``NUMERIC_MATRIX``, so it accepts any
   featurizer that outputs numeric blocks on both sides.

``get_state``, ``set_state`` and ``is_fitted``
   Let DrEvalPy save, restore and check the trained estimator.

:doc:`/how-to/extend/write_predictor` shows the other predictor interfaces.

Register the components
-----------------------

A ``@register_*`` decorator runs when its module is imported. Save the two listings next to
your script, as ``toy_sample_featurizer.py`` and ``toy_ridge_predictor.py`` (the repository
keeps them in ``docs/examples/``), and import them:

..
   The snippet runner starts in an empty directory, so this block, which Sphinx does not
   render, copies the two example files there. A reader saves them as described above.

   .. code-block:: python

      import shutil
      import sys
      from pathlib import Path

      examples = Path(__file__).resolve().parents[1] / "examples"
      for name in ("toy_sample_featurizer", "toy_ridge_predictor"):
          shutil.copy(examples / f"{name}.py", f"{name}.py")
          sys.modules.pop(name, None)
      sys.path.insert(0, str(Path.cwd()))

.. code-block:: python

   import toy_ridge_predictor
   import toy_sample_featurizer

Importing is all it takes. For a directory of components that every ``drevalpy`` command should
load, see :doc:`/how-to/extend/load_extensions`.

Assemble a model
----------------

A recipe names a model by its parts: ``sample featurizer : drug featurizer : predictor``. Your
model combines ``toySample`` with the built-in ``fingerprints`` drug featurizer and your
``toyRidge`` predictor. ``ModelSpec`` gives the configuration the name its results are reported
under.

.. code-block:: python

   from drevalpy import ModelSpec
   from drevalpy.models.config import from_spec

   config = from_spec("toySample:fingerprints:toyRidge")
   toy_ridge = ModelSpec("ToyRidge", config)
   print(config.model_id)

The output:

.. code-block:: text

   toySample:fingerprints:toyRidge

``ModelSpec.from_name`` would also accept this recipe, but it would report the model under the
recipe string. Building the spec yourself lets you choose the name.
:doc:`/reference/recipes` describes the full grammar.

Benchmark it
------------

Run your model next to ``NaiveMeanEffectsPredictor`` in leave-pair-out (LPO), as in the first
tutorial. Tuning is off, so ``toyRidge`` uses its default ``alpha``.

.. code-block:: python

   baseline = ModelSpec.from_name("NaiveMeanEffectsPredictor")
   results = drevalpy.run([baseline, toy_ridge], dataset, "LPO", hyperparameter_tuning=False)
   print(results)

The output ends with the pooled metrics of both models:

.. code-block:: text

   ExperimentResult
       Dataset: SYNTHETIC
       Split mode: LPO
       Normalized by: None
       Models: 2
           NaiveMeanEffectsPredictor (5 folds): MSE=0.8399, RMSE=0.9144, MAE=0.7218, R^2=0.6851, Pearson=0.8298, Spearman=0.8409, Kendall=0.6466
           ToyRidge (5 folds): MSE=0.5572, RMSE=0.7457, MAE=0.5800, R^2=0.7912, Pearson=0.8916, Spearman=0.8989, Kendall=0.7227

Over all pairs your model leads by about 0.06 Pearson. The first tutorial showed why that number
alone is not enough. Compare the models within each drug:

.. code-block:: python

   import pandas as pd

   per_drug = pd.DataFrame(results.summary_table(group_by="drug")).T
   print(per_drug[["Pearson", "Spearman", "RMSE"]].round(3))

.. code-block:: text

                              Pearson  Spearman   RMSE
   NaiveMeanEffectsPredictor    0.386     0.372  0.895
   ToyRidge                     0.502     0.459  0.727

Within drugs your model correlates at 0.50 against 0.39 for the naive baseline. ``ToyRidge``
reads the gene expression of the sample, which tells it something about which samples respond.
The naive baseline sees only identities.

What you learned
----------------

- A component is a class with a registration decorator; a featurizer prepares the views of one
  side and a predictor turns blocks into predictions.
- A recipe such as ``toySample:fingerprints:toyRidge`` combines registered components into a
  model, and ``ModelSpec`` names it.
- A new predictor earns its place by beating ``NaiveMeanEffectsPredictor`` within drugs, not
  only over all pairs.

To tune ``alpha`` and test how much the model depends on its features, continue with
:doc:`tune_and_stress_test`. To check your components against DrEvalPy's conformance tests, see
:doc:`/how-to/extend/test_components`.
