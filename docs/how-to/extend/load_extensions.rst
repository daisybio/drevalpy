Load extension files and directories
====================================

.. interface: both

A ``@register_*`` decorator only runs when its module is imported. This page shows how to make
your featurizers, predictors, splitters, visualizations and model presets visible to DrEvalPy
without installing a package: put them in an **extension directory** and load it. For
components you want to install and share, see :doc:`publish_plugin`.

An extension directory holds two kinds of files:

- ``*.py`` files are imported, which triggers every ``register_*`` decorator inside them.
- ``*.yaml`` files are read as **zoo presets** (named model configurations built from registered
  components) or as dataset declarations (see :doc:`add_dataset_source`).

Create an extension directory
-----------------------------

The directory below has one Python file that registers a featurizer and a predictor, and one
YAML file that gives two model presets a name. The command line selects models by preset name
only, so a predictor needs a preset before ``drevalpy run`` can use it.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      Create ``my_extensions/components.py``:

      .. snippet: skip

      .. code-block:: python

         import numpy as np

         from drevalpy.plugin import (
             DenseViewSampleFeaturizer,
             FeatureFormat,
             FeatureFreePredictor,
             register_predictor,
             register_sample_featurizer,
         )


         @register_sample_featurizer(
             "logExpression",
             description="log1p of the gene_expression view.",
             output_format=FeatureFormat.NUMERIC_MATRIX,
         )
         class LogExpressionFeaturizer(DenseViewSampleFeaturizer):
             input_views = ("gene_expression",)

             def _compute_matrix(self, source, matrix):
                 return np.log1p(np.clip(matrix, 0.0, None))


         @register_predictor(
             "meanBaseline",
             description="Predict the mean training response.",
             sample_input_format=FeatureFormat.NUMERIC_MATRIX,
             drug_input_format=FeatureFormat.NUMERIC_MATRIX,
         )
         class MeanBaseline(FeatureFreePredictor):
             def _fit(self, batch):
                 self._mean = float(np.mean(batch.response))

             def _predict(self, batch):
                 return np.full(batch.n_pairs, self._mean)

             def get_state(self):
                 return {"mean": self._mean} if hasattr(self, "_mean") else {}

             def set_state(self, state):
                 if "mean" in state:
                     self._mean = state["mean"]

      Create ``my_extensions/zoo.yaml``:

      .. code-block:: yaml

         LogElasticNet:
           sample_featurizer: logExpression
           drug_featurizer: fingerprints
           predictor: elasticNet
         MeanBaseline:
           predictor: meanBaseline

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         from pathlib import Path

         extensions = Path("my_extensions")
         extensions.mkdir(exist_ok=True)

         (extensions / "components.py").write_text('''
         import numpy as np

         from drevalpy.plugin import (
             DenseViewSampleFeaturizer,
             FeatureFormat,
             FeatureFreePredictor,
             register_predictor,
             register_sample_featurizer,
         )


         @register_sample_featurizer(
             "logExpression",
             description="log1p of the gene_expression view.",
             output_format=FeatureFormat.NUMERIC_MATRIX,
         )
         class LogExpressionFeaturizer(DenseViewSampleFeaturizer):
             input_views = ("gene_expression",)

             def _compute_matrix(self, source, matrix):
                 return np.log1p(np.clip(matrix, 0.0, None))


         @register_predictor(
             "meanBaseline",
             description="Predict the mean training response.",
             sample_input_format=FeatureFormat.NUMERIC_MATRIX,
             drug_input_format=FeatureFormat.NUMERIC_MATRIX,
         )
         class MeanBaseline(FeatureFreePredictor):
             def _fit(self, batch):
                 self._mean = float(np.mean(batch.response))

             def _predict(self, batch):
                 return np.full(batch.n_pairs, self._mean)

             def get_state(self):
                 return {"mean": self._mean} if hasattr(self, "_mean") else {}

             def set_state(self, state):
                 if "mean" in state:
                     self._mean = state["mean"]
         ''')

         (extensions / "zoo.yaml").write_text("""
         LogElasticNet:
           sample_featurizer: logExpression
           drug_featurizer: fingerprints
           predictor: elasticNet
         MeanBaseline:
           predictor: meanBaseline
         """)

The loader imports ``*.py`` files first (sorted by name, skipping ``__init__.py``) and reads
``*.yaml`` files afterwards, so a preset can name a component defined in the same directory. It
does not descend into subdirectories.

In a zoo file, each top-level key is a preset name and its value is the model configuration:
``sample_featurizer``, ``drug_featurizer`` and ``predictor``. A feature-free predictor takes
only ``predictor``, and the configuration check rejects featurizers for it. A file may instead
hold a single preset: put ``predictor`` (and the featurizers) at the top level and the preset
takes its name from a ``name:`` key, or from the file name when there is none. A preset name
that matches a built-in preset raises ``ValueError``. The configuration keys and recipe strings
are listed in :doc:`/reference/extension_files` and :doc:`/reference/recipes`.

Load the directory
------------------

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      ``--extensions-dir`` (``-e``) is a global option, so it goes **before** the subcommand.
      Repeat it to load several directories.

      .. code-block:: bash

         drevalpy -e my_extensions list predictors meanBaseline
         drevalpy -e my_extensions list zoo-presets

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         from drevalpy.registry import load_extensions, predictor

         load_extensions(directories=["my_extensions"])
         assert "meanBaseline" in predictor.list()

Loading the same file again is a no-op while its registrations remain. If a file fails part-way,
the registries are rolled back, so a module that registers two components and then raises
leaves neither behind. A directory that does not exist raises ``FileNotFoundError``.

Run a model from the extension
------------------------------

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      ``drevalpy run`` takes preset names, so use the name from ``zoo.yaml``. ``--no-hpo``
      skips hyperparameter search to keep the run short.

      .. code-block:: bash

         drevalpy data synthetic --output toy.h5mu
         drevalpy -e my_extensions run MeanBaseline --dataset toy.h5mu --split-mode LSO --no-hpo -o results

   .. tab-item:: Python
      :sync: python

      A ``ModelSpec`` pairs a name with a ``ModelConfig``. ``ModelSpec.from_name`` resolves a
      preset loaded from a zoo file; to build a configuration yourself, use
      ``ModelSpec(name, from_spec(recipe))``.

      .. code-block:: python

         import drevalpy
         from drevalpy import ModelSpec

         dataset = drevalpy.load("toy.h5mu")
         spec = ModelSpec.from_name("MeanBaseline")
         results = drevalpy.run([spec], dataset, "LSO", hyperparameter_tuning=False)

Load a directory automatically
------------------------------

Set ``DREVALPY_EXTENSIONS_DIR`` to a directory (or several, separated by ``os.pathsep``: ``:``
on Linux and macOS, ``;`` on Windows). The variable is read every time DrEvalPy loads
extensions, and its directories are loaded **before** any ``-e`` directory.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      The CLI reads the variable before it handles ``-e``, so the two combine.

      .. snippet: skip

      .. code-block:: bash

         export DREVALPY_EXTENSIONS_DIR=my_extensions
         drevalpy run MeanBaseline --dataset toy.h5mu --split-mode LSO --no-hpo

   .. tab-item:: Python
      :sync: python

      ``load_extensions()`` with no ``directories`` argument reads the variable. Passing any
      list, even an empty one, means the variable is ignored. The example sets it for one call
      only.

      .. code-block:: python

         import os
         from unittest import mock

         from drevalpy.registry import load_extensions

         with mock.patch.dict(os.environ, {"DREVALPY_EXTENSIONS_DIR": "my_extensions"}):
             load_extensions()

Load individual files, modules and zoo files
--------------------------------------------

The command line loads whole directories. From Python, ``load_extensions`` also takes
single items:

- ``modules``: dotted names of importable modules, with the same effect as ``import``.
- ``files``: paths of single ``.py`` files, executed as temporary modules.
- ``directories``: the extension directories described above.
- ``zoo_files``: YAML presets that map a name to an already-registered stack. These are not
  Python classes and not experiment hyperparameter files.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      Put the files in a directory and pass the directory to ``-e``. A package that is installed
      in the environment needs no flag when it is a :doc:`plugin <publish_plugin>`.

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         from drevalpy.registry import load_extensions

         load_extensions(
             files=["my_extensions/components.py"],
             zoo_files=["my_extensions/zoo.yaml"],
             directories=[],
         )

If your components are an importable package, a normal ``import my_components`` registers them
too.

Save and load a model with custom components
--------------------------------------------

A checkpoint is a ZIP archive that stores the resolved ``ModelConfig`` (component **names**)
and the fitted state, not the Python classes. On load, DrEvalPy looks the names up in the
registries again. A custom component that is not registered in the loading process makes
``DRPModel.load`` fail, so load the same extensions first. Built-in components need no extra
step.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      The command line does not write or read single-model checkpoints. Use the Python tab for
      this task.

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         from drevalpy import DRPModel

         folds = drevalpy.split(dataset, mode="LSO", n_splits=3)
         model = DRPModel(ModelSpec.from_name("MeanBaseline"))
         model.train(dataset, folds[0].train)
         model.save("mean_baseline.zip")

         restored = DRPModel.load("mean_baseline.zip")
         predictions = restored.predict(dataset, folds[0].test)

.. warning::

   Checkpoints are unpickled when loaded, which can execute arbitrary code. Load only
   checkpoints you trust.

The general save and load workflow is in :doc:`/how-to/train_save_load`.
