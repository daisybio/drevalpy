.. include:: readme.rst

.. role:: small

.. role:: smaller

DrEvalPy documentation is organized into shared **concepts** plus separate **CLI**
and **Python** tracks. Both interfaces run the same evaluation pipeline; only
the wiring differs.

Suggested path
--------------

1. :doc:`getting_started/installation` — Python version, pip/Conda/Docker, and
   verify with ``drevalpy --help``.
2. Pick one getting-started track:

   - :doc:`cli/quickstart` — smallest ``drevalpy`` command, where results land,
     and ``drevalpy report``.
   - :doc:`python/quickstart` — ``load_dataset``, ``construct_model``, and
     ``drug_response_experiment``.

3. Read :doc:`concepts/datasets` and :doc:`concepts/evaluation` when you change
   datasets, split modes (LPO/LCO/LTO/LDO), or metrics.
4. :doc:`concepts/from_components_to_models` and :doc:`concepts/model_zoo` —
   how zoo preset names map to featurizer + predictor stacks.
5. Go deeper on your track: :doc:`cli/experiment` or :doc:`python/experiments`,
   plus :doc:`python/models` for composition in code.

For demanding or highly reproducible runs, use the Nextflow pipeline
`nf-core/drugresponseeval <https://nf-co.re/drugresponseeval/dev/>`_ (see
:doc:`cli/pipeline_commands`).

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   getting_started/installation

.. toctree::
   :maxdepth: 1
   :caption: Concepts

   concepts/datasets
   concepts/evaluation
   concepts/from_components_to_models
   concepts/model_zoo

.. toctree::
   :maxdepth: 1
   :caption: CLI guide

   cli/quickstart
   cli/experiment
   cli/hyperparameter_tuning
   cli/reporting
   cli/wandb
   cli/custom_splits
   cli/pipeline_commands
   cli/reference

.. toctree::
   :maxdepth: 2
   :caption: Python guide

   python/quickstart
   python/datasets
   python/experiments
   python/models
   python/model_inputs
   python/hyperparameter_tuning
   python/visualization
   python/persistence
   python/architecture
   python/component_catalog
   python/custom_models
   python/api/index

.. toctree::
   :maxdepth: 1
   :caption: Project

   project/contributing
   project/citing
   project/news
   project/contributors
   project/memes

.. _github: https://github.com/daisybio/drevalpy
