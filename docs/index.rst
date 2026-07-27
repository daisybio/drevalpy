.. include:: readme.rst

.. role:: small

.. role:: smaller

DrEvalPy documentation is organized into shared concepts plus separate CLI and
Python tracks. Start with installation, then choose the interface you want to use.

- :doc:`cli/quickstart` — run a first experiment with ``drevalpy`` and open the HTML report.
- :doc:`python/quickstart` — load data, construct a model, and call ``drug_response_experiment``.

For demanding or highly reproducible runs, we also provide the Nextflow pipeline
`nf-core/drugresponseeval <https://nf-co.re/drugresponseeval/dev/>`_.

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   getting_started/installation

.. toctree::
   :maxdepth: 1
   :caption: Concepts

   concepts/datasets
   concepts/evaluation
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
