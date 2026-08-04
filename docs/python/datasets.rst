Datasets
========

If you are reading this, we assume you are already familiar with these
concepts:

- :doc:`/concepts/datasets`
- :doc:`/concepts/evaluation`

This page explains how to load built-in and custom datasets, and how to split them for training and evaluation.

Built-in names
--------------

``AVAILABLE_DATASETS`` maps each built-in name to its loader:

.. code-block:: python

   from drevalpy.datasets import AVAILABLE_DATASETS
   from drevalpy.datasets.loader import load_dataset

   print(sorted(AVAILABLE_DATASETS))
   # CCLE, CTRPv1, CTRPv2, BeatAML2, GDSC1, GDSC2, PDX_Bruna, TOYv1, TOYv2

   response = load_dataset("TOYv1", path_data="data", measure="LN_IC50")

Built-in loaders download into ``path_data`` on first use. Pass ``measure`` to
select the response column (for example ``LN_IC50``, ``AUC``, ``response``).
When CurveCurator refitting is enabled for a workflow, measure names gain a
``_curvecurator`` suffix — see :doc:`/concepts/datasets`.

Custom raw and prefit tables
----------------------------

An unknown ``dataset_name`` is treated as a custom load path under
``{path_data}/{dataset_name}/``.

**Prefit response CSV** at
``{path_data}/{dataset_name}/{dataset_name}.csv`` needs at least
``cell_line_id``, ``drug_id``, and a measure column. Leave-Tissue-Out also
needs ``tissue`` (pass ``tissue_column`` when the column name differs):

.. code-block:: python

   response = load_dataset(
       "MyStudy",
       path_data="data",
       measure="LN_IC50",
       tissue_column="tissue",
   )

**Raw viability** (long format with ``dose``, ``response``, ``sample``,
``drug``, optional ``replicate``; doses in µM) lives at
``{path_data}/{dataset_name}/{dataset_name}_raw.csv``. Set
``curve_curator=True`` so CurveCurator fits curves and writes the prefit CSV:

.. code-block:: python

   response = load_dataset(
       "MyRawStudy",
       path_data="data",
       measure="response",
       curve_curator=True,
       cores=4,
   )

Splits
------

:func:`~drevalpy.experiment.drug_response_experiment` splits the loaded
``DrugResponseDataset`` for you (``test_mode`` of ``LPO``, ``LCO``, ``LTO``,
or ``LDO``). You can also call ``split_dataset`` yourself before a custom
training loop:

.. code-block:: python

   response = load_dataset("TOYv1", path_data="data")
   response.split_dataset(n_cv_splits=5, mode="LCO")

For external split scripts, pass ``custom_splitter`` to
``drug_response_experiment`` — see :doc:`experiments`. Split semantics are
documented in :doc:`/concepts/evaluation`.
