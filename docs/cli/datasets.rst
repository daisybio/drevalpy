Datasets
========

If you are reading this, we assume you are already familiar with this
concept:

- :doc:`/concepts/datasets`

This page covers dataset loading and splitting from the CLI.

Loading datasets
----------------

``drevalpy data load`` downloads a built-in dataset (or resolves a custom
path) and writes it as a ``.h5mu`` file:

.. code-block:: bash

   drevalpy data load GDSC1 data/GDSC1.h5mu

The first positional argument is the dataset name (or path to an existing
``.h5mu`` file). The second is the output file path. Built-in datasets are
downloaded into the system cache on first use (see
:doc:`/getting_started/installation` for ``DREVALPY_CACHE_DIR``).

Built-in dataset names are: ``BeatAML2``, ``CCLE``, ``CTRPv1``, ``CTRPv2``,
``GDSC1``, ``GDSC2``, ``PDX_Bruna``. Sizes and provenance for each are in
:doc:`/concepts/datasets`.

Splitting datasets
------------------

``drevalpy data split`` generates cross-validation fold files from a dataset:

.. code-block:: bash

   drevalpy data split GDSC1 splits/ --mode LCO --n-splits 5

Options:

- ``--mode`` / ``-m`` — split mode: ``LPO``, ``LCO``, ``LDO``, or ``LTO``
  (default ``LPO``)
- ``--n-splits`` / ``-n`` — number of CV folds (default ``5``)
- ``--validation-ratio`` — fraction of training data for validation (default
  ``0.1``)
- ``--random-state`` — random seed (default ``42``)

Each fold is written as a ``.npz`` file (``fold_0.npz``, ``fold_1.npz``, …)
in the output directory. These files can be passed to ``drevalpy single`` for
per-fold execution.

Split semantics (leakage constraints per mode) are documented in
:doc:`/concepts/evaluation`.

Using ``drevalpy run`` with datasets
-------------------------------------

The ``drevalpy run`` command handles loading and splitting automatically. Pass
``--dataset`` with a built-in name or file path:

.. code-block:: bash

   drevalpy run ElasticNet --dataset GDSC1 --split-mode LCO

For more on the ``run`` command, see :doc:`experiments`.
