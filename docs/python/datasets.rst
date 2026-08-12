Datasets
========

If you are reading this, we assume you are already familiar with these
concepts:

- :doc:`/concepts/datasets`
- :doc:`/concepts/evaluation`

This page explains how to load built-in and custom datasets, and how to split
them for training and evaluation.

Built-in datasets
-----------------

Built-in names are listed in the packaged registry. Use the dataset registry
to discover them and :func:`~drevalpy.data.load` to load:

.. code-block:: python

   from drevalpy.data import load

   dataset = load("GDSC1")

Built-in loaders download into the system cache directory on first use (see
:doc:`/getting_started/installation` for ``DREVALPY_CACHE_DIR``). The response
measure depends on the dataset; when CurveCurator refitting is enabled for a
workflow, measure names gain a ``_curvecurator`` suffix — see
:doc:`/concepts/datasets`.

Custom .h5mu files
------------------

Point :func:`~drevalpy.data.load` at a ``.h5mu`` file path
directly:

.. code-block:: python

   dataset = load("/path/to/MyStudy.h5mu")

Any path that is not a recognized built-in name is treated as a file path.

Splits
------

:func:`~drevalpy.run` splits the loaded dataset for you (``split_mode``
of ``LPO``, ``LCO``, ``LTO``, or ``LDO``). You can also use
:func:`~drevalpy.data.split` yourself before a custom training loop:

.. code-block:: python

   from drevalpy.data import load, split

   dataset = load("GDSC1")
   folds = split(dataset, mode="LCO", n_splits=5)

Each fold is a :class:`~drevalpy.types.SplitMasks` object containing boolean
masks for train, validation, and test sets. Pass a fold directly to
:func:`~drevalpy.single` for per-fold execution:

.. code-block:: python

   from drevalpy.models import construct_model
   from drevalpy import single

   ElasticNet = construct_model("ElasticNet")
   result = single(ElasticNet, dataset, folds[0], hyperparameter_tuning=False)

Split semantics (leakage constraints per mode) are documented in
:doc:`/concepts/evaluation`.
