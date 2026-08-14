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
:doc:`/getting_started/installation` for ``DREVALPY_CACHE_DIR``). ``X`` holds
``pEC50``; other measures are ``response.layers`` entries — see
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

Curve quality
~~~~~~~~~~~~~

Every built-in splitter first drops the pairs whose fitted dose-response curve
fails ``relevance_score >= -log10(0.05)`` and ``abs(fold_change) >= 0.45``, so a
fold never contains a curve the fit itself calls meaningless. There is no knob
for this on :func:`~drevalpy.data.split`: a comparison between models is only
meaningful over the same pairs.

To see how many pairs a mode dropped, compare the fold masks against the
measured pairs:

.. code-block:: python

   import numpy as np

   measured = ~np.isnan(dataset.response_matrix)
   used = np.zeros_like(measured)
   for fold in folds:
       used |= fold.train.mask | fold.test.mask | fold.val.mask
   dropped = int((measured & ~used).sum())

See :ref:`curve-quality` for what the metrics mean, and
:func:`~drevalpy.plugin.curve_quality_mask` to filter on the others from a
custom splitter.

Split semantics (leakage constraints per mode) are documented in
:doc:`/concepts/evaluation`.
