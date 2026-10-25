Load a dataset and split it into folds
======================================

.. interface: both

This page shows how to load a built-in dataset or a dataset file, check what it holds, and split
it into cross-validation folds with a split mode. You do not need to split by hand to run an
experiment: ``run`` splits for you. Split yourself to inspect the folds, to reuse the same folds
across tools, or to train the folds in parallel; see :doc:`/how-to/run_folds_in_parallel`.

The examples use the simulated dataset, so they run offline. :doc:`/reference/datasets`
describes the file format and the curve-quality filter.

Load a built-in dataset
-----------------------

Built-in datasets are loaded by name. The first load downloads the file into the cache
directory; later loads read it from there. ``drevalpy list datasets`` shows the registered names;
see :doc:`/how-to/list_available`. ``TOYv1`` is a small dataset of 8 MB, well suited to try
this out. ``GDSC1``, ``CTRPv2`` and the other screens are much larger.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      The command writes the loaded dataset to a file, which you can pass to any command as
      ``--dataset``. You can also pass the name itself to ``run`` and the other commands that
      take ``--dataset``.

      .. snippet: network

      .. code-block:: bash

         drevalpy data load TOYv1 toyv1.h5mu

      The command prints:

      .. code-block:: text

         Wrote TOYv1 to toyv1.h5mu

   .. tab-item:: Python
      :sync: python

      .. snippet: network

      .. code-block:: python

         import drevalpy

         toy = drevalpy.load("TOYv1")
         print(toy)

      The call prints:

      .. code-block:: text

         Dataset
             Name: TOYv1
             Samples: 32
             Drugs: 12
             Measured pairs: 384
             Randomization: None
             Modalities:
                 gene_expression: 32 × 11711
                 mutations: 32 × 2998
                 copy_number_variation_gistic: 32 × 2952
                 proteomics: 32 × 758
                 response_rr1: 32 × 300
                 response_rr2: 32 × 300

Where the downloads go, and how to move them, is in :doc:`/how-to/install` and
:doc:`/reference/environment`. To make your own datasets loadable by name, see
:doc:`/how-to/extend/add_dataset_source`.

Load a dataset from a file
--------------------------

``load`` also accepts the path to an existing ``.h5mu`` file. DrEvalPy has no loader for CSV or
other formats; to turn your own measurements into a dataset file, see
:doc:`/how-to/bring_your_own_data`. This example first writes the simulated dataset to a file
and then loads that file.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         drevalpy data synthetic --output toy.h5mu

      Every command that takes a dataset accepts the path as ``--dataset toy.h5mu``.

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         import drevalpy

         drevalpy.synthetic().save("toy.h5mu")
         dataset = drevalpy.load("toy.h5mu")
         print(dataset)

      The ``print`` shows:

      .. code-block:: text

         Dataset
             Name: SYNTHETIC
             Samples: 60
             Drugs: 8
             Measured pairs: 456
             Randomization: None
             Modalities:
                 gene_expression: 60 × 16

A file you pass by path is used as it is. A name that matches a file in the cache directory or a
registered dataset is looked up first; the order is in :doc:`/reference/datasets`.

Look inside a dataset
---------------------

The command line has no command that prints a dataset; use Python. A ``Dataset`` gives the
identifiers of its samples and drugs, the tissue of each sample, and the response matrix with
the pEC50 of every pair. Unmeasured pairs are ``NaN``.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      ``drevalpy list datasets`` shows the registered datasets, not their content.

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         print(dataset.sample_ids[:3], dataset.drug_ids[:3])
         print(dataset.sample_meta.head(3))
         print(dataset.response_matrix.shape)

      The calls print:

      .. code-block:: text

         ['SYNTH-000' 'SYNTH-001' 'SYNTH-002'] ['100000' '100001' '100002']
                   tissue
         SYNTH-000   Lung
         SYNTH-001  Blood
         SYNTH-002   Skin
         (60, 8)

Split a dataset into folds
--------------------------

A split mode decides what is held out together; see :doc:`/reference/split_modes`. This example
holds out whole samples (``LSO``) in 5 folds. ``--mode`` takes ``LPO``, ``LSO``, ``LDO`` or
``LTO``; ``--n-splits`` and ``--validation-ratio`` change the number of folds and the share of
validation data. The default ``random_state`` (``--seed`` on the command line) is 42, so repeating a split gives the same folds.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      The command takes the path of a ``.h5mu`` file or a registered dataset name, like
      ``run --dataset``.

      .. code-block:: bash

         drevalpy data split toy.h5mu splits --mode LSO

      It writes one file per fold to the ``splits`` directory and prints:

      .. code-block:: text

         Wrote 5 folds to splits

      The directory holds ``fold_0.npz`` to ``fold_4.npz``.

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         folds = drevalpy.split(dataset, "LSO")
         print(len(folds))
         print(folds[0])

      ``split`` returns one ``SplitMasks`` per fold, with a ``train``, a ``test`` and a ``val``
      mask:

      .. code-block:: text

         5
         SplitMasks
             Shape: (60, 8)
             Train: 338 pairs
             Test: 89 pairs
             Val: 29 pairs
             Metadata:
                 split_mode: LSO
                 fold_index: 0
                 n_splits: 5
                 validation_ratio: 0.1
                 random_state: 42
                 dataset: SYNTHETIC
                 fold_id: 35d304f70093
                 quality: {'min_relevance_score': 1.3010299956639813, 'min_abs_fold_change': 0.45}

Each mask is a boolean matrix of the same shape as the response matrix: samples by drugs. The
three masks of a fold do not overlap. With the built-in split modes, only pairs that pass the curve-quality rule in
:doc:`/reference/datasets` are in any mask, so the three counts can add up to less than the
number of measured pairs. The rule is the default one unless you pass ``quality=`` to ``split``
(``CurveQuality.off()`` keeps every measured pair). On the command line, ``drevalpy data split``
has ``--no-quality-filter``, ``--min-relevance-score`` and ``--min-abs-fold-change``.

Check a split
-------------

The fold above holds out samples, so no test sample may occur in the training pairs. The check
uses the sample index of each pair in a mask:

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      The command line has no check command. Use the Python tab, which reads the ``.npz`` files
      written above with ``SplitMasks.load("splits/fold_0.npz")``.

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         fold = folds[0]
         train_samples = set(dataset.sample_ids[fold.train.pairs[:, 0]])
         test_samples = set(dataset.sample_ids[fold.test.pairs[:, 0]])
         print(len(train_samples), len(test_samples), train_samples & test_samples)

      The ``&`` is the overlap, which is empty:

      .. code-block:: text

         44 12 set()

DrEvalPy also validates every fold when it creates it and raises ``SplitValidationError`` if a
fold leaks.

Save and reload folds
---------------------

A fold is a small ``.npz`` file. Save the folds you want to share, and load them where you need
them:

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      ``drevalpy data split`` already saved them. :doc:`/how-to/run_folds_in_parallel` passes
      them to ``drevalpy single``.

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         from drevalpy.types import SplitMasks

         fold.save("fold_0.npz")
         print(SplitMasks.load("fold_0.npz").metadata["fold_id"])

      .. code-block:: text

         35d304f70093

The ``fold_id`` identifies the exact masks, and DrEvalPy stores it in the results of every model
that ran on the fold.
