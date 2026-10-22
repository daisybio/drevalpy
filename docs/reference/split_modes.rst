Split modes and stress tests
============================

Split modes
-----------

A split mode decides what is held out of training. ``drevalpy.split(dataset, mode)`` and
``drevalpy data split`` produce one set of masks per fold. ``drevalpy data split`` takes a
registered dataset name or the path of a ``.h5mu`` file.

.. list-table::
   :header-rows: 1
   :widths: 14 30 56

   * - Split mode
     - Held out together
     - Test set
   * - ``LPO``
     - Observed sample-drug pairs
     - Pairs; the sample and the drug may occur in training
   * - ``LSO``
     - Samples
     - Samples that never occur in training; drugs may overlap
   * - ``LTO``
     - Tissues
     - Tissues that never occur in training
   * - ``LDO``
     - Drugs
     - Drugs that never occur in training; samples may overlap

.. image:: /_static/img/LPO.png
   :width: 24%
   :alt: Leave-Pair-Out split mode

.. image:: /_static/img/LSO.png
   :width: 24%
   :alt: Leave-Sample-Out split mode

.. image:: /_static/img/LTO.png
   :width: 24%
   :alt: Leave-Tissue-Out split mode

.. image:: /_static/img/LDO.png
   :width: 24%
   :alt: Leave-Drug-Out split mode

``LCO`` is accepted as an alias of ``LSO``, the former name of that split mode. A split made
with ``LCO`` is recorded as ``LSO``.

Folds
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Property
     - Value
   * - Signature
     - ``split(dataset, mode, n_splits=5, validation_ratio=0.1, random_state=42)``
   * - Fold assignment
     - ``KFold(shuffle=True)`` over the groups of the mode: observed pairs, samples, tissues
       or drugs
   * - Validation set
     - ``max(1, int(n * validation_ratio))`` of the ``n`` groups left after the test groups
       are removed
   * - Pairs considered
     - Pairs that pass the curve-quality filter in :doc:`datasets`
   * - Leakage check
     - Each fold is validated; a violation raises ``SplitValidationError``
   * - Returned value
     - A list of ``SplitMasks``, each holding ``train``, ``test`` and ``val`` masks
   * - ``drevalpy run`` and ``drevalpy.run``
     - Always ``n_splits=5``, ``random_state=seed``

Each fold carries this metadata:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Key
     - Content
   * - ``mode``, ``split_mode``
     - The canonical split mode
   * - ``fold_index``
     - Position of the fold in the list
   * - ``n_splits``, ``validation_ratio``, ``random_state``
     - The arguments of ``split``
   * - ``dataset``
     - The dataset name
   * - ``fold_id``
     - First 12 characters of the SHA-256 of the train, test and validation masks

``drevalpy data split DATASET OUTPUT_DIR`` writes one ``fold_<i>.npz`` per fold with the
boolean arrays ``train``, ``test`` and ``val``, the array ``_pair_seeds`` and a JSON
``_metadata`` entry.

Randomization tests
-------------------

A randomization test trains a model on a copy of the dataset in which some of the model's
views are shuffled, to show how much the model depends on them.
``drevalpy.randomization(model_spec, dataset, randomization_mode, *, randomization_type="permutation",
random_state=42)`` returns the randomized datasets; ``drevalpy run -r MODE`` runs them.

.. list-table::
   :header-rows: 1
   :widths: 20 12 68

   * - Randomization mode
     - Views of
     - Datasets produced
   * - ``SVRC``
     - samples
     - One per sample view, with only that view randomized
   * - ``SVCC``
     - samples
     - One per sample view, with all other sample views randomized
   * - ``SVRD``
     - drugs
     - One per drug view, with only that view randomized
   * - ``SVCD``
     - drugs
     - One per drug view, with all other drug views randomized

The views come from the model's configuration. A test that would change nothing is skipped
with a warning and produces no dataset and no run:

- ``SVCC`` and ``SVCD`` on a model with a single view, because no other view is left to
  randomize;
- a test whose view the dataset does not store, such as ``SVRD`` on a dataset without the
  drug view.

``Dataset.with_randomized_views`` itself raises ``KeyError`` for a view the dataset does not
store. A model with no views, such as ``NaiveMeanEffectsPredictor``, gets no randomization
runs. A mode code that is not in the table is ignored without an error. ``drevalpy.run``
applies the tests to every model it is given.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Type
     - Effect
   * - ``permutation``
     - The default. Rows are permuted across the samples or the drugs
   * - ``invariant``
     - Each row is drawn from a Gaussian with that row's mean and standard deviation (at
       least ``1e-8``). The edges of drug graphs are also rewired, keeping the degree of every
       node

Each randomized dataset has ``randomization = (mode, view)`` and
``uns["randomization_provenance"]``, which records ``root_seed``, ``effective_seed``, ``type``
and ``views``. The effective seed is derived from the SHA-256 of
``[random_state, type, mode, view, sorted views]``, so the same inputs give the same shuffle.
``random_state`` is ``--randomization-random-state``, which defaults to ``--seed``.

``drevalpy experiments randomization MODEL DATASET OUTPUT_DIR`` takes a registered dataset
name or the path of a ``.h5mu`` file. It writes one file per randomized dataset with
``Dataset.save``, named ``<MODE>:<view>.h5mu``, and prints the number of files. Its default
mode is ``SVRC``. When every test is skipped it writes 0 files and exits with status 0.
``Dataset.load`` restores the ``randomization`` label of such a file, so ``drevalpy single``
records it in the run and ``drevalpy aggregate`` keeps those runs apart from the ordinary
ones. ``drevalpy single`` reads only a file path.

Robustness tests
----------------

A robustness test repeats training on the same fold with the order of the pairs shuffled, to
show how much the result depends on it.

``drevalpy.robustness(split_masks, n_permutations)`` returns ``n_permutations`` copies of a
fold. Copy ``trial`` has its ``train``, ``test`` and ``val`` pairs shuffled with
``shuffled(seed=trial)``, for ``trial`` in ``range(n_permutations)``. The masks stay the same
and the metadata gets ``robustness_trial``.

``drevalpy run --robustness-trials N`` runs, for each fold, the original and then the ``N``
variants. ``drevalpy experiments robustness`` reads each fold ``.npz`` and writes
``<stem>_trial_<i>.npz``.

Seeds of a run
~~~~~~~~~~~~~~

The root ``seed`` (``--seed``, default 42) is the same for every run of an experiment, and so
is the seed of the hyperparameter sampler unless ``hpo_random_state`` is given. The random
number generators are seeded again before tuning and before the final fit with a value derived
from the root seed and the identity of the run. That value is the first four bytes of the
SHA-256 of the JSON list ``[seed, model name, dataset name, split mode, fold index, fold id,
randomization, robustness trial]``. It differs between robustness variants and between
randomization settings of the same fold.

How the runs appear in the results is described in :doc:`results`.
