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
   * - ``LStO``
     - Studies
     - One study per fold; the study never occurs in training, but its cell lines and drugs may
       occur under another study

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

``LStO`` needs a dataset with a ``study`` sample column, which ``drevalpy.data.concat`` builds;
see :doc:`/how-to/run_cross_study`. It raises ``KeyError`` without the column and ``ValueError``
with fewer than two studies.

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
     - ``split(dataset, mode, n_splits=5, validation_ratio=0.1, random_state=42, *, quality=None, measure=None)``
   * - Fold assignment
     - ``KFold(shuffle=True)`` over the groups of the mode: observed pairs, samples, tissues
       or drugs. ``LStO`` makes one fold per study and ignores ``n_splits``
   * - Validation set
     - ``max(1, int(n * validation_ratio))`` of the ``n`` groups left after the test groups
       are removed. For ``LStO`` the groups are the pairs of the training studies: a random
       share of them, so a validation pair may come from any training study
   * - Pairs considered
     - Pairs with a finite value of the response measure that pass the curve-quality rule in
       :doc:`datasets`
   * - Response measure
     - ``measure``, ``pEC50`` or a response layer; ``None`` keeps the dataset's measure, ``LN_IC50``
       unless it was set
   * - Curve-quality rule
     - ``quality``, a ``CurveQuality``; ``None`` means ``CurveQuality.default()``. ``split``
       passes it to the splitter, and the built-in modes leave out the pairs that fail it
   * - Leakage check
     - Each fold is validated; a violation raises ``SplitValidationError``
   * - Returned value
     - A list of ``Fold``, each holding ``train``, ``test`` and ``val`` masks
   * - ``drevalpy experiment`` and ``drevalpy.experiment``
     - Always ``n_splits=5``, ``random_state=seed``

Each fold carries this metadata:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Key
     - Content
   * - ``split_mode``
     - The canonical split mode
   * - ``fold_index``
     - Position of the fold in the list
   * - ``n_splits``, ``validation_ratio``, ``random_state``
     - The arguments of ``split``
   * - ``dataset``
     - The dataset name
   * - ``fold_id``
     - First 12 characters of the SHA-256 of the train, test and validation masks
   * - ``quality``
     - The curve-quality rule of the split, as written by ``CurveQuality.to_metadata()``: a
       mapping from each threshold that is set to its value, such as ``min_abs_fold_change``
       ``0.45``; empty for ``CurveQuality.off()``
   * - ``measure``
     - The response measure of the split, such as ``LN_IC50``. A run on the fold must use the
       same measure
   * - ``test_study``
     - ``LStO`` only: the name of the study the fold tests on

In ``LStO`` folds ``n_splits`` is the number of studies. Folds are ordered by study name. The
sample IDs of a concatenated dataset are ``<study>::<sample_id>``, so a cell line measured in
two studies is two samples, one per study, and the test study's cell lines may appear in training
under the other studies' prefixes.

``drevalpy data split DATASET OUTPUT_DIR`` writes one ``fold_<i>.npz`` per fold with the
boolean arrays ``train``, ``test`` and ``val``, and two JSON entries: ``pair_seeds``, the pair-order seed of each mask, and
``_metadata``.

Randomization tests
-------------------

A randomization test trains a model with some of its views shuffled, to show how much the
model depends on them.
``drevalpy.randomization(model_spec, dataset, randomization_mode, *, randomization_type="permutation",
random_state=42)`` returns the randomized datasets; ``drevalpy experiment -r MODE`` runs them.

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

The views come from the model's configuration. ``SVCC`` and ``SVCD`` on a model with a single
view would change nothing, because no other view is left to randomize: such a test is skipped
with a warning and produces no dataset and no run.

A randomized dataset holds the same data as the original. It records which views to randomize,
and a model trained on it randomizes the output of each featurizer that reads one of those views,
leaving the other featurizers alone, inside a concatenation too. The featurizer is fitted on
the real input and runs at its own settings, so ``fingerprints`` tuned to another ``n_bits``
randomizes fingerprints of that length, and a view the featurizer computes, such as a drug
graph from the SMILES, is randomized as well. A featurizer that computes only from the SMILES,
such as ``bpePharmaformer`` and ``smilesTokens``, reads no view, so a model built on it has no
drug view to randomize. A model with no views, such as ``NaiveMeanEffectsPredictor``, gets no
randomization runs. A mode code that is not in the table is ignored without an error.
``drevalpy.experiment`` applies the tests to every model it is given.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Type
     - Effect
   * - ``permutation``
     - The default. Each sample or drug gets the featurizer output of another, its donor, by one
       seeded permutation of the samples or drugs the featurizer can use. A sample or drug has the
       same donor when the model is fitted and when it predicts
   * - ``invariant``
     - Each row of the output (the row of a sample or drug, or of each atom of a graph or
       per-atom matrix) is drawn from a Gaussian with that row's mean and standard deviation
       (at least ``1e-8``), seeded per sample or drug. The bonds of drug graphs are also
       rewired, keeping the degree of every atom

Each randomized dataset has ``randomization = (mode, view)`` and
``uns["randomization_provenance"]``, which records ``root_seed``, ``effective_seed``, ``type``
and ``views``; ``Dataset.view_randomization`` reads the last three, and a model trained on the
dataset randomizes by them. The effective seed is derived from the SHA-256 of
``[random_state, type, mode, view, sorted views]``, so the same inputs give the same shuffle.
``random_state`` is ``--randomization-random-state``, which defaults to ``--seed``.

``drevalpy data randomization MODEL DATASET OUTPUT_DIR`` takes a registered dataset
name or the path of a ``.h5mu`` file. It writes one file per randomized dataset with
``Dataset.save``, named ``<MODE>__<view>.h5mu``, and prints the number of files. Its default
mode is ``SVRC``. When every test is skipped it writes 0 files and exits with status 0.
``Dataset.load`` restores the ``randomization`` label and the provenance of such a file, so
``drevalpy run`` randomizes the same views and records the label in the run and ``drevalpy aggregate`` keeps those runs apart from the ordinary
ones.

Robustness tests
----------------

A robustness test repeats training on the same fold with the order of the pairs shuffled, to
show how much the result depends on it.

``drevalpy.robustness(fold, n_repeats)`` returns ``n_repeats`` copies of a
fold. Copy ``repeat`` has its ``train``, ``test`` and ``val`` pairs shuffled with
``shuffled(seed=repeat)``, for ``repeat`` in ``range(n_repeats)``. The masks stay the same
and the metadata gets ``robustness_repeat``.

``drevalpy experiment --robustness-repeats N`` runs, for each fold, the original and then the ``N``
variants. ``drevalpy data split --robustness-repeats N`` writes ``fold_<i>_repeat_<r>.npz``
alongside each ``fold_<i>.npz``.

Seeds of a run
~~~~~~~~~~~~~~

The root ``seed`` (``--seed``, default 42) is the same for every run of an experiment, and so
is the seed of the hyperparameter sampler unless ``hpo_random_state`` is given. The random
number generators are seeded again before tuning and before the final fit with a value derived
from the root seed and the identity of the run. That value is the first four bytes of the
SHA-256 of the JSON list ``[seed, model name, dataset name, split mode, fold index, fold id,
randomization, robustness repeat]``. It differs between robustness variants and between
randomization settings of the same fold.

How the runs appear in the results is described in :doc:`results`.
