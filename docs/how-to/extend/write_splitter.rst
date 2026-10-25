Write a splitter
================

.. interface: python

A splitter turns a dataset into cross-validation folds. It is a plain function, registered under
a mode name. Once registered, the mode works anywhere a split mode is accepted: in
``drevalpy.split``, in the experiment runner, and on the command line.

Register a splitter function
----------------------------

Decorate a function with ``register_splitter(mode, description, validation, *, override=False,
aliases=())``. The registry calls it as
``fn(dataset, n_splits, validation_ratio, random_state, quality=quality)`` and expects a list
of ``SplitMasks``, one per fold. The ``quality`` keyword is required: registering a function
without a ``quality`` parameter raises ``InvalidInputError`` (a ``ValueError``). Each ``SplitMasks`` holds ``train``, ``test`` and ``val`` masks shaped like the
response matrix.

The dataset argument is typed as ``Dataset`` from ``drevalpy.plugin``, a runtime-checkable
protocol with seven members: the ``sample_ids``, ``drug_ids`` and ``response_matrix`` properties,
plus ``get_tissue(ids)``, ``sample_column(column, ids)``, ``response_layer_names()`` and
``get_response_layer(name)``. A real ``Dataset`` satisfies it. If you write a test double by hand,
implement all seven, or ``isinstance`` checks against the protocol fail. ``sample_column`` returns one
per-sample metadata column, such as ``study``, and raises ``KeyError`` for a column the dataset
does not have.

``drevalpy.split`` and ``drevalpy.run`` hand your function the unfiltered dataset and the
curve-quality rule as ``quality``, a ``CurveQuality`` from ``drevalpy.plugin``. Call
``observed_mask(dataset, quality)`` to get the pairs you may use. See `Filter on curve quality`_.

.. literalinclude:: /examples/toy_splitter.py
   :language: python
   :caption: docs/examples/toy_splitter.py

Declare the leakage constraint
------------------------------

``validation`` names the constraint the registry enforces after every split. Pick one of:

- ``"LSO"``: no sample appears in both train and test.
- ``"LDO"``: no drug appears in both.
- ``"LPO"``: no sample-drug pair appears in both.
- ``"LTO"``: no tissue appears in both.
- ``"LStO"``: no study appears in both train and test (the ``study`` sample column).

The registry wraps your function, so the check also runs when someone calls the registered
splitter directly. A violation raises ``SplitValidationError``. The registry also
accepts a legacy alias for ``"LSO"`` here.

Avoid name clashes and keep old names working
---------------------------------------------

Registering a mode that already exists raises ``ValueError``. Pass ``override=True`` when you
mean to replace it, so one package cannot quietly change another package's split semantics.

Pass ``aliases=("OLD_NAME",)`` to keep a renamed mode reachable under its former name. An alias
resolves to the canonical mode wherever a mode is looked up (``drevalpy.split``,
``drevalpy list splitters``, ``--split-mode``), while results and file names always carry the
canonical name. An alias that collides with an existing mode or alias raises. The built-in ``LSO``
is itself registered with an alias for its former name.

Try it
------

This compact splitter holds out the even-numbered samples. It calls the registry with the alias
and gets the canonical mode back.

.. code-block:: python

   import numpy as np

   import drevalpy
   from drevalpy.plugin import SplitMask, SplitMasks, observed_mask, register_splitter
   from drevalpy.testing import build_synthetic_dataset


   @register_splitter(
       "EVEN_ODD_LSO",
       "Leave-Sample-Out: even-numbered samples are the test set",
       validation="LSO",
       aliases=("PARITY",),
   )
   def even_odd(dataset, n_splits=5, validation_ratio=0.1, random_state=42, *, quality):
       observed = observed_mask(dataset, quality)
       rows = np.arange(observed.shape[0])
       test_rows, train_rows = rows[rows % 2 == 0], rows[rows % 2 == 1]

       def keep(selected):
           mask = np.zeros_like(observed)
           mask[selected, :] = observed[selected, :]
           return mask

       return [
           SplitMasks(
               train=SplitMask(keep(train_rows[1:])),
               test=SplitMask(keep(test_rows)),
               val=SplitMask(keep(train_rows[:1])),
           )
       ]


   dataset = build_synthetic_dataset()
   folds = drevalpy.split(dataset, mode="PARITY", n_splits=1)
   assert folds[0].metadata["split_mode"] == "EVEN_ODD_LSO"

For the command line, put the function in a ``.py`` file in an extension directory and pass it
with ``-e``; :doc:`load_extensions` shows how. The mode then works with ``--split-mode`` and
with ``drevalpy data split --mode``.

Filter on curve quality
-----------------------

The datasets ship every fitted dose-response curve, including the meaningless ones, so treating
"has a response" as "usable" would train on junk. ``drevalpy.split`` and ``drevalpy.run`` therefore
pass the user's ``CurveQuality`` rule to your splitter as ``quality`` (see :doc:`/reference/datasets`),
together with the unfiltered dataset. ``observed_mask(dataset, quality)`` returns the boolean
sample-by-drug matrix of pairs that have a finite response *and* pass the rule. The user's choice
(``quality=`` in Python, ``--no-quality-filter`` on the command line) reaches your splitter without
further code. With ``CurveQuality.off()`` the mask is plain finiteness and no quality layer is read.
A rule that needs a layer the dataset lacks raises ``UnknownNameError``, a ``KeyError``.

The rule is yours to interpret. ``split`` records the rule the user asked for in each fold's
``quality`` metadata whatever the splitter does with it, so a splitter that deliberately ignores
it, for instance to compare filtered and unfiltered pairs, should say so in its description.
To tighten the rule, change it before the call:

.. code-block:: python

   from dataclasses import replace

   from drevalpy.plugin import CurveQuality, observed_mask
   from drevalpy.testing import build_synthetic_dataset

   dataset = build_synthetic_dataset()
   strict = observed_mask(dataset, replace(CurveQuality.default(), min_r2=0.8))
   everything = observed_mask(dataset, CurveQuality.off())
   assert strict.sum() <= everything.sum()

``curve_quality_mask`` computes the same mask from keyword options when you want a rule of your
own instead of the user's. Called bare, it applies the default rule, ``relevance_score >=
-log10(0.05)`` and ``abs(fold_change) >= 0.45``. Every other quality metric is a keyword option
(and a ``CurveQuality`` field), and its default ``None`` means "not checked":

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Option
     - Effect
   * - ``min_relevance_score``, ``min_abs_fold_change``
     - The default rule. Pass ``None`` to switch off one half of it.
   * - ``max_p_value``, ``min_log_p_value``
     - Raw, uncorrected significance. ``min_relevance_score`` is the corrected equivalent, which is
       why these are off by default.
   * - ``min_f_value``, ``min_f_value_sam``
     - F statistic, raw and s0-corrected.
   * - ``min_r2``, ``max_rmse``, ``min_signal_quality``
     - Goodness of fit.
   * - ``min_abs_slope``, ``max_abs_slope``
     - Hill slope bounds. The upper bound excludes step-like fits.
   * - ``min_front``, ``max_back``
     - Plateau bounds of a well-formed descending curve.
   * - ``min_pec50``, ``max_pec50``
     - Inflection point range, read from the response matrix.
   * - ``regulation``
     - Keep only CurveCurator's own labels. Pass a collection from ``"up"``, ``"down"`` and
       ``"not"``, for example ``("down",)``.

A metric that is NaN always fails, because a curve CurveCurator could not score is not worth
training on. Requesting a metric whose layer is absent from the dataset raises ``KeyError``
rather than being skipped silently. That includes the default rule: a splitter that calls
``curve_quality_mask`` reads the layers even when the run used ``CurveQuality.off()``, so it
fails on a dataset that has none. Prefer ``observed_mask(dataset, quality)`` unless you need a
rule that ignores the user's.

.. code-block:: python

   from drevalpy.plugin import curve_quality_mask

   strict = curve_quality_mask(dataset, min_r2=0.8, regulation=("down",))
   assert strict.shape == dataset.response_matrix.shape

Test the splitter
-----------------

Resolve your mode through the registry rather than calling the function directly, so the
leakage check runs, and pass ``quality=CurveQuality.default()`` (or ``CurveQuality.off()``). :doc:`test_components` shows how to use ``drevalpy.testing`` for this.
