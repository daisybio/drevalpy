MORGOTH
=============================

``SingleDrugMorgoth`` wraps `MORGOTH <https://github.com/volkamerlab/MORGOTH>`_ (Rolli et al., *Increasing
trustworthiness of machine learning-based drug sensitivity prediction with a multivariate random forest approach*,
Digital Discovery, 2026, `doi:10.1039/D5DD00284B <https://doi.org/10.1039/D5DD00284B>`_). MORGOTH is a random forest
whose trees learn the response value and a class simultaneously. The split criterion combines a classification and a
regression impurity; ``impact_classification`` (λ in the paper) is the share of the classification. drevalpy uses the
regression output as the prediction.

Fitting, sample weights, the class-aware tree weights and the prediction are the functions of the ``morgoth`` package;
the model only adds what drevalpy has to provide:

* **Class labels.** If the response has the CurveCurator regulation, responses with regulation ``NA`` are dropped and
  the class is 1 (sensitive) for ``down``, 0 (resistant) for ``not`` and ``up``. The threshold is then the quantile of
  the training responses at the fraction of sensitive responses, taken from the sensitive end (low for ln(IC50), high
  for pEC50), so that it separates the classes. Without the regulation, the threshold is the ``threshold_quantile`` of
  the training responses (training fold only); responses below it are class 1 (sensitive for ln(IC50)), the others
  class 0. The threshold is also used for the ``simple`` / ``linear`` sample weights of the package.
* **Features.** Gene expression of the genes in ``gene_list`` (default: landmark genes), arcsinh-transformed and
  standardized with a scaler fitted on the training cell lines. Of these, the ``mrmr_k`` genes (default: 50, as in the
  paper) are selected by mRMR on the training rows only.
* **Fast split.** With ``fast_split`` (default), the split search of the package is replaced by an equivalent
  vectorized one. The replacement only exists in the current process, so the trees are then always grown in a single
  process.
* **Parallelization.** Otherwise, the package grows the trees in a ``multiprocessing.Pool`` with all cores, as it does
  for the prediction. If the drugs already run in parallel (``n_jobs > 1`` of the experiment, see ``run_per_drug``),
  each drug uses a single process and grows its trees in the worker itself.

Installation
------------

The ``morgoth`` package is an optional extra of drevalpy:

.. code-block:: bash

   pip install "drevalpy[morgoth]"

or, from source, ``poetry install -E morgoth``.

Hyperparameters
---------------

``n_estimators``, ``min_samples_leaf``, ``max_features``, ``max_depth``, ``criterion_class`` and ``criterion_reg`` are
passed on to the forest of the package. ``sample_weights`` is ``simple``, ``linear`` or ``none``; ``tree_weights``
switches on the class-aware aggregation over the trees (SAURON-RF). ``fast_split`` switches between the vectorized
(default) and the original split search. The forest is seeded with 42.

Training is fast with small ``mrmr_k`` and running the drugs in parallel.

SingleDrugMorgoth Model
----------------------------------

.. automodule:: drevalpy.models.Morgoth.morgoth
   :members:
   :undoc-members:
   :show-inheritance:
