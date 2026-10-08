Create the leaderboard
======================

After your run on the precomputed splits (see :doc:`precomputed_splits`), the leaderboard is created in three steps.

1. Evaluate the run
-------------------

.. code-block:: bash

    drevalpy report --run_id my_model --dataset_name CTRPv2

Besides the HTML report, this writes the following files into ``results/my_model/``:

* ``evaluation_results.csv``: evaluation metrics per model and CV split,
* ``true_vs_pred.csv``: true and predicted responses,
* ``evaluation_results_per_drug.csv`` and ``evaluation_results_per_cl.csv``.

2. Add your model to the existing leaderboard results
-----------------------------------------------------

The results of the models that are already on the leaderboard can be downloaded from `Zenodo <https://doi.org/10.5281/zenodo.12633909>`_
(``leaderboard_Oct26.zip``). Unzip it. The leaderboard folder contains ``evaluation_results.csv`` and ``true_vs_pred.csv``.
Append the lines of your model (without the header line) from your own files to the respective files of the leaderboard folder.
The ``NaiveMeanEffectsPredictor`` is always run as a baseline and is already part of the leaderboard results, so ``grep -v`` leaves its lines out:

.. code-block:: bash

    tail -n +2 results/my_model/evaluation_results.csv | grep -v NaiveMeanEffectsPredictor >> leaderboard_Oct26/evaluation_results.csv
    tail -n +2 results/my_model/true_vs_pred.csv | grep -v NaiveMeanEffectsPredictor >> leaderboard_Oct26/true_vs_pred.csv

Make sure that the downloaded files end with a line break before you append.

3. Create the leaderboard
-------------------------

For the leaderboard, the test mode must be ``LCO`` and the dataset must be ``CTRPv2``. Run ``create_leaderboard`` on the merged files:

.. code-block:: bash

    python -m drevalpy.visualization.create_leaderboard \
        --results_path leaderboard_Oct26/evaluation_results.csv \
        --true_vs_pred_path leaderboard_Oct26/true_vs_pred.csv \
        --test_mode LCO \
        --dataset CTRPv2 \
        --output_dir my_leaderboard

This creates ``leaderboard_dark.png`` and ``leaderboard_light.png`` (normalized Pearson, RMSE and raw Pearson per model)
and a critical difference diagram in the output directory. The baselines (models starting with ``Naive``) are marked as such.
To only look at your own model and the baselines, use your own ``results/my_model`` files directly instead of the merged ones.

Options:

* ``--results_path``: path to ``evaluation_results.csv`` (required).
* ``--true_vs_pred_path``: path to ``true_vs_pred.csv``. If given, a per-drug Pearson panel is added.
* ``--test_mode``: ``LCO``, ``LDO``, ``LPO`` or ``LTO``. Only results of this test mode are used. Default: ``LCO``.
* ``--dataset``, ``--measure``: names shown in the plot title. Defaults: ``CTRPv2``, ``LN_IC50_curvecurator``.
* ``--output_dir``: where to save the images. Default: ``docs/_static/img``.
* ``--top_n``: only show the top N models.
* ``--cd_metric``: metric for the critical difference diagram. Default: ``Pearson: normalized``.
* ``--cd_width``, ``--cd_height``: size of the critical difference diagram in inches.
* ``--font_adder``: increase the font size.

.. note::

    The critical difference diagram (Friedman test with post-hoc Conover test over the CV splits) needs several CV splits.
    We recommend at least 7, which are provided in ``splits.zip``.
