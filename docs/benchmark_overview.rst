Benchmark your own model
========================

This page is the shortest path from "I have a model" to "my model is on the leaderboard".
The leaderboard compares models on fixed, precomputed cross-validation splits, so that every model is evaluated on exactly the same data.

1. **Implement your model.** Subclass ``DRPModel``, register it in ``drevalpy/models/__init__.py``.
   See :doc:`runyourmodel` (and :doc:`example_tinynn` for a complete example).
2. **Run it on the precomputed splits.** Download ``splits.zip`` from `Zenodo <https://doi.org/10.5281/zenodo.12633909>`_
   and run ``drevalpy`` either standalone or with the Nextflow pipeline.
   See :doc:`precomputed_splits`.
3. **Create the leaderboard.** ``drevalpy report`` evaluates your predictions,
   ``create_leaderboard`` turns the evaluation into the leaderboard and the critical difference diagram.
   You can add your model to the existing leaderboard results from Zenodo. See :doc:`leaderboard`.

In short, for a model called ``YourModel`` evaluated on CTRPv2 in the leave-cell-line-out setting:

.. code-block:: bash

    # 1. put the unzipped splits into the result folder (see "Run on precomputed splits")
    mkdir -p results/my_model/CTRPv2/LCO
    cp -r splits results/my_model/CTRPv2/LCO/splits

    # 2. run your model on these splits
    drevalpy --run_id my_model --models YourModel --dataset_name CTRPv2 --test_mode LCO

    # 3. evaluate, then add your model to the downloaded leaderboard results (leaderboard_Oct26.zip from Zenodo)
    drevalpy report --run_id my_model --dataset_name CTRPv2
    tail -n +2 results/my_model/evaluation_results.csv | grep -v NaiveMeanEffectsPredictor >> leaderboard_Oct26/evaluation_results.csv
    tail -n +2 results/my_model/true_vs_pred.csv | grep -v NaiveMeanEffectsPredictor >> leaderboard_Oct26/true_vs_pred.csv

    # 4. create the leaderboard (must be LCO and CTRPv2)
    python -m drevalpy.visualization.create_leaderboard \
        --results_path leaderboard_Oct26/evaluation_results.csv \
        --true_vs_pred_path leaderboard_Oct26/true_vs_pred.csv \
        --test_mode LCO --dataset CTRPv2 --output_dir my_leaderboard

.. toctree::
   :hidden:
   :maxdepth: 1

   runyourmodel
   example_tinynn
   precomputed_splits
   leaderboard
