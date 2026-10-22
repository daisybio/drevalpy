How-to guides
=============

Each guide solves one task and assumes you know the basics from the tutorials. Most guides show
the command line and Python side by side; pick a tab once and the page remembers it. The
guides run on the simulated dataset (``drevalpy data synthetic`` or ``drevalpy.synthetic()``),
so you can try every step offline.

**Set up and prepare data**

- :doc:`install`: install DrEvalPy from source, with pip or conda, or run the Docker image.
- :doc:`load_and_split`: load a registered dataset and split it by pair, sample, drug or tissue.
- :doc:`bring_your_own_data`: turn your own dose-response screen into a dataset.

**Run and tune experiments**

- :doc:`choose_models`: pick baselines and zoo presets, or combine components into a recipe.
- :doc:`run_experiment`: evaluate models over the folds of a split mode.
- :doc:`run_folds_in_parallel`: train each fold as its own job and combine the runs.
- :doc:`tune_hyperparameters`: set the search, the metric, a narrower space or a fixed value.
- :doc:`run_stress_tests`: run randomization and robustness tests.

**Report, save and look things up**

- :doc:`make_report`: write the HTML report and single plots.
- :doc:`train_save_load`: train one model, save it and load it again.
- :doc:`list_available`: list models, datasets, metrics and the other registered names.

Guides for writing your own components are under :doc:`extend/index`, and guides for moving
from DrEvalPy 1 are under :doc:`migrate/index`.

.. toctree::
   :maxdepth: 1

   install
   load_and_split
   bring_your_own_data
   choose_models
   run_experiment
   run_folds_in_parallel
   tune_hyperparameters
   run_stress_tests
   make_report
   train_save_load
   list_available
   extend/index
   migrate/index
