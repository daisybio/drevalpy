What makes a run reproducible
=============================

Running the same experiment twice with the same inputs should give the same
folds, the same fitted models and the same scores. This page explains where
randomness enters a DrEvalPy run, how it is controlled, and what is recorded so
that you can check the result afterwards. The commands for a repeatable run are in
:doc:`/how-to/run_experiment`; the parameter names are in :doc:`/reference/python_api`.

One root seed
-------------

An experiment has one root ``seed``, which defaults to 42. Everything that is
random derives from it:

- The folds. ``run`` splits the dataset with the root seed as its random state,
  so the same dataset, split mode and seed give the same folds.
- Hyperparameter search. The sampler for the search (a tree-structured Parzen
  estimator) is seeded, and trials run one after another rather than in
  parallel. Its seed, ``hpo_random_state``, defaults to the root seed.
- Randomization tests. Their seed, ``randomization_random_state``, defaults to
  the root seed too.
- The training of each model, as described next.

You can set the three seeds separately, for example to keep the folds fixed while
you vary the search, but a single value is enough for a repeatable experiment.

Every fit gets its own seed
---------------------------

Before each hyperparameter search and before each final fit, DrEvalPy seeds the
Python, NumPy and PyTorch random number generators. The seed for that fit is not
the root seed itself. It is a hash of the root seed together with the *identity*
of the fit: the model, the dataset, the split mode, the fold, and, where they
apply, the randomization test and the robustness trial.

Deriving the seed from the identity has two effects:

- Different fits get different seeds, so the models in a fold do not all start
  from the identical random draw.
- A fit's seed does not depend on what else ran. Adding a model to an experiment,
  removing one, or running folds in separate processes leaves the seed of every
  other fit unchanged. A result can be reproduced on its own.

Each fold's metadata has an execution section that records what the fit was
given: the model configuration, the root seed, the hyperparameter search settings
(whether it ran, the metric, the number of trials and the search seed), and the
pair counts and checksums of the scopes used for fitting, stopping, validation
and testing (see :doc:`training_lifecycle`). A result file therefore says which
configuration and which pairs produced it.

Seeding fixes DrEvalPy's own sources of randomness. It does not turn every
numerical routine in every library into a bit-exact one, and a result that
depends on the number of threads or on the hardware can still differ in the last
digits. The literature models that draw random triplets use the globally seeded
NumPy stream, which the per-fit seeding fixes (see :doc:`literature_models`).

Randomization tests record how the data were altered
----------------------------------------------------

A randomization test scrambles an input view (see :doc:`evaluation_design`). Each
combination of randomization mode and view gets its own effective seed, derived
from the randomization seed. The record of what was done is stored in the
fold's metadata, with the root seed, the effective seed, the type of scrambling and the views
involved. If a randomization result looks surprising, you can tell exactly which
scramble produced it.

What a robustness test varies
-----------------------------

A robustness test repeats the ordinary evaluation several times, and the trials
differ in the order of the pairs within the training, validation and test sets.
The trial number is part of the fit's identity. That has a consequence worth
stating plainly: the root seed does not change between trials, but the seed
derived for the fit does, and so does the 25% of the validation pairs that is
reserved for early stopping, because it is taken from the shuffled order. A
robustness test therefore measures sensitivity to pair order together with the
training randomness that comes with it, not pair order alone. The spread of the
scores across trials indicates how much a single reported score can be trusted.

What a reproducible run needs from you
--------------------------------------

DrEvalPy controls its side of the experiment. The rest is your inputs:

- the same dataset file. A dataset downloaded again later may have been updated,
  so keep the file you used or its version (see :doc:`datasets_provenance`);
- the same model configurations, including hyperparameters, and the same extension
  or plugin versions if you registered your own components;
- the same split mode, seed and number of robustness trials;
- compatible versions of the numerical libraries, since fitted models and tuning
  results can differ between versions.

To keep a fitted model rather than refit it, save it and load it again; see
:doc:`/how-to/train_save_load`.
