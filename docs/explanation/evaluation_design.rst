Why split modes and baselines matter
====================================

A drug response score means little until you know what the model was asked to
predict and what a trivial predictor would have scored on the same task. This
page explains the two design choices that make DrEvalPy scores comparable:
fixed split modes and naive baselines. The third choice, the metrics used to
score them, has its own page: :doc:`metrics`.

Split modes decide the question being asked
-------------------------------------------

A *split mode* is the rule that decides which pairs are held out for testing. A
pair is one sample-drug measurement. The same model can look excellent in one
split mode and useless in another, because each mode asks a different question
about what the model has already seen.

.. image:: /_static/img/LPO.png
    :width: 24%
    :alt: Leave-pair-out split mode

.. image:: /_static/img/LSO.png
    :width: 24%
    :alt: Leave-sample-out split mode

.. image:: /_static/img/LTO.png
    :width: 24%
    :alt: Leave-tissue-out split mode

.. image:: /_static/img/LDO.png
    :width: 24%
    :alt: Leave-drug-out split mode

**Leave-pair-out (LPO).**
   Random pairs are held out. The test sample and the test drug have both
   appeared in training, paired with other partners. The question is whether a
   model can fill in missing entries of a matrix it has mostly seen. This is the
   easiest mode, and it is the one where sample and drug averages alone already
   explain a great deal.

**Leave-sample-out (LSO).**
   Whole samples are held out; the drugs still appear in training. The question
   is whether the model can predict the response of a sample it has never seen
   from that sample's features. This is the relevant mode for personalized
   medicine. The 1.x name is still accepted as an alias; see
   :doc:`/reference/split_modes`.

**Leave-tissue-out (LTO).**
   Whole tissues are held out. Every sample from the test tissue is absent from
   training, so the model cannot lean on a close relative of the test sample.
   This probes transfer between tissues and is harder than LSO.

**Leave-drug-out (LDO).**
   Whole drugs are held out; the samples still appear in training. The question
   is whether the model can predict a drug it has never seen from that drug's
   features. This is the relevant mode for new-drug discovery, and it is
   usually the hardest. A model that has no way to describe a drug, such as one
   that only looks up a drug identity, has nothing to go on for a held-out drug beyond an average.

**Leave-study-out (LStO).**
   One whole study is held out per fold, and the model trains on the others. It
   needs a dataset built from several studies; see `Evaluating across studies`_.

Each mode is a splitter registered in the splitter registry (see
:doc:`registries_and_plugins`), and every splitter checks its own output after
splitting: a leave-sample-out split fails loudly if a test sample leaks into
training. The same checks apply to the validation split used for tuning and
early stopping (see :doc:`training_lifecycle`).

Before a splitter runs, ``split`` drops the pairs whose fitted dose-response
curve fails a curve-quality rule, so that models are trained and scored only on
measurements that are distinguishable from noise. The default rule keeps pairs
with a relevance score of at least ``-log10(0.05)`` and an absolute fold change
of at least 0.45. It is a setting of the experiment, chosen once for all models,
and it is recorded in the metadata of every fold; see :doc:`/reference/datasets`.

Why not report one split mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A single mode can mislead in both directions. A model that scores well in LPO
may have learned nothing about sample or drug features, and a model that
struggles in LDO may still be useful for prioritising samples. Reporting a model
under several split modes shows which kind of generalization it actually
provides. The :doc:`/how-to/run_experiment` guide shows how to run several modes
in one experiment.

Naive baselines set the floor
-----------------------------

Drug response values have strong structure that needs no model of biology. Some
drugs are potent against almost every sample, and some samples are sensitive to
almost every drug. A predictor that only recovers those averages can look
strong on correlation or :math:`R^2`.

DrEvalPy therefore ships a family of naive baselines, predictors that use no
omics or chemical features at all. They are registered in the zoo like any other
model:

- the overall mean of the training responses;
- the mean per sample, per drug, or per tissue;
- the mean per tissue-drug combination;
- ``NaiveMeanEffectsPredictor``, which adds a dataset mean, a tissue effect
  (when tissue labels are available), a sample effect and a drug effect.

``NaiveMeanEffectsPredictor`` is usually the strongest of these,
because it combines everything that the simple averages capture separately. It
is additive rather than a strict ANOVA: the sample effect is estimated on the
residual after the dataset and tissue means, while the drug effect is estimated
against the overall mean. An entity it has never seen contributes no effect, so
in leave-sample-out and leave-tissue-out splits it falls back on the dataset
mean plus whatever effects are known, and in leave-drug-out splits it has no
drug effect to offer. Its scores in those modes therefore show how much of the
response is explained by what the training data already knew about the
remaining entities.

A model that does not beat the best naive baseline in a split mode has not shown
that its features matter in that mode. That is why DrEvalPy asks you to run the
naive baselines next to your model, and why the report can score every model
relative to ``NaiveMeanEffectsPredictor``. The reasoning behind that comparison
is on the :doc:`metrics` page. The naive baselines are part of the experiment
like any other model. ``drevalpy.run`` treats every model alike and does not add
them on your behalf, so you list them yourself (see
:doc:`/how-to/run_experiment`).

Stress tests check that the features are doing the work
-------------------------------------------------------

Beating a naive baseline shows a model has some signal. Stress tests ask whether
that signal comes from the input features. They are run on the models you
choose, in the same experiment, and each has a different target.

**Randomization tests** scramble input views while the rest of the experiment
stays fixed, and ask how much performance drops. If a model scores the same
with a randomized gene expression view, then gene expression was never used.
There are four randomization modes, named by what stays intact:

- *SVRC* and *SVRD* randomize a single sample view or a single drug view and
  leave the others intact.
- *SVCC* and *SVCD* keep a single sample view or a single drug view intact and
  randomize all the others.

A test that would change nothing is skipped with a warning and produces no run: *SVCC* or
*SVCD* on a model with a single view has no other view to randomize, and a drug-view mode
on a dataset without that drug view has nothing to randomize. Such a run would only repeat
the ordinary one under a randomization label.

Two kinds of scrambling are available. A *permutation* shuffles rows across
entities, which keeps each feature's distribution but breaks the link to the
response. An *invariant* scramble draws new values that match each row's mean
and standard deviation, and rewires graphs while preserving their degrees, so
that coarse statistics survive but the specific content does not. Naive
baselines have no views, so there is nothing to randomize and they produce no
randomization tests.

**Robustness tests** repeat the evaluation with the pair order reshuffled for
each trial. They measure how much a model's score depends on incidental
ordering and on the training run that the order feeds into, rather than on the
data. The number of trials controls how many repetitions you collect. What stays
fixed and what varies across trials is explained in :doc:`reproducibility`.

The tests are described here conceptually; the commands and options are in
:doc:`/how-to/run_stress_tests`, and the names of the modes are listed in
:doc:`/reference/split_modes`.

Evaluating across studies
-------------------------

Training on one dataset and scoring on another is a valuable check, because
two screens can disagree about the same drug and sample. DrEvalPy 2 does this
in two steps: ``concat`` joins the datasets of several studies into one dataset
in which every sample carries the name of its study, and the **leave-study-out
(LStO)** split mode holds out one study per fold. The model trains on the other
studies and is scored on the study it has not seen. The question is whether a
model transfers between laboratories, assays and cell line collections, which a
random split inside one study cannot show. Unlike LSO, LTO and LDO, LStO does
not keep the held-out entities out of training: a cell line or drug of the test
study can also be in a training study, under that study's name. The studies are
the unit that never overlaps, so the test is about the study, not about new
samples or drugs. The commands are in :doc:`/how-to/run_cross_study`.

Two cautions apply to a cross-study score. First, studies do not measure the
response on one scale. The pEC50 values of GDSC2 average about 6.31 and those of
CTRPv2 about 5.55, so a model trained on one study can predict with an offset on
the other. Errors and R² then mix the offset with how well the model ranks the
pairs, while the correlations are not affected by an offset. Read the
correlations first, and compare the error metrics only with a baseline on the
same folds. Second, the studies may have fitted their curves differently. Prefer
responses refit with the same curve fitting procedure, so that the response
definition itself is aligned (see :doc:`datasets_provenance`).
