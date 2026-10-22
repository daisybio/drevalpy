Migrating to v2
===============

v2 tightens the contracts between splitting, featurization, training and
reporting. Re-run an experiment rather than combining its old output with new
output: split archives now retain ordering seeds, randomized datasets record
their effective seed, and result files identify diagnostic variants explicitly.
Likewise, re-train and re-save models after upgrading. Component checkpoints
store the current featurizer and predictor state, not a compatibility adapter
for earlier fitted objects.

Early stopping: A, E and V
--------------------------

The supplied validation mask may serve two purposes for models that support
early stopping. v2 calls the original training mask **A**, derives an
early-stopping holdout **E** from the validation mask, and uses the remaining
validation pairs **V** for HPO scoring. The final model trains on ``A | V`` and
continues to use ``E`` for stopping. Test pairs remain disjoint from all three.

``SplitMasks.early_stopping_mask()`` derives E/V deterministically from the
stored validation-pair order. A model only gets this split when it declares
early-stopping support and the validation mask has more than one pair. A
predictor or featurizer that sets ``requires_early_stopping = True`` requires a usable E
holdout after invalid feature rows have been removed; otherwise fitting raises a
``ValueError``. Custom training entry points should pass the training response
as ``pair_expanded_ids`` and the stopping response as
``pair_expanded_es_ids``. Original train/validation/test masks must have the
response-matrix shape, disjoint response pairs and the chosen mode's entity
separation. The derived E and V masks require only pair disjointness: entities
may be shared between those two parts of the original validation set.

Featurizer contract
-------------------

Custom featurizers no longer implement separate ``_transform`` and
``_transform_blocks`` hooks. Implement these two hooks instead:

* ``_transform_features(source, entity_ids) -> FeaturizerResult`` computes a
  matrix and named ``FeatureBlock`` mapping for nonempty, valid IDs in one pass.
* ``_empty_transform_features(source) -> FeaturizerResult`` returns zero rows
  with the fitted output shape and retained schema. It must not ask an external
  backend to transform an empty batch.

``FeaturizerResult`` is available from ``drevalpy.plugin`` and contains
``matrix`` and ``blocks``. The public ``transform`` and ``transform_blocks``
methods are projections of ``transform_features``. For invalid requested rows,
v2 pads numeric matrices and blocks with NaN; object matrices and ragged/graph
payloads are padded with ``None``. Non-entity metadata blocks are retained as
they are. ``FeaturizerResult.from_blocks`` remains useful when the matrix is the
ordered concatenation of numeric entity-aligned blocks.

Training and persistence
------------------------

The component stack has three explicit states: ``UNTRAINED``, ``EMPTY`` and
``FITTED``. A new ``train`` call rebuilds fresh components before fitting, so a
failed or empty retrain cannot expose a prior predictor. ``EMPTY`` means that no
usable training pairs survived response/feature filtering; ``predict`` returns
NaN for its requested pairs. ``FITTED`` is the only state that can be saved and
loaded. Empty and untrained stacks intentionally have no model checkpoint;
handle an empty fit as a result of that run rather than persisting it as a
reusable model artifact.

Results, variants and reports
-----------------------------

One logical fold can now produce multiple executions: its ordinary run, feature
randomization runs and robustness trials. ``ModelResult.n_folds`` counts unique
fold identities, while ``n_runs`` counts every execution. Use
``ModelResult.select_runs()`` to select results:

* with no arguments, it returns ordinary, unrandomized, non-robustness runs;
* ``randomization=(mode, view)`` or ``robustness=trial`` selects one variant;
* ``randomization="all"`` and/or ``robustness="all"`` explicitly include a
  variant dimension.

Reports and ordinary aggregates use the default selection, keeping diagnostic
variants out of leaderboards, critical-difference plots, scatter plots and
group correlations. Their report keys use the full execution identity so two
variants cannot overwrite each other. Normalization matches every variant to
the ordinary reference execution with the same fold identity. If a reference
model has more than one ordinary execution for a fold, normalization rejects the
ambiguous result.

Randomization and subsets
-------------------------

``randomization(..., random_state=...)`` now treats the supplied state as a
root seed. Every generated ``(mode, view)`` obtains a stable, distinct effective
seed derived from its full identity; the result records both seeds, type and
affected views in ``dataset.mdata.uns["randomization_provenance"]``. Set the
root seed in scripts and CLI runs when reproducing prior comparisons.

``Dataset.subset_samples`` and ``Dataset.subset_drugs`` return independent
datasets rather than backed views. Their ``uns`` payload is deep-copied,
randomization metadata is retained, and drug subsetting removes unretained
entries from ``uns["drug_graphs"]``. Code that relied on mutating a subset to
mutate its parent must update the parent explicitly.

What to re-run
--------------

Re-run folds and HPO when they use early stopping, custom featurizers, feature
randomization, robustness trials, or a subset dataset. Rebuild reports from the
new ``RunResult`` files. Recreate saved model archives after a successful
FITTED training run. Old split and result archives may load where their format
is still supported, but they do not gain v2 ordering seeds, provenance or
variant identity retroactively.

Also re-run MOLIR experiments after the loss weighting, variance ranking and
feature alignment corrections; SuperFELTR experiments after the expression
transform and shared omics corrections; SRMF cold-start evaluations; and DrugGNN experiments
whose best validation checkpoint preceded their final epoch. Triplet sampling
now draws from the NumPy stream that ``single`` seeds from ``seed`` before each
fit, so re-run triplet-based training when reproducibility matters. The same
call also seeds Python and torch, so neural fits are repeatable for a fixed
``seed``; results from earlier versions, which did not seed them, cannot be
reproduced. Recreate affected
SuperFELTR, BPE and pathway checkpoints; repeat SparseGO runs with ontology
metadata that previously failed or used inconsistent gene ordering. The
:doc:`v2_correctness_review` records these verified corrections separately from
the protocol and API changes.
