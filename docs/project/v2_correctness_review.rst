V2 correctness review and regression evidence
=============================================

The starting implementation is commit
``cf5eeb6a0e020ab7e98c9829c0d81dc00a8cdd5c``. Its kernel fixtures and training
predictions remain recorded separately from expected corrections. The source
revisions, audited files, artifact checksums and runtime inventories are in
``docs/project/literature_audit/manifest.json``. Reproduction tools live in
``tools/literature_reference/`` and ``tools/literature_training_benchmark.py``.
The reference runtimes are optional, isolated audit environments; TensorFlow,
Keras and Octave are not library dependencies.

Eight-model comparison
----------------------

.. list-table:: Numerical evidence and preserved adaptations
   :header-rows: 1
   :widths: 12 44 44

   * - Model
     - Independent upstream comparison
     - Interpretation
   * - DrugGNN
     - GraphDRP's GCN variant: mapped three-layer graph convolution stack,
       forward values, loss, gradients and optimizer update.
     - The local mean pooling, dense cell-line branch and unbounded head are
       explicit architectural adaptations. Full-network output equality with
       GraphDRP is neither expected nor claimed. Best validation checkpoint
       selection is now restored, as in GraphDRP.
   * - Precily
     - Published pathways ``Model1.hdf5``: all trained Dense weights mapped
       from Keras to PyTorch; forward, MSE, gradients and SGD update pass in
       TensorFlow float64.
     - Hidden widths match that particular published model. A single model,
       configurable drug features, fold-specific GSVA, PyTorch initialization
       and optimizer defaults differ from the published ensemble workflow.
   * - SRMF
     - Original ``compute_loss.m``, ``alg_update.m`` and ``CMF.m`` executed
       in Octave: nonzero similarity penalties, zero/full/partial observation
       rows, both alternating updates and five-step best-factor selection.
     - Initial factors are shared to isolate MATLAB versus NumPy RNG streams.
       Local similarities come from configured features; upstream benchmark
       similarity matrices are not substituted for those features.
   * - MOLIR
     - Multi-Omics_analysis encoders plus the original MOLI fusion/head
       ancestry: regression-adapted intermediate tensors, outputs, weighted
       triplet plus MSE loss, gradients and optimizer update.
     - Preserve continuous-response triplets and MSE. Encoder ordering follows
       Multi-Omics_analysis; batch-axis normalization and head ordering follow
       original MOLI. This is a documented composition, not an unchanged
       classifier. The ignored triplet coefficient is corrected.
   * - SuperFELTR
     - Super.FELT encoders and classifier layers before sigmoid: intermediate
       fusion, regression outputs, loss, active gradients and optimizer update;
       encoder triplet losses and gradients with shared triplet indices.
     - Preserve continuous-response sampling, MSE and frozen encoders while
       fitting the regression head. Binary and continuous-response triplet
       samplers are intentionally different.
   * - PharmaFormer
     - Upstream attention network: mapped weights, complete forward, loss,
       parameter gradients and optimizer update.
     - Preserve fold-trained feature construction and disjoint validation
       stopping instead of upstream training-loss stopping. Pretraining and
       published benchmark data reproduction are outside this audit.
   * - DIPK
     - Upstream autoencoder encoder/decoder and complete attention/prediction
       network: intermediate tensors, masked ragged input adaptation, forward,
       loss, active gradients and optimizer update.
     - Compare one attention head because upstream hardcodes one. Keep the
       configurable local head count and batch-size-one support. The original
       autoencoder saves its final epoch; the local early-stopped encoder also
       returns its final epoch. No best-encoder replay was imposed.
   * - SparseGO
     - Upstream ontology sorting, virtual-node wiring, sparse layer and full
       network: intermediate tensors, training/evaluation forward, MSE,
       gradients and optimizer update.
     - The upstream ``torch_sparse.coalesce`` constructor operation is adapted
       to native PyTorch COO coalescing and disclosed in the evidence. Preserve
       the local fixed-epoch MSE fit; upstream selects correlation checkpoints.

Float32 comparisons use ``rtol=1e-5, atol=1e-6``; float64 uses
``rtol=1e-7, atol=1e-9``. The initial Precily attempt with Keras's PyTorch backend
had a gradient difference of about ``1.4e-6`` exceeding the entrywise tolerance. Investigation
identified that backend's float64-to-float32 type promotion. The independent
TensorFlow float64 run passes the original float64 tolerances; no tolerance was
loosened to hide that observation. Raw results retain the comparison scopes.

Real integration coverage
-------------------------

The current-versus-starting-v2 training gate covers seeds ``0``, ``1`` and ``2``
for all eight models. The benchmark uses actual optimizer/factorization updates, finite predictions,
fit losses, checkpoint evidence and numerical state round-trips. Starting-v2
and corrected results are separate artifacts. Extended runs exercise multiple
batches and epochs, alongside focused tests of checkpoint selection and tiny
holdout fallbacks.

``tests/models/test_global_models.py`` now exercises DrugGNN's portable graph
payloads and SparseGO's embedded ontology through the full model facade.
``TestDIPKArtifactIntegration`` uses the real MolGNet checkpoint and PPI/gene
artifacts, verifies their pinned checksums, trains the gene-expression
autoencoder and predictor, and compares predictions after save/load for all
three seeds. It is a network test because a cold cache downloads artifacts.
Missing or mismatched artifacts fail this gate.

Expected changes from verified defects
--------------------------------------

* DrugGNN previously retained the last epoch after early stopping. It now loads
  the validation-MSE winner. A deliberately worsening four-epoch run verifies
  that prediction uses the earlier winning checkpoint.
* MOLIR now applies ``gamma`` to triplet loss. Both source families apply this
  coefficient; storing it without using it was an accidental deviation.
* MOLIR ranks expression variance before z-scoring, aligns equally wide omics
  matrices by feature names, and shares the experiment's seeded NumPy stream for
  triplet sampling. SuperFELTR shares the alignment and sampling fixes.
* SuperFELTR again applies arcsinh to expression before variance selection and
  inference, as the established regression adaptation does.
* SRMF uses a direct training-mean prediction when either entity is unseen; its
  earlier mean-filled latent vectors could return K times mean squared.
* SuperFELTR checkpoints now store actual encoder input widths even when
  optional feature names are absent. Such checkpoints previously restored
  without a regressor and predicted NaNs.
* SparseGO accepts embedded ontology text and file-backed metadata, validates
  gene indices and orders input values consistently with the connectivity.
  Inconsistent explicit index/order metadata fails clearly.
* Portable DrugGNN graph dictionaries are converted to PyG objects at the
  featurizer boundary. Missing optional drug views permit DIPK's real MolGNet
  computation fallback.
* Learned BPE merges and fitted pathway scores survive checkpoints. Transforming
  held-out entities no longer replaces cached training pathway scores with
  scores recomputed using those held-out covariates. Incomplete pathway
  checkpoints fail on load. Fitted pathway names and GMT content are saved and
  reused, protecting output identity from changed source gene sets.

The training/evaluation protocol correction is a separate intentional change:
HPO fits training A, stops on E and scores V; final training starts fresh on
A union V and stops on the same E. Test data remain untouched. Models without
early stopping fit A union validation. Former scores produced with E in final
training are not a valid performance floor.

Review and migration
--------------------

Review the artifacts in this order: pinned references and immutable fixtures;
model-specific corrections; training and evaluation; lifecycle and combined
features; variants and seeds; subsets and documentation. The repository checks
remain serial where registries are shared. Coverage and size policy floors are
preserved; SparseGO's two coverage exemptions were removed after actual facade
coverage, and the component-stack size ceiling was lowered to its measured size.

See :doc:`v2_migration` for extension API changes, result selection, mask seeds,
checkpoint behavior and the experiments that must be rerun. Benchmark changes
caused by the corrected protocol, MOLIR coefficient and DrugGNN checkpoint
selection must be reported separately from structural regression comparisons.

Repository verification
-----------------------

The final serial non-network run passed **4,624 tests**, with eight skips and
11 network tests deselected. Aggregate coverage is **95.15%**. The per-module
coverage gate passed for 292 modules and the size gate for 293 modules, without
lowering policy floors. The three real-artifact DIPK integration runs passed
separately at seeds 0, 1 and 2.

The plotting subprocess and offline wheel-build checks pass with access to the
existing uv cache and a writable Matplotlib cache. Frozen literature training
tests explicitly select CPU, so availability of the Mac GPU cannot change the
numerical baseline. They restore process-wide random state and write checkpoints
inside their temporary test directories.

Commands used for the final verification::

    uv run pytest -m "not network" --cov --cov-report=json --cov-report=term-missing
    uv run pytest -m network tests/models/test_global_models.py
    uv run python tools/coverage_gate.py
    uv run python tools/size_gate.py
    uv run prek run --all-files
    uv run ty check
    uv run --group docs sphinx-build -W -E docs /tmp/drevalpy-docs-build

The packaged upstream runners were replayed after formatting; Precily used the
isolated TensorFlow environment and SRMF replayed the actual Octave output.
These checks establish the scopes listed above. The source comparison in
``docs/project/literature_audit/differences.md`` records preprocessing, feature
identity, initialization, optimizer, batching and checkpoint differences for
each model. Neural upstream one-update checks use a shared SGD configuration;
source-specific training policies are compared separately. They do not constitute a full
published benchmark reproduction or equality of intentionally adapted pipelines.
