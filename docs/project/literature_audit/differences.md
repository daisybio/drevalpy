# Source comparison and difference classification

References and file SHA256 values are pinned in `manifest.json`. The starting v2
commit is `cf5eeb6a0e020ab7e98c9829c0d81dc00a8cdd5c`. Historical regression
adaptations were also checked against drevalpy commit
`528685182164cb1412637bcb95d9fccd0be99291`; its source hashes are recorded separately.
These distinctions define what the numerical comparisons can establish.

## Evidence boundary

The independent neural runs compare synchronized weights or published weights,
intermediate values where structurally comparable, forward outputs, losses,
gradients and a shared one-step SGD update. They establish algebra and gradient
agreement in the recorded scope. They do not establish identical default RNG
streams or equivalence of every native optimizer configuration. The source
review below compares those defaults and training policies explicitly. SRMF
instead executes its native alternating updates and best-factor selection.

The seeded small-training benchmarks compare starting v2 and corrected v2 using
real production training, checkpoint selection and numerical save/load checks.
They are not executions of the upstream dataset-specific training scripts.
Architectural, classification-to-regression and leakage-avoidance adaptations
make raw end-to-end upstream equality an inappropriate acceptance target.
Published benchmark reproduction and reconstruction of opaque published feature
matrices are outside this audit. Missing runtime results are never counted as
successful comparisons; the required Python, TensorFlow and Octave runs executed.

## DrugGNN / GraphDRP

- **Retained adaptation:** GraphDRP `preprocess.py:123–159` constructs normalized
  78-component atom descriptors. The local graph featurizer constructs its own
  unnormalized atom vocabulary with ring, charge and hybridization categories.
  The source uses binary mutation inputs and sigmoid-transformed responses
  (`preprocess.py:186–265`); the local recipe uses landmark expression and the
  caller's response units. Feature widths and identities are intentionally different.
- **Retained adaptation:** `models/gcn.py:15–88` uses max pooling, a Conv1d cell
  branch and sigmoid output. Local DrugGNN retains the three GCN operations but
  uses mean pooling, a dense branch and unbounded output. Full-model initialization
  and output equality are not meaningful across these different architectures.
  Both use ordinary PyTorch/PyG initialization; shared-layer weights are mapped.
- **Compared training policy:** both use MSE, Adam and shuffled graph training
  batches with ordered validation. Source default learning rate is 1e-4; local
  is 1e-3 and tunable. Adam updates are equivalent only with matched configuration.
  Local patience-based stopping extends source fixed-budget training.
- **Corrected defect:** GraphDRP `training.py:89–110` saves the best validation-MSE
  model. Local training now reloads that winner. A worsening multi-epoch benchmark
  distinguishes it from the former final-epoch predictor.

## Precily

- **Retained adaptation:** `Fig4/Fig4b/DrugsPred.R:5–45` reads a fixed Features.csv:
  1,329 pathways followed by 100 drug coordinates, then averages five models.
  Local training concatenates its own fold-derived GSVA order with configured
  drug features and trains one model. Published weights cannot be applied to
  arbitrary local features just because both matrices have 1,429 columns.
- **Numerical comparison:** the published pathways Model1 dense widths are exactly
  1429→512→140→200→1. Its mapped trained weights pass TensorFlow float64 output,
  MSE, gradient and SGD checks. Keras Glorot initialization and PyTorch Linear
  initialization differ. Keras Adam epsilon is 1e-7 versus PyTorch 1e-8, so native
  Adam equivalence is not claimed from a matched learning rate alone.
- **Compared training policy:** the source tuner uses patience-5 stopping, but
  final models use 50 epochs and batch size 128. Local Precily retains fixed-budget
  training; its defaults match those final-fit budgets, with explicit shuffling.
- **Retained set dependence:** source GSVA runs over a supplied cohort
  (`Fig2/Fig2b/Fig2b.R:25–28`). Local fit scores use training entities only;
  unseen scores use the union of training and requested entities. Consequently,
  unseen scores can change with the requested cohort. This is not a frozen
  inductive transform or an R-versus-gseapy preprocessing parity claim.
- **Corrected defects:** held-out transformation no longer replaces cached
  training scores. Checkpoints retain fitted pathway order and GMT content;
  later GSVA outputs are aligned to that schema and missing pathways fail clearly.

## SRMF

- **Numerical equivalence:** `compute_loss.m`, `alg_update.m` and `CMF.m` agree
  with local loss, both updates, relative-loss stopping and best-factor selection
  for matched masks, responses, similarities, hyperparameters and initial factors.
  The executed Octave fixture includes zero-, fully- and partially-observed rows
  and nonzero penalties on both similarity matrices.
- **Retained wrapper adaptation:** source `predict.m:2–24` max-abs scales responses
  and loads opaque similarity matrices. The historical drevalpy regression wrapper
  instead consumes caller-provided response units (including any experiment
  response transform), computes fingerprint Jaccard and expression Pearson
  similarities, and uses its own regularization defaults. This remains explicit;
  it is not full-pipeline equivalence with the MATLAB demonstration. Regularization
  means response scaling can change the solution. NumPy and MATLAB RNG streams
  also differ, which is why the solver check shares initial factors.
- **Corrected local policy defect:** upstream predicts only rows/columns represented
  by fitted factors. The local extension intends a training-mean fallback for new
  entities. It previously put the mean into each latent coordinate, producing
  K×mean² for two unknown entities. Either unknown entity now yields the mean
  response directly. Known-pair factorization and predictions remain unchanged.

## MOLIR

- **Established ancestry:** MultiOmics `src/models/moli_model.py:5–16` supplies
  Linear/ReLU/BatchNorm encoders; original MOLI's Erlotinib script supplies
  batch-axis normalization and Linear/Dropout head ordering. MSE and a linear
  output replace classification losses/sigmoid. The reference runner deliberately
  composes these selected source operations, rather than claiming one unchanged
  source classifier. Default module initialization matches these PyTorch building
  blocks at corresponding dimensions; RNG trajectories of entire different
  classifiers are not compared.
- **Retained preprocessing adaptation:** arcsinh and top-N expression selection
  replace source variance thresholds and gene-set intersections. Mutation/GISTIC
  values are supplied by the dataset; they are not automatically rebinarized.
  Scaling and selection fit only training rows. The user must provide semantically
  compatible mutation/CNV views when comparing datasets.
- **Corrected defects:** select variance before z-scoring, since z-scoring makes
  nonconstant variances approximately one; align equally wide omics matrices by
  feature names; apply gamma to triplet loss; and draw unseeded-call triplets from
  the experiment's seeded NumPy stream. Explicit isolated triplet seeds retain
  their original Generator behavior.
- **Compared training policy:** both families use Adagrad; local per-group learning
  rates share one configured value whereas upstream exposes separate values.
  Native updates agree only with matched group parameters. Binary class-weighted
  sampling is intentionally removed for continuous regression. Local train loaders
  retain drop-last behavior, validation retains its remainder, and disjoint E
  stopping/best checkpoints extend the source fixed-epoch workflow.

## SuperFELTR

- **Retained regression adaptation:** Super.FELT supplies the three encoders and
  classifier layers. Continuous-response triplets and MSE replace binary sampling
  and BCE; encoders remain frozen while fitting the regression head.
- **Confirmed historical choices:** the drevalpy regression predecessor explicitly
  uses Adam encoders and an Adagrad head (`models/SuperFELTR/utils.py:71–78,238–244`),
  unlike the upstream classifier's Adagrad encoders. It uses arcsinh expression
  and per-view selection, rather than adding source StandardScaler and class
  balancing. These choices are preserved. Standard PyTorch layer initialization
  remains; matched-weight checks isolate the network operations from RNG draws.
- **Corrected defects:** v2 had dropped the predecessor's expression arcsinh
  (`models/SuperFELTR/superfeltr.py:258–263`). It is restored before selection and
  inference. Same-width feature alignment and seeded triplets share the MOLIR
  corrections. Saved encoder widths now come from actual layers when names are
  absent. Empty precomputed output uses the fitted expression width and block set.
- **Compared training policy:** staged encoder/head fits, local regression sampling,
  disjoint E stopping and best checkpoints are retained. The source binary-class
  training loop is not replayed as though it were the local regression algorithm.

## PharmaFormer

- **Established intended variant:** local architecture matches `pretrain.py:37–94`.
  The extra extractor dropout in `finetune.py` belongs to a different variant and
  is not silently added to from-scratch local fits. Native initialization, Adam
  without weight decay, MSE and shuffled batches follow the pretraining branch.
- **Retained input contract:** the local and historical drevalpy wrappers scale
  dataset-provided expression without another logarithm. Upstream prepares raw
  RPKM with log2(x+1); no automatic equivalence is claimed for a dataset with an
  unspecified expression scale. Supply the intended log-scale view when reproducing
  that source preparation. BPE construction/order also follows the configured
  fold pipeline, not an assumed identity with published BPE codes.
- **Leakage correction preserved:** upstream fits its scaling before KFold and
  stops on training loss. Local StandardScaler/MinMax and BPE fit only the current
  training data, and the predictor stops on disjoint E and restores its winner.
  The full attention network passes mapped-weight forward/loss/gradient/SGD checks;
  local seeded training traces match starting v2 under the same supplied features.

## DIPK

- **Numerical evidence:** source DAE encoder/decoder and full attention/head were
  executed. Intermediate tensors, training/evaluation outputs, MSE, active
  gradients and one update agree. The source hardcodes one head; that is the
  comparison setting. Local configurable heads, batch-size-one handling and
  padded ragged inputs are explicit extensions. Both paths use standard PyTorch
  initialization and Adam/MSE; matched-weight checks separate initialization from
  architecture. Local actual training exercises shuffled batches.
- **Checkpoint policy:** source `PreTrain/PreTrain.py` saves final AE weights;
  local AE also returns the final executed epoch, including after local early
  stopping. A real worsening fit pins this behavior. Source predictor training
  evaluates held-out test data each epoch; local E-based stopping and best-E
  checkpoint selection replace that leakage.
- **Artifact boundary:** real MolGNet, PPI and gene-list artifacts are checksum
  verified and exercised through raw-input training/save-load at three seeds.
  This checks the shipped artifact path, not equality with every external RMA/PPI
  table used in the publication. Feature identities/order are pinned in the
  fixtures and the full-network ragged comparison.

## SparseGO

- **Numerical evidence:** actual source ontology helpers, virtual-node wiring,
  sparse layer and complete network execute with the frozen ontology. They agree
  on connectivity, intermediates, train/eval outputs, MSE, gradients and one SGD
  update. The source torch_sparse constructor operation is replaced by equivalent
  native COO coalescing, explicitly recorded in the runner.
- **Retained training adaptation:** local SGD/MSE with epoch learning-rate decay
  and fixed budget differs from upstream correlation-selected checkpoints and
  early stopping. Module initialization uses corresponding PyTorch defaults;
  the numerical gate synchronizes weights rather than claiming identical random
  draws across arbitrary ontologies. Batches and gene/fingerprint construction
  are configured locally and exercised in actual local training.
- **Corrected input defects:** embedded/file ontology metadata now resolves;
  declared gene indices/order must agree and named expression columns are aligned
  to connectivity. Real facade tests exercise the ontology path. Full published
  ontology, fingerprint and dataset reproduction is outside scope.
