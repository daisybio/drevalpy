How the literature models differ from their papers
==================================================

DrEvalPy includes ten literature models: DrugGNN (from GraphDRP), Precily, SRMF,
MOLIR, SuperFELTR, PharmaFormer, DIPK, SparseGO, EnsembleMF and PaccMann. They are
reimplementations in the same framework as every other model, trained and scored under
the same split modes. A reimplementation is not a copy, and results from DrEvalPy will not match
a paper's table digit for digit. This page lists what was changed relative to the
published method and why, so that you know what a score represents.

The scripts that compare these reimplementations numerically against the published code
live in the separate ``drevalpy-parity`` repository, next to the 1.6.1 comparison
(:doc:`parity_1_6_1`).

Some changes apply to every literature model and are not repeated below:

- *No peeking at held-out data.* Several published training scripts fit scalers
  on all data before cross-validation, or monitor the test set while training.
  Here, everything that is fitted (scalers, vocabularies, selected features) uses
  the training entities only, and early stopping watches the early-stopping
  scope, never the test set (see :doc:`training_lifecycle`).
- *Regression instead of classification.* Several of the methods were published
  for sensitive-versus-resistant labels. Here they predict a continuous response, so
  classification losses and output activations become mean squared error and a
  linear output.
- *Framework defaults.* Layers use the ordinary PyTorch (or scikit-learn) default
  initialization, and each model is seeded per fit (see :doc:`reproducibility`).
  A model trained here does not follow the random number stream of the original
  code.
- *Features come from the dataset.* Models read the views of the dataset
  they run on, rather than the published feature files. A model whose paper used a
  specific processed feature matrix is therefore comparable in method, not in
  inputs.

The models also differ in how they behave under the evaluation. SRMF, Precily and
SparseGO do not stop early; they train for a fixed budget or run their own solver.
DIPK and PharmaFormer require an early-stopping scope.

DrugGNN
-------

DrugGNN is based on GraphDRP. A graph network encodes the drug, a second branch
encodes the sample, and a regression head combines them.

- The drug graph uses its own featurizer, with unnormalized one-hot descriptors
  for each atomic node (atomic number, degree, charge, number of hydrogens,
  hybridization, aromaticity and ring membership). The published code uses a
  different, normalized descriptor vector. The feature widths and meaning differ.
- The sample branch reads landmark gene expression through a dense layer. The
  published model reads binary mutation inputs through a one-dimensional
  convolution and applies a sigmoid to the response. Here the response keeps the
  units of the dataset and the output is unbounded.
- The three graph convolution layers are kept, but pooling is by mean instead of
  by maximum.
- Training uses Adam and mean squared error. The learning rate defaults to
  1e-3 and is tunable, while the published default is 1e-4. Training stops early
  when the validation loss has not improved for five epochs, which the published
  fixed-budget training does not do.
- The model restores the weights of the epoch with the best validation loss before
  predicting.

Precily
-------

Precily predicts the response from pathway activity scores and drug
descriptors.

- The published model reads a fixed feature file of 1,329 pathways followed by 100
  drug coordinates and averages five networks. Here the pathway scores are
  computed per fold, from the dataset, and concatenated with the configured drug
  features, and one network is trained. Published weights cannot be applied to
  these features even if the widths happen to match.
- The gene sets are the MSigDB C2 CP v6.1 collection, the one the published feature file was
  built from. A dataset that stores its own gene sets in ``uns["pathways_gmt"]`` is scored
  against those. Otherwise the collection is downloaded from MSigDB once, checked against a
  pinned hash and kept in the cache directory.
- Training is a fixed budget with 50 epochs, batch size 128 and shuffling,
  matching the budgets of the published final models, not the patience-based
  stopping of the published tuner. The default learning rate is 1e-3. PyTorch and Keras
  also differ in the default epsilon of the Adam optimizer, so a matching learning
  rate alone does not give identical updates.
- Pathway activity is computed with gene set variation analysis over a cohort of
  samples. Scores for the training samples are computed from the training samples
  only. Scores for unseen samples are computed over the training and requested
  samples together, which means an unseen sample's score can change slightly with
  which other samples are scored with it. It is not a frozen transformation of
  one sample.

SRMF
----

SRMF is a similarity-regularized matrix factorization. It learns latent factors
for samples and drugs and regularizes them with similarity matrices.

- The solver follows the published algorithm: the same loss, the alternating
  updates, the stopping rule on relative loss, and selection of the best factors.
- The published demonstration scales responses and loads precomputed similarity
  matrices. Here the similarities are computed from the data, as Jaccard
  similarity of drug fingerprints and Pearson correlation of sample gene
  expression, and the model uses the response in whatever units the experiment
  provides. Because the method is regularized, response scaling can change the
  solution. The regularization defaults are the local ones.
- Samples and drugs are factorized in sorted identifier order, as in drevalpy 1.6.1,
  so a seeded fit starts from the same initial factors for the same entities.
- The published method predicts only for samples and drugs represented in the
  fitted factors. Here, a sample or a drug that was not seen in training receives
  the training mean response: if *either* the sample or the drug is unknown, the
  prediction is the mean. Known pairs are predicted by the factorization.
- The model does not support early stopping.

MOLIR
-----

MOLIR is based on MOLI, a multi-omics network with one encoder per omics type.
It is a single-drug model: one network is trained for each drug.

- The encoder layers (linear, ReLU, batch normalization) follow the published
  model, and the classification loss and sigmoid output are replaced by mean
  squared error and a linear output. The model is a selection of published
  operations composed for regression, not one unchanged published classifier.
- Expression is transformed with the inverse hyperbolic sine, and the genes with
  the highest variance are kept, in place of the published variance thresholds
  and gene-set intersections. Variance is selected *before* z-scoring, since
  after z-scoring all variances are about one. Scaling and selection use the
  training rows only. DrEvalPy 1.6.1 selected after z-scoring, so its kept genes
  were decided by floating-point rounding.
- Mutation and copy-number inputs are used as the dataset provides them, and are not
  re-binarized. When you compare datasets, make sure these views mean the same
  thing in both.
- Omics matrices of equal width are aligned by feature name rather than by
  position.
- The triplet loss that shapes the encoders is weighted by gamma, and triplets are
  drawn from the NumPy random stream, which is seeded for each fit.
- Training uses Adagrad as in the published model, but with one shared learning
  rate for all parameter groups, where the published code exposes separate rates.
  The published class-weighted sampling is removed because the response is
  continuous. Early stopping and best-weights selection on the early-stopping scope
  are added.

SuperFELTR
----------

SuperFELTR (Super.FELT) trains one encoder per omics type, then a classifier.

- Encoders and head follow the published layers. Encoders are trained with
  continuous-response triplets and mean squared error, and then frozen while the
  regression head is fitted.
- The encoders use Adam, and the head uses Adagrad. The published classifier used
  Adagrad for the encoders. This choice comes from the earlier DrEvalPy regression
  implementation and is kept.
- Expression is transformed with the inverse hyperbolic sine and features are
  selected per view, instead of the published standard scaling and class balancing.
- The alignment of equally wide inputs by feature name and the seeded triplets are
  shared with MOLIR.
- Early stopping and best-weights selection use the early-stopping scope.

PharmaFormer
------------

PharmaFormer is a transformer that reads gene expression and a tokenized
molecular string.

- The network follows the published pretraining variant, with Adam without weight
  decay, mean squared error and shuffled batches. The extra dropout in the
  published fine-tuning variant is not added to models trained from scratch here.
- Expression is used as the dataset provides it, with no additional logarithm.
  The published pipeline applies log2(x+1) to raw RPKM values. To reproduce that
  preparation, supply a log-scale expression view. The vocabulary for the
  molecular string is learned from the current fold and so may differ from the
  published one.
- The published code fits its scalers before cross-validation and stops on the
  training loss. Here a standard scaler and a min-max scaler, and the vocabulary,
  are fitted on the training data only, and training stops on the early-stopping
  scope and restores the best weights.
- The scaler is fitted on each training sample once, not once per sample-drug pair,
  so a sample that was measured with many drugs does not weigh more. An earlier
  DrEvalPy release fitted it on every pair; this was corrected to follow DrEvalPy 1.6.1.

DIPK
----

DIPK combines a denoising autoencoder on gene expression and pretrained molecular
graph embeddings of the drug with an attention head.

- The published code uses a single attention head. Here the number of heads is
  configurable, and the default is two. Fits with batches of one pair and padded
  variable-length inputs are handled.
- The autoencoder masks 20% of its input during training. It returns the weights
  of its final epoch, including when it stopped early, as in the published
  pre-training.
- The autoencoder is fitted on each training sample once, not once per
  sample-drug pair, with the early-stopping samples as its validation set. Each
  sample profile is z-scored on its own (sample standard deviation) after the gene
  selection. A log2(x + 1) step is added first only when the expression view
  records a linear unit such as TPM; the curated datasets are already
  log2(TPM + 1). Its learning rate, patience and batch size are the
  ``lr_autoencoder``, ``patience_autoencoder`` and ``batch_size_autoencoder``
  hyperparameters, and ``epochs_autoencoder`` defaults to 1000. Only mini-batches
  are moved to the GPU. An earlier DrEvalPy release fitted the autoencoder on every
  pair, repeating each sample once per drug it was measured with; this was
  corrected to follow DrEvalPy 1.6.1.
- The published predictor evaluates the test set at every epoch. Here the predictor
  stops on the early-stopping scope and restores the weights of the best epoch.
- The pretrained molecular embeddings and the protein-interaction table are
  external artifacts downloaded on first use. They are the shipped versions, not
  necessarily identical to every table used in the paper.

SparseGO
--------

SparseGO is a sparse network whose connections follow the gene ontology.

- The ontology wiring, the virtual nodes and the sparse layer follow the published
  implementation. The published dependency on a sparse-tensor library is replaced
  with an equivalent native sparse representation.
- Training uses stochastic gradient descent with mean squared error, a
  learning-rate decay per epoch and a fixed budget of 400 epochs by default. The
  published code selects checkpoints by correlation and stops early. Here the
  training is not stopped early.
- Gene order matters because the network is wired by the ontology. Named
  expression genes are aligned to the order of the ontology, and the declared gene
  order must agree with it. The ontology can be given as a file or as embedded
  metadata.
- Reproducing the published ontology, the drug fingerprints and the published
  dataset as a whole is outside what DrEvalPy checks.

EnsembleMF
----------

EnsembleMF is a two-tower matrix factorization. It was contributed to DrEvalPy 1.6.1
and has no publication of its own, so there is no paper to differ from. This page
lists how the port differs from the 1.6.1 implementation.

- Two residual networks turn the expression profile of a sample and the fingerprint
  of a drug into latent factors. The score is their dot product plus a bias per sample,
  per drug and overall, plus a small interaction network. A free embedding per
  drug is added to the drug factor, but only for drugs that were in a training batch.
  A drug that was held out of training is scored from its fingerprint alone.
- The prediction is the mean of ``n_ensemble`` members (20 by default). Member *i*
  is initialized and shuffled with seed ``seed + i`` and keeps the weights of its
  epoch with the best error on the early-stopping scope. A negative ``seed`` leaves
  the members unseeded.
- The expression is read through the ``scaledGeneExpression`` featurizer with
  ``transform: rank``, on the ``landmark_genes`` list. Each value becomes its position
  among the sorted values of the training samples for that gene, and the result is
  standardized with a scaler fitted on the same samples. A sample is therefore scored
  against a fixed reference, whatever else is predicted with it. ``transform: arcsinh``
  gives the strictly pointwise alternative. The fingerprints have 256 bits.
- Expression values reach the featurizer as float32, as everywhere in v2, where the
  1.6.1 model read them in double precision. The rank positions are the same for
  values that float32 represents exactly. The features differ only by rounding.
- A pair whose sample or drug has no features is not scored: like every v2 predictor,
  EnsembleMF returns NaN for it. The 1.6.1 model returned the training mean.
- Without an early-stopping scope, 10% of the training pairs are held out with a
  generator seeded from ``seed``. The 1.6.1 model drew them from the global random
  state.
- Checkpoints are the usual v2 checkpoints; those of 1.6.1 do not load.

PaccMann
--------

PaccMann (Manica et al., Molecular Pharmaceutics 2019) embeds the tokenized SMILES
of a drug, convolves the embedding at three window sizes, and lets the gene
expression and each SMILES representation attend over each other. The attended
vectors go through dense layers to one response. The network is the 1.6.1 one,
ported without change of architecture.

- The SMILES are tokenized by the regular expression of the original ``pytoda``
  package, with padding (0) and unknown (1) as the first two indices. The
  ``smilesTokens`` featurizer builds the vocabulary from the training drugs.
  Padding is 512 tokens and a drug longer than that is truncated.
- The 1.6.1 model augmented the SMILES by drawing a random equivalent SMILES from
  RDKit at every step. Here ``smilesTokens`` draws a fixed bank of 20 equivalent
  SMILES per drug once, when it is fitted, and each row of a batch picks one of them
  at random in every epoch. The bank and the training draw are seeded, so a fit is
  repeatable, but the bank is not the random stream of the original code. Prediction
  uses the canonical SMILES of the drug as given. A drug seen only at prediction
  time is tokenized with the vocabulary of the training drugs; its unknown tokens
  map to index 1. Without RDKit, or with ``augment: false``, the bank has one
  entry and a warning is emitted.
- The expression is the ``gene_list_paccmann_network_prop`` list with a scaler
  fitted on the training *pairs*, so a sample counts once for every drug it was
  measured with. That is what 1.6.1 did, and the ``fit_on_pairs`` option of
  ``landmarkGenes`` keeps it. The list has 1,955 genes here, not 1,957: the
  re-curated cell-line screens no longer measure ``PRRC2B`` and ``S100A7L2``. The
  features are float32, where 1.6.1 used double precision.
- The predictor stops on the early-stopping scope and restores the weights of its
  best epoch. Without such a scope the full budget of epochs runs and the last
  weights are kept. A trailing training batch of one pair is dropped, because batch
  normalisation cannot process it.

What was checked
----------------

The layers of the neural models were compared with the original code under
identical weights and inputs, and SRMF's solver was compared with the original
algorithm. End-to-end training on the published datasets was not reproduced,
because the changes listed above make exact equality the wrong goal. If you need to
reproduce a published number, use the published code, and use the notes above to
see where the two will diverge.
