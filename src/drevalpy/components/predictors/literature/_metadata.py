"""Shared literature references for literature predictors."""

from __future__ import annotations

from drevalpy.types.enums.literature_reference import LiteratureReference

DRUGGNN_REFERENCE = LiteratureReference(
    repo_url="https://github.com/hauldhut/GraphDRP",
    citation_doi="10.1109/TCBB.2021.3060430",
    citation_text="DrugGNN-style GCN on molecular graphs with dense sample features (GraphDRP codebase).",
    deviations=(
        "GraphDRP GCN variant: retain its three graph convolutions; use mean pooling, a dense sample "
        "branch and an unbounded regression head instead of max pooling, Conv1d and sigmoid."
    ),
)

PRECILY_REFERENCE = LiteratureReference(
    repo_url="https://github.com/SmritiChawla/Precily",
    citation_doi="10.1038/s41467-022-33291-z",
    citation_text="Precily pathway and SMILESVec drug response model.",
    deviations=(
        "Hidden widths match the published pathways Model1 checkpoint. Use one regressor instead of the "
        "five-model ensemble, fold-specific GSVA and configurable drug features; PyTorch "
        "initialization/optimizer defaults differ from Keras."
    ),
)

SRMF_REFERENCE = LiteratureReference(
    repo_url="https://github.com/linwang1982/SRMF",
    citation_doi="10.1186/s12885-017-3500-5",
    citation_text="Similarity-regularized matrix factorization for drug response prediction.",
    deviations=(
        "Alternating updates and best-objective factor selection match the MATLAB reference with shared "
        "initial factors. Retain the regression wrapper's response units and configured feature "
        "similarities instead of the MATLAB demonstration's max-abs scaling and fixed matrices. "
        "NumPy/MATLAB RNG streams differ; unseen entities use a mean-response fallback absent upstream."
    ),
)

MOLIR_REFERENCE = LiteratureReference(
    repo_url="https://github.com/kramerlab/Multi-Omics_analysis",
    citation_doi="10.1186/s12859-023-05166-7",
    citation_text=(
        "MOLIR is a regression adaptation of MOLI (Sharifi-Noghabi et al., Bioinformatics 2019, "
        "doi:10.1093/bioinformatics/btz318); the DOI above is the Hauptmann and Kramer benchmark that "
        "compares MOLI and Super.FELT."
    ),
    deviations=(
        "Regression adaptation of kramerlab Multi-Omics_analysis MOLI: preserve encoders and late "
        "integration, replace classification sigmoid/BCE with continuous outputs/MSE and response-based "
        "triplet sampling."
    ),
)

SUPERFELTR_REFERENCE = LiteratureReference(
    repo_url="https://github.com/DMCB-GIST/Super.FELT",
    citation_doi="10.1186/s12859-023-05166-7",
    citation_text=(
        "SuperFELTR is a regression adaptation of Super.FELT (Park et al., BMC Bioinformatics 2021, "
        "doi:10.1186/s12859-021-04146-z); the DOI above is the Hauptmann and Kramer benchmark that "
        "compares MOLI and Super.FELT."
    ),
    deviations=(
        "Regression adaptation: preserve supervised omics encoders and triplet training; use continuous "
        "response triplets and MSE regression without the binary classifier sigmoid/BCE."
    ),
)

PHARMAFORMER_REFERENCE = LiteratureReference(
    repo_url="https://github.com/zhouyuru1205/PharmaFormer",
    citation_doi="10.1038/s41698-025-01082-6",
    citation_text="PharmaFormer integrates gene expression and compound views via a transformer encoder.",
    deviations=(
        "Architecture follows upstream pretrain.py, not its extra-dropout finetuning variant. Scale "
        "dataset-provided expression without an additional log2 transform, learn fold-specific BPE, "
        "and stop on disjoint validation instead of training loss."
    ),
)

DIPK_REFERENCE = LiteratureReference(
    repo_url="https://github.com/user15632/DIPK",
    citation_doi="10.1093/bib/bbae153",
    citation_text="DIPK deep integration model with BIONIC and MolGNet features.",
    deviations=(
        "Preserve the upstream autoencoder architecture, 20% masking and final-epoch encoder return. Add "
        "disjoint validation stopping to autoencoder/predictor; predictor restores best validation "
        "weights. Real MolGNet/PPI artifacts are required for raw-input integration."
    ),
)

SPARSEGO_REFERENCE = LiteratureReference(
    repo_url="https://github.com/KatynaSada/SparseGO_lightning",
    citation_doi="10.1016/j.ebiom.2023.104767",
    citation_text="SparseGO visible neural network structured by the Gene Ontology hierarchy.",
    deviations=(
        "Preserve ontology-indexed sparse connections. Align named input genes explicitly; train on "
        "training-fold features. Local fixed-epoch MSE training differs from upstream correlation-based "
        "checkpoint winners."
    ),
)
