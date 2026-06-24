"""Shared registry metadata for literature predictors."""

from __future__ import annotations

LITERATURE_INTEGRATION_DEVIATIONS = (
    "Modular drevalpy port with explicit featurizers; trainable encoders remain in the "
    "predictor. Preprocessing, tensor layout, and default hyperparameters may differ from "
    "reference repository scripts."
)

DRUGGNN_METADATA = {
    "category": "literature",
    "template_repo_url": "https://github.com/hauldhut/GraphDRP",
    "citation_text": ("DrugGNN-style GCN on molecular graphs with dense cell-line features (GraphDRP codebase)."),
    "deviations": LITERATURE_INTEGRATION_DEVIATIONS,
}

PRECIILY_METADATA = {
    "category": "literature",
    "template_repo_url": "https://github.com/SmritiChawla/Precily",
    "citation_text": "Precily pathway and SMILESVec drug response model.",
    "deviations": LITERATURE_INTEGRATION_DEVIATIONS,
}

SRMF_METADATA = {
    "category": "literature",
    "template_repo_url": "https://github.com/linwang1982/SRMF",
    "citation_text": "Similarity-regularized matrix factorization for drug response prediction.",
    "deviations": LITERATURE_INTEGRATION_DEVIATIONS,
}

MOLIR_METADATA = {
    "category": "literature",
    "template_repo_url": "https://github.com/hosseinshn/MOLI",
    "citation_doi": "10.1186/s12859-023-05166-7",
    "citation_text": "Multi-omics late integration regression (MOLIR / MOLI family).",
    "deviations": LITERATURE_INTEGRATION_DEVIATIONS,
}

SUPERFELTR_METADATA = {
    "category": "literature",
    "template_repo_url": "https://github.com/DMCB-GIST/Super.FELT",
    "citation_doi": "10.1186/s12859-023-05166-7",
    "citation_text": "SuperFELTR multi-omics feature extraction and late integration model.",
    "deviations": LITERATURE_INTEGRATION_DEVIATIONS,
}

PHARMAFORMER_METADATA = {
    "category": "literature",
    "template_repo_url": "https://github.com/zhouyuru1205/PharmaFormer",
    "citation_doi": "10.1038/s41698-025-01082-6",
    "citation_text": ("PharmaFormer integrates gene expression and compound views via a transformer encoder."),
    "deviations": LITERATURE_INTEGRATION_DEVIATIONS,
}

DIPK_METADATA = {
    "category": "literature",
    "template_repo_url": "https://github.com/user15632/DIPK",
    "citation_text": "DIPK deep integration model with BIONIC and MolGNet features.",
    "deviations": LITERATURE_INTEGRATION_DEVIATIONS,
}
