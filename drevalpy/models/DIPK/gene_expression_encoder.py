"""Compatibility re-export for moved DIPK gene expression encoder."""

from drevalpy.components.predictors.literature.impl.dipk.gene_expression_encoder import (
    CollateFn,
    DataSet,
    GeneExpressionDecoder,
    GeneExpressionEncoder,
    encode_gene_expression,
    train_gene_expession_autoencoder,
)

__all__ = [
    "CollateFn",
    "DataSet",
    "GeneExpressionDecoder",
    "GeneExpressionEncoder",
    "encode_gene_expression",
    "train_gene_expession_autoencoder",
]
