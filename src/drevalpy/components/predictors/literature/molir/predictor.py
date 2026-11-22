"""MOLIR literature predictor consuming PredictorInput directly."""

from __future__ import annotations

from typing import TYPE_CHECKING

from drevalpy.components.predictors.literature._omics import OMIC_BLOCK_NAMES, SingleDrugOmicsPredictor
from drevalpy.registry._component import LiteratureReference
from drevalpy.registry.predictor import register
from drevalpy.types.data.predictor_input import PredictorInput

# MOLIModel lives in .utils, which imports pytorch_lightning at module scope. This
# module is imported eagerly by register_builtins(), so the import is
# deferred to the methods that construct a model - keeping the training stack off
# the critical path of `import drevalpy`. Guarded by tests/test_import_cost_policy.py.
if TYPE_CHECKING:
    from drevalpy.components.predictors.literature.molir.utils import MOLIModel

_REFERENCE = LiteratureReference(
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


@register(
    "molir",
    description="MOLIR single-drug multi-omics model.",
    reference=_REFERENCE,
    license="GPL-3.0-only",
)
class MOLIRPredictor(SingleDrugOmicsPredictor):
    """MOLIR predictor: per-drug multi-omics late integration model."""

    _models: dict[str, MOLIModel]

    fixed_hyperparameters = {
        "mini_batch": 32,
        "h_dim1": 64,
        "h_dim2": 64,
        "h_dim3": 64,
        "learning_rate": 0.01,
        "dropout_rate": 0.5,
        "weight_decay": 0.0001,
        "gamma": 0.5,
        "epochs": 30,
        "margin": 1.5,
    }

    def _fit_drug(self, drug_id: str, data: PredictorInput) -> MOLIModel:
        """Fit a single-drug MOLIR model.

        :param drug_id: Identifier of the drug to train on.
        :param data: Subset batch for this drug.
        :returns: The model; left untrained when the drug has fewer pairs than one mini-batch.
        """
        widths = tuple(data.sample.blocks[name].values.shape[1] for name in OMIC_BLOCK_NAMES)
        model = self._build_model(widths)

        if len(data) >= self._hyperparameters["mini_batch"]:
            model.fit(data)

        return model

    def _build_model(self, widths: tuple[int, ...]) -> MOLIModel:
        """Construct an untrained MOLIR model sized to the three omic views.

        :param widths: Feature widths of the expression, mutation and CNV views.
        :returns: The initialized model.
        """
        from drevalpy.components.predictors.literature.molir.utils import MOLIModel

        return MOLIModel(
            hpams=dict(self._hyperparameters),
            input_dim_expr=widths[0],
            input_dim_mut=widths[1],
            input_dim_cnv=widths[2],
        )
