"""Raw feature-matrix assembly for hand-rolled ``DRPModel`` subclasses.

Neither method here touches ``_stack``: they read only the view names the resolved
config asks for, and pull those views straight out of a pair of feature sources.
That is what separates them from the fit/predict path in ``_training.py``, which
goes through the component stack and never assembles a matrix itself.

A model built by ``construct_model`` never calls either - the stack featurizes -
so these exist for callers that drive featurization by hand.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class DRPFeatureMatrixMixin:
    """Assemble feature matrices for the views a model's config requires."""

    @property
    def cell_line_views(self) -> list[str]:
        """Return required cell-line views; implemented by ``DRPModel``.

        :raises NotImplementedError: If the subclass does not implement this hook.
        """
        raise NotImplementedError

    @property
    def drug_views(self) -> list[str]:
        """Return required drug views; implemented by ``DRPModel``.

        :raises NotImplementedError: If the subclass does not implement this hook.
        """
        raise NotImplementedError

    def get_concatenated_features(
        self,
        cell_line_view: str | None,
        drug_view: str | None,
        cell_line_ids_output: np.ndarray,
        drug_ids_output: np.ndarray,
        cell_line_input: Any,
        drug_input: Any,
    ) -> np.ndarray:
        """Concatenate selected cell-line and drug feature views into matrix ``X``.

        :param cell_line_view: Cell-line view name, or ``None`` to omit cell-line features.
        :param drug_view: Drug view name, or ``None`` to omit drug features.
        :param cell_line_ids_output: Cell-line identifiers for the output pairs.
        :param drug_ids_output: Drug identifiers for the output pairs.
        :param cell_line_input: Cell-line feature source, or ``None``.
        :param drug_input: Drug feature source, or ``None``.
        :returns: Feature matrix with one row per output pair.
        :raises ValueError: If a requested view is missing from the inputs, or if
            neither side requested one.
        """
        inputs = self.get_feature_matrices(
            cell_line_ids=cell_line_ids_output,
            drug_ids=drug_ids_output,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )
        drug_features = _require_view(inputs, drug_view, "drug_view")
        cell_line_features = _require_view(inputs, cell_line_view, "cell_line_view")

        if cell_line_features is not None and drug_features is not None:
            return np.concatenate((cell_line_features, drug_features), axis=1)
        if cell_line_features is not None:
            return cell_line_features
        if drug_features is not None:
            return drug_features
        raise ValueError("No features provided.")

    def get_feature_matrices(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: Any,
        drug_input: Any,
    ) -> dict[str, np.ndarray]:
        """Return feature matrices for the model's required views.

        :param cell_line_ids: Cell-line identifiers, one per pair.
        :param drug_ids: Drug identifiers, one per pair.
        :param cell_line_input: Cell-line feature source, or ``None``.
        :param drug_input: Drug feature source, or ``None``.
        :returns: Mapping from view name to feature matrix aligned with the ids.
        :raises ValueError: If a required view is missing from the inputs.
        """
        return {
            **_matrices_for_side(cell_line_input, self.cell_line_views, cell_line_ids, "Cell line"),
            **_matrices_for_side(drug_input, self.drug_views, drug_ids, "Drug"),
        }


def _matrices_for_side(
    source: Any,
    views: list[str],
    identifiers: np.ndarray,
    label: str,
) -> dict[str, np.ndarray]:
    """Pull every required view of one entity side out of its feature source.

    :param source: Feature source for the side, or ``None`` to contribute nothing.
    :param views: View names the model config requires from this side.
    :param identifiers: Entity identifiers, one per pair.
    :param label: Side name used in the error message.
    :returns: Mapping from view name to matrix, empty when *source* is ``None``.
    :raises ValueError: If *source* does not carry one of *views*.
    """
    if source is None:
        return {}
    matrices = {}
    for view in views:
        if view not in source.view_names:
            raise ValueError(f"{label} input does not contain view {view}")
        matrices[view] = source.get_feature_matrix(view=view, identifiers=identifiers)
    return matrices


def _require_view(
    inputs: dict[str, np.ndarray],
    view: str | None,
    label: str,
) -> np.ndarray | None:
    """Look up one requested view, insisting it was actually assembled.

    :param inputs: Assembled matrices, keyed by view name.
    :param view: Requested view name, or ``None`` to skip this side.
    :param label: Parameter name used in the error message.
    :returns: The matrix for *view*, or ``None`` when *view* is ``None``.
    :raises ValueError: If *view* was requested but is absent from *inputs*.
    """
    if view is None:
        return None
    if view not in inputs:
        raise ValueError(f"Expected {label} '{view}' to be in inputs, but it was not. Inputs: {inputs}")
    return inputs[view]
