"""Build `~drevalpy.models.config.ModelConfig` from a zoo name or a recipe string.

A recipe is turned into a config in two steps: this module reads the syntax into a plain
field mapping, and ``drevalpy.models.config._from_dict.from_dict`` then resolves the names
against the registry. Only the scope is looked up here, since a recipe never spells it out.
"""

from __future__ import annotations

from typing import Any

from drevalpy.components.model_id import parse_model_id
from drevalpy.components.registry import get_predictor
from drevalpy.models.config._from_dict import from_dict
from drevalpy.models.config.model import ModelConfig
from drevalpy.models.config.resolved import ResolvedModelConfig
from drevalpy.types.model_scope import ModelScope
from drevalpy.types.prediction_mode import PredictionMode


def _default_scope_for_predictor(pred_cls: type[Any]) -> ModelScope:
    supported_scopes = getattr(pred_cls, "supported_scopes", None)
    if supported_scopes is not None and len(supported_scopes) == 1:
        return next(iter(supported_scopes))
    return ModelScope.MULTI_DRUG


def _apply_optional_hyperparameters(
    config: ModelConfig,
    hyperparameters: dict[str, Any] | None,
) -> ModelConfig | ResolvedModelConfig:
    """Apply public hyperparameters by returning a resolved config when needed.

    Recipe builders historically returned ``ModelConfig``. When hyperparameters
    are provided, return the resolved object so callers that only need a template
    without overrides still receive ``ModelConfig``, while override paths receive
    ``ResolvedModelConfig`` via the public-flat helper.

    :param config: Template config.
    :param hyperparameters: Optional public overrides.
    :returns: Template or resolved config.
    """
    if not hyperparameters:
        return config
    from drevalpy.components.tuning.public_flat import apply_public_hyperparameters_to_config

    return apply_public_hyperparameters_to_config(config, hyperparameters)


def _recipe_slots(recipe: str) -> tuple[str | None, str | None, str, ModelScope]:
    """Read a recipe's syntax and look up the scope its predictor implies.

    A bare token that names no known predictor is most likely a mistyped zoo name, so that
    one case is reported in terms of both options. Anything with a colon is unambiguously
    meant as a recipe, and keeps the grammar's or the registry's own message.

    :param recipe: ``predictor``, ``cell:predictor``, or ``cell:drug:predictor``.
    :returns: Cell-line slot, drug slot, predictor name, and the implied scope.
    :raises ValueError: If the recipe is malformed or names an unknown predictor.
    :raises ImportError: If a recipe's predictor is registered but its module fails to load.
    """
    try:
        cell_line, drug, predictor = parse_model_id(recipe)
        scope = _default_scope_for_predictor(get_predictor(predictor))
    except (ValueError, ImportError) as exc:
        if ":" in recipe:
            raise
        msg = (
            f"Unknown model spec {recipe!r}. Use a recipe triple "
            "(cellLine:drug:predictor), zoo name, or feature-free predictor token."
        )
        raise ValueError(msg) from exc
    return cell_line, drug, predictor, scope


def _recipe_payload(
    recipe: str,
    *,
    prediction_mode: PredictionMode | str = PredictionMode.REGRESSION,
) -> dict[str, Any]:
    """Turn a recipe string into the field mapping ``from_dict`` expects.

    This is the syntax half of ``from_spec``: it splits the recipe into its slots and picks
    the scope the predictor implies, but leaves the featurizer and predictor slots as recipe
    strings, since their own validators know how to read them. Only the scope needs a
    registry lookup, because it is the one field a recipe never spells out.

    A two-part recipe omits the drug slot; ``ModelConfig`` fills in the identity featurizer
    that single-drug routing requires.

    :param recipe: ``predictor``, ``cell:predictor``, or ``cell:drug:predictor``.
    :param prediction_mode: Regression or classification mode for the predictor.
    :returns: Mapping of ``ModelConfig`` fields, with slots left as recipe strings.
    :raises ValueError: If a two-part recipe names a predictor that is not single-drug.
    """
    cell_line, drug, predictor, scope = _recipe_slots(recipe)
    if cell_line is not None and drug is None and scope != ModelScope.SINGLE_DRUG:
        msg = "two-part recipes require a single-drug predictor"
        raise ValueError(msg)
    return {
        "cell_line_featurizer": cell_line,
        "drug_featurizer": drug,
        "predictor": predictor,
        "prediction_mode": prediction_mode,
        "scope": scope,
    }


def _zoo_config(
    name: str,
    hyperparameters: dict[str, Any] | None,
    prediction_mode: PredictionMode | str,
) -> ModelConfig | ResolvedModelConfig | None:
    """Resolve a registered zoo preset, or report that *name* is not one.

    :param name: Candidate zoo preset name.
    :param hyperparameters: Optional flat public hyperparameter overrides.
    :param prediction_mode: Mode to apply, honoured only when no overrides are given.
    :returns: The preset's config, or ``None`` when *name* is not a zoo entry.
    """
    # Imported lazily: drevalpy.models.factory imports the config package, which imports
    # this module, so a module-scope import here would be circular.
    from drevalpy.models.factory import model_config_for_name

    try:
        # When hyperparameters are given, model_config_for_name resolves them and the
        # requested prediction mode is not applied on that path (historical behaviour).
        return model_config_for_name(
            name,
            hyperparameters,
            prediction_mode=None if hyperparameters else PredictionMode(prediction_mode),
        )
    except KeyError:
        return None


def _build_from_spec(
    spec: str,
    *,
    hyperparameters: dict[str, Any] | None = None,
    prediction_mode: PredictionMode | str = PredictionMode.REGRESSION,
) -> ModelConfig | ResolvedModelConfig:
    """Parse a model specification string.

    A spec is either the name of a registered zoo preset or a recipe naming the parts
    directly. Zoo names win, so a preset can shadow a bare predictor name. Recipes are
    handled in two steps: ``_recipe_payload`` reads the syntax, then ``from_dict`` resolves
    the names against the registry and checks that the combination is legal.

    :param spec: Zoo preset name, or a recipe of one to three colon-separated parts.
    :param hyperparameters: Optional flat public hyperparameter overrides.
    :param prediction_mode: Regression or classification mode for the predictor.
    :returns: Validated ``ModelConfig`` instance, or ``ResolvedModelConfig`` when
        *hyperparameters* are provided.
    :raises ValueError: If ``spec`` is unknown or validation fails.
    """
    trimmed = spec.strip()
    if not trimmed:
        msg = "model spec must be a non-empty string"
        raise ValueError(msg)

    zoo_config = _zoo_config(trimmed, hyperparameters, prediction_mode)
    if zoo_config is not None:
        return zoo_config

    payload = _recipe_payload(trimmed, prediction_mode=prediction_mode)
    config = from_dict(payload, source=f"recipe {trimmed!r}")
    return _apply_optional_hyperparameters(config, hyperparameters)
