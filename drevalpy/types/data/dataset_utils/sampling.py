"""Hyperparameter sampling utility using Optuna."""

from __future__ import annotations


def _sample_hp_configs(featurizer_cls: type, n: int) -> list[dict]:
    """Sample N hyperparameter configs from a featurizer's HP space using Optuna.

    Respects declared distributions (log-uniform, integer, categorical, etc.).
    """
    import optuna

    from drevalpy.components.core.tuning.search_space import sample_from_optuna_trial

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    space = featurizer_cls.get_hyperparameter_space()
    if not space:
        return [{}] * n

    study = optuna.create_study()
    configs: list[dict] = []
    for _ in range(n):
        trial = study.ask()
        config = sample_from_optuna_trial(trial, space)
        study.tell(trial, 0.0)
        configs.append(config)
    return configs
