"""Main evaluation pipeline entry point."""

from __future__ import annotations


def main(args) -> None:
    """Run the drug response evaluation pipeline.

    :param args: Parsed command-line arguments for the evaluation pipeline.
    """
    from .validation import validate_models, validate_test_modes

    validate_models(args)
    validate_test_modes(args)

    from drevalpy.data import load
    from drevalpy.experiment.run import mu_experiment
    from drevalpy.models._model_lookup import get_model_class

    from .response_transform import get_response_transformation

    mudataset = load(args.dataset_name)

    models = [get_model_class(model) for model in args.models]
    baselines = [get_model_class(b) for b in args.baselines] if args.baselines else []

    response_transformation = get_response_transformation(args.response_transformation)

    for test_mode in args.test_mode:
        mu_experiment(
            models=models,
            mudataset=mudataset,
            dataset_name=args.dataset_name,
            baselines=baselines,
            response_transformation=response_transformation,
            run_id=args.run_id,
            test_mode=test_mode,
            hpam_optimization_metric=args.optim_metric,
            n_cv_splits=args.n_cv_splits,
            path_out=args.path_out,
            overwrite=args.overwrite,
            model_checkpoint_dir=args.model_checkpoint_dir,
            hyperparameter_tuning=not args.no_hyperparameter_tuning,
            wandb_project=args.wandb_project,
            hpo_num_samples=getattr(args, "hpo_num_samples", 16),
            hpo_random_state=getattr(args, "hpo_random_state", 42),
            hpo_resources_per_trial=getattr(args, "hpo_resources_per_trial", None),
        )
