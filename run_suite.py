from drevalpy.datasets import RESPONSE_DATASET_FACTORY
from drevalpy.experiment import drug_response_experiment
from drevalpy.models import MODEL_FACTORY
from drevalpy.utils import get_parser, check_arguments, get_response_transformation


if __name__ == "__main__":
    args = get_parser().parse_args()
    check_arguments(args)
    models = [MODEL_FACTORY[model] for model in args.models]

    if args.baselines is not None:
        baselines = [MODEL_FACTORY[baseline] for baseline in args.baselines]
    else:
        baselines = []
    response_data = RESPONSE_DATASET_FACTORY[args.dataset_name]()
    cross_study_datasets = [
        RESPONSE_DATASET_FACTORY[dataset]() for dataset in args.cross_study_datasets
    ]
    # TODO Allow for custom randomization tests maybe via config file

    if args.randomization_mode[0] == "None":
        args.randomization_mode = None
    response_transformation = get_response_transformation(args.response_transformation)

    for test_mode in args.test_mode:
        drug_response_experiment(
            models=models,
            baselines=baselines,
            response_data=response_data,
            response_transformation=response_transformation,
            metric=args.optim_metric,
            n_cv_splits=args.n_cv_splits,
            multiprocessing=args.multiprocessing,
            test_mode=test_mode,
            randomization_mode=args.randomization_mode,
            randomization_type=args.randomization_type,
            n_trials_robustness=args.n_trials_robustness,
            cross_study_datasets=cross_study_datasets,
            path_out=args.path_out,
            run_id=args.run_id,
            overwrite=args.overwrite,
        )

    # TODO now do evaluation, visualization, etc.
