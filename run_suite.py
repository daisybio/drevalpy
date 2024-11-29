"""Main script to run the drug response evaluation pipeline."""

from drevalpy.utils import get_parser, main

if __name__ == "__main__":
    # PIPELINE: PARAMS_CHECK
    arguments = get_parser().parse_args()
    main(arguments)
