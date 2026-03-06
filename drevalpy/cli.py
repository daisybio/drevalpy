"""Main script to run the drug response evaluation pipeline."""

from drevalpy.utils import get_parser, main


def cli_main():
    """Command line interface entry point for the drug response evaluation pipeline."""
    args = get_parser().parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
