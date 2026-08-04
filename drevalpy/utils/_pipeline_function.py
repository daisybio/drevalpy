"""Decorator to mark a function as a pipeline function."""


def pipeline_function(func):
    """Mark a function as part of the evaluation pipeline.

    Args:
        func: Callable to decorate.

    Returns:
        The same callable with ``is_pipeline_function`` set to ``True``.
    """
    func.is_pipeline_function = True  # Adds a custom attribute to the function
    return func


pipeline_function.__module__ = "drevalpy.utils"
