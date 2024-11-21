"""Decorator to mark a function as a pipeline function."""


def pipeline_function(func):
    """
    Decorator to mark a function as a pipeline function.

    :param func: function to decorate
    :return: function with custom attribute
    """
    func.is_pipeline_function = True  # Adds a custom attribute to the function
    return func
