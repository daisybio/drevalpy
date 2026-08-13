"""Runnable plugin examples that the extensions guide includes verbatim.

Every module here is a real, importable plugin component. The docs build imports
all of them and runs the conformance checks over them before rendering a single
page, and ``docs/python/extensions.rst`` pulls each file in with
``literalinclude``. So the code on the page is code that ran: an example that
stops working fails ``sphinx-build`` instead of quietly rotting.

Importing a module here executes its ``@register_*`` decorator and mutates the
process-wide registries, which is why nothing imports this package implicitly.
The docs build goes through ``docs/_examples.py``, which rolls the registries
back afterwards, and ``tests/docs/test_examples.py`` does the import in a
subprocess so drevalpy's own registry-count assertions stay unaffected.
"""
