.. Template for a pure re-export module: documents the members but registers no
.. index entries, so aliasing a class does not make every unqualified reference
.. to it ambiguous. Selected with ``:template: facade.rst`` on an autosummary
.. directive. It lives at the root of ``templates_path`` rather than under
.. ``autosummary/`` because that is where the ``:template:`` option is resolved
.. from; a copy in the subdirectory is silently ignored in favour of
.. ``autosummary/base.rst``.

{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource
   :no-index:
