{{ fullname | escape | underline}}

{% if modules %}
.. automodule:: {{ fullname }}
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource
   :no-index:

.. rubric:: Submodules

.. autosummary::
   :toctree:
   :recursive:
{% for item in modules %}
{%- if not item.split('.')[-1].startswith('_') %}
   {{ item }}
{%- endif %}
{%- endfor %}
{% else %}
.. automodule:: {{ fullname }}
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource
{% endif %}
