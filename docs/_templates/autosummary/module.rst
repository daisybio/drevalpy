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
{%- if not item.split('.')[-1].startswith('_') and item.split('.')[-1] != 'custom_splits' %}
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
