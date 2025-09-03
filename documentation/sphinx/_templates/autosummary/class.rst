{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

:mod:`{{ module }}`\:

.. autoclass:: {{ objname}}
    :no-members:
    {% if attributes -%}
    :members: {{ attributes | join(', ') }}
    {% endif %}

{% if methods %}
.. rubric:: Summary of Methods

.. autosummary::
   :nosignatures:
   :toctree:

   {% for item in methods | reject("==", "__init__") -%}
   ~{{ name }}.{{ item }}
   {% endfor %}
{% endif %}

|

