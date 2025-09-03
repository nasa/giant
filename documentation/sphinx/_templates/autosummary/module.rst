{{ objname | escape | underline }}

.. automodule:: {{ fullname }}
    :no-members:

.. currentmodule:: {{ fullname }}

{% if modules %}
.. rubric:: Modules

.. autosummary::
    :nosignatures:
    :toctree: {{ name }}
    :recursive:

    {% for item in modules -%}
    {{ item.split('.')[-1] }}
    {% endfor %}

{% endif %}
{% if classes %}
.. rubric:: Classes

.. autosummary::
    :nosignatures:
    :toctree: {{ name }}

    {% for item in classes -%}
    {{ item.split('.')[-1] }}
    {% endfor %}

{% endif %}
{% if functions %}
.. rubric:: Function

.. autosummary::
    :nosignatures:
    :toctree: {{ name }}

    {% for item in functions -%}
    {{ item.split('.')[-1] }}
    {% endfor %}

{% endif %}
{% if attributes %}
.. rubric:: Constants

.. autosummary::
    :nosignatures:
    :toctree: {{ name }}

    {% for item in attributes -%}
    {{ item.split('.')[-1] }}
    {% endfor %}

{% endif %}

|

