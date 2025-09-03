{{ name | escape | underline }}

.. currentmodule:: {{ module }}

:mod:`{{ module }}`

.. automodule:: {{ module }}
    :no-members:

.. rubric:: Usage

.. argparse::
    :module: {{ module }}
    :func: _get_parser
    :prog: {{ name }}

{% if classes %}
.. rubric:: Classes

.. autosummary::
    :nosignatures:

    {% for item in classes -%}
    {{ item }}
    {% endfor %}

{% endif %}
{% if functions|length > 1 %}
.. rubric:: Function

.. autosummary::
    :nosignatures:

    {% for item in functions | reject("==", "main") -%}
    {{ item }}
    {% endfor %}

{% endif %}
{% if attributes %}
.. rubric:: Constants

.. autosummary::
    :nosignatures:

    {% for item in attributes -%}
    {{ item }}
    {% endfor %}

{% endif %}

|

