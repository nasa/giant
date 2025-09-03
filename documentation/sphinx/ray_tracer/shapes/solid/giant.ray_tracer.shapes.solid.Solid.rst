Solid
=====

.. currentmodule:: giant.ray_tracer.shapes.solid

:mod:`giant.ray_tracer.shapes.solid`\:

.. autoclass:: Solid
    :no-members:

    .. attribute:: bounding_box
        :type: AxisAlignedBoundingBox

        The :class:`.AxisAlignedBoundingBox` that fully contains this solid.

    .. attribute:: id
        :type: int

        The unique identifier for this object as an integer.

.. rubric:: Summary of Methods

.. autosummary::
   :nosignatures:
   :toctree:

   ~Solid.compute_intersect
   ~Solid.rotate
   ~Solid.trace
   ~Solid.translate
   ~Solid.find_limbs
   ~Solid.compute_limb_jacobian


|
