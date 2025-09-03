Surface
=======

.. currentmodule:: giant.ray_tracer.shapes.surface

:mod:`giant.ray_tracer.shapes.surface`\:

.. autoclass:: Surface
    :no-members:

    .. attribute:: bounding_box
        :type: AxisAlignedBoundingBox

        The :class:`.AxisAlignedBoundingBox` that fully contains this solid.

    .. attribute:: reference_ellipsoid
        :type: Ellipsoid

        The :class:`.Ellipsoid` that best represents the surface.


.. rubric:: Summary of Methods

.. autosummary::
   :nosignatures:
   :toctree:

   ~Surface.compute_intersect
   ~Surface.compute_limb_jacobian
   ~Surface.find_limbs
   ~Surface.rotate
   ~Surface.trace
   ~Surface.translate
   


|
