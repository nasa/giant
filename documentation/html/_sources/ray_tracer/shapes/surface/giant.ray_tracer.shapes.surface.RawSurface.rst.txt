RawSurface
==========

.. currentmodule:: giant.ray_tracer.shapes.surface

:mod:`giant.ray_tracer.shapes.surface`\:

.. autoclass:: RawSurface
    :no-members:
    :members: albedos, facets, normals, num_faces, stacked_vertices, vertices

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

   ~RawSurface.compute_bounding_box
   ~RawSurface.compute_intersect
   ~RawSurface.compute_limb_jacobian
   ~RawSurface.compute_reference_ellipsoid
   ~RawSurface.find_limbs
   ~RawSurface.merge
   ~RawSurface.rotate
   ~RawSurface.trace
   ~RawSurface.translate
   


|
