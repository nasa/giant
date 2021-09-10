Surface32
=========

.. currentmodule:: giant.ray_tracer.shapes.surface

:mod:`giant.ray_tracer.shapes.surface`\:

.. autoclass:: Surface32
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

   ~Surface32.compute_bounding_box
   ~Surface32.compute_intersect
   ~Surface32.compute_limb_jacobian
   ~Surface32.compute_reference_ellipsoid
   ~Surface32.find_limbs
   ~Surface32.merge
   ~Surface32.rotate
   ~Surface32.trace
   ~Surface32.translate
   


|
