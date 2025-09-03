Surface64
=========

.. currentmodule:: giant.ray_tracer.shapes.surface

:mod:`giant.ray_tracer.shapes.surface`\:

.. autoclass:: Surface64
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

   ~Surface64.compute_bounding_box
   ~Surface64.compute_intersect
   ~Surface64.compute_limb_jacobian
   ~Surface64.compute_normals
   ~Surface64.compute_reference_ellipsoid
   ~Surface64.find_limbs
   ~Surface64.merge
   ~Surface64.rotate
   ~Surface64.trace
   ~Surface64.translate
   


|
