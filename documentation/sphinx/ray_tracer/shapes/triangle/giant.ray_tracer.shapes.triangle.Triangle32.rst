Triangle32
==========

.. currentmodule:: giant.ray_tracer.shapes.triangle

:mod:`giant.ray_tracer.shapes.triangle`\:

.. autoclass:: Triangle32
    :no-members:
    :members: albedos, facets, normals, num_faces, sides, stacked_vertices, vertices

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

   ~Triangle32.compute_bounding_box
   ~Triangle32.compute_intersect
   ~Triangle32.compute_limb_jacobian
   ~Triangle32.compute_normals
   ~Triangle32.compute_reference_ellipsoid
   ~Triangle32.find_limbs
   ~Triangle32.get_albedo
   ~Triangle32.merge
   ~Triangle32.rotate
   ~Triangle32.trace
   ~Triangle32.translate
   


|
