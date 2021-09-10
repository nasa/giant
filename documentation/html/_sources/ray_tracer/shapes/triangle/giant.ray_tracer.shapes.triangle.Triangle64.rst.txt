Triangle64
==========

.. currentmodule:: giant.ray_tracer.shapes.triangle

:mod:`giant.ray_tracer.shapes.triangle`\:

.. autoclass:: Triangle64
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

   ~Triangle64.compute_bounding_box
   ~Triangle64.compute_intersect
   ~Triangle64.compute_limb_jacobian
   ~Triangle64.compute_normals
   ~Triangle64.compute_reference_ellipsoid
   ~Triangle64.find_limbs
   ~Triangle64.get_albedo
   ~Triangle64.merge
   ~Triangle64.rotate
   ~Triangle64.trace
   ~Triangle64.translate
   


|
