Ellipsoid
=========

.. currentmodule:: giant.ray_tracer.shapes.ellipsoid

:mod:`giant.ray_tracer.shapes.ellipsoid`\:

.. autoclass:: Ellipsoid
    :no-members:
    :members: center, ellipsoid_matrix, orientation, principal_axes
    
    .. attribute:: bounding_box
        :type: AxisAlignedBoundingBox

        The :class:`.AxisAlignedBoundingBox` that fully contains this Ellipsoid.

    .. attribute:: id
        :type: int

        The unique identifier for this ellipsoid as an integer.

    .. attribute:: albedo_map
        :type: Optional[Callable[[numpy.ndarray], numpy.ndarray]]

        The albedo map for the ellipsoid as a callable object which takes in an nx2 array of latitude/longitude in
        radians and returns a length n array of the albedo values for each point.

        Typically this is an instance of :class:`scipy.interpolate.RegularGridInterpolator` or similar.

.. rubric:: Summary of Methods

.. autosummary::
   :nosignatures:
   :toctree:

   ~Ellipsoid.compute_albedos
   ~Ellipsoid.compute_bounding_box
   ~Ellipsoid.compute_intersect
   ~Ellipsoid.compute_limb_jacobian
   ~Ellipsoid.compute_normals
   ~Ellipsoid.find_limbs
   ~Ellipsoid.intersect
   ~Ellipsoid.rotate
   ~Ellipsoid.trace
   ~Ellipsoid.translate
   


|
