


from giant.ray_tracer.shapes.solid cimport Solid
from giant.ray_tracer.shapes.axis_aligned_bounding_box cimport AxisAlignedBoundingBox

cimport numpy as cnp


cdef class Ellipsoid(Solid):


    cdef:
        double[:] _center
        double[:, :] _orientation
        double[:, :] _ellipsoid_matrix

    cdef public:
        double[:] _principal_axes
        object albedo_map

