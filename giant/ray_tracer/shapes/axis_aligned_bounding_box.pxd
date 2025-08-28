


cimport numpy as cnp

cdef class AxisAlignedBoundingBox:

    cdef public double[:] _min_sides
    cdef public double[:] _max_sides
    cdef public object _rotation

    # functions
    cdef void _compute_intersect(self, const double[:] start, const double[:] inv_direction, cnp.uint8_t *res,
                                 double *near_distance, double *far_distance) noexcept nogil
    cdef void _trace(self, const double[:, :] starts, const double[:, :] inv_directions, cnp.uint8_t[:] res,
                     double[:, :] distances, cnp.uint32_t numrays) noexcept nogil

