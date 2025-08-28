#cython: language_level=3




from giant.ray_tracer.shapes.shape cimport Shape
from giant.ray_tracer.shapes.axis_aligned_bounding_box cimport AxisAlignedBoundingBox
from giant.ray_tracer.shapes.ellipsoid cimport Ellipsoid

cimport numpy as cnp

cdef class Surface(Shape):
    cdef public:
        Ellipsoid reference_ellipsoid

    # functions
    cdef void _compute_intersect(self, const double[:] start, const double[:] direction, const double[:] inv_direction,
                                 const cnp.int64_t[] ignore, const cnp.uint32_t num_ignore,
                                 cnp.uint8_t *hit, double[:] intersect, double[:] normal, double *albedo,
                                 cnp.int64_t *facet, double *hit_distance) noexcept nogil

    cdef void _trace(self, const double[:, :] starts, const double[:, :] directions, const double[:, :] inv_directions,
                     const cnp.int64_t[:, :] ignore, const cnp.uint32_t numrays, const bint omp,
                     cnp.uint8_t[:] hit, double[:, :] intersect, double[:, :] normal, double[:] albedo,
                     cnp.int64_t[:] facet, double[:] hit_distances) noexcept nogil


cdef class RawSurface(Surface):
    cdef readonly:
        cnp.uint32_t[:, :] _facets
        bint _single_albedo


cdef class Surface64(RawSurface):

    cdef readonly:
        double[:, :] _normals
        double[:, :] _vertices
        double[:] _albedo_array
        double _albedo


cdef class Surface32(RawSurface):

    cdef readonly:
        float[:, :] _normals
        float[:, :] _vertices
        float[:] _albedo_array
        float _albedo


