# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


cimport numpy as cnp

from giant.ray_tracer.shapes.shapes cimport AxisAlignedBoundingBox, Ellipsoid, Shape
from giant.ray_tracer.shapes.surface cimport RawSurface, Surface

cdef class KDNode:

    cdef:
        double[:, :] _centers
        cnp.uint32_t _order

    cdef public:
        KDNode left
        KDNode right
        AxisAlignedBoundingBox bounding_box
        RawSurface surface
        bint has_surface
        cnp.uint32_t id
        cnp.uint32_t id_order

    cdef void _compute_intersect(self, const double[:] start, const double[:] direction, const double[:] inv_direction,
                                  const cnp.int64_t[] ignore, const cnp.uint32_t num_ignore, cnp.int64_t[] shape_ignore,
                                  cnp.uint8_t *hit, double[:] intersect, double[:] normal, double *albedo,
                                  cnp.int64_t *facet, double *previous_hit_distance) noexcept nogil

    cpdef compute_bounding_box(self)

        
cdef class KDTree(Surface):
    cdef public:
        int max_depth
        KDNode root
        RawSurface surface

    cdef:
        object _rotation
        double[:] _position
        cnp.int64_t[:, :] _shape_ignore

cpdef list get_ignore_inds(KDNode node, size_t vertex_id)

cpdef cnp.ndarray get_facet_vertices(KDNode node, size_t facet_id)
