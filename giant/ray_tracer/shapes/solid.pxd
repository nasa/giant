


cimport numpy as cnp

from giant.ray_tracer.shapes.shape cimport Shape


cdef class Solid(Shape):

    cdef public:
        cnp.int64_t id
