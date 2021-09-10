# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


cimport numpy as cnp

from giant.ray_tracer.shapes.surface cimport Surface64, Surface32


cdef class Triangle64(Surface64):

    # functions
    cdef double _get_albedo(self, const double[3] rhs, const int face) nogil
    cdef void _get_sides(self, const cnp.uint32_t face, double[3] side1, double[3] side2) nogil

cdef class Triangle32(Surface32):

    # functions
    cdef float _get_albedo(self, const double[3] rhs, const int face) nogil
    cdef void _get_sides(self, const cnp.uint32_t face, float[3] side1, float[3] side2) nogil

