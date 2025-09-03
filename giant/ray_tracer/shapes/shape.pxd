



from giant.ray_tracer.shapes.axis_aligned_bounding_box cimport AxisAlignedBoundingBox


cdef class Shape:
    cdef public:
        AxisAlignedBoundingBox bounding_box

    pass
