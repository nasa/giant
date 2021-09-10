# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.



from giant.ray_tracer.shapes.axis_aligned_bounding_box cimport AxisAlignedBoundingBox


cdef class Shape:
    cdef public:
        AxisAlignedBoundingBox bounding_box

    pass
