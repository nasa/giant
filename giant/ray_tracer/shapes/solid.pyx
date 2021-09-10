# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
This cython module defines the base class for all solids in GIANT.

Description
-----------

A solid is considered a 3D object that can be represented without resorting to tesselation.

Use
---

In general users do not need to concern themselves with this module unless you are doing development work.
"""


from giant.ray_tracer.shapes.shape cimport Shape


cdef class Solid(Shape):
    """
    A solid represents an 3D object that can be mathematically represented without resorting to tesselation (for
    instance a tri-axial ellipsoid).

    This is essentially an abstract base class that is primarily used to identify what are solids in GIANT.  It only
    adds an :attr:`id` attribute above the standard :class:`.Shape` base class from which it is defined.

    In general a user should not have to interact with this class unless they are defining a new solid for GIANT, in
    which case they should subclass this class and ensure that they set the :attr:`id` attribute.
    """

    pass
