# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
This cython module defines the :class:`.AxisAlignedBoundingBox` used internally by the GIANT cpu ray tracer for
acceleration purpose.

Use
---

Typically a user doesn't directly interact with the objects in this module, since it is primarily used internally by the
ray tracer.  There are a few use cases though, as documented in the class documentation.

Note that you cannot render an :class:`.AxisAlignedBoundingBox` in an image because the returns from the
:meth:`.AxisAlignedBoundingBox.trace` method are not what is required.  If you need to render a box consider using
triangles to form the sides (the :attr:`.AxisAlignedBoundingBox.vertices` attribute can be useful for this purpose).
"""

from copy import copy

from typing import Union, Tuple, Optional

import cython

import numpy as np
cimport numpy as cnp

from giant.rotations import Rotation

from giant.ray_tracer.rays import Rays
from giant._typing import ARRAY_LIKE


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:, :] _min_max_to_bounding_box(const double[:] min_sides, const double[:] max_sides):
    """
    This internal c function converts minimum/maximum values for each axis into the vertices of the bounding box.

    The returned result is a 3x8 array of doubles, where the column 0 is the origin (min) of the AABB, column 1
    is +x, column 2 is +y, column 3 is +z, column 4 is +xy, column 5 is +yz, column 6 is +xz, and column 7 is +xyz

    These will be returned in the un-rotated frame of the bounding box.  To get the vertices in the current frame see
    :attr:`.AxisAlignedBoundingBox.vertices`.
    """

    cdef:
        double[:] distance = np.zeros(3, dtype=np.float64)
        size_t i
        double[:, :] out = np.zeros((3, 8), dtype=np.float64)

    for i in range(3):
        # compute the width of the bounding box in each axis
        distance[i] = max_sides[i] - min_sides[i]

        # set initial location for all vertices
        out[i, :] = min_sides[i]
        out[i, :] = min_sides[i]
        out[i, :] = min_sides[i]

    # v1 (x)
    out[0, 1] += distance[0]
    # v2 (y)
    out[1, 2] += distance[1]
    # v3 (z)
    out[2, 3] += distance[2]
    # v4 (xy)
    out[0, 4] += distance[0]
    out[1, 4] += distance[1]
    # v5 (yz)
    out[1, 5] += distance[1]
    out[2, 5] += distance[2]
    # v6 (xz)
    out[0, 6] += distance[0]
    out[2, 6] += distance[2]
    # v7 (xyz)
    out[0, 7] += distance[0]
    out[1, 7] += distance[1]
    out[2, 7] += distance[2]

    return out


def min_max_to_bounding_box(min_sides, max_sides):
    """
    min_max_to_bounding_box(min_sides, max_sides)

    This function provides a python interface to the internal c function.

    It determines the vertices of the bounding box given the min/max values for each axis.

    The vertices are returned as an array of shape 3x6 where each column represents a new vertex. column 0 is the origin
    (min) of the AABB, column 1 is +x, column 2 is +y, column 3 is +z, column 4 is +xy, column 5 is +yz,
    column 6 is +xz, and column 7 is +xyz.

    These will be returned in the un-rotated frame of the bounding box.  To get the vertices in the current frame see
    :attr:`.AxisAlignedBoundingBox.vertices`.

    :param min_sides: the minimum sides of the bounding box as a length 3 array
    :type min_sides: np.ndarray
    :param max_sides: the maximum sides of the bounding box as a length 3 array
    :type max_sides: np.ndarray
    :return: a 3x8 array of the vertices of the bounding box in the translated but not rotated frame.  The data type
             will always be double precision
    :rtype: np.ndarray
    """

    min_sides = np.asarray(min_sides).ravel().astype(np.float64)
    max_sides = np.asarray(max_sides).ravel().astype(np.float64)

    return np.asarray(_min_max_to_bounding_box(min_sides, max_sides))


cdef class AxisAlignedBoundingBox:
    """
    __init__(self, min_sides, max_sides, _rotation=None)

    This class provides an efficient implementation of an axis aligned bounding box.

    An axis aligned bounding box is defined by 6 floats plus a rotation.  The first 3 floats specify the minimum values
    of the contained data for each axis (x, y, z) while the second 3 specify the maximum values for each axis.  The
    rotation defines the rotation from the current frame to the original frame the bounding box was built for, since the
    axis aligned bounding box is only valid in the original frame.

    The benefit to using an axis aligned bounding box is that it is very efficient to check intersection with a ray,
    which makes it a good choice for building acceleration structures (as is done in the :class:`.KDTree`).

    Typically a user doesn't create an AABB by themselves, especially since an AABB is not actually renderable (that is,
    you can't use it as a target in a scene).  Instead, each object has its own ``bounding_box`` attribute which is used
    internally by the ray tracer.

    To trace the bounding box, simply use the :meth:`.compute_intersect` or :meth:`trace` methods.  These will check
    whether the provided rays hit the bounding box, as well as the minimum/maximum distance from the ray origin that the
    intersection occurs at (if requested).  Note that this is different than other shapes in the ray tracer, hence why it
    can't be used for rendering.

    Additionally, users may find use for the :attr:`vertices` attribute.  This gives the 8 vertices of the corners of
    the AABB in the current frame, which can be useful for determining which pixels actually need to be traced for a
    given object.
    """

    def __init__(self, min_sides, max_sides, _rotation=None):
        """
        :param min_sides: A length 3 array specifying the minimum values for each axis.
        :type min_sides: ARRAY_LIKE
        :param max_sides: A length 3 array specifying the maximum values for each axis.
        :type max_sides: ARRAY_LIKE
        :param _rotation: The current rotation of the bounding box (primarily used for pickling/unpickling)
        :type _rotation: Optional[Rotation]
        """

        self._min_sides = np.zeros(3, dtype=np.float64)
        self._max_sides = np.zeros(3, dtype=np.float64)

        self.min_sides = min_sides
        self.max_sides = max_sides
        self._rotation = _rotation


    def __reduce__(self):
        """
        __reduce__(self)

        This method is used to pickle the object for saving to a file.

        The return gives the class object, and the arguments to the constructor as min_sides, max_sides, _rotation as
        expected by pickle.

        :return: The class object, and a tuple of the min_sides, max_sides, and rotation
        :rtype: Tuple[AxisAlignedBoundingBox, Tuple[np.ndarray, np.ndarray, Optional[Rotation]]]
        """

        return self.__class__, (self.min_sides, self.max_sides, self._rotation)

    @property
    def min_sides(self):
        """
        The minimum axes of the axis aligned bounding box in the translated (but not rotated) frame as a numpy array
        of length 3 [x, y, z] as double numpy array.
        """

        return np.asarray(self._min_sides)

    @min_sides.setter
    def min_sides(self, val):

        # make sure an array, flat, and double
        self._min_sides = np.asarray(val).ravel().astype(np.float64)

    @property
    def max_sides(self):
        """
        The maximum axes of the axis aligned bounding box in the translated (but not rotated) frame as a numpy array
        of length 3 [x, y, z] as double numpy array.
        """

        return np.asarray(self._max_sides)

    @max_sides.setter
    def max_sides(self, val):

        # make sure an array, flat, and double
        self._max_sides = np.asarray(val).ravel().astype(np.float64)

    @property
    def vertices(self):
        """
        This property provides the vertices of the bounding box in the current frame (rotated/translated) as a double
        numpy array.

        The vertices are returned as an array of shape 3x6 where each column represents a new vertex. column 0 is the
        origin (min) of the AABB, column 1 is +x, column 2 is +y, column 3 is +z, column 4 is +xy, column 5 is +yz,
        column 6 is +xz, and column 7 is +xyz.

        This is done by a call to :func:`.min_max_to_bounding_box` and then rotating into the current frame using the
        :attr:`._rotation` attribute.
        """

        # convert the min max to bounding box vertices
        verts = np.asarray(_min_max_to_bounding_box(self._min_sides, self._max_sides))

        if self._rotation is not None:
            # rotate the vertices if we need to
            verts = self._rotation.matrix.T @ verts

        return verts

    def __eq__(self, other):
        """
        __eq__(self, other)

        This method compares one bounding box to another, checking for equality.

        This is primarily just used for testing purposes.

        :param other: The other bounding box to consider
        :type other: Optional[AxisAlignedBoundingBox]
        :return: True if the bounding boxes contain equivalent data, False otherwise
        :rtype: bool
        """

        if other is None:
            return False

        if not np.allclose(self.min_sides, other.min_sides):
            return False

        if not np.allclose(self.max_sides, other.max_sides):
            return False

        if (self._rotation is not None) and (other._rotation is None):
            return False

        if (self._rotation is None) and (other._rotation is not None):
            return False

        if (self._rotation is not None) and (other._rotation is not None) and not np.allclose(self._rotation.matrix,
                                                                                              other._rotation.matrix):
            return False

        return True

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _compute_intersect(self, const double[:] start, const double[:] inv_direction,
                                 cnp.uint8_t *res, double *near_distance, double *far_distance) nogil:
        """
        This c method is used to compute the intersect of a single ray with this bounding box.

        We assume that the ray has already been rotated into the correct frame.

        We check the intersection and distances using
        https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection

        Arguments start and inv_direction are inputs.  Arguments res, near_distance, and far_distance are outputs.
        Argument res is an int* because cython doesn't (didn't?) like bint pointers.
        """

        cdef int i

        cdef double t1, t2, tmin, tmax

        # algorithm based off of
        # https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
        t1 = (self._min_sides[0] - start[0]) * inv_direction[0]
        t2 = (self._max_sides[0] - start[0]) * inv_direction[0]

        tmin = min(t1, t2)
        tmax = max(t1, t2)

        for i in range(1, 3):
            t1 = (self._min_sides[i] - start[i]) * inv_direction[i]
            t2 = (self._max_sides[i] - start[i]) * inv_direction[i]

            tmin = max(tmin, min(t1, t2))
            tmax = min(tmax, max(t1, t2))

        res[0] =  tmax >= max(tmin, 0)
        near_distance[0] = tmin
        far_distance[0] = tmax

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _trace(self, double[:, :] starts, double[:, :] inv_directions,
                     cnp.uint8_t[:] res, double[:, :] distances, cnp.uint32_t num_rays) nogil:
        """
        This c method checks multiple rays to see if they intersect with this box by making multiple calls to
        the _compute_intersect method.

        We do not use parallel here because it might be called as part of another parallel block.

        We also assume that the ray has already been rotated into the correct frame.

        Arguments starts, inv_directions, and num_rays are inputs.  Arguments res, and distances are outputs.
        Argument res is an int[:] because cython doesn't (didn't?) like bint memory views.
        """

        cdef cnp.uint32_t ray

        for ray in range(num_rays):

            self._compute_intersect(starts[:, ray], inv_directions[:, ray], &res[ray],
                                    &distances[ray, 0], &distances[ray, 1])

    def trace(self, rays, return_distances = False):
        """
        trace(self, rays, return_distances = False)

        This python method traces multiple rays with the AABB and returns the results as a boolean array (True if
        intersected) as well as the minimum and maximum distance to the intersect for each ray (if requested).

        This is done by making a call to the c version of this method.

        If this box has been rotated a copy of the rays are first rotated into the box's frame and then traced (because
        AABB only works when it is in the original frame).  Once we have the rays in the AABB frame, we then use the
        equations from https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
        to both check if the bounding box is intersected, and to get the minimum intersect distance.  The results are
        returned as 1 or 2 numpy arrays, where the first is a boolean array with ``True`` for each element where the
        corresponding ray struck the surface and ``False`` otherwise and the optional second (if
        ``return_distances=True``) is a 2D array of doubles where the first column has the closest intersect distance,
        and the second column has the further intersect distance.  These numbers are only valid when the corresponding
        element in the hit check array (the first return) is ``True``.

        :param rays: The rays to trace through the bounding box.
        :type rays: Rays
        :return: The boolean array specifying whether the ray hit the bounding box and optionally the 2D double array
                 containing the near/far distance of the intersect for each ray (only valid when the ray actually hits
                 the box)
        :rtype: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
        """

        if self._rotation is not None:
            rays = copy(rays)
            rays.rotate(self._rotation)

        # extracted the required components of the rays
        start = rays.start.reshape(3, -1)
        inverse_direction = rays.inv_direction.reshape(3, -1)

        # initialize the results array as an integer array
        res = np.zeros(rays.num_rays, dtype=np.uint8)

        # the near/far distances array
        distances = np.zeros((rays.num_rays, 2), dtype=np.float64)

        # do the tracing
        self._trace(start, inverse_direction, res, distances, rays.num_rays)

        if return_distances:
            # convert the results array to bool and return
            return res.astype(bool), distances
        else:
            return res.astype(bool)

    def compute_intersect(self, ray, return_distances=False):
        """
        compute_intersect(self, ray, return_distances = False)

        This python method traces a single ray with the AABB and returns the results as a boolean value (True if
        intersected) as well as the minimum and maximum distance to the intersect as a length 2 double numpy array
        (if requested).

        This is done by making a call to the c version of this method.

        If this box has been rotated a copy of the ray is first rotated into the box's frame and then traced (because
        AABB only works when it is in the original frame).  Once we have the ray in the AABB frame, we then use the
        equations from https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
        to both check if the bounding box is intersected, and to get the minimum and maximum intersect distance.  The
        results are returned as a boolean and optionally a numpy array, where the first is ``True`` when the
        ray struck the surface and ``False`` otherwise and the optional second (if
        ``return_distances=True``) is a length 2 array of doubles where the first column has the closest intersect
        distance, and the second column has the further intersect distance.  These numbers are only valid when the
        first return is ``True``.

        :param ray: The ray to trace through the bounding box.
        :type ray: Rays
        :return: The boolean value whether the ray hit the bounding box and optionally the length 2 double array
                 containing the near/far distance of the intersect for the ray (only valid when the ray actually hits
                 the box)
        :rtype: Union[bool, Tuple[bool, np.ndarray]]
        """

        # initialize the result integer
        cdef cnp.uint8_t res = False

        # rotate the ray if we need to
        if self._rotation is not None:
            ray = copy(ray)
            ray.rotate(self._rotation)

        # extract the required components of the ray
        start = ray.start.ravel()
        inverse_direction = ray.inv_direction.ravel()

        cdef double[:] distances = np.zeros(2, dtype=np.float64)

        # compute the intersection
        self._compute_intersect(start, inverse_direction, &res, &distances[0], &distances[1])

        if return_distances:
            # cast the result back to a boolean
            return bool(res), np.asarray(distances)
        else:
            return bool(res)


    def rotate(self, rotation):
        """
        rotate(self, rotation)

        This method is used to rotate the axis aligned bound box.

        In actuality the box is not rotated because that would make it invalid, instead the inverse rotation is stored
        and used to rotate all rays that are traced against this box.  This is done in :attr:`._rotation`.

        :param rotation: The rotation by which to rotate the bounding box.  For valid inputs see the :class:`.Rotation`
                         documentation.
        :type rotation: Union[Rotation, ARRAY_LIKE]
        """

        if isinstance(rotation, Rotation):
            if self._rotation is not None:
                # if the box has already been rotated, right multiply by the inverse of the new rotation
                self._rotation = self._rotation * rotation.inv()

            else:

                # if the box hasn't been rotated store the inverse rotation
                self._rotation = rotation.inv()

        else:

            if self._rotation is not None:
                # if the box has already been rotated, right multiply by the inverse of the new rotation
                self._rotation = self._rotation * Rotation(rotation).inv()

            else:

                # if the box hasn't been rotated store the inverse rotation
                self._rotation = Rotation(rotation).inv()

    def translate(self, translation):
        """
        translate(self, translation)

        This method translates the AABB.

        This is done by simply adding the translation to the min/max sides of the box in the original AABB frame. That
        is, if the bounding box has been rotated, we first rotate the provided translation into the original AABB frame,
        and then translate the min/max sides.

        :param translation: the translation the the AABB is undergoing
        :type translation: ARRAY_LIKE
        """

        # make sure the translation is a numpy array
        trans_array = np.asarray(translation).ravel().astype(np.float64)

        # rotate the translation into the AABB frame
        if self._rotation is not None:

            trans_array = self._rotation.matrix @ trans_array

        # apply the translation to both the min and max sides
        self._min_sides += trans_array

        self._max_sides += trans_array
