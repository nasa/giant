# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
This module defines a class for representing rays in GIANT, a function to generate them from a camera model, and the
numpy structured data type used to store the results of a ray trace.

Description
-----------

In GIANT, A :class:`.Rays` is used in ray tracing to intersect with a surface.  It is defined fully by a start, a
direction, and an inverse direction (1/direction).  Rays are defined to be efficient in GIANT, making use of numpy
broadcasting so that you can efficiently have a single start for many directions or many starts for a single direction,
and also support translation and rotation like other ray tracer objects.

Use
---

Rays are used frequently in GIANT.  They're pretty simple to create (simply provide the start and direction as numpy
arrays) and can be directly created for pixels from a camera in the camera frame using :func:`.compute_rays`.  Once
they're created they can be used for tracing :mod:`.shapes`, :class:`.KDTree`, :class:`.Scene`, and
:class:`.SceneObject` and for generating illumination inputs.  In addition, you can ignore certain facets (for surfaces)
or entire solids when tracing rays, using the :attr:`.ignore` attribute of the :class:`.Rays` class.  The format that
the ids for the ignore attribute take are somewhat complicated, so be sure to read the documentation carefully if you
are planning to use this feature.
"""


from typing import Optional, Iterable, Union, Tuple

import numpy as np

from giant.rotations import Rotation
from giant.camera_models.camera_model import CameraModel
from giant._typing import ARRAY_LIKE, Real


INTERSECT_DTYPE: np.dtype = np.dtype([('check', bool), ('distance', np.float64),
                                      ('intersect', np.float64, (3,)), ('normal', np.float64, (3,)),
                                      ('albedo', np.float64), ('facet', np.int64)])
"""
The numpy datatype returned when rays are traced with a :mod:`.shapes` or :class:`.KDTree` in GIANT.

For an overview of how structured data types work in numpy, refer to https://numpy.org/doc/stable/user/basics.rec.html

The following table describes the purpose of each field.  

================ ================ ======================================================================================
Field            Type             Description
================ ================ ======================================================================================
check            bool             A boolean flag specifying if the ray hit the object
distance         double           The distance between the ray start and the intersect location (if the ray struck the 
                                  object)
intersect        3 element double The location that the ray struck the object in the current frame (if the ray struck 
                 the object)
normal           3 element double The unit vector that is normal to the surface at the point the ray struck the object
                                  as a 3 element array (if the ray struck the object)
albedo           double           The albedo of the surface as a float
facet            int64            The id of the facet(if traced against a surface)/solid (if traced against a solid) 
                                  that the ray struck (if the ray struck the object).  This will be the fully encoded id
                                  so that you could use it in subsequent calls to trace to ignore that facet if so 
                                  desired.
================ ================ ======================================================================================

In general, anywhere that ``check`` is not ``True`` has no guarantee on the values of the other elements.
"""


# todo: consider updating the init to accept the inv directions for faster indexing

class Rays:
    """
    A class to store/manipulate rays.

    In GIANT a ray is defined by a start and a direction (and optionally a list of ids to ignore when tracing a ray
    through a scene/to a shape.  In addition, given these inputs, this class automatically computes the inverse
    direction (1/direction) which is used when intersecting the rays with an axis aligned bounding box.

    When creating rays, if you are creating multiple rays that share the same :attr:`start` or the same
    :attr:`direction` you should input these as a single array of length 3 (not 3x1).  This will allow us to use
    numpy broadcasting rules to efficiently represent the rays without having to duplicate memory.  In addition, you can
    update the start/direction of the rays by using the :meth:`rotate` and :meth:`translate` methods, or by setting to
    the :attr:`start` and :attr:`direction` directly.

    To use the :attr:`ignore` attribute of the rays you need to encode the full id of whatever you are trying to ignore.
    This can be very tricky and depends on what your tracing the rays against.  For instance, if you are directly
    tracing the rays with a :class:`.RawSurface` like a :class:`.Triangle64` then the id is simply the facet number (row
    number in the :attr:`.RawSurface.facets` array).  If you are tracing with a :class:`.Solid`, then the id is the
    :attr:`.Solid.id`.  If you are tracing with a :class:`.KDTree` then the id needs to be a combination of the ids of
    the path through the tree to the leaf node containing the facet, plus the row number for the facet in the
    :attr:`.KDNode.surface` of the leaf node (this can conveniently be determined using :func:`.get_ignore_inds`).  If
    you are tracing with a scene, then you need to take the id of the geometry combined with index of the scene object
    in the :attr:`.Scene.target_objs` list.  All of these get combined into a single integer.  Given this, it is rare to
    set the :attr:`ignore` attribute yourself.  Instead, typically you either let GIANT set the attribute for you
    automatically, or you get the value from a previous trace (the ``facet`` component of :data:`.INTERSECT_DTYPE` will
    fully encode the id for whatever you traced through).

    This class supports iterating through rays one at a time using the normal python syntax (``for ray in rays: ...``).
    That being said, this is not super efficient and is not the way GIANT handles multiple rays internally.  You can
    also use indexing on the :class:`.Rays` object, which will return another :class:`.Rays` object.  This can be useful
    for things like boolean indexing and slicing.
    """

    def __init__(self, start: ARRAY_LIKE, direction: ARRAY_LIKE, ignore: Optional[ARRAY_LIKE] = None):
        """
        :param start: Where the rays begin at as a length 3 array or a 3xn array
        :param direction: The direction that the rays proceed in as a length 3 array or a 3xn array (typically this
                          should be unit vectors)
        :param ignore: The ids to ignore when tracing the rays.  This should be either ``None`` for no ignores, a length
                       n array for a single ignore per ray (set to -1 for rays where you don't want any ignores), or a
                       length n Sequence of arrays (where there are multiple (possibly different numbers) ignores for
                       each ray)
        """
        # set up the hidden attributes
        self._start = None
        self._direction = None
        # inv direction is 1/direction
        self._inv_direction = None
        self._ignore = None

        self.num_rays: int = 1
        """
        The number of rays contained in the object.
        """

        # use the property setters to handle things effectively
        self.start = start
        self.direction = direction
        self.ignore = ignore

    def __iter__(self) -> Iterable['Rays']:
        """
        Iterate 1 at a time through the rays contained in this object.
        """

        if self.num_rays > 1:

            starts, directions = np.broadcast_arrays(self._start.reshape(3, -1), self._direction.reshape(3, -1))

            if (self.ignore is not None) and isinstance(self.ignore, (list, np.ndarray, tuple)):

                for start, direction, ignore in zip(starts.T, directions.T, self.ignore):
                    yield Rays(start, direction, ignore=ignore)

            elif self.ignore is not None:

                for start, direction in zip(starts.T, directions.T):
                    yield Rays(start, direction, ignore=self.ignore)

            else:
                for start, direction in zip(starts.T, directions.T):
                    yield Rays(start, direction)

        else:

            yield self

    def __getitem__(self, item: Union[int, ARRAY_LIKE, slice]) -> 'Rays':
        """
        Select a subset of the rays contained in this object.

        Typically this is used with slicing and boolean indexing

        :param item: The value to use to index with
        """

        if self.num_rays > 1:
            starts, directions = np.broadcast_arrays(self._start.reshape(3, -1), self._direction.reshape(3, -1))

            if (self.ignore is not None) and isinstance(self.ignore, (list, np.ndarray, tuple)):

                return Rays(starts.T[item].T, directions.T[item].T, ignore=self.ignore[item])

            elif self.ignore is not None:

                return Rays(starts.T[item].T, directions.T[item].T, ignore=self.ignore)

            else:

                return Rays(starts.T[item].T, directions.T[item].T)

        else:

            raise ValueError('Cannot get item from a single ray.')

    def __len__(self) -> int:
        """
        Returns the number of rays contained in the object
        """

        return self.num_rays

    def rotate(self, rotation: Union[Rotation, ARRAY_LIKE]):
        """
        Rotates the start location(s) and the direction(s) of the ray(s) in place.

        :param rotation:  an array representing a rotation or a :class:`.Rotation` object by which to rotate the rays
        """

        if isinstance(rotation, Rotation):

            self._direction = np.matmul(rotation.matrix, self._direction)
            self._start = np.matmul(rotation.matrix, self._start)

        else:

            self._direction = np.matmul(Rotation(rotation).matrix, self._direction)
            self._start = np.matmul(Rotation(rotation).matrix, self._start)

        self._inv_direction = 1 / self._direction

    def translate(self, translation: ARRAY_LIKE):
        """
        Translates the start location(s) of the ray(s) in place.

        The directions are not affected by this method.

        :param translation: an array like vector
        """

        translation_array = np.asarray(translation).astype(np.float64).squeeze()

        if (self._start.ndim > 1) and (translation.ndim == 1):
            translation_array = translation_array.reshape(3, -1)

        self._start += translation_array

    @property
    def start(self) -> np.ndarray:
        """
        The beginning location(s) of the ray(s) as an 3xn array of start locations (if n==1 then a flat 3 component
        array is returned).

        If there are multiple directions and only a single start then this will still return a 3xn array but will use
        numpy broadcasting so that memory is not duplicated.
        """

        if self.num_rays > 1:
            starts, _ = np.broadcast_arrays(self._start.reshape(3, -1), self._direction.reshape(3, -1))

            return starts

        return self._start

    @start.setter
    def start(self, val: ARRAY_LIKE):

        val_array = np.asarray(val).squeeze().astype(np.float64)

        if val_array.shape and (val_array.shape[0] != 3):
            raise ValueError("The first axis must have a length of 3")

        if val_array.ndim == 2:
            if ((self._direction is not None) and (self._direction.ndim == 2) and
                    (val_array.shape[-1] != self._direction.shape[-1])):

                raise ValueError("The start and direction arrays must have the same shape.")

            self.num_rays = max(self.num_rays, val_array.shape[-1])
        else:
            self.num_rays = max(self.num_rays, 1)

        self._start = val_array

    @property
    def direction(self) -> np.ndarray:
        """
        The direction vector(s) of the ray(s) as an 3xn array of vector(s)  (if n==1 then a flat 3 component array is
        returned).

        If there are multiple starts and only a single direction then this will still return a 3xn array but will use
        numpy broadcasting so that memory is not duplicated.
        """

        if self.num_rays > 1:
            _, directions = np.broadcast_arrays(self._start.reshape(3, -1), self._direction.reshape(3, -1))

            return directions

        return self._direction

    @direction.setter
    def direction(self, val: ARRAY_LIKE):

        val_array = np.asarray(val).squeeze().astype(np.float64)

        if (not val_array.shape) or (val_array.shape[0] != 3):
            raise ValueError("The first axis must have a length of 3")

        if val_array.ndim == 2:
            if ((self._start is not None) and (self._start.ndim == 2) and
                    (val_array.shape[-1] != self._start.shape[-1])):

                raise ValueError("The start and direction arrays must have the same shape.")

            self.num_rays = max(self.num_rays, val_array.shape[-1])
        else:
            self.num_rays = max(self.num_rays, 1)

        self._direction = val_array
        self._inv_direction = 1 / self._direction

    @property
    def inv_direction(self) -> np.ndarray:
        """
        The inverse of the direction vectors (1/directions) as a 3xn array (if n==1 then a flat 3 component array is
        returned).

        If there are multiple starts and only a single direction then this will still return a 3xn array but will use
        numpy broadcasting so that memory is not duplicated.
        """

        if self.num_rays > 1:
            _, inv_directions = np.broadcast_arrays(self._start.reshape(3, -1), self._inv_direction.reshape(3, -1))

            return inv_directions

        return self._inv_direction

    @property
    def ignore(self) -> Optional[ARRAY_LIKE]:
        """
        An array of the full ids of whatever you are trying to ignore for specific rays or None.

        This is generally used when illuminating a scene as a way to ignore the surface that a ray starts at.  See the
        class documentation for more details on how this mush be set.
        """

        return self._ignore

    @ignore.setter
    def ignore(self, val):

        self._ignore = val


def compute_rays(model: CameraModel, rows: ARRAY_LIKE, cols: ARRAY_LIKE, grid_size: int = 1, temperature: Real = 0,
                 image_number: int = 0) -> Tuple[Rays, np.ndarray]:
    """
    Compute rays passing through the given row, col pairs for the given camera in the camera frame.

    The rays are assumed to start at the origin of the camera frame (0, 0, 0).  The directions are formed by calls to
    :meth:`.CameraModel.pixels_to_unit` and are generate all at once.  The pixel values corresponding to the generated
    rays are returned second as a 2xn array.

    When creating the rays, you can specify how many rays per pixel you want to generate (evenly distributed in a square
    subpixel pattern) using the ``grid_size`` argument.  The result will be

    .. code::

           1  2        grid_size
        +------ ... ------+
        |                 |
        |                 |
        |  x  x ... x  x  | 1
        |                 |
        |                 |
        |  x  x ... x  x  | 2
        ...             ...
        |  x  x ... x  x  | grid_size -1
        |                 |
        |                 |
        |  x  x ... x  x  | grid_size
        |                 |
        |                 |
        +------ ... ------+

    where +-| indicates the bounds of the pixel, and x indices subpixel locations for the rays (with even spacing
    between each sub-pixel location and the edges of the pixel.

    :param model: A camera model object which is used to set the direction for the rays
    :param rows: An array like list of rows to generate rays through (paired with cols).  If there are only 2 elements
                 then it is assumed that this is a min, max pair (inclusive on both sides)  and you want to generate
                 rays for every pixel between min and max
    :param cols: An array like list of cols to generate rays through (paired with rows).  If there are only 2 elements
                 then it is assumed that this is a min, max pair (inclusive on both sides)  and you want to generate
                 rays for every pixel between min and max
    :param grid_size: The number of rays per edge of pixel (sqrt of number of rays).  There will be
                      ``grid_size*grid_size`` rays evenly distributed through the area of each pixel pair requested
    :param temperature: The temperature of the image the rays are being computed for passed to
                        :meth:`.CameraModel.pixels_to_unit`
    :param image_number: The number of the image the rays are being created for pass to
                         :meth:`.CameraModel.pixels_to_unit`
    :return: The rays passing through the requested pixels in the camera frame and the subpixel locations
    """

    if len(rows) == 2:
        # determine the spacing between each grid point in pixels
        grid_dist = 1 / grid_size

        # determine where in the pixel the grid will start at
        grid_start = 0.5 - grid_dist / 2

        # get the subpixel locations for each ray
        cols, rows = np.meshgrid(np.arange(cols[0] - grid_start, cols[1] + 0.5, grid_dist),
                                 np.arange(rows[0] - grid_start, rows[1] + 0.5, grid_dist))

    # stack the columns and rows into a single xy matrix
    uv = np.vstack([cols.ravel(), rows.ravel()])

    # get the direction vectors for each ray
    directions = model.pixels_to_unit(uv, temperature=temperature, image=image_number)

    # store the origin as the origin of the camera frame
    starts = np.zeros(3)

    return Rays(starts, directions), uv
