# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
This module implements a basic point object for GIANT, used to represent objects that cannot be traced but that exist in
a scene.

Description
-----------

:class:`.Point` objects in GIANT are essentially just a size 3 numpy array containing the location of the point in the
current frame.  It contains no geometry and will not produce anything visible when rendering a scene.  It is primarily
used to represent the light source (the sun) in the scene, though you could also use it if you are doing unresolved
navigation using :mod:`.unresolved`.
"""

from typing import Union, Self

import numpy as np

from giant.rotations import Rotation

from giant._typing import ARRAY_LIKE


class Point:
    """
    Represents a single, unrenderable point.

    The point is just a shell around a size 3 numpy array to keep track of the location (but not orientation) of
    something in a scene that you don't actually want to render when doing ray tracing.  Therefore, this is most
    frequently used to represent the location of the sun in the scene.  Technically you can also use this if you are
    only doing :mod:`.unresolved` relative navigation; however, this is not recommended since no knowledge of the size
    of the object is available.

    To use this class simply provide the initial location as a length 3 array like object.  It can then be
    rotated/translated like other objects in a scene (though again note that it won't be rendered, and its orientation
    is not tracked, only the position).  If you are looking for something that will track both position and orientation,
    and can be rendered then we recommend checking out the :class:`.Ellipsoid` class instead.
    """

    def __init__(self, location: ARRAY_LIKE):
        """
        :param location: The starting location in the current frame for the point as a size 3 array like object.
        """

        self.position: np.ndarray = np.asarray(location).reshape(3).astype(np.float64)
        """
        The location of the point as a length 3 double array.
        """

    def rotate(self, rotation: Union[Rotation, ARRAY_LIKE]) -> Self:
        """
        This rotates the point into a new frame in place.

        Only the location is rotated, since the point itself doesn't have any orientation.

        :param rotation: The rotation from the previous frame to the new frame.  For possible inputs see the
                         :class:`.Rotation` documentation.
        """

        if isinstance(rotation, Rotation):
            self.position = np.matmul(rotation.matrix, self.position)

        else:
            self.position = np.matmul(Rotation(rotation).matrix, self.position)
            
        return self

    def translate(self, translation: ARRAY_LIKE) -> Self:
        """
        This translates the location of the point.

        :param translation: an array like object of size 3
        :raises ValueError: if the provided translation is not size 3
        """

        if np.size(translation) == 3:
            trans_array = np.asarray(translation).astype(np.float64)

            self.position += trans_array.reshape(3)

        else:
            raise ValueError("You have entered an improperly sized translation.\n"
                             "Only length 3 translations are allowed.\n"
                             "You entered {0}".format(np.size(translation)))
            
        return self
