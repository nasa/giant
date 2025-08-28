


"""
This cython module defines what is essentially the abstract base class for all renderable objects in GIANT (defining the
minimum interface required for an object to be renderable).

Use
---

Users typically will not use this module/class directly unless they are doing advanced development work, in which case
refer to the following class documentation.
"""

from typing import Union

import numpy as np

from giant.rotations import Rotation

from giant._typing import ARRAY_LIKE


cdef class Shape:
    """
    This represents the minimum required interface for an object to be considered traceable in GIANT.

    This is essentially an abstract base class (though abstract base classes do not actually exist as c extensions).  It
    defines the minimum interface for geometry in GIANT to be considered traceable in the normal :class:`.Scene` setup
    in GIANT.  All of the built in GIANT geometry that is traceable inherits from this class.

    In general, a user will never actually use this class unless they are (a) defining a new geometry, in which case
    they should inherit from this class and override its 4 methods or (b) creating something where they want to accept
    renderable geometry, in which case they should use this as the base class for the type of class that they accept.
    """

    def rotate(self, rotation):
        """
        This method rotates the shape in place.

        :param rotation: The rotation to be applied as a numpy array representing a rotation (see :class:`.Rotation` for
                         possible representations) or a :class:`.Rotation` object itself.
        :type rotation: Union[Rotation, ARRAY_LIKE]
        """

        pass

    def translate(self, translation):
        """
        This method translates the shape in place.

        :param translation: a size 3 array with which to translate the shape.
        :type translation: ARRAY_LIKE
        """

        pass

    def compute_intersect(self, ray):
        """
        This method computes the intersections between a ray and the geometry defined by the class, returning the
        results as a numpy array with type :attr:`.INTERSECT_DTYPE`.

        :param ray:  The ray to trace
        :type ray: Rays
        :return: The intersection result as a length 1 numpy array of type :attr:`.INTERSECT_DTYPE`
        :rtype: np.ndarray
        """
        return

    def trace(self, rays):
        """
        This method computes the intersections between a series of rays and the geometry defined by the class, returning
        the results as a numpy array with type :attr:`.INTERSECT_DTYPE`.

        :param rays:  The rays to trace
        :type rays: Rays
        :return: The intersection results as a length n numpy array of type :attr:`.INTERSECT_DTYPE`
        :rtype: np.ndarray
        """

        return

    def find_limbs(self, scan_center_dir, scan_dirs, observer_position=None):
        """
        find_limbs(self, scan_center_dir, scan_dirs, observer_position=None)

        The method determines the limb points (visible edge of the shape) that would be an observed for an observer
        located at ``observer_position`` looking toward ``scan_center_dir`` along the directions given by ``scan_dirs``.

        Typically it is assumed that the location of the observer is at the origin of the current frame and therefore
        ``observer_position`` can be left as ``None``.

        The returned limbs are expressed as vectors from the observer to the limb point in the current frame.

        :param scan_center_dir: the unit vector which the scan is to begin at in the current frame as a length 3 array
        :type scan_center_dir: np.ndarray
        :param scan_dirs: the unit vectors along with the scan is to proceed as a 3xn array in the current frame where
                          each column represents a new limb point we wish to find (should be nearly orthogonal to the
                          ``scan_center_dir`` in most cases).
        :type scan_dirs: np.ndarray
        :param observer_position: The location of the observer in the current frame.  If ``None`` then it is assumed
                                  the observer is at the origin of the current frame
        :type observer_position: Optional[np.ndarray]
        :return: the vectors from the observer to the limbs in the current frame as a 3xn array
        :rtype: np.ndarray
        """

        return

    def compute_limb_jacobian(self, scan_center_dir, scan_dirs, limb_points, observer_position=None):
        r"""
        compute_limb_jacobian(self, scan_center_dir, scan_dirs, limb_points, observer_position=None)

        This method computes the linear change in the limb location given a change in the relative position between the
        ellipsoid and the shape.

        :param scan_center_dir: the unit vector which the scan is to begin at in the current frame as a length 3 array
        :type scan_center_dir: np.ndarray
        :param scan_dirs: the unit vectors along with the scan is to proceed as a 3xn array in the current frame where
                          each column represents a new limb point we wish to find (should be nearly orthogonal to the
                          ``scan_center_dir`` in most cases).
        :type scan_dirs: np.ndarray
        :param limb_points: The vectors from the observer to the limb points in the current frame as a 3xn numpy array
                            where each column corresponds to the same column in the :attr:`scan_dirs` attribute.
        :type limb_points: np.ndarray
        :param observer_position: The location of the observer in the current frame.  If ``None`` then it is assumed
                                  the observer is at the origin of the current frame
        :type observer_position: Optional[np.ndarray]
        :return: The jacobian matrix as a nx3x3 array where each panel corresponds to the column in the ``limb_points``
                 input.
        """

        return
