"""
This module contains helper functions for transforming from/to spherical coordinates.
"""

from typing import Sequence

import numpy as np

from giant._typing import DOUBLE_ARRAY, SCALAR_OR_ARRAY, F_SCALAR_OR_ARRAY


def radec_to_unit(ra: SCALAR_OR_ARRAY, dec: SCALAR_OR_ARRAY) -> DOUBLE_ARRAY:
    r"""
    This utility converts (a) right ascension and declination pair(s) expressed in units of radians into (an) unit
    vector(s).

    The conversion to a unit vector is given by:

    .. math::
        \hat{\mathbf{x}}=\left[\begin{array}{c}\text{cos}(\delta)\text{cos}(\alpha)\\
        \text{cos}(\delta)\text{sin}(\alpha)\\
        \text{sin}(\delta)\end{array}\right]

    where :math:`\alpha` is the right ascension, :math:`\delta` is the declination, and :math:`\hat{\mathbf{x}}` is
    the resulting unit vector.

    This method is vectorized, therefore multiple unit vectors can be created from multiple ra, dec pairs at the
    same time.  When multiple conversions are performed, each unit vector is specified as a column in the array.

    This function performs broadcast rules using numpy conventions, therefore you can provide inputs with different
    shapes, so long as they are able to be broadcast (you could add them together and numpy wouldn't complain).  If you
    provide >1D arrays then they will be raveled using c convention.

    :param ra: The right ascension(s) to convert to unit vector(s) in units of radians
    :param dec: The declination(s) to convert to unit vector(s) in units of radians
    :return: A 3xn array of unit vectors corresponding to the right ascension, declination pair(s)
    """

    # make sure the arrays are the same length
    ra, dec = np.broadcast_arrays(ra, dec)

    ra = ra.ravel()  # ensure our arrays are flat
    dec = dec.ravel()  # ensure our arrays are flat

    unit_x = np.cos(dec) * np.cos(ra)
    unit_y = np.cos(dec) * np.sin(ra)
    unit_z = np.sin(dec)

    return np.vstack([unit_x, unit_y, unit_z]).squeeze()


def unit_to_radec(unit: Sequence[float] | DOUBLE_ARRAY) -> tuple[float, float] | \
                                                           tuple[DOUBLE_ARRAY, DOUBLE_ARRAY]:
    r"""
    This function converts a unit vector(s) into a right ascension/declination bearing(s).  
    
    The right ascension is defined as the angle between the x axis and the projection of the unit vector onto the
    xy-plane and is output between 0 and pi.  The declination is defined as the angle between the xy-plane and the 
    unit vector and is output between -pi/2 and pi/2 (positive values indicate the vector has a positive z component and
    negative values indicate the vector has a negative z component. These are computed using

    .. math::

        dec = \text{sin}^{-1}(z) \\
        ra = \text{tan}^{-1}\left(\frac{y}{x}\right)
    
    Note that the vector input should be along the first axis (or as columns if there are multiple vectors), and that 
    the vector(s) needs to be of unit length or the results from this function will be invalid.

    If the input contains more than 1 vector, then the output will be 2 arrays.  Otherwise, if it is a single vector,
    the output will be 2 floats as a tuple.
    
    :param unit: The unit vector to be converted to a ra/dec bearing
    :return: The right ascension(s) and declination(s) as a tuple in units of radians
    """

    if np.shape(unit)[0] != 3:
        raise ValueError('The length of the first axis must be 3')

    dec = np.arcsin(unit[2])
    ra = np.arctan2(unit[1], unit[0])

    ra_check = ra < 0

    if np.ndim(ra):
        ra[ra_check] += 2 * np.pi

    elif ra_check:
        ra += 2 * np.pi

    return ra, dec


def radec_distance(ra1: SCALAR_OR_ARRAY, dec1: SCALAR_OR_ARRAY,
                   ra2: SCALAR_OR_ARRAY, dec2: SCALAR_OR_ARRAY) -> F_SCALAR_OR_ARRAY:
    r"""
    This function computes the great-circle angular distance in units of radians between ra/dec pairs.

    The distance is computed using

    .. math::

        \text{cos}^{-1}\left(\text{cos}(\delta_1)\text{cos}(\delta_2)\text{cos}(\alpha_1-\alpha_2) +
        \text{sin}(\delta_1)\text{sin}(\delta_2)\right)

    where :math:`\delta_1` is the first declination, :math:`\delta_2` is the second declination, :math:`\alpha_1` is the
    first right ascension, and :math:`\alpha_2` is the second right ascension, all in radians.

    This function is vectorized and uses broadcasting rules, therefore you can specify the inputs as mixtures of scalars
    and arrays, as long as they can all be broadcast to a common shape.  The output will be either a scalar (if all
    scalars are input) or an array if any of the inputs are an array.

    :param ra1: The right ascension values for the first parts of the pairs with units of radians
    :param dec1: The declination values for the first parts of the pairs with units of radians
    :param ra2: The right ascension values for the second parts of the pairs with units of radians
    :param dec2: The declination values for the second parts of the pairs with units of radians
    :return: The great circle angular distance between the points with units of radians
    """
    ra1 = np.asanyarray(ra1)
    return np.arccos(np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2) + np.sin(dec1) * np.sin(dec2))
