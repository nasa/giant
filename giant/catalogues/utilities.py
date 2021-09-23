# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


r"""
This submodule provides utility constants and functions for working with star data in GIANT.

Most of these utilities are focused on conversions of either units or of representations (ie a bearing to a unit
vector) with the exception of applying proper motion and computing the distance between 2 bearings.  For more details on
what is contained refer to the following summary tables and the documentation for each constant/function.
"""

import datetime

from typing import Tuple, Union

import numpy as np

import pandas as pd

from giant._typing import SCALAR_OR_ARRAY, ARRAY_LIKE, Real


__all__ = ['DEG2RAD', 'RAD2DEG', 'DEG2MAS', 'MAS2DEG', 'RAD2MAS', 'MAS2RAD', 'PARSEC2KM', 'STAR_DIST',
           'SI_DAYS_PER_YEAR', 'SI_SECONDS_PER_DAY', 'MJD_EPOCH', 'radec_to_unit', 'unit_to_radec',
           'timedelta_to_si_years', 'datetime_to_mjd_years', 'apply_proper_motion', 'radec_distance']
"""
Things to import if someone wants to do from giant.catalogues.utilities import *
"""

# CONSTANTS.

DEG2RAD: float = np.pi / 180  # rad/deg
r"""
This constant converts from units of degrees to units of radians through multiplication.

That is ``angle_rad = angle_deg*DEG2RAD`` where ``angle_rad`` is the angle in radians and ``angle_deg`` is the angle in 
degrees.

Mathematically this is :math:`\frac{\pi}{180}`.
"""

RAD2DEG: float = 180 / np.pi  # deg/rad
r"""
This constant converts from units of radians to units of degrees through multiplication.

That is ``angle_deg = angle_rad*RAD2DEG`` where ``angle_rad`` is the angle in radians and ``angle_deg`` is the angle in 
degrees.

Mathematically this is :math:`\frac{180}{\pi}`.
"""

DEG2MAS: float = 3600 * 1000  # mas/deg
r"""
This constant converts from units of degrees to units of milli-arc-seconds through multiplication.

That is ``angle_mas = angle_deg*DEG2MAS`` where ``angle_mas`` is the angle in milli-arc-seconds and ``angle_deg`` is the 
angle in degrees.

Mathematically this is :math:`3600000`.
"""

MAS2DEG: float = 1 / DEG2MAS  # deg/mas
r"""
This constant converts from units of milli-arc-seconds to units of degrees through multiplication.

That is ``angle_deg = angle_mas*MAS2DEG`` where ``angle_mas`` is the angle in milli-arc-seconds and ``angle_deg`` is the 
angle in degrees.

Mathematically this is :math:`\frac{1}{3600000}`.
"""

RAD2MAS: float = RAD2DEG * DEG2MAS
r"""
This constant converts from units of radians to units of milli-arc-seconds through multiplication.

That is ``angle_mas = angle_rad*RAD2MAS`` where ``angle_mas`` is the angle in milli-arc-seconds and ``angle_rad`` is the 
angle in radians.

Mathematically this is :math:`\frac{180}{3600000\pi}`.
"""

MAS2RAD: float = 1 / RAD2MAS
r"""
This constant converts from units of milli-arc-seconds to units of radians through multiplication.

That is ``angle_rad = angle_mas*MAS2RAD`` where ``angle_mas`` is the angle in milli-arc-seconds and ``angle_rad`` is the 
angle in radians.

Mathematically this is :math:`\frac{3600000\pi}{180}`.
"""

PARSEC2KM: float = 30856775814913.673  # km
r"""
This constant converts from units of parsecs to units of kilometers through multiplication.

That is ``distance_km = distance_parsec*PARSEC2KM`` where ``distance_km`` is the distance in kilometers and 
``distance_parsec`` is the distance in parsecs.

Mathematically this is :math:`\frac{3600000\pi}{180}`.
"""

STAR_DIST: float = 5.428047027e15  # km
"""
The average distance of stars from the UCAC4 catalogue in kilometers.  

This value is used to set the distance for stars for which there is no distance information available.
"""

SI_DAYS_PER_YEAR: float = 365.25  # days
"""
This constant provides the number of SI days in a SI year.
"""

SI_SECONDS_PER_DAY: int = 86400  # seconds
"""
This constant provides the number of SI seconds in a SI day.
"""

MJD_EPOCH: datetime.datetime = datetime.datetime(1858, 11, 17)  # November 17, 1858
"""
This constant provides the standard modified Julian date epoch, November 17, 1858, as a datetime
"""


# UTILITY FUNCTIONS
def radec_to_unit(ra: SCALAR_OR_ARRAY, dec: SCALAR_OR_ARRAY) -> np.ndarray:
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


def unit_to_radec(unit: ARRAY_LIKE) -> Tuple[SCALAR_OR_ARRAY, SCALAR_OR_ARRAY]:
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


def timedelta_to_si_years(delta: datetime.timedelta) -> float:
    """
    This function converts a python timedelta object to a fractional number of SI years.

    The timedelta object is first converted to seconds using method ``total_seconds``, and this is then converted to SI
    years.
    
    :param delta: The python timedelta object to be converted to fractional SI years
    :return: The length of time covered by the time delta in units of fractional SI years
    """

    return delta.total_seconds() / SI_SECONDS_PER_DAY / SI_DAYS_PER_YEAR


def datetime_to_mjd_years(date: datetime) -> float:
    """
    This function converts a python datetime objects to the number of SI years since the MJD Epoch of November 17, 1858.

    This is computed by computing the time delta between the :attr:`.MJD_EPOCH` and the input datetime object, and then
    using :func:`.timedelta_to_si_years` to convert to the fractional years since the epoch.
    
    :param date: the python datetime object to be converted to MJD years
    :return: the number of SI years since November 17, 1858
    """

    return timedelta_to_si_years(date - MJD_EPOCH)


def apply_proper_motion(star_records: pd.DataFrame, new_time: Union[Real, datetime.datetime],
                        copy: bool = True) -> pd.DataFrame:
    """
    This function adjusts the right ascension and declination of stars to a new time.  
    
    The right ascension and declination are updated using the corresponding proper motion of the stars.  The formulation
    used here assumes constant linear velocity as described in section 1.2.8 of "The Hipparcos and Tycho2 Catalogues".
    The bearing measurement is converted to a unit vector, which is then updated using vector addition with the delta 
    applied along the vectors of increasing right ascension and increasing declination.  This model also allows for 
    consideration of a radial velocity, but that is currently not implemented.
    
    The stars input into this method should be a pandas dataframe with the GIANT format.  Specifically, this function
    requires the dataframe to have columns of ``['ra', 'dec', 'ra_proper_motion', 'dec_proper_motion', 'epoch']`` with
    units of degrees, degrees/year, and SI years (since January 1, 1) respectively.  The updated bearing can be stored
    either in a copy of the dataframe, or in-place, depending on the ``copy`` key word argument.  Either way the
    resulting dataframe is returned.
    
    The ``new_time`` parameter should either be a datetime object, or a float of the modified julian years for the 
    desired time. The ``copy`` flag states whether to return a copy of the dataframe with the updates applied
    (recommended), or to make the updates in place.

    :param star_records: a pandas dataframe containing the bearing and proper motion of star records to be updated
    :param new_time: the new epoch to calculate the star positions at expressed as a mjy float or python datetime object
    :param copy: An option flag indicating whether to make a copy of star_records before applying proper motion
    :return: a pandas dataframe containing the star records with bearing values updated to the new epoch
    """

    if copy:
        out = star_records.copy()

    else:
        out = star_records

    # convert the ra and dec values into radians
    ra0 = out['ra'].values * DEG2RAD  # type: np.ndarray
    dec0 = out['dec'].values * DEG2RAD  # type: np.ndarray

    # assume linear motion based on Hipparcos Vol 1
    # compute the unit vector for each star
    r_unit = radec_to_unit(ra0, dec0).reshape(3, -1)

    # compute the unit vector in the direction of increasing right ascension
    p_unit = np.vstack([-np.sin(ra0), np.cos(ra0), np.zeros(ra0.shape)])

    # compute the unit vector in the direction of increasing declination
    q_unit = np.vstack([-np.sin(dec0) * np.cos(ra0), -np.sin(dec0) * np.sin(ra0), np.cos(dec0)])

    # compute the change in distance per year (future expansion)
    zeta0 = 0

    start_time = out['epoch'].values

    # compute the time delta
    if isinstance(new_time, datetime.datetime):

        new_time = timedelta_to_si_years(new_time - datetime.datetime(1, 1, 1))

    timedelta = new_time - start_time

    # compute the new direction vector
    r_unit_new = r_unit * (1 + zeta0 * timedelta) + (p_unit * out['ra_proper_motion'].values * DEG2RAD +
                                                     q_unit * out['dec_proper_motion'].values * DEG2RAD) * timedelta
    r_unit_new /= np.linalg.norm(r_unit_new, axis=0, keepdims=True)

    # compute the new ra/dec and store it
    ra, dec = unit_to_radec(r_unit_new)

    out['ra'] = ra * RAD2DEG
    out['dec'] = dec * RAD2DEG

    # compute the new uncertainties for ra and dec
    out['ra_sigma'] = np.sqrt(out['ra_sigma'] ** 2 + timedelta ** 2 * out['ra_pm_sigma'] ** 2)
    out['dec_sigma'] = np.sqrt(out['dec_sigma'] ** 2 + timedelta ** 2 * out['dec_pm_sigma'] ** 2)

    # update the epochs
    out["epoch"] = new_time

    return out


def radec_distance(ra1: SCALAR_OR_ARRAY, dec1: SCALAR_OR_ARRAY,
                   ra2: SCALAR_OR_ARRAY, dec2: SCALAR_OR_ARRAY) -> SCALAR_OR_ARRAY:
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
    return np.arccos(np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2) + np.sin(dec1) * np.sin(dec2))
