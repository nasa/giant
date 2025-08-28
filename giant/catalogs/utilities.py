# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


r"""
This submodule provides utility constants and functions for working with star data in GIANT.

Most of these utilities are focused on conversions of either units or of representations (ie a bearing to a unit
vector) with the exception of applying proper motion and computing the distance between 2 bearings.  For more details on
what is contained refer to the following summary tables and the documentation for each constant/function.
"""

import datetime

from typing import Union

import numpy as np

import pandas as pd

from giant.utilities.spherical_coordinates import unit_to_radec, radec_to_unit


__all__ = ['DEG2RAD', 'RAD2DEG', 'DEG2MAS', 'MAS2DEG', 'RAD2MAS', 'MAS2RAD', 'PARSEC2KM', 'AVG_STAR_DIST',
           'SI_DAYS_PER_YEAR', 'SI_SECONDS_PER_DAY', 'MJD_EPOCH', 
           'timedelta_to_si_years', 'datetime_to_mjd_years', 'apply_proper_motion']
"""
Things to import if someone wants to do from giant.catalogs.utilities import *
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

AVG_STAR_DIST: float = 5.428047027e15  # km
"""
The average distance of stars from the UCAC4 catalog in kilometers.  

This value is used to set the distance for stars for which there is no distance information available.
"""

AVG_STAR_DIST_SIGMA = 20 / (PARSEC2KM*1000/AVG_STAR_DIST) ** 2 * 1000 * PARSEC2KM
"""
The sigma value to use for stars with parallax that is not well known or known at all.

This is meant to be very high so that the stars are mostly ignored if using weighted estimation.
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
def timedelta_to_si_years(delta: datetime.timedelta) -> float:
    """
    This function converts a python timedelta object to a fractional number of SI years.

    The timedelta object is first converted to seconds using method ``total_seconds``, and this is then converted to SI
    years.
    
    :param delta: The python timedelta object to be converted to fractional SI years
    :return: The length of time covered by the time delta in units of fractional SI years
    """

    return delta.total_seconds() / SI_SECONDS_PER_DAY / SI_DAYS_PER_YEAR


def datetime_to_mjd_years(date: datetime.datetime) -> float:
    """
    This function converts a python datetime objects to the number of SI years since the MJD Epoch of November 17, 1858.

    This is computed by computing the time delta between the :attr:`.MJD_EPOCH` and the input datetime object, and then
    using :func:`.timedelta_to_si_years` to convert to the fractional years since the epoch.
    
    :param date: the python datetime object to be converted to MJD years
    :return: the number of SI years since November 17, 1858
    """

    return timedelta_to_si_years(date - MJD_EPOCH)


def apply_proper_motion(star_records: pd.DataFrame, new_time: Union[float, datetime.datetime],
                        copy: bool = True) -> pd.DataFrame:
    """
    This function adjusts the right ascension and declination of stars to a new time.  
    
    The right ascension and declination are updated using the corresponding proper motion of the stars.  The formulation
    used here assumes constant linear velocity as described in section 1.2.8 of "The Hipparcos and Tycho2 Catalogs".
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
    ra0: np.ndarray = out['ra'].to_numpy() * DEG2RAD 
    dec0: np.ndarray = out['dec'].to_numpy() * DEG2RAD 

    # assume linear motion based on Hipparcos Vol 1
    # compute the unit vector for each star
    r_unit = radec_to_unit(ra0, dec0).reshape(3, -1)

    # compute the unit vector in the direction of increasing right ascension
    p_unit = np.vstack([-np.sin(ra0), np.cos(ra0), np.zeros(ra0.shape)])

    # compute the unit vector in the direction of increasing declination
    q_unit = np.vstack([-np.sin(dec0) * np.cos(ra0), -np.sin(dec0) * np.sin(ra0), np.cos(dec0)])

    # compute the change in distance per year (future expansion)
    zeta0 = 0

    start_time = out['epoch'].to_numpy()

    # compute the time delta
    if isinstance(new_time, datetime.datetime):

        new_time = timedelta_to_si_years(new_time - datetime.datetime(1, 1, 1))

    timedelta = new_time - start_time

    # compute the new direction vector
    r_unit_new = r_unit * (1 + zeta0 * timedelta) + (p_unit * out['ra_proper_motion'].to_numpy() * DEG2RAD +
                                                     q_unit * out['dec_proper_motion'].to_numpy() * DEG2RAD) * timedelta
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

