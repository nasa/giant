# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
This module defines the abstract base class (abc) for defining GIANT star catalogues that will work for Stellar OpNav
and camera calibration as well as the column definitions for the dataframe used to contain stars in GIANT.

The abc documents the required interface that must be implemented for each star catalogue for it to be fully
functional in GIANT.  As such, when you define a new catalogue in GIANT you should subclass this class and be sure to
implement all of its abstract methods. You should only worry about this abc when you are defining a new catalogue.  If
you are using an existing catalogue then you can ignore this documentation and read the documentation for the catalogue
you are using for more specific details.

The column definitions are stored as 2 module attributes :attr:`GIANT_COLUMNS` and :attr:`GIANT_TYPES` which specify the
column names and the column types respectively of the dataframe used to store star records for use in GIANT.

Use
---

To implement a full function GIANT catalogue, you must implement the following instance attribute

============================================= ==========================================================================
Instance Attribute                            Description
============================================= ==========================================================================
:attr:`~.Catalogue.include_proper_motion`     A boolean flag specifying whether to apply proper motion to the queried
                                              locations to translate them to the specified time before returning.
                                              Technically this isn't actually required, but it is strongly recommended.
============================================= ==========================================================================

In addition, you need to implement the following method

============================================= ==========================================================================
Method                                        Description
============================================= ==========================================================================
:attr:`~.Catalogue.query_catalogue`           A method which queries the catalogue for requested Ra/Dec/Mag of stars
                                              and, if requested, applies proper motion to translate those stars to the
                                              requested time.
============================================= ==========================================================================

So long as your class implements these things it will be fully functional in GIANT and can be used for star
identification purposes.
"""

from abc import ABCMeta, abstractmethod

from datetime import datetime

from typing import Union, Optional, List, Type

import numpy as np
import pandas as pd

from giant._typing import Real, ARRAY_LIKE


GIANT_COLUMNS: List[str] = ['ra', 'dec', 'distance', 'ra_proper_motion', 'dec_proper_motion', 'mag',
                            'ra_sigma', 'dec_sigma', 'distance_sigma', 'ra_pm_sigma', 'dec_pm_sigma', 'epoch']
"""
This specifies the name of the DataFrame columns used to store star observations in GIANT.

These columns represent the minimum set of data that GIANT needs to know about a star to use it for star identification.
A description of each column follows.

===================== ======== =================================================================================
column                units    description
===================== ======== =================================================================================
`'ra'`                deg      The right ascension of the star after correcting for proper motion
`'dec'`               deg      The declination of the star after correcting for proper motion
`'distance'`          km       The distance to the star from the Solar system barycenter (converted from
                               parallax). This column has a default value of 5.428047027e15 if no parallax
                               information is provided by the catalogue.
`'ra_proper_motion'`  deg/year The proper motion for the right ascension
`'dec_proper_motion'` deg/year The proper motion for the declination
`'mag'`               N/A      The apparent magnitude of the star according to the star catalogue
`'ra_sigma'`          deg      The formal uncertainty in the right ascension according to the catalogue
`'dec_sigma'`         deg      The formal uncertainty in the declination according to the catalogue
`'distance_sigma'`    km       The formal uncertainty in the distance according to the catalogue 
                               (converted from parallax).  This has a default value of 
                               1.9949433041226756e+19 km for stars with no parallax information.
`'ra_pm_sigma'`       deg/year The formal uncertainty in the right ascension proper motion according to 
                               the catalogue
`'dec_pm_sigma'`      deg/year The formal uncertainty in the declination proper motion according to the
                               catalogue
`'epoch'`             MJD year The current epoch of the ra/dec/proper motion as a float
===================== ======== =================================================================================

"""


GIANT_TYPES: List[Type] = [np.float64] * len(GIANT_COLUMNS)
"""
This specifies the data type for each column of the GIANT star dataframe.

This is generally all double precision float values, though that could change in a future release.

This can be used with the :attr:`GIANT_COLUMNS` list to create a numpy structured dtype that mimics the Pandas DataFrame
``np.dtype(list(zip(GIANT_COLUMNS, GIANT_NAMES)))``
"""


class Catalogue(metaclass=ABCMeta):
    """
    This is the abstract base class for star catalogues that GIANT can use for star identification.

    This class defines the minimum required attributes and methods that GIANT expects when interfacing with a star
    catalogue.  It is also set up to implement duck typing, so if you don't want to you don't need to subclass this
    class directly when defining a new catalogue, though it is still strongly encouraged.

    To define a new Catalogue you pretty much only need to implement a :meth:`query_catalogue` method with the call
    signature specified in the abstract method documentation.  This is what GIANT will use when retrieving stars from
    the catalogue.  Beyond that, you should probably also implement and use an instance attribute
    :attr:`include_proper_motion` (although this is not required) as a flag that the user could use to turn proper
    motion applying on or off on a call to :meth:`query_catalogue`.

    If you are just trying to use a GIANT catalogue, you don't need to worry about this class, instead see one of the
    concrete implementations in :mod:`.ucac`, :mod:`.tycho`, or :mod:`.giant_catalogue`

    .. note:: Because this is an ABC, you cannot create an instance of this class.
    """

    def __init__(self, include_proper_motion: bool = True):
        """
        :param include_proper_motion: A flag indicating that proper motion should be applied to query results from this
                                      catalogue
        """

        self.include_proper_motion: bool = include_proper_motion
        """
        Apply proper motion to queried star locations to get the location at the requested time
        """

    @abstractmethod
    def query_catalogue(self, ids: Optional[ARRAY_LIKE] = None, min_ra: Real = 0, max_ra: Real = 360,
                        min_dec: Real = -90, max_dec: Real = 90, min_mag: Real = -4, max_mag: Real = 20,
                        search_center: Optional[ARRAY_LIKE] = None, search_radius: Optional[Real] = None,
                        new_epoch: Optional[Union[datetime, Real]] = None) -> pd.DataFrame:
        """
        This method queries stars from the catalogue that meet specified constraints and returns them as a DataFrame
        with columns of :attr:`.GIANT_COLUMNS`.

        Stars can either be queried by ID directly or by right ascension/declination/magnitude. You cannot filter using
        both with this method.  If :attr:`apply_proper_motion` is ``True`` then this will shift the stars to the new
        epoch input by the user (``new_epoch``) using proper motion.

        :param ids: A sequence of star ids to retrieve from the catalogue.  What these ids are vary from catalogue to
                    catalogue so see the catalogue documentation for details.
        :param min_ra: The minimum ra bound to query stars from in degrees
        :param max_ra: The maximum ra bound to query stars from in degrees
        :param min_dec: The minimum declination to query stars from in degrees
        :param max_dec: The maximum declination to query stars from in degrees
        :param min_mag: The minimum magnitude to query stars from.  Recall that magnitude is inverse (so lower
                        magnitude is a dimmer star)
        :param max_mag: The maximum magnitude to query stars from.  Recall that magnitude is inverse (so higher
                        magnitude is a dimmer star)
        :param search_center: The center of a search cone as a ra/dec pair.
        :param search_radius: The radius about the center of the search cone
        :param new_epoch: The epoch to translate the stars to using proper motion if :attr:`apply_proper_motion` is
                          turned on
        :return: A Pandas dataframe with columns :attr:`GIANT_COLUMNS`.
        """

        pass

    @classmethod
    def __subclasshook__(cls, other: type) -> bool:

        if cls is Catalogue:
            return hasattr(other, 'query_catalogue')

        return NotImplemented
