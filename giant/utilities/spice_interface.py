# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
This module provides utility functions and classes for quickly creating callable objects to NAIF spice functions as well
as a function to convert a datetime object to spice ephemeris time without using spice itself.

There are 2 different interfaes made available in this module.  The perfered interface is a set of 3 classes plus two
functions.  The classes, :class:`.SpicePosition`, :class:`.SpiceState`, and :class:`.SpiceOrientation` are wrappers
around calls to spice through the third party library `spiceypy <https://spiceypy.readthedocs.io/en/master/>` which
compute the relative position vector, relative state vector (position and velocity), and rotation with preset options.
This makes it really easy to use spice to drive :class:`.SceneObject` instances in GIANT (among other things).  These
only work when spiceypy is installed, but that should almost always be the case.  They also require you to load the
appropriate data before calling them using the
`furnsh <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/spicelib/furnsh.html>` spice function as is typical for
spice.  If you don't load the data you will get a ``'SpiceyError'``.

Suppose we would frequently like to retrieve the position of the Moon with respect to the Earth in the J2000 inertial
frame.  Each time that we want to get this value we could do something like

    >>> import spiceypy
    >>> import datetime
    >>> spiceypy.furnsh('/path/to/metakernel.tm')
    >>> observation_date = datetime.datetime.now()
    >>> et = spiceypy.str2et(observation_date.isoformat())
    >>> pos, _ = spiceypy.spkpos('Moon', et, 'J2000', 'LT+S', 'Earth')

Or, if we use this interface we could do something like the following:

    >>> import giant.utilities.spice_interface as spint
    >>> moon_pos_earth = spint.SpicePosition("Moon", "J2000", "LT+S", "Earth")
    >>> pos = moon_pos_earth(observation_date)

The 2 functions in the prefered interface, :func:`leap_seconds` and :func:`datetime_to_et` are useful time operations
that are available whether spice is being used or not.  The :func:`leap_seconds` function inputs a datetime object and
outputs the number of leap seconds that occurred between the input observation_date and the J2000 epoch.  The
:func:`datetime_to_et` function converts and input datetime object into its corresponding ephemeris time.

The next group of functions are not prefered and also are only available when
`spiceypy <https://spiceypy.readthedocs.io/en/master/>`_ is part of the current python installation and you have loaded
the apropriate data.  These functions create callable objects (actually partial function objects)
that only take in a time and return a position, state, or orientation.  The first function,
:func:`create_callable_position`, returns a partial function object wrapped around the spkpos spice routine that accepts
a ephemeris time as input and returns a position.  The next function, :func:`create_callable_state`, does the same thing
but returns a state vector (position and velocity) instead of just a position vector (and wraps the spkezr routine
instead of spkpos). :func:`create_callable_orientation` wraps the pxform routine and returns a partial function which
outputs a rotation given as an :class:`.Rotation` object when given an ephemeris time float as input.  The final
function, :func:`et_callable_to_datetime_callable` takes a callable object that inputs an ephemeris time as a float as
input (most likely the result of one of the previous functions) and outputs a callable object (a function object) that
accepts a datetime object as input instead.  While this interface is fully functional, it does not play nicely with
pickle due to the partial functions being anonymous, thus why we recommend the class interface instead which does work
with pickle.

We could use these (not recommended) routines in this module and do something like

    >>> moon_pos_earth_et = spint.create_callable_position('Moon', 'J2000', 'LT+S', 'EARTH')
    >>> moon_pos_earth = spint.et_callable_to_datetime_callable(moon_pos_earth_et)
    >>> pos = moon_pos_earth(observation_date)

Note that in the above, the light time output is dropped from the call to spkpos.

Note that all GIANT routines that require a function to return position, state, or orientation requires that the
function work with only a datetime object input, making this module extremely useful (both interfaces).
"""

import datetime
from functools import partial
from typing import Callable, Union, Tuple

import numpy as np
import pandas as pd

from ..rotations import Rotation


HAS_SPICE: bool = False
"""
This flag specifies whether spiceypy is available in the current python environment
"""


try:
    import spiceypy as spice

    HAS_SPICE = True

except ImportError:

    spice = ''


J2000_EPOCH = datetime.datetime(2000, 1, 1, 12, 0, 0)
"""
The UTC Epoch for spice ephemeris time
"""


# info from naif0012.tls -- list of leap_seconds since 1972 as datetime objects
LEAP_SECONDS_LIST = np.array([datetime.datetime(1972, 1, 1, 0, 0),  # -23
                              datetime.datetime(1972, 7, 1, 0, 0),  # -22
                              datetime.datetime(1973, 1, 1, 0, 0),  # -21
                              datetime.datetime(1974, 1, 1, 0, 0),  # -20
                              datetime.datetime(1975, 1, 1, 0, 0),  # -19
                              datetime.datetime(1976, 1, 1, 0, 0),  # -18
                              datetime.datetime(1977, 1, 1, 0, 0),  # -17
                              datetime.datetime(1978, 1, 1, 0, 0),  # -16
                              datetime.datetime(1979, 1, 1, 0, 0),  # -15
                              datetime.datetime(1980, 1, 1, 0, 0),  # -14
                              datetime.datetime(1981, 7, 1, 0, 0),  # -13
                              datetime.datetime(1982, 7, 1, 0, 0),  # -12
                              datetime.datetime(1983, 7, 1, 0, 0),  # -11
                              datetime.datetime(1985, 7, 1, 0, 0),  # -10
                              datetime.datetime(1988, 1, 1, 0, 0),  # -9
                              datetime.datetime(1990, 1, 1, 0, 0),  # -8
                              datetime.datetime(1991, 1, 1, 0, 0),  # -7
                              datetime.datetime(1992, 7, 1, 0, 0),  # -6
                              datetime.datetime(1993, 7, 1, 0, 0),  # -5
                              datetime.datetime(1994, 7, 1, 0, 0),  # -4
                              datetime.datetime(1996, 1, 1, 0, 0),  # -3
                              datetime.datetime(1997, 7, 1, 0, 0),  # -2
                              datetime.datetime(1999, 1, 1, 0, 0),  # -1
                              # J2000 epoch                            32 past TAI
                              datetime.datetime(2006, 1, 1, 0, 0),  # +1
                              datetime.datetime(2009, 1, 1, 0, 0),  # +2
                              datetime.datetime(2012, 7, 1, 0, 0),  # +3
                              datetime.datetime(2015, 7, 1, 0, 0),  # +4
                              datetime.datetime(2017, 1, 1, 0, 0)  # +5
                              ], 'datetime64[us]')
"""
A numpy datetime64 array of leap seconds taken from naif0012.tls
"""


# constants take from naif0012.tls
_DELTA_T_A = 32.184
_K_CONST = 1.657e-3
_EB_CONST = 1.671e-2
_MEAN_ANOM_START = 6.239996
_MEAN_ANOM_VELO = 1.99096871e-7


def leap_seconds(date: datetime.datetime) -> int:
    """
    This function returns the number of leap seconds between observation_date and January 1, 12:00:00.000 (TDB).

    For dates before the epoch, leap seconds are returned as negative integers.  For dates after the epoch
    leap seconds are returned as positive integers.

    :param date: a python datetime object
    :return: The number of leap seconds between the input observation_date and the epoch
    """

    positive_leaps = ((date >= LEAP_SECONDS_LIST) & (J2000_EPOCH < LEAP_SECONDS_LIST)).sum()

    negative_leaps = ((date <= LEAP_SECONDS_LIST) & (J2000_EPOCH > LEAP_SECONDS_LIST)).sum()

    return positive_leaps - negative_leaps


def datetime_to_et(date: Union[datetime.datetime, np.datetime64, pd.DatetimeIndex]) -> float:
    """
    This function converts a python datetime object to ephemeris time correcting for leap seconds

    If you have spiceypy installed in your python distribution then this is essentially just a wrapper around the
    str2et function from spice.  If you don't have spiceypy installed then this emulates str2et in python code
    with a hardcoded version of the tls kernel in this module.  If the tls kernel is changed and you are not using
    spiceypy then this module needs to be updated!

    :param date: The datetime instance to be converted
    :return: The ephemeris time corresponding to observation_date for use in the spice system
    """

    if not HAS_SPICE:

        leapsecs = leap_seconds(date)

        delta_at = 32 + leapsecs

        time_since_j2000 = (date - J2000_EPOCH).total_seconds() + leapsecs

        mean_anom = _MEAN_ANOM_START + _MEAN_ANOM_VELO * time_since_j2000

        eccentricity = mean_anom + _EB_CONST * np.sin(mean_anom)

        et_m_tai = _DELTA_T_A + _K_CONST * np.sin(eccentricity)

        delta_et = et_m_tai + delta_at

        return delta_et + (date - J2000_EPOCH).total_seconds()

    else:

        try:
            return spice.str2et(date.isoformat())
        except AttributeError:
            return spice.str2et(date.astype(datetime.datetime).isoformat())


def _move_et_end_and_drop_lt(func: Callable, targ: str, ref: str, abcorr: str, obs: str, et: float) -> np.ndarray:
    """
    This helper function simply rearranges the order of inputs for a call to spkpos or spkezr and drops the light time
    output from those functions so only the position or position and velocity are returned
    :param func: spkpos or spkezr function
    :param targ: the target
    :param ref: the reference frame
    :param abcorr: the aberrations and lightime corrections flag
    :param obs: the observer
    :param et: the ephemeris time
    :return: The position or position and velocity at the requested time/settings
    """
    res, _ = func(targ, et, ref, abcorr, obs)

    return res


def create_callable_position(target: str, frame: str, corrections: str, observer: str) -> Callable:
    """
    This function generates a partial function of the spkpos method from spice with the target, frame, abcorr, and
    observer inputs already set (so that the only remaining input is the ephemeris time).

    This function can only be used if the spiceypy package is installed and available for the current python
    implementation.  The resulting callable from this method will return the position vector from the observer to the
    target expressed in the specified frame using the input corrections.

    Calls to the output from this function are equivalent to

        >>> import spiceypy
        >>> et = 5.23
        >>> state, _ = spiceypy.spkpos(target, et, frame, corrections, observer)

    where ``et`` is the ephemeris time passed to the output from this function.

    :param target: The target to return the position of (targ)
    :param frame: The frame to return the position vector in (ref)
    :param corrections: The flag for aberration and light time corrections (abcorr)
    :param observer: The origin of the position vector (obs)
    :return: A partial function wrapper around the spiceypy.spkpos function which only requires the ephemeris time to be
             input
    """

    if HAS_SPICE:

        return partial(_move_et_end_and_drop_lt, spice.spkpos, target, frame, corrections, observer)

    else:

        raise ImportError('Spiceypy must be installed to use this utility\n')


def create_callable_state(target: str, frame: str, corrections: str, observer: str) -> Callable:
    """
    This function generates a partial function of the spkezr method from spice with the target, frame, abcorr, and
    observer inputs already set (so that the only remaining input is the ephemeris time).

    This function can only be used if the spiceypy package is installed and available for the current python
    implementation.  The resulting callable from this method will return the position and velocity vector from the
    observer to the target expressed in the specified frame using the input corrections.

    Calls to the output from this function are equivalent to

        >>> import spiceypy
        >>> et = 5.23
        >>> state, _ = spiceypy.spkezr(target, et, frame, corrections, observer)

    where ``et`` is the ephemeris time passed to the output from this function

    :param target: The target to return the state of (targ)
    :param frame: The frame to return the state vector in (ref)
    :param corrections: The flag for aberration and light time corrections (abcorr)
    :param observer: The origin of the state vector (obs)
    :return: A partial function wrapper around the spiceypy.spkpos function which only requires the ephemeris time to be
             input
    """

    if HAS_SPICE:

        return partial(_move_et_end_and_drop_lt, spice.spkezr, target, frame, corrections, observer)

    else:

        raise ImportError('Spiceypy must be installed to use this utility')


def _rotation_to_attitude(func: Callable) -> Callable:
    """
    This function changes an output from a 3x3 rotation matrix to an Rotation object

    :param func:  A function that returns a 3x3 rotation matrix
    :return: A callable which returns and Rotation object
    """
    def return_rotation(et: float) -> Rotation:
        """
        Convert the output of func into a GIANT Rotation object

        :param et: The ephemeris time in seconds the rotation is requested at
        :return: The Rotation object at the requested time.
        """
        return Rotation(func(et))

    return return_rotation


def create_callable_orientation(from_frame: str, to_frame: str) -> Callable:
    """
    This function generates a partial function of the pxform function from spice with the from and the to frames
    specified.

    This function can only be used if the spiceypy package is installed and available for the current python
    implementation.  The resulting callable from this method will return the rotation to go from frame from_frame to
    frame to_frame at time et as an Rotation object.

    Calls to the output from this function are equivalent to

        >>> import giant.rotations as at
        >>> import spiceypy
        >>> et = 5.23
        >>> rot = Rotation(spiceypy.pxform(from_frame, to_frame, et))

    where ``et`` is the input you pass to the result from this function

    :param from_frame: The NAIF frame name of the reference frame
    :param to_frame: The NAIF frame name of the target frame for the transformation
    :return: A partial function wrapper around the spiceypy.pxform function which only requires the et input
    """

    if HAS_SPICE:

        return _rotation_to_attitude(partial(spice.pxform, from_frame, to_frame))

    else:

        raise ImportError('Spiceypy must be installed to use this utility\n')


def et_callable_to_datetime_callable(func: Callable) -> Callable:
    """
    This function takes a callable object that takes a time in ephemeris time and returns a callable object that takes
    a time as a python datetime object.

    Calls to the output from this function are equivalent to

        >>> import spiceypy
        >>> import datetime
        >>> observation_date = datetime.datetime.now()
        >>> et = spiceypy.str2et(observation_date.isoformat())
        >>> res = func(et)

    where ``observation_date`` is the datetime object that would be passed to the output from this function and ``func``
    is the input to this function.

    :param func: The callable to be converted to datetime callable
    :return: An object which accepts a datetime object inplace of an et float
    """

    def datetime_callable(date: datetime.datetime):
        """
        This function returns a callable object which accepts a python datetime object instead of an et float.

        :param date: A python datetime object to be used to call the function
        :type date: datetime.datetime
        :return: The result of func(datetime_to_et(observation_date)) which is generally a numpy array containing either
                 the position or state of an object
        """

        et = datetime_to_et(date)
        return func(et)

    return datetime_callable


class SpicePosition:
    """
    This class creates a callable that returns the position (in km) of one object to another given a python datetime.

    This class works by storing the "static" inputs to the call to the spice function ``spkpos`` (specifically the
    target, reference frame, corrections, and observer). This makes it easy to define an object that can be used
    throughout GIANT in the :class:`.SceneObj`.  In addition, this class provides access to the light time computation
    from spice using :meth:`light_time` and both the position and light time using :meth:`position_light_time`.  The
    interface to spice is provided through spiceypy.

    For more details, refer to the NAIF spice documentation for spkpos at
    https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/spicelib/spkpos.html.

    This class should be preferred over the :func:`.create_callable_position` because it works much better with pickle
    and provides more functionality.
    """

    def __init__(self, target: str, reference_frame: str, corrections: str, observer: str):
        """
        :param target: The name/integer spice id for the object we are computing the position vector to as a string.
        :param reference_frame: The frame we are computing the position vector in as a string
        :param corrections: The corrections to use when computing the position vector (usually for GIANT this should be
                            ``'NONE'``
        :param observer: The name/integer spice id for the object we are computing the position vector from as a string.
        """

        self.target: str = target
        """
        The object we are computing the position vector to as a string.  
        
        This usually is the name of the object or its integer spice id as a string.  This is passed to the 
        ``TARG`` input for spkpos.
        """

        self.reference_frame: str = reference_frame
        """
        The frame we are to compute the position vector in as a string

        For GIANT this should usually be the inertial frame ``'J200'``.  This is passed to the ``REF`` input for spkpos.
        """

        self.corrections: str = corrections
        """
        The corrections we are to apply when computing the position vector.
        
        Valid inputs are ``'NONE'`` for no correction, ``'LT'`` for light time only corrections, ``'LT+S'`` for light 
        time plus stellar aberration corrections, ``'CN'`` for converged light time only corrections, and ``'CN+S'`` for 
        converged light time plus aberration corrections.

        For GIANT this should usually be ``'NONE'`` since it does its own corrections.  This is passed to the ``ABCORR`` 
        input for spkpos.
        """


        self.observer: str = observer
        """
        The object we are computing the position vector from as a string.  

        This usually is the name of the object or its integer spice id as a string.  This is passed to the 
        ``OBS`` input for spkpos.
        """


    def __call__(self, date: datetime.datetime) -> np.ndarray:
        """
        Make the call to spkpos given the stored settings at the input date returning the position vector in kilometers.

        Specifically the call is

        .. code::

            spiceypy.spkpos(self.target, datetime_to_et(date), self.reference_frame, self.corrections, self.observer)[0]

        where :func:`.datetime_to_et` converts a datetime object into ephemeris (TDB) seconds since the J2000 epoch
        and we take the first return to get the position vector and not the light time.

        :param date: The date we are querying spice at as a datetime object
        :return: The position vector from :attr:`observer` to :attr:`target` in frame :attr:`reference_frame` using
                 corrections :attr:`corrections` at ``date``
        """

        return spice.spkpos(self.target, datetime_to_et(date), self.reference_frame, self.corrections, self.observer)[0]

    def light_time(self, date: datetime.datetime) -> float:
        """
        Make the call to spkpos given the stored settings at the input date returning only the light time in TDB
        seconds.

        Specifically the call is

        .. code::

            spiceypy.spkpos(self.target, datetime_to_et(date), self.reference_frame, self.corrections, self.observer)[1]

        where :func:`.datetime_to_et` converts a datetime object into ephemeris (TDB) seconds since the J2000 epoch
        and we take the second return to get the light time and not the position vector

        :param date: The date we are querying spice at as a datetime object
        :return: The one way light time between the observer and the target in TDB seconds.
        """

        return spice.spkpos(self.target, datetime_to_et(date), self.reference_frame, self.corrections, self.observer)[1]

    def position_light_time(self, date: datetime.datetime) -> Tuple[np.ndarray, float]:
        """
        Make the call to spkpos given the stored settings at the input date returning both the position vector in
        kilometers and the light time in TDB
        seconds.

        Specifically the call is

        .. code::

            spiceypy.spkpos(self.target, datetime_to_et(date), self.reference_frame, self.corrections, self.observer)

        where :func:`.datetime_to_et` converts a datetime object into ephemeris (TDB) seconds since the J2000 epoch.

        :param date: The date we are querying spice at as a datetime object
        :return: The relative position vector in kilometers and the one way light time between the observer and the
                 target in TDB seconds as a tuple.
        """

        return spice.spkpos(self.target, datetime_to_et(date), self.reference_frame, self.corrections, self.observer)


class SpiceState:
    """
    This class creates a callable that returns the state vector (position and velocity, km and km/s respectively) of one
    object to another given a python datetime.

    This class works by storing the "static" inputs to the call to the spice function ``spkezr`` (specifically the
    target, reference frame, corrections, and observer). In addition, this class provides access to the light time
    computation from spice using :meth:`light_time` and both the state and light time using :meth:`state_light_time`.
    The interface to spice is provided through spiceypy.

    For more details, refer to the NAIF spice documentation for spkezr at
    https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/spicelib/spkezr.html.

    This class should be preferred over the :func:`.create_callable_state` because it works much better with pickle
    and provides more functionality.
    """

    def __init__(self, target: str, reference_frame: str, corrections: str, observer: str):
        """
        :param target: The name/integer spice id for the object we are computing the state vector to as a string.
        :param reference_frame: The frame we are computing the state vector in as a string
        :param corrections: The corrections to use when computing the state vector (usually for GIANT this should be
                            ``'NONE'``
        :param observer: The name/integer spice id for the object we are computing the state vector from as a string.
        """

        self.target: str = target
        """
        The object we are computing the state vector to as a string.  

        This usually is the name of the object or its integer spice id as a string.  This is passed to the 
        ``TARG`` input for spkezr.
        """

        self.reference_frame: str = reference_frame
        """
        The frame we are to compute the state vector in as a string

        For GIANT this should usually be the inertial frame ``'J200'``.  This is passed to the ``REF`` input for spkezr.
        """

        self.corrections: str = corrections
        """
        The corrections we are to apply when computing the state vector.

        Valid inputs are ``'NONE'`` for no correction, ``'LT'`` for light time only corrections, ``'LT+S'`` for light 
        time plus stellar aberration corrections, ``'CN'`` for converged light time only corrections, ``'CN+S'`` for 
        converged light time plus aberration corrections, ``'XLT'`` for transmission light time only, ``'XLT+S` for 
        transmission light time and stellar aberration, ``'XCN'`` for transmission converged light time only, and 
        ``'XCN+S'`` for transmission converged light time plus stellar aberration.

        For GIANT this should usually be ``'NONE'`` since it does its own corrections.  This is passed to the ``ABCORR`` 
        input for spkezr.
        """

        self.observer: str = observer
        """
        The object we are computing the state vector from as a string.  

        This usually is the name of the object or its integer spice id as a string.  This is passed to the 
        ``OBS`` input for spkezr.
        """

    def __call__(self, date: datetime.datetime) -> np.ndarray:
        """
        Make the call to spkezr given the stored settings at the input date returning the state vector
        (position, velocity) in kilometers and kilometers per second respectively.

        Specifically the call is

        .. code::

            spiceypy.spkezr(self.target, datetime_to_et(date), self.reference_frame, self.corrections, self.observer)[0]

        where :func:`.datetime_to_et` converts a datetime object into ephemeris (TDB) seconds since the J2000 epoch
        and we take the first return to get the state vector and not the light time.

        :param date: The date we are querying spice at as a datetime object
        :return: The state vector from :attr:`observer` to :attr:`target` in frame :attr:`reference_frame` using
                 corrections :attr:`corrections` at ``date``
        """

        return spice.spkpos(self.target, datetime_to_et(date), self.reference_frame, self.corrections, self.observer)[0]

    def light_time(self, date: datetime.datetime) -> float:
        """
        Make the call to spkezr given the stored settings at the input date returning only the light time in TDB
        seconds.

        Specifically the call is

        .. code::

            spiceypy.spkezr(self.target, datetime_to_et(date), self.reference_frame, self.corrections, self.observer)[1]

        where :func:`.datetime_to_et` converts a datetime object into ephemeris (TDB) seconds since the J2000 epoch
        and we take the second return to get the light time and not the position vector

        :param date: The date we are querying spice at as a datetime object
        :return: The one way light time between the observer and the target in TDB seconds.
        """

        return spice.spkezr(self.target, datetime_to_et(date), self.reference_frame, self.corrections, self.observer)[1]

    def position_light_time(self, date: datetime.datetime) -> Tuple[np.ndarray, float]:
        """
        Make the call to spkezr given the stored settings at the input date returning both the state vector in
        kilometers and kilometers per second and the light time in TDB seconds.

        Specifically the call is

        .. code::

            spiceypy.spkezr(self.target, datetime_to_et(date), self.reference_frame, self.corrections, self.observer)

        where :func:`.datetime_to_et` converts a datetime object into ephemeris (TDB) seconds since the J2000 epoch.

        :param date: The date we are querying spice at as a datetime object
        :return: The relative state vector in kilometers and kilometers per second and the one way light time between
                 the observer and the target in TDB seconds as a tuple.
        """

        return spice.spkezr(self.target, datetime_to_et(date), self.reference_frame, self.corrections, self.observer)


class SpiceOrientation:
    """
    This class creates a callable that returns the rotation  from one frame to another as a :class:`.Rotation` given a
    python datetime.

    This class works by storing the "static" inputs to the call to the spice function ``pxform`` (specifically the
    from frame and to frame). The interface to spice is provided through spiceypy.

    For more details, refer to the NAIF spice documentation for pxform at
    https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/spicelib/pxform.html.

    This class should be preferred over the :func:`.create_callable_orientation` because it works much better with
    pickle.
    """

    def __init__(self, from_frame: str, to_frame: str):
        """
        :param from_frame: The name/integer spice id for the starting frame as a string.
        :param to_frame: The name/integer spice id for the ending frame as a string.
        """

        self.from_frame: str = from_frame
        """
        The frame that we are starting in as a string.

        This usually is the name of the frame or its integer spice id as a string.  This is passed to the 
        ``FROM`` input for pxform.
        """

        self.to_frame: str = to_frame
        """
        The frame that we are rotating to as a string.

        This usually is the name of the frame or its integer spice id as a string.  This is passed to the 
        ``TO`` input for pxform.
        """

    def __call__(self, date: datetime.datetime) -> np.ndarray:
        """
        Make the call to pxform given the stored settings at the input date returning the rotation as a
        :class:`.Rotation` object.

        Specifically the call is

        .. code::

            giant.rotations.Rotation(spiceypy.pxform(self.from_frame, self.to_frame, datetime_to_et(date)))

        where :func:`.datetime_to_et` converts a datetime object into ephemeris (TDB) seconds since the J2000 epoch.

        :param date: The date we are querying spice at as a datetime object
        :return: The rotation from :attr:`from_frame` to :attr:`to_fram` at time ``date`` as a :class:`.Rotation`.
        """

        return Rotation(spice.pxform(self.from_frame, self.to_frame, datetime_to_et(date)))
