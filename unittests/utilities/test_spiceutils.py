"""
test_spiceutils
===============

Tests the functionality of the routines in the spice_interface module.

The majority of these tests need the spiceypy library to be installed to run (just like the majority of the spice_interface
functions require spiceypy).

Test Cases
__________
"""

from unittest import TestCase

import datetime

import numpy as np

import giant.utilities.spice_interface as spint

import os

import spiceypy as spice


LOCALDIR = os.path.dirname(os.path.realpath(__file__))


class TestDatetime2Et(TestCase):
    """
    Test to ensure that the conversions between UTC and ET are correct
    """

    def test_datetime2et(self):

        spice.furnsh(os.path.join(LOCALDIR, '..', 'test_data', 'naif0012.tls'))

        date = datetime.datetime(2017, 2, 1, 0, 0, 0)

        et = spint.datetime_to_et(date)

        et_spice = spice.str2et(date.isoformat())

        self.assertEqual(et, et_spice)

        date = datetime.datetime(1950, 2, 1, 0, 0, 0)

        et = spint.datetime_to_et(date)

        et_spice = spice.str2et(date.isoformat())

        self.assertEqual(et, et_spice)

        date = datetime.datetime(2000, 1, 1, 0, 0, 0)

        et = spint.datetime_to_et(date)

        et_spice = spice.str2et(date.isoformat())

        self.assertEqual(et, et_spice)

        date = datetime.datetime(2006, 1, 1, 0, 0, 0)

        et = spint.datetime_to_et(date)

        et_spice = spice.str2et(date.isoformat())

        self.assertEqual(et, et_spice)

        spice.kclear()


class TestCreateCallablePosition(TestCase):
    """
    This test ensures that the partial objects returned by the create_callable_position function return the correct 
    results.
    """

    def test_create_callable_position(self):
        spice.furnsh(os.path.join(LOCALDIR, '..', 'test_data', 'de424.bsp'))
        targ = 'Moon'
        frame = 'J2000'
        corrections = 'LT+S'
        observer = 'Earth'

        posfun = spint.create_callable_position(targ, frame, corrections, observer)

        et = 123.23

        pos1 = posfun(et)

        pos2, _ = spice.spkpos(targ, et, frame, corrections, observer)

        np.testing.assert_array_equal(pos1, pos2)

        spice.kclear()


class TestCreateCallableState(TestCase):
    """
    This test ensures that the partial objects returned by the create_callable_state function return the correct 
    results.
    """

    def test_create_callable_state(self):
        spice.furnsh(os.path.join(LOCALDIR, '..', 'test_data', 'de424.bsp'))
        targ = 'Moon'
        frame = 'J2000'
        corrections = 'LT+S'
        observer = 'Earth'

        statefun = spint.create_callable_state(targ, frame, corrections, observer)

        et = 123.23

        state1 = statefun(et)

        state2, _ = spice.spkezr(targ, et, frame, corrections, observer)

        np.testing.assert_array_equal(state1, state2)

        spice.kclear()


class TestCreateCallableOrientation(TestCase):
    """
    This test ensures that the partial objects returned by the create_callable_orientation function return the correct 
    results.
    """

    def test_create_callable_state(self):
        spice.furnsh(os.path.join(LOCALDIR, '..', 'test_data', 'pck00010.tpc'))
        targframe = 'J2000'
        curframe = 'IAU_EARTH'

        orfun = spint.create_callable_orientation(targframe, curframe)

        et = 123.23

        orientation = orfun(et)

        rotmat = spice.pxform(targframe, curframe, et)

        np.testing.assert_array_almost_equal(orientation.matrix, rotmat)

        spice.kclear()


class TestEtCallableToDatetimeCallable(TestCase):
    """
    This test ensures that the function object returned by the et_callable_to_datetime_callable function returns the 
    right result
    """

    def test_et_callable_to_datetime_callable(self):

        spice.furnsh(os.path.join(LOCALDIR, '..', 'test_data', 'naif0012.tls'))

        def et_fun(ephemtime):

            return ephemtime

        dt_fun = spint.et_callable_to_datetime_callable(et_fun)

        date = datetime.datetime(2011, 3, 5, 3, 44, 23)

        et = spint.datetime_to_et(date)

        self.assertAlmostEqual(et, dt_fun(date))

        spice.kclear()
