"""
test_spiceutils
===============

Tests the functionality of the routines in the spice_interface module.

The majority of these tests need the spiceypy library to be installed to run (just like the majority of the spice_interface
functions require spiceypy).

Test Cases
__________
"""

from unittest import TestCase, skipUnless

import datetime

import numpy as np

import giant.utilities.spice_interface as spint

import os

try:
    import spiceypy as spice
    HASSPICE = True

except ImportError:
    spice = None
    HASSPICE = False


LOCALDIR = os.path.dirname(os.path.realpath(__file__))


class TestLeapseconds(TestCase):
    """
    Test to ensure that the correct number of leap_seconds are returned by the leap_seconds function
    """

    def test_leapseconds(self):

        date = datetime.datetime(2017, 2, 1, 0, 0, 0)

        ls = spint.leap_seconds(date)

        self.assertEqual(ls, 5)

        date = datetime.datetime(1950, 2, 1, 0, 0, 0)

        ls = spint.leap_seconds(date)

        self.assertEqual(ls, -23)

        date = datetime.datetime(2000, 1, 1, 0, 0, 0)

        ls = spint.leap_seconds(date)

        self.assertEqual(ls, 0)

        date = datetime.datetime(2006, 1, 1, 0, 0, 0)

        ls = spint.leap_seconds(date)

        self.assertEqual(ls, 1)


class TestDatetime2Et(TestCase):
    """
    Test to ensure that the conversions between UTC and ET are correct
    """

    def test_datetime2et(self):

        if HASSPICE:
            spice.furnsh(os.path.join(LOCALDIR, '..', 'test_data', 'naif0012.tls'))

            with self.subTest(sputil_hasspice=True):  # this is something of a silly test
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

            with self.subTest(sputil_hasspice=False):  # this tests the manual conversion from datetime to et is correct
                spint.HAS_SPICE = False
                date = datetime.datetime(2017, 2, 1, 0, 0, 0)

                et = spint.datetime_to_et(date)

                et_spice = spice.str2et(date.isoformat())

                self.assertAlmostEqual(et, et_spice)

                date = datetime.datetime(1950, 2, 1, 0, 0, 0)

                et = spint.datetime_to_et(date)

                et_spice = spice.str2et(date.isoformat())

                self.assertAlmostEqual(et, et_spice)

                date = datetime.datetime(2000, 1, 1, 0, 0, 0)

                et = spint.datetime_to_et(date)

                et_spice = spice.str2et(date.isoformat())

                self.assertAlmostEqual(et, et_spice)

                date = datetime.datetime(2006, 1, 1, 0, 0, 0)

                et = spint.datetime_to_et(date)

                et_spice = spice.str2et(date.isoformat())

                self.assertAlmostEqual(et, et_spice)

                spint.HAS_SPICE = True

            spice.kclear()

        else:  # if we don't have spice then use hard coded checks

            date = datetime.datetime(2017, 2, 1, 0, 0, 0)

            et = spint.datetime_to_et(date)

            et_spice = 539179269.1847936

            self.assertAlmostEqual(et, et_spice)

            date = datetime.datetime(1950, 2, 1, 0, 0, 0)

            et = spint.datetime_to_et(date)

            et_spice = -1575201558.8151963

            self.assertAlmostEqual(et, et_spice)

            date = datetime.datetime(2000, 1, 1, 0, 0, 0)

            et = spint.datetime_to_et(date)

            et_spice = -43135.816087188054

            self.assertAlmostEqual(et, et_spice)

            date = datetime.datetime(2006, 1, 1, 0, 0, 0)

            et = spint.datetime_to_et(date)

            et_spice = 189345665.1839256

            self.assertAlmostEqual(et, et_spice)


class TestCreateCallablePosition(TestCase):
    """
    This test ensures that the partial objects returned by the create_callable_position function return the correct 
    results.
    """

    @skipUnless(HASSPICE, "Cannot perform CreateCallablePosition test if spice is not installed")
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

    @skipUnless(HASSPICE, "Cannot perform CreateCallableState test if spice is not installed")
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

    @skipUnless(HASSPICE, "Cannot perform CreateCallableState test if spice is not installed")
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

        if HASSPICE:
            spice.furnsh(os.path.join(LOCALDIR, '..', 'test_data', 'naif0012.tls'))

        def et_fun(ephemtime):

            return ephemtime

        dt_fun = spint.et_callable_to_datetime_callable(et_fun)

        date = datetime.datetime(2011, 3, 5, 3, 44, 23)

        et = spint.datetime_to_et(date)

        self.assertAlmostEqual(et, dt_fun(date))

        if HASSPICE:
            spice.kclear()
