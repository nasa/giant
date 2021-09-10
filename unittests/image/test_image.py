"""
test_image
===============

Tests the functionality of the OpNavImage class.  Only tests the ability to read/create an image


Test Cases
__________
"""

from unittest import TestCase

import numpy as np

from giant.image import OpNavImage, ExposureType
from giant.rotations import Rotation

from datetime import datetime

import cv2

import os


LOCALDIR = os.path.dirname(os.path.realpath(__file__))


class TestOpNavImage(TestCase):
    """
    Test the OpNavImage class
    """

    def test___init__(self):

        x = np.random.randn(100, 100)

        im = OpNavImage(x.copy(), observation_date=datetime(1, 1, 1), instrument='camera', spacecraft='awesome',
                        rotation_inertial_to_camera=Rotation([0, 0, 0]), position=[1, 2, 3], velocity=[4, 5, 6],
                        file='hello/world', exposure=20.5, parse_data=False,
                        exposure_type='long', saturation=100, dark_pixels=[1, 2, 3, 4, 5], temperature=20)

        np.testing.assert_array_equal(x, im)

        self.assertEqual(im.observation_date, datetime(1, 1, 1))
        self.assertEqual(im.instrument, 'camera')
        self.assertEqual(im.spacecraft, 'awesome')
        self.assertEqual(im.rotation_inertial_to_camera, Rotation([0, 0, 0]))
        np.testing.assert_array_equal(im.position, [1, 2, 3])
        np.testing.assert_array_equal(im.velocity, [4, 5, 6])
        self.assertEqual(im.file, 'hello/world')
        self.assertEqual(im.exposure, 20.5)
        self.assertEqual(im.exposure_type, ExposureType.LONG)
        self.assertEqual(im.saturation, 100)
        np.testing.assert_array_equal(im.dark_pixels, [1, 2, 3, 4, 5])
        self.assertEqual(im.temperature, 20)

        self.assertIsInstance(im, OpNavImage)

        with self.assertRaises(NotImplementedError):
            OpNavImage(x, parse_data=True)

    def test_view(self):

        x = np.random.randn(100, 100)

        y = x.view(OpNavImage)

        np.testing.assert_array_equal(x, y)

        self.assertIsInstance(y, OpNavImage)

    def test_load_image(self):
        exts = ['.bmp', '.jpeg', '.jpg', '.jpe',
                '.png', '.pgm', '.ppm', '.ras',
                '.tiff', '.tif', '.fits']

        base = os.path.join(LOCALDIR, '..', 'test_data', 'logo')

        comp = cv2.imread(base + '.png', cv2.IMREAD_GRAYSCALE).astype(np.float64)
        comp /= comp.max()

        comp[comp == 0] = 1

        for ext in exts:

            with self.subTest(ext=ext):

                im = OpNavImage.load_image(base + ext).astype(np.float64)

                im /= im.max()

                im[im == 0] = 1

                if ext == '.fits':

                    im = np.flipud(im)

                self.assertLessEqual(np.median(np.abs(im-comp)), 0.1)
