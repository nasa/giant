"""
test_point_source_finder
========================

Tests the methods and classes contained in the point_source_finder submodule of GIANT.
"""

from unittest import TestCase
import numpy as np
from giant.image_processing import point_source_finder as gpsf
from giant.point_spread_functions.gaussians import IterativeGaussian


class TestPointSourceFinder(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = np.abs(np.random.randn(1000, 1000))

        x, y = np.meshgrid(np.arange(20, 30), np.arange(50, 60))  # form the underlying grid
        z = 20000 * np.exp(-(np.power(x - 25, 2) / (2 * (3 ** 2)) +
                             (np.power(y - 55, 2) / (2 * (1 ** 2)))))  # generate the height data

        cls.image[y, x] += z

        x, y = np.meshgrid(np.arange(500, 530), np.arange(800, 820))  # form the underlying grid
        z = 20000 * np.exp(-(np.power(x - 520.2, 2) / (2 * (2 ** 2)) + (np.power(y - 810.9, 2) / (2 * (2 ** 2)))))
        # generate the height data

        cls.image[y, x] += z

        cls.psf = gpsf.PointOfInterestFinder()
        cls.psf.min_size = 5
        cls.psf.max_size = 200
        cls.psf.centroid_size = 2
        cls.psf.point_spread_function = IterativeGaussian

    def test_corners_to_roi(self):
        im = np.random.randn(500, 600)
        corn_row = [5.5, 3, 6.5, 8.9]
        corn_col = [4.3, 2.7, 3.3, 7.8]
        roi = im[self.psf.corners_to_roi(corn_row, corn_col)]
        np.testing.assert_array_equal(roi.shape, (7, 7))

    def test_find_poi_in_roi(self):
        with self.subTest(region=None):
            poi, _, __ = self.psf.find_poi_in_roi(self.image)
            poi = np.array(poi)
            np.testing.assert_array_equal(poi, [[25, 55], [520, 811]])

        with self.subTest(region=(slice(30, 70), slice(5, 50))):
            region = self.psf.corners_to_roi([30, 70], [5, 50])

            poi, _, __ = self.psf.find_poi_in_roi(self.image, region=region)

            np.testing.assert_array_equal(poi, [[25, 55]])

        with self.subTest(region=(slice(790, 840), slice(500, 550))):
            region = self.psf.corners_to_roi([790, 840], [500, 550])

            poi, _, __ = self.psf.find_poi_in_roi(self.image, region=region)

            np.testing.assert_array_equal(poi, [[520, 811]])

        with self.subTest(region=(slice(200, 400), slice(300, 450))):
            region = self.psf.corners_to_roi([200, 400], [300, 450])

            poi, _, __ = self.psf.find_poi_in_roi(self.image, region=region)

            self.assertFalse(poi)

    def test_refine_locations(self):
        poi_in = gpsf.FindPPOIInROIOut(peak_location=[np.array([25, 55]), np.array([520, 810])], stats=[[], []], peak_snr=[0, 0])
        refined_locs, _, _, _, _ = self.psf.refine_locations(self.image, poi_in)

        np.testing.assert_array_almost_equal(refined_locs, [[25, 55], [520.2, 810.9]], decimal=2)

    def test_locate_subpixel_poi_in_roi(self):
        with self.subTest(region=None):
            out = self.psf(self.image)
            refined_locs = out.centroids

            np.testing.assert_array_almost_equal(refined_locs, [[25, 55], [520.2, 810.9]], decimal=2)

        with self.subTest(region=(slice(30, 70), slice(5, 50))):
            region = self.psf.corners_to_roi([30, 70], [5, 50])

            out = self.psf(self.image, region=region)
            refined_locs = out.centroids

            np.testing.assert_array_almost_equal(refined_locs, [[25, 55]], decimal=2)

        with self.subTest(region=(slice(790, 840), slice(500, 550))):
            region = self.psf.corners_to_roi([790, 840], [500, 550])

            out = self.psf(self.image, region=region)
            refined_locs = out.centroids

            np.testing.assert_array_almost_equal(refined_locs, [[520.2, 810.9]], decimal=2)

        with self.subTest(region=(slice(200, 400), slice(300, 450))):
            region = self.psf.corners_to_roi([200, 400], [300, 450])

            out = self.psf(self.image, region=region)
            refined_locs = out.centroids

            self.assertEqual(refined_locs.size, 0)


if __name__ == '__main__':
    import unittest

    unittest.main()