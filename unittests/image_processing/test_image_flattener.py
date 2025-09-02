"""
test_image_flattener
====================

Tests the methods and classes contained in the image_flattener submodule of GIANT.
"""

from unittest import TestCase
import numpy as np
from typing import cast
from giant.image_processing import image_flattener


class TestImageFlattener(TestCase):
    def test_flatten_image_and_get_noise_level(self):
        test = np.arange(100).reshape(10, 10)

        res = image_flattener.ImageFlattener()(test)

        true_flat = np.array([[-2, -2, -2, -2, -2, -2, -2, -2, -1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 2, 2, 2, 2, 2, 2, 2, 2]])

        np.testing.assert_array_equal(true_flat, res.flattened)

        # 0.5 is the maximum standard deviation that can be computed given the test image
        self.assertLessEqual(cast(float, res.estimated_noise_level), 0.5)
        
    def test_local_flatten_image_and_get_noise_level(self):
        test = np.arange(100).reshape(10, 10)

        res = image_flattener.ImageFlattener(image_flattener.ImageFlattenerOptions(image_flattening_noise_approximation=image_flattener.ImageFlatteningNoiseApprox.LOCAL,
                                                                                   local_flattening_kernel_size=1))(test)

        true_flat = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 10],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 20],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 20],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 20],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 20],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 20],
                              [0, 0, 0, 0, 0, 0, 0, 0, -59, 10],
                              [0, 0, 0, 0, 0, 0, 0, -69, -79, 0],
                              [0, 1, 2, 2, 2, 2, 2, 2, 2, 2]])

        np.testing.assert_array_almost_equal(true_flat, res.flattened)

        # 0.5 is the maximum standard deviation that can be computed given the test image
        np.testing.assert_array_almost_equal(cast(list[float], res.estimated_noise_level), [0, 0, 0, 0, 0, 0, 0, 0, 32.866734])
        

if __name__ == '__main__':
    import unittest
    unittest.main()