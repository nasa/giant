"""
test_local_maxima
=================

Tests the methods and classes contained in the local_maxima submodule of GIANT.
"""

from unittest import TestCase
import numpy as np
from giant.image_processing import local_maxima


class TestLocalMaxima(TestCase):
    def test_local_maxima(self):
        im = [[0, 1, 2, 20, 1],
              [5, 2, 1, 3, 1],
              [0, 1, 2, 10, 1],
              [1, 2, 10, -2, -5],
              [50, 2, -1, -2, 30]]

        desired = [[False, False, False, True, False],
                   [True, False, False, False, False],
                   [False, False, False, True, False],
                   [False, False, True, False, False],
                   [True, False, False, False, True]]

        np.testing.assert_array_equal(local_maxima(im), desired)


if __name__ == '__main__':
    import unittest
    unittest.main()