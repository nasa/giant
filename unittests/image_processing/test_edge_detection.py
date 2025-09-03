"""
test_edge_detection
===================

Tests the methods and classes contained in the edge_detection submodule of GIANT.
"""

from unittest import TestCase
import numpy as np
from giant.image_processing.edge_detection import pae_subpixel_edge_detector as gpae
from giant.image_processing.edge_detection import pixel_edge_detector as gped
from giant.image_processing.edge_detection import zernike_ramp_edge_detector as gpre


class TestPixelEdgeDetector(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = np.abs(np.random.randn(1000, 1000))

        grid_size = 7
        grid_dist = 1 / grid_size
        grid_start = 0.5 - grid_dist / 2

        radius, center = 50, (200, 300)
        rows, cols = np.meshgrid(np.arange(np.floor(center[0] - radius * 1.1) - grid_start,
                                           np.ceil(center[0] + radius * 1.1) + 0.5,
                                           grid_dist),
                                 np.arange(np.floor(center[1] - radius * 1.1) - grid_start,
                                           np.ceil(center[1] + radius * 1.1) + 0.5,
                                           grid_dist), indexing='ij')

        illum_vals = 1000 * (((rows - center[0]) ** 2 + (cols - center[1]) ** 2) <= radius ** 2).astype(np.float64)
        
        np.add.at(cls.image, (np.round(rows.ravel()).astype(np.int64), 
                              np.round(cols.ravel()).astype(np.int64)), illum_vals.ravel())

        cls.ped = gped.PixelEdgeDetector()

        cls.radius = radius
        cls.center = np.array(center)

    def test_pae_edges(self):
        edges = self.ped.identify_edges(self.image)

        radius_est = np.sqrt(((edges - self.center[::-1].reshape(2, 1)) ** 2).sum(axis=0))

        radius_err = self.radius - radius_est

        self.assertTrue((radius_err < 1).all())

        # TODO: need to figure out why these are so high.
        self.assertLess(radius_err.std(), 0.5)

        self.assertLess(np.abs(radius_err.mean()), 0.11)
        
        
        
class TestPaeEdgeDetector(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = np.abs(np.random.randn(1000, 1000))

        grid_size = 7
        grid_dist = 1 / grid_size
        grid_start = 0.5 - grid_dist / 2

        radius, center = 50, (200, 300)
        rows, cols = np.meshgrid(np.arange(np.floor(center[0] - radius * 1.1) - grid_start,
                                           np.ceil(center[0] + radius * 1.1) + 0.5,
                                           grid_dist),
                                 np.arange(np.floor(center[1] - radius * 1.1) - grid_start,
                                           np.ceil(center[1] + radius * 1.1) + 0.5,
                                           grid_dist), indexing='ij')

        illum_vals = 1000 * (((rows - center[0]) ** 2 + (cols - center[1]) ** 2) <= radius ** 2).astype(np.float64)
        
        np.add.at(cls.image, (np.round(rows.ravel()).astype(np.int64), 
                              np.round(cols.ravel()).astype(np.int64)), illum_vals.ravel())

        cls.ped = gpae.PAESubpixelEdgeDetector(None)

        cls.radius = radius
        cls.center = np.array(center)

    def test_pae_edges(self):
        edges = self.ped.identify_edges(self.image)

        radius_est = np.sqrt(((edges - self.center[::-1].reshape(2, 1)) ** 2).sum(axis=0))

        radius_err = self.radius - radius_est

        self.assertTrue((radius_err < 1).all())

        # TODO: need to figure out why these are so high.
        self.assertLess(radius_err.std(), 0.1)

        self.assertLess(np.abs(radius_err.mean()), 0.01)
        
        
class TestZernikeRampEdgeDetector(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = np.abs(np.random.randn(1000, 1000))

        grid_size = 7
        grid_dist = 1 / grid_size
        grid_start = 0.5 - grid_dist / 2

        radius, center = 50, (200, 300)
        rows, cols = np.meshgrid(np.arange(np.floor(center[0] - radius * 1.1) - grid_start,
                                           np.ceil(center[0] + radius * 1.1) + 0.5,
                                           grid_dist),
                                 np.arange(np.floor(center[1] - radius * 1.1) - grid_start,
                                           np.ceil(center[1] + radius * 1.1) + 0.5,
                                           grid_dist), indexing='ij')

        illum_vals = 1000 * (((rows - center[0]) ** 2 + (cols - center[1]) ** 2) <= radius ** 2).astype(np.float64)
        
        np.add.at(cls.image, (np.round(rows.ravel()).astype(np.int64), 
                              np.round(cols.ravel()).astype(np.int64)), illum_vals.ravel())

        cls.ped = gpre.ZernikeRampEdgeDetector(None)

        cls.radius = radius
        cls.center = np.array(center)

    def test_pae_edges(self):
        edges = self.ped.identify_edges(self.image)

        radius_est = np.sqrt(((edges - self.center[::-1].reshape(2, 1)) ** 2).sum(axis=0))

        radius_err = self.radius - radius_est

        self.assertTrue((radius_err < 1).all())

        # TODO: need to figure out why these are so high.
        self.assertLess(radius_err.std(), 0.1)

        self.assertLess(np.abs(radius_err.mean()), 0.01)
        
if __name__ == '__main__':
    import unittest
    unittest.main()
