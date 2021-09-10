import unittest
from unittest import TestCase
import giant.ray_tracer.rays as g_rays
import giant.rotations as at
import numpy as np
import copy


class TestRay(TestCase):

    def setUp(self):

        self.ray_start = np.array([1, 2, 3])
        self.ray_direction = np.array([4, 5, 6])
        self.ray = g_rays.Rays(self.ray_start, self.ray_direction)

        self.rays_start = np.array([[1, 3], [4, 5], [6, 7]])
        self.rays_direction = np.array([[4, 5], [5, 6], [7, 8]])
        self.rays = g_rays.Rays(self.rays_start, self.rays_direction)
        self.num_rays = 2

        self.rotation = at.Rotation([np.pi, np.pi / 2, np.pi / 4])

    def test_creation_one_ray(self):

        for seq_type in [list, np.array, tuple]:

            with self.subTest(seq_type=seq_type):

                ray = g_rays.Rays(seq_type(self.ray_start), seq_type(self.ray_direction))

                np.testing.assert_array_equal(ray._start, self.ray_start)
                np.testing.assert_array_equal(ray._direction, self.ray_direction)

                self.assertEqual(ray.num_rays, 1)

    def test_creation_multi_ray(self):

        for seq_type in [list, np.array, tuple]:

            with self.subTest(seq_type=seq_type):

                rays = g_rays.Rays(seq_type(self.rays_start), seq_type(self.rays_direction))

                np.testing.assert_array_equal(rays._start, self.rays_start)
                np.testing.assert_array_equal(rays._direction, self.rays_direction)

                self.assertEqual(rays.num_rays, self.num_rays)

    def test_single_iter(self):

        for ray in self.ray:
            self.assertIsInstance(ray, g_rays.Rays)

            np.testing.assert_array_equal(ray._start, self.ray_start)
            np.testing.assert_array_equal(ray._direction, self.ray_direction)

    def test_multi_iter(self):

        for ind, ray in enumerate(self.rays):
            self.assertIsInstance(ray, g_rays.Rays)

            np.testing.assert_array_equal(ray._start, self.rays_start[:, ind])
            np.testing.assert_array_equal(ray._direction, self.rays_direction[:, ind])

    def test_single_getitem(self):

        with self.assertRaises(ValueError):
            _ = self.ray[0]

    def test_multi_getitem(self):

        second = self.rays[1]

        np.testing.assert_array_equal(second._start, self.rays_start[:, 1])
        np.testing.assert_array_equal(second._direction, self.rays_direction[:, 1])

    def test_single_len(self):

        self.assertEqual(len(self.ray), 1)

    def test_multi_len(self):

        self.assertEqual(len(self.rays), 2)

    # TODO: Multiplication tests -- maybe

    def test_single_rotate(self):

        ray_copy = copy.deepcopy(self.ray)

        ray_copy.rotate(self.rotation)

        np.testing.assert_array_almost_equal(ray_copy._start, np.matmul(self.rotation.matrix, self.ray._start),
                                             err_msg="Rotation rotation ray start")
        np.testing.assert_array_almost_equal(ray_copy._direction,
                                             np.matmul(self.rotation.matrix, self.ray._direction),
                                             err_msg="Rotation rotation ray direction")

        ray_copy = copy.deepcopy(self.ray)

        ray_copy.rotate(self.rotation.matrix)

        np.testing.assert_array_almost_equal(ray_copy._start, np.matmul(self.rotation.matrix, self.ray._start),
                                             err_msg="Matrix rotation ray start")
        np.testing.assert_array_almost_equal(ray_copy._direction,
                                             np.matmul(self.rotation.matrix, self.ray._direction),
                                             err_msg="Matrix rotation ray direction")

    def test_multi_rotate(self):

        ray_copy = copy.deepcopy(self.rays)

        ray_copy.rotate(self.rotation)

        np.testing.assert_array_almost_equal(ray_copy._start, np.matmul(self.rotation.matrix, self.rays._start),
                                             err_msg="Rotation rotation ray start")
        np.testing.assert_array_almost_equal(ray_copy._direction,
                                             np.matmul(self.rotation.matrix, self.rays._direction),
                                             err_msg="Rotation rotation ray direction")

        ray_copy = copy.deepcopy(self.rays)

        ray_copy.rotate(self.rotation.matrix)

        np.testing.assert_array_almost_equal(ray_copy._start, np.matmul(self.rotation.matrix, self.rays._start),
                                             err_msg="Matrix rotation ray start")
        np.testing.assert_array_almost_equal(ray_copy._direction,
                                             np.matmul(self.rotation.matrix, self.rays._direction),
                                             err_msg="Matrix rotation ray direction")

    def test_single_translate(self):

        ray_copy = copy.deepcopy(self.ray)

        ray_copy.translate(5)

        np.testing.assert_array_equal(ray_copy._start, self.ray_start + 5,
                                      err_msg="Scalar Translation")

        ray_copy = copy.deepcopy(self.ray)

        ray_copy.translate(self.ray_start)

        np.testing.assert_array_equal(ray_copy._start, self.ray_start + self.ray_start,
                                      err_msg="Vector Translation")

    def test_multi_translate(self):

        # ray_copy = copy.deepcopy(self.rays)
        #
        # ray_copy.translate(5)
        #
        # np.testing.assert_array_equal(ray_copy._start, self.rays_start + 5,
        #                               err_msg="Scalar Translation")

        ray_copy = copy.deepcopy(self.rays)

        ray_copy.translate(self.ray_start)

        np.testing.assert_array_equal(ray_copy._start, self.rays_start + self.ray_start[..., np.newaxis],
                                      err_msg="Vector Translation")

        ray_copy = copy.deepcopy(self.rays)

        ray_copy.translate(self.rays_start)

        np.testing.assert_array_equal(ray_copy._start, self.rays_start + self.rays_start,
                                      err_msg="Matrix Translation")
