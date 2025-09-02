from typing import cast

from unittest import TestCase
import giant.ray_tracer.shapes as g_shapes
from giant.ray_tracer.shapes.axis_aligned_bounding_box import min_max_to_bounding_box
import giant.ray_tracer.rays as g_rays
import giant.rotations as at
import numpy as np
import copy


# noinspection PyTypeChecker
class TestTriangle(TestCase):
    def setUp(self):

        self.single_verts = np.array([[-1, 1, 0],
                                      [-1, -1, 1],
                                      [0, 0, 0]], dtype=np.float64).T

        self.single_facet = np.array([[0, 1, 2]])

        self.single_sides = np.array([[2, 1],
                                      [0, 2],
                                      [0, 0]], dtype=np.float64)

        self.single_normals = np.array([0, 0, 1], dtype=np.float64)

        self.single_albedos = np.array([0, 1, 2])

        self.single_triangle = g_shapes.Triangle64(self.single_verts,
                                                 self.single_albedos,
                                                 self.single_facet)

        # self.multi_verts = np.array([[[-1, 1, 0],
        #                               [-1, -1, 1],
        #                               [0, 0, 0]],
        #                              [[1, 0, 0],
        #                               [-1, 1, 0],
        #                               [0, 0, -1]]])
        self.multi_verts = np.array([[-1, 1, 0, 0],
                                     [-1, -1, 1, 0],
                                     [0, 0, 0, -1]], dtype=np.float64).T

        self.multi_sides = np.array([[[2, 1],
                                      [0, 2],
                                      [0, 0]],
                                     [[-1, -1],
                                      [2, 1],
                                      [0, -1]]], dtype=np.float64)

        self.multi_normals = np.array([[0, 0, 1],
                                       [-2 / np.sqrt(6), -1 / np.sqrt(6), 1 / np.sqrt(6)]])

        self.multi_albedos = np.array([0, 1, 2, 1.5])

        self.multi_facets = np.array([[0, 1, 2], [1, 2, 3]])

        self.multi_triangles = g_shapes.Triangle64(self.multi_verts,
                                                 self.multi_albedos,
                                                 self.multi_facets)

        self.rotation = at.Rotation([np.pi, np.pi / 2, np.pi / 4])

    def test_creation_singletri(self):

        for seq_type in [list, np.array, tuple]:
            with self.subTest(seq_type=seq_type):
                tri = g_shapes.Triangle64(seq_type(self.single_verts),
                                        seq_type(self.single_albedos),
                                        seq_type(self.single_facet))

                np.testing.assert_array_equal(tri.stacked_vertices, self.single_verts.T.reshape(-1, 3, 3), 'verts')
                np.testing.assert_array_equal(tri.albedos, self.single_albedos, 'albedos')
                np.testing.assert_array_equal(tri.sides, self.single_sides.reshape(-1, 3, 2), 'sides')
                np.testing.assert_array_equal(tri.normals, self.single_normals.reshape(-1, 3), 'normals')
                self.assertEqual(tri.num_faces, 1)

    def test_creation_multitri(self):

        for seq_type in [list, np.array, tuple]:
            with self.subTest(seq_type=seq_type):
                tri = g_shapes.Triangle64(seq_type(self.multi_verts),
                                        seq_type(self.multi_albedos),
                                        seq_type(self.multi_facets))

                np.testing.assert_array_equal(tri.stacked_vertices,
                                              self.multi_verts[self.multi_facets].swapaxes(-1, -2),
                                              'verts')
                np.testing.assert_array_equal(tri.albedos, self.multi_albedos, 'albedos')
                np.testing.assert_array_equal(tri.sides, self.multi_sides, 'sides')
                np.testing.assert_array_equal(tri.normals, self.multi_normals, 'normals')
                self.assertEqual(tri.num_faces, 2)

    def test_single_bounding_box(self):

        min_sides = self.single_triangle.bounding_box.min_sides.reshape(1, 3)
        max_sides = self.single_triangle.bounding_box.max_sides.reshape(1, 3)

        self.assertTrue((self.single_triangle.vertices >= min_sides).all())
        self.assertTrue((self.single_triangle.vertices <= max_sides).all())

    def test_multi_bounding_box(self):

        min_sides = self.multi_triangles.bounding_box.min_sides.reshape(1, 3)
        max_sides = self.multi_triangles.bounding_box.max_sides.reshape(1, 3)

        self.assertTrue((self.multi_triangles.vertices >= min_sides).all())
        self.assertTrue((self.multi_triangles.vertices <= max_sides).all())

    def test_single_rotate(self):

        tri_copy = copy.deepcopy(self.single_triangle)

        tri_copy.rotate(self.rotation)

        np.testing.assert_array_almost_equal(tri_copy.vertices, np.matmul(self.rotation.matrix,
                                                                          self.single_triangle.vertices.T).T,
                                             err_msg="Rotation rotation vertices")
        np.testing.assert_array_almost_equal(tri_copy.normals,
                                             np.matmul(self.rotation.matrix, self.single_triangle.normals.T).T,
                                             err_msg="Rotation rotation normals")
        np.testing.assert_array_almost_equal(tri_copy.sides,
                                             np.matmul(self.rotation.matrix, self.single_triangle.sides),
                                             err_msg="Rotation rotation sides")

        tri_copy = copy.deepcopy(self.single_triangle)

        tri_copy.rotate(self.rotation.matrix)

        np.testing.assert_array_almost_equal(tri_copy.vertices, np.matmul(self.rotation.matrix,
                                                                          self.single_triangle.vertices.T).T,
                                             err_msg="Matrix rotation vertices")
        np.testing.assert_array_almost_equal(tri_copy.normals,
                                             np.matmul(self.rotation.matrix, self.single_triangle.normals.T).T,
                                             err_msg="Matrix rotation normals")
        np.testing.assert_array_almost_equal(tri_copy.sides,
                                             np.matmul(self.rotation.matrix, self.single_triangle.sides),
                                             err_msg="Matrix rotation sides")

    def test_multi_rotate(self):

        tri_copy = copy.deepcopy(self.multi_triangles)

        tri_copy.rotate(self.rotation)

        np.testing.assert_array_almost_equal(tri_copy.vertices, np.matmul(self.rotation.matrix,
                                                                          self.multi_triangles.vertices.T).T,
                                             err_msg="Rotation rotation vertices")
        np.testing.assert_array_almost_equal(tri_copy.normals,
                                             np.matmul(self.rotation.matrix, self.multi_triangles.normals.T).T,
                                             err_msg="Rotation rotation normals")
        np.testing.assert_array_almost_equal(tri_copy.sides,
                                             np.matmul(self.rotation.matrix, self.multi_triangles.sides),
                                             err_msg="Rotation rotation sides")

        tri_copy = copy.deepcopy(self.multi_triangles)

        tri_copy.rotate(self.rotation.matrix)

        np.testing.assert_array_almost_equal(tri_copy.vertices, np.matmul(self.rotation.matrix,
                                                                          self.multi_triangles.vertices.T).T,
                                             err_msg="Matrix rotation vertices")
        np.testing.assert_array_almost_equal(tri_copy.normals,
                                             np.matmul(self.rotation.matrix, self.multi_triangles.normals.T).T,
                                             err_msg="Matrix rotation normals")
        np.testing.assert_array_almost_equal(tri_copy.sides,
                                             np.matmul(self.rotation.matrix, self.multi_triangles.sides),
                                             err_msg="Matrix rotation sides")

    def test_single_translation(self):

        tri_copy = copy.deepcopy(self.single_triangle)

        tri_copy.translate([5, 4, 3])

        np.testing.assert_array_equal(tri_copy.vertices,
                                      self.single_triangle.vertices + np.array([5, 4, 3]).reshape(1, 3))

        tri_copy = copy.deepcopy(self.single_triangle)

        tri_copy.translate([[5, 4, 3]])

        np.testing.assert_array_equal(tri_copy.vertices,
                                      self.single_triangle.vertices + np.array([5, 4, 3]).reshape(1, 3))

        tri_copy = copy.deepcopy(self.single_triangle)

        tri_copy.translate([[5],
                            [4],
                            [3]])

        np.testing.assert_array_equal(tri_copy.vertices,
                                      self.single_triangle.vertices + np.array([5, 4, 3]).reshape(1, 3))

        with self.assertRaises(ValueError):
            tri_copy = copy.deepcopy(self.single_triangle)

            tri_copy.translate([1, 2, 3, 4])

    def test_multi_translation(self):

        tri_copy = copy.deepcopy(self.multi_triangles)

        tri_copy.translate([5, 4, 3])

        np.testing.assert_array_equal(tri_copy.vertices,
                                      self.multi_triangles.vertices + np.array([5, 4, 3]).reshape(1, 3))

        tri_copy = copy.deepcopy(self.multi_triangles)

        tri_copy.translate([[5, 4, 3]])

        np.testing.assert_array_equal(tri_copy.vertices,
                                      self.multi_triangles.vertices + np.array([5, 4, 3]).reshape(1, 3))

        tri_copy = copy.deepcopy(self.multi_triangles)

        tri_copy.translate([[5],
                            [4],
                            [3]])

        np.testing.assert_array_equal(tri_copy.vertices,
                                      self.multi_triangles.vertices + np.array([5, 4, 3]).reshape(1, 3))

        with self.assertRaises(ValueError):
            tri_copy = copy.deepcopy(self.multi_triangles)

            tri_copy.translate([1, 2, 3, 4])

    def test_single_get_albedo(self):

        albedo = cast(float, self.single_triangle.get_albedo(self.single_triangle.stacked_vertices[0, :, 0], 0))

        self.assertAlmostEqual(albedo, 0)

        albedo = cast(float, self.single_triangle.get_albedo(self.single_triangle.stacked_vertices[0, :, 1], 0))

        self.assertAlmostEqual(albedo, 1)

        albedo = cast(float, self.single_triangle.get_albedo(self.single_triangle.stacked_vertices[0, :, 2], 0))

        self.assertAlmostEqual(albedo, 2)

        albedo = cast(float, self.single_triangle.get_albedo(self.single_triangle.stacked_vertices[0, :, :2].mean(axis=-1), 0))

        self.assertAlmostEqual(albedo, 0.5)

        albedo = cast(float, self.single_triangle.get_albedo(self.single_triangle.stacked_vertices[0, :, ::2].mean(axis=-1), 0))

        self.assertAlmostEqual(albedo, 1)

        albedo = cast(float, self.single_triangle.get_albedo(self.single_triangle.stacked_vertices[0, :, 1:].mean(axis=-1), 0))

        self.assertAlmostEqual(albedo, 1.5)

        albedo = cast(float, self.single_triangle.get_albedo(self.single_triangle.stacked_vertices[0].mean(axis=-1), 0))

        self.assertAlmostEqual(albedo, 1)

    def test_multi_get_albedo(self):

        # first triangle
        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[0, :, 0], 0))

        self.assertAlmostEqual(albedo, 0)

        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[0, :, 1], 0))

        self.assertAlmostEqual(albedo, 1)

        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[0, :, 2], 0))

        self.assertAlmostEqual(albedo, 2)

        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[0, :, :2].mean(axis=-1), 0))

        self.assertAlmostEqual(albedo, 0.5)

        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[0, :, ::2].mean(axis=-1), 0))

        self.assertAlmostEqual(albedo, 1)

        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[0, :, 1:].mean(axis=-1), 0))

        self.assertAlmostEqual(albedo, 1.5)

        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[0].mean(axis=-1), 0))

        self.assertAlmostEqual(albedo, 1)

        # second triangle
        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[1, :, 0], 1))

        self.assertAlmostEqual(albedo, 1)

        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[1, :, 1], 1))

        self.assertAlmostEqual(albedo, 2)

        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[1, :, 2], 1))

        self.assertAlmostEqual(albedo, 1.5)

        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[1, :, :2].mean(axis=-1), 1))

        self.assertAlmostEqual(albedo, 1.5)

        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[1, :, ::2].mean(axis=-1), 1))

        self.assertAlmostEqual(albedo, 1.25)

        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[1, :, 1:].mean(axis=-1), 1))

        self.assertAlmostEqual(albedo, 1.75)

        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[1].mean(axis=-1), 1))

        self.assertAlmostEqual(albedo, 1.5)

        # multi triangles
        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[:2, :, 0].T, [0, 1]))

        np.testing.assert_array_almost_equal(albedo, [0, 1])

        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[:2, :, 1].T, [0, 1]))

        np.testing.assert_array_almost_equal(albedo, [1, 2])

        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[:2, :, 2].T, [0, 1]))

        np.testing.assert_array_almost_equal(albedo, [2, 1.5])

        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[:2, :, :2].mean(axis=-1).T,
                                                 [0, 1]))

        np.testing.assert_array_almost_equal(albedo, [0.5, 1.5])

        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[:2, :, ::2].mean(axis=-1).T,
                                                 [0, 1]))

        np.testing.assert_array_almost_equal(albedo, [1, 1.25])

        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[:2, :, 1:].mean(axis=-1).T,
                                                 [0, 1]))

        np.testing.assert_array_almost_equal(albedo, [1.5, 1.75])

        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[:2].mean(axis=-1).T, [0, 1]))

        np.testing.assert_array_almost_equal(albedo, [1, 1.5])

    def test_single_compute_intersect(self):

        single_ray = g_rays.Rays([0, 0, 1], [0, 0, -1])

        results = self.single_triangle.compute_intersect(single_ray)[0]

        self.assertTrue(results["check"])
        np.testing.assert_array_equal(results["intersect"], [0, 0, 0])
        np.testing.assert_array_equal(results["normal"], [0, 0, 1])
        self.assertEqual(results["albedo"], 1.25)
        self.assertEqual(results["facet"], 0)

        single_ray = g_rays.Rays([1, 1, 1], [-1, -1, -1])

        results = self.single_triangle.compute_intersect(single_ray)[0]

        self.assertTrue(results["check"])
        np.testing.assert_array_equal(results["intersect"], [0, 0, 0])
        np.testing.assert_array_equal(results["normal"], [0, 0, 1])
        self.assertEqual(results["albedo"], 1.25)
        self.assertEqual(results["facet"], 0)

        single_ray = g_rays.Rays([0, 0, 1], [0, 0, 1])

        results = self.single_triangle.compute_intersect(single_ray)[0]

        self.assertFalse(results["check"])
        np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["albedo"], np.nan)
        self.assertEqual(results["facet"], -1)

        single_ray = g_rays.Rays([1, 0, 0], [-1, 0, 0])

        results = self.single_triangle.compute_intersect(single_ray)[0]

        self.assertFalse(results["check"])
        np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["albedo"], np.nan)
        self.assertEqual(results["facet"], -1)

        single_ray = g_rays.Rays([-100, 0, 1], [0, 0, -1])

        results = self.single_triangle.compute_intersect(single_ray)[0]

        self.assertFalse(results["check"])
        np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["albedo"], np.nan)
        self.assertEqual(results["facet"], -1)

        single_ray = g_rays.Rays([100, 0, 1], [0, 0, -1])

        results = self.single_triangle.compute_intersect(single_ray)[0]

        self.assertFalse(results["check"])
        np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["albedo"], np.nan)
        self.assertEqual(results["facet"], -1)

    def test_multi_compute_intersect(self):
        single_ray = g_rays.Rays([0, 0, 1], [0, 0, -1])

        results = self.multi_triangles.compute_intersect(single_ray)[0]

        self.assertTrue(results["check"])
        np.testing.assert_array_equal(results["intersect"], [0, 0, 0])
        np.testing.assert_array_equal(results["normal"], [0, 0, 1])
        self.assertEqual(results["albedo"], 1.25)
        self.assertEqual(results["facet"], 0)

        single_ray = g_rays.Rays([1, 1, 1], [-1, -1, -1])

        results = self.multi_triangles.compute_intersect(single_ray)[0]

        self.assertTrue(results["check"])
        np.testing.assert_array_equal(results["intersect"], [0, 0, 0])
        np.testing.assert_array_equal(results["normal"], [0, 0, 1])
        self.assertEqual(results["albedo"], 1.25)
        self.assertEqual(results["facet"], 0)

        single_ray = g_rays.Rays([0, 0, 1], [0, 0, 1])

        results = self.multi_triangles.compute_intersect(single_ray)[0]

        self.assertFalse(results["check"])
        np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["albedo"], np.nan)
        self.assertEqual(results["facet"], -1)

        single_ray = g_rays.Rays([100, 0, 1], [0, 0, -1])

        results = self.multi_triangles.compute_intersect(single_ray)[0]

        self.assertFalse(results["check"])
        np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["albedo"], np.nan)
        self.assertEqual(results["facet"], -1)

        single_ray = g_rays.Rays([-100, 0, 1], [0, 0, -1])

        results = self.multi_triangles.compute_intersect(single_ray)[0]

        self.assertFalse(results["check"])
        np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["albedo"], np.nan)
        self.assertEqual(results["facet"], -1)

        single_ray = g_rays.Rays([-1, 0, 0], [1, 0, 0])

        results = self.multi_triangles.compute_intersect(single_ray)[0]

        self.assertTrue(results["check"])
        np.testing.assert_array_equal(results["intersect"], [0.5, 0, 0])
        np.testing.assert_array_equal(results["normal"], self.multi_normals[1])
        self.assertEqual(results["albedo"], 1.5)
        self.assertEqual(results["facet"], 1)

    def test_single_trace(self):
        single_ray = g_rays.Rays([0, 0, 1], [0, 0, -1])

        results = self.single_triangle.trace(single_ray)[0]

        self.assertTrue(results["check"])
        np.testing.assert_array_equal(results["intersect"], [0, 0, 0])
        np.testing.assert_array_equal(results["normal"], [0, 0, 1])
        self.assertEqual(results["albedo"], 1.25)
        self.assertEqual(results["facet"], 0)

        single_ray = g_rays.Rays([1, 1, 1], [-1, -1, -1])

        results = self.single_triangle.trace(single_ray)[0]

        self.assertTrue(results["check"])
        np.testing.assert_array_equal(results["intersect"], [0, 0, 0])
        np.testing.assert_array_equal(results["normal"], [0, 0, 1])
        self.assertEqual(results["albedo"], 1.25)
        self.assertEqual(results["facet"], 0)

        single_ray = g_rays.Rays([0, 0, 1], [0, 0, 1])

        results = self.single_triangle.trace(single_ray)[0]

        self.assertFalse(results["check"])
        np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["albedo"], np.nan)
        self.assertEqual(results["facet"], -1)

        single_ray = g_rays.Rays([1, 0, 0], [-1, 0, 0])

        results = self.single_triangle.trace(single_ray)[0]

        self.assertFalse(results["check"])
        np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["albedo"], np.nan)
        self.assertEqual(results["facet"], -1)

        single_ray = g_rays.Rays([-100, 0, 1], [0, 0, -1])

        results = self.single_triangle.trace(single_ray)[0]

        self.assertFalse(results["check"])
        np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["albedo"], np.nan)
        self.assertEqual(results["facet"], -1)

        single_ray = g_rays.Rays([100, 0, 1], [0, 0, -1])

        results = self.single_triangle.trace(single_ray)[0]

        self.assertFalse(results["check"])
        np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["albedo"], np.nan)
        self.assertEqual(results["facet"], -1)

        multi_ray = g_rays.Rays([[0, 1], [0, 1], [1, 1]], [[0, -1], [0, -1], [-1, -1]])

        results = self.single_triangle.trace(multi_ray)

        self.assertTrue(results["check"].all())
        np.testing.assert_array_equal(results["intersect"], [[0, 0, 0]] * 2)
        np.testing.assert_array_equal(results["normal"], [[0, 0, 1]] * 2)
        np.testing.assert_array_equal(results["albedo"], [1.25] * 2)
        np.testing.assert_array_equal(results["facet"], [0] * 2)

    def test_multi_trace(self):
        single_ray = g_rays.Rays([0, 0, 1], [0, 0, -1])

        results = self.multi_triangles.trace(single_ray)[0]

        self.assertTrue(results["check"])
        np.testing.assert_array_equal(results["intersect"], [0, 0, 0])
        np.testing.assert_array_equal(results["normal"], [0, 0, 1])
        self.assertEqual(results["albedo"], 1.25)
        self.assertEqual(results["facet"], 0)

        single_ray = g_rays.Rays([1, 1, 1], [-1, -1, -1])

        results = self.multi_triangles.trace(single_ray)[0]

        self.assertTrue(results["check"])
        np.testing.assert_array_equal(results["intersect"], [0, 0, 0])
        np.testing.assert_array_equal(results["normal"], [0, 0, 1])
        self.assertEqual(results["albedo"], 1.25)
        self.assertEqual(results["facet"], 0)

        single_ray = g_rays.Rays([0, 0, 1], [0, 0, 1])

        results = self.multi_triangles.trace(single_ray)[0]

        self.assertFalse(results["check"])
        np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["albedo"], np.nan)
        self.assertEqual(results["facet"], -1)

        single_ray = g_rays.Rays([100, 0, 1], [0, 0, -1])

        results = self.multi_triangles.trace(single_ray)[0]

        self.assertFalse(results["check"])
        np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["albedo"], np.nan)
        self.assertEqual(results["facet"], -1)

        single_ray = g_rays.Rays([-100, 0, 1], [0, 0, -1])

        results = self.multi_triangles.trace(single_ray)[0]

        self.assertFalse(results["check"])
        np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["albedo"], np.nan)
        self.assertEqual(results["facet"], -1)

        single_ray = g_rays.Rays([-1, 0, 0], [1, 0, 0])

        results = self.multi_triangles.trace(single_ray)[0]

        self.assertTrue(results["check"])
        np.testing.assert_array_equal(results["intersect"], [0.5, 0, 0])
        np.testing.assert_array_equal(results["normal"], self.multi_normals[1])
        self.assertEqual(results["albedo"], 1.5)
        self.assertEqual(results["facet"], 1)

        multi_ray = g_rays.Rays([[0, 0, 100],
                                 [0, 0, 100],
                                 [1, -1, 100]],
                                [[0, 0, 1],
                                [0, 0, 1],
                                [-1, 1, 1]])

        results = self.multi_triangles.trace(multi_ray)

        self.assertTrue(results[0]["check"])
        np.testing.assert_array_equal(results[0]["intersect"], [0, 0, 0])
        np.testing.assert_array_equal(results[0]["normal"], self.multi_normals[0])
        self.assertEqual(results[0]["albedo"], 1.25)
        self.assertEqual(results[0]["facet"], 0)

        self.assertTrue(results[1]["check"])
        np.testing.assert_array_equal(results[1]["intersect"], [0, 0, 0])
        np.testing.assert_array_equal(results[1]["normal"], self.multi_normals[0])
        self.assertEqual(results[1]["albedo"], 1.25)
        self.assertEqual(results[1]["facet"], 0)

        self.assertFalse(results[2]["check"])
        np.testing.assert_array_equal(results[2]["intersect"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results[2]["normal"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results[2]["albedo"], np.nan)
        self.assertEqual(results[2]["facet"], -1)


class TestTriangle32(TestCase):
    def setUp(self):

        self.single_verts = np.array([[-1, 1, 0],
                                      [-1, -1, 1],
                                      [0, 0, 0]], dtype=np.float32).T

        self.single_facet = np.array([[0, 1, 2]])

        self.single_sides = np.array([[2, 1],
                                      [0, 2],
                                      [0, 0]], dtype=np.float32)

        self.single_normals = np.array([0, 0, 1], dtype=np.float32)

        self.single_albedos = np.array([0, 1, 2], dtype=np.float32)

        self.single_triangle = g_shapes.Triangle32(self.single_verts,
                                                   self.single_albedos,
                                                   self.single_facet)

        # self.multi_verts = np.array([[[-1, 1, 0],
        #                               [-1, -1, 1],
        #                               [0, 0, 0]],
        #                              [[1, 0, 0],
        #                               [-1, 1, 0],
        #                               [0, 0, -1]]])
        self.multi_verts = np.array([[-1, 1, 0, 0],
                                     [-1, -1, 1, 0],
                                     [0, 0, 0, -1]], dtype=np.float32).T

        self.multi_sides = np.array([[[2, 1],
                                      [0, 2],
                                      [0, 0]],
                                     [[-1, -1],
                                      [2, 1],
                                      [0, -1]]], dtype=np.float32)

        self.multi_normals = np.array([[0, 0, 1],
                                       [-2 / np.sqrt(np.float32(6)),
                                        -1 / np.sqrt(np.float32(6)),
                                        1 / np.sqrt(np.float32(6))]],
                                      dtype=np.float32)

        self.multi_albedos = np.array([0, 1, 2, 1.5], dtype=np.float32)

        self.multi_facets = np.array([[0, 1, 2], [1, 2, 3]])

        self.multi_triangles = g_shapes.Triangle32(self.multi_verts,
                                                   self.multi_albedos,
                                                   self.multi_facets)

        self.rotation = at.Rotation([np.pi, np.pi / 2, np.pi / 4])

    def test_creation_singletri(self):

        for seq_type in [list, np.array, tuple]:
            with self.subTest(seq_type=seq_type):
                tri = g_shapes.Triangle32(seq_type(self.single_verts),
                                          seq_type(self.single_albedos),
                                          seq_type(self.single_facet))

                np.testing.assert_array_equal(tri.stacked_vertices, self.single_verts.T.reshape(-1, 3, 3), 'verts')
                np.testing.assert_array_equal(tri.albedos, self.single_albedos, 'albedos')
                np.testing.assert_array_equal(tri.sides, self.single_sides.reshape(-1, 3, 2), 'sides')
                np.testing.assert_array_equal(tri.normals, self.single_normals.reshape(-1, 3), 'normals')
                self.assertEqual(tri.num_faces, 1)

    def test_creation_multitri(self):

        for seq_type in [list, np.array, tuple]:
            with self.subTest(seq_type=seq_type):
                tri = g_shapes.Triangle32(seq_type(self.multi_verts),
                                          seq_type(self.multi_albedos),
                                          seq_type(self.multi_facets))

                np.testing.assert_array_equal(tri.stacked_vertices,
                                              self.multi_verts[self.multi_facets].swapaxes(-1, -2),
                                              'verts')
                np.testing.assert_array_equal(tri.albedos, self.multi_albedos, 'albedos')
                np.testing.assert_array_equal(tri.sides, self.multi_sides, 'sides')
                np.testing.assert_array_equal(tri.normals, self.multi_normals, 'normals')
                self.assertEqual(tri.num_faces, 2)

    def test_single_bounding_box(self):

        min_sides = self.single_triangle.bounding_box.min_sides.reshape(1, 3)
        max_sides = self.single_triangle.bounding_box.max_sides.reshape(1, 3)

        self.assertTrue((self.single_triangle.vertices >= min_sides).all())
        self.assertTrue((self.single_triangle.vertices <= max_sides).all())

    def test_multi_bounding_box(self):

        min_sides = self.multi_triangles.bounding_box.min_sides.reshape(1, 3)
        max_sides = self.multi_triangles.bounding_box.max_sides.reshape(1, 3)

        self.assertTrue((self.multi_triangles.vertices >= min_sides).all())
        self.assertTrue((self.multi_triangles.vertices <= max_sides).all())

    def test_single_rotate(self):

        tri_copy = copy.deepcopy(self.single_triangle)

        tri_copy.rotate(self.rotation)

        np.testing.assert_array_almost_equal(tri_copy.vertices, np.matmul(self.rotation.matrix,
                                                                          self.single_triangle.vertices.T).T,
                                             err_msg="Rotation rotation vertices")
        np.testing.assert_array_almost_equal(tri_copy.normals,
                                             np.matmul(self.rotation.matrix, self.single_triangle.normals.T).T,
                                             err_msg="Rotation rotation normals")
        np.testing.assert_array_almost_equal(tri_copy.sides,
                                             np.matmul(self.rotation.matrix, self.single_triangle.sides),
                                             err_msg="Rotation rotation sides")

        tri_copy = copy.deepcopy(self.single_triangle)

        tri_copy.rotate(self.rotation.matrix)

        np.testing.assert_array_almost_equal(tri_copy.vertices, np.matmul(self.rotation.matrix,
                                                                          self.single_triangle.vertices.T).T,
                                             err_msg="Matrix rotation vertices")
        np.testing.assert_array_almost_equal(tri_copy.normals,
                                             np.matmul(self.rotation.matrix, self.single_triangle.normals.T).T,
                                             err_msg="Matrix rotation normals")
        np.testing.assert_array_almost_equal(tri_copy.sides,
                                             np.matmul(self.rotation.matrix, self.single_triangle.sides),
                                             err_msg="Matrix rotation sides")

    def test_multi_rotate(self):

        tri_copy = copy.deepcopy(self.multi_triangles)

        tri_copy.rotate(self.rotation)

        np.testing.assert_array_almost_equal(tri_copy.vertices, np.matmul(self.rotation.matrix,
                                                                          self.multi_triangles.vertices.T).T,
                                             err_msg="Rotation rotation vertices")
        np.testing.assert_array_almost_equal(tri_copy.normals,
                                             np.matmul(self.rotation.matrix, self.multi_triangles.normals.T).T,
                                             err_msg="Rotation rotation normals")
        np.testing.assert_array_almost_equal(tri_copy.sides,
                                             np.matmul(self.rotation.matrix, self.multi_triangles.sides),
                                             err_msg="Rotation rotation sides")

        tri_copy = copy.deepcopy(self.multi_triangles)

        tri_copy.rotate(self.rotation.matrix)

        np.testing.assert_array_almost_equal(tri_copy.vertices, np.matmul(self.rotation.matrix,
                                                                          self.multi_triangles.vertices.T).T,
                                             err_msg="Matrix rotation vertices")
        np.testing.assert_array_almost_equal(tri_copy.normals,
                                             np.matmul(self.rotation.matrix, self.multi_triangles.normals.T).T,
                                             err_msg="Matrix rotation normals")
        np.testing.assert_array_almost_equal(tri_copy.sides,
                                             np.matmul(self.rotation.matrix, self.multi_triangles.sides),
                                             err_msg="Matrix rotation sides")

    def test_single_translation(self):

        tri_copy = copy.deepcopy(self.single_triangle)

        tri_copy.translate([5, 4, 3])

        np.testing.assert_array_equal(tri_copy.vertices,
                                      self.single_triangle.vertices + np.array([5, 4, 3]).reshape(1, 3))

        tri_copy = copy.deepcopy(self.single_triangle)

        tri_copy.translate([[5, 4, 3]])

        np.testing.assert_array_equal(tri_copy.vertices,
                                      self.single_triangle.vertices + np.array([5, 4, 3]).reshape(1, 3))

        tri_copy = copy.deepcopy(self.single_triangle)

        tri_copy.translate([[5],
                            [4],
                            [3]])

        np.testing.assert_array_equal(tri_copy.vertices,
                                      self.single_triangle.vertices + np.array([5, 4, 3]).reshape(1, 3))

        with self.assertRaises(ValueError):
            tri_copy = copy.deepcopy(self.single_triangle)

            tri_copy.translate([1, 2, 3, 4])

    def test_multi_translation(self):

        tri_copy = copy.deepcopy(self.multi_triangles)

        tri_copy.translate([5, 4, 3])

        np.testing.assert_array_equal(tri_copy.vertices,
                                      self.multi_triangles.vertices + np.array([5, 4, 3]).reshape(1, 3))

        tri_copy = copy.deepcopy(self.multi_triangles)

        tri_copy.translate([[5, 4, 3]])

        np.testing.assert_array_equal(tri_copy.vertices,
                                      self.multi_triangles.vertices + np.array([5, 4, 3]).reshape(1, 3))

        tri_copy = copy.deepcopy(self.multi_triangles)

        tri_copy.translate([[5],
                            [4],
                            [3]])

        np.testing.assert_array_equal(tri_copy.vertices,
                                      self.multi_triangles.vertices + np.array([5, 4, 3]).reshape(1, 3))

        with self.assertRaises(ValueError):
            tri_copy = copy.deepcopy(self.multi_triangles)

            tri_copy.translate([1, 2, 3, 4])

    def test_single_get_albedo(self):

        albedo = cast(float, self.single_triangle.get_albedo(self.single_triangle.stacked_vertices[0, :, 0], 0))

        self.assertAlmostEqual(albedo, 0)

        albedo = cast(float, self.single_triangle.get_albedo(self.single_triangle.stacked_vertices[0, :, 1], 0))

        self.assertAlmostEqual(albedo, 1)

        albedo = cast(float, self.single_triangle.get_albedo(self.single_triangle.stacked_vertices[0, :, 2], 0))

        self.assertAlmostEqual(albedo, 2)

        albedo = cast(float, self.single_triangle.get_albedo(self.single_triangle.stacked_vertices[0, :, :2].mean(axis=-1), 0))

        self.assertAlmostEqual(albedo, 0.5)

        albedo = cast(float, self.single_triangle.get_albedo(self.single_triangle.stacked_vertices[0, :, ::2].mean(axis=-1), 0))

        self.assertEqual(albedo, 1)

        albedo = cast(float, self.single_triangle.get_albedo(self.single_triangle.stacked_vertices[0, :, 1:].mean(axis=-1), 0))

        self.assertEqual(albedo, 1.5)

        albedo = cast(float, self.single_triangle.get_albedo(self.single_triangle.stacked_vertices[0].mean(axis=-1), 0))

        self.assertEqual(albedo, 1)

    def test_multi_get_albedo(self):

        # first triangle
        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[0, :, 0], 0))

        self.assertAlmostEqual(albedo, 0)

        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[0, :, 1], 0))

        self.assertAlmostEqual(albedo, 1)

        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[0, :, 2], 0))

        self.assertAlmostEqual(albedo, 2)

        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[0, :, :2].mean(axis=-1), 0))

        self.assertAlmostEqual(albedo, 0.5)

        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[0, :, ::2].mean(axis=-1), 0))

        self.assertAlmostEqual(albedo, 1)

        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[0, :, 1:].mean(axis=-1), 0))

        self.assertAlmostEqual(albedo, 1.5)

        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[0].mean(axis=-1), 0))

        self.assertAlmostEqual(albedo, 1)

        # second triangle
        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[1, :, 0], 1))

        self.assertAlmostEqual(albedo, 1)

        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[1, :, 1], 1))

        self.assertAlmostEqual(albedo, 2)

        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[1, :, 2], 1))

        self.assertAlmostEqual(albedo, 1.5)

        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[1, :, :2].mean(axis=-1), 1))

        self.assertAlmostEqual(albedo, 1.5)

        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[1, :, ::2].mean(axis=-1), 1))

        self.assertAlmostEqual(albedo, 1.25)

        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[1, :, 1:].mean(axis=-1), 1))

        self.assertAlmostEqual(albedo, 1.75)

        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[1].mean(axis=-1), 1))

        self.assertAlmostEqual(albedo, 1.5)

        # multi triangles
        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[:2, :, 0].T, [0, 1]))

        np.testing.assert_array_almost_equal(albedo, [0, 1])

        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[:2, :, 1].T, [0, 1]))

        np.testing.assert_array_almost_equal(albedo, [1, 2])

        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[:2, :, 2].T, [0, 1]))

        np.testing.assert_array_almost_equal(albedo, [2, 1.5])

        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[:2, :, :2].mean(axis=-1).T,
                                                 [0, 1]))

        np.testing.assert_array_almost_equal(albedo, [0.5, 1.5])

        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[:2, :, ::2].mean(axis=-1).T,
                                                 [0, 1]))

        np.testing.assert_array_almost_equal(albedo, [1, 1.25])

        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[:2, :, 1:].mean(axis=-1).T,
                                                 [0, 1]))

        np.testing.assert_array_almost_equal(albedo, [1.5, 1.75])

        albedo = cast(float, self.multi_triangles.get_albedo(self.multi_triangles.stacked_vertices[:2].mean(axis=-1).T, [0, 1]))

        np.testing.assert_array_almost_equal(albedo, [1, 1.5])

    def test_single_compute_intersect(self):

        single_ray = g_rays.Rays([0, 0, 1], [0, 0, -1])

        results = self.single_triangle.compute_intersect(single_ray)[0]

        self.assertTrue(results["check"])
        np.testing.assert_array_equal(results["intersect"], [0, 0, 0])
        np.testing.assert_array_equal(results["normal"], [0, 0, 1])
        self.assertEqual(results["albedo"], 1.25)
        self.assertEqual(results["facet"], 0)

        single_ray = g_rays.Rays([1, 1, 1], [-1, -1, -1])

        results = self.single_triangle.compute_intersect(single_ray)[0]

        self.assertTrue(results["check"])
        np.testing.assert_array_equal(results["intersect"], [0, 0, 0])
        np.testing.assert_array_equal(results["normal"], [0, 0, 1])
        self.assertEqual(results["albedo"], 1.25)
        self.assertEqual(results["facet"], 0)

        single_ray = g_rays.Rays([0, 0, 1], [0, 0, 1])

        results = self.single_triangle.compute_intersect(single_ray)[0]

        self.assertFalse(results["check"])
        np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["albedo"], np.nan)
        self.assertEqual(results["facet"], -1)

        single_ray = g_rays.Rays([1, 0, 0], [-1, 0, 0])

        results = self.single_triangle.compute_intersect(single_ray)[0]

        self.assertFalse(results["check"])
        np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["albedo"], np.nan)
        self.assertEqual(results["facet"], -1)

        single_ray = g_rays.Rays([-100, 0, 1], [0, 0, -1])

        results = self.single_triangle.compute_intersect(single_ray)[0]

        self.assertFalse(results["check"])
        np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["albedo"], np.nan)
        self.assertEqual(results["facet"], -1)

        single_ray = g_rays.Rays([100, 0, 1], [0, 0, -1])

        results = self.single_triangle.compute_intersect(single_ray)[0]

        self.assertFalse(results["check"])
        np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["albedo"], np.nan)
        self.assertEqual(results["facet"], -1)

    def test_multi_compute_intersect(self):
        single_ray = g_rays.Rays([0, 0, 1], [0, 0, -1])

        results = self.multi_triangles.compute_intersect(single_ray)[0]

        self.assertTrue(results["check"])
        np.testing.assert_array_equal(results["intersect"], [0, 0, 0])
        np.testing.assert_array_equal(results["normal"], [0, 0, 1])
        self.assertEqual(results["albedo"], 1.25)
        self.assertEqual(results["facet"], 0)

        single_ray = g_rays.Rays([1, 1, 1], [-1, -1, -1])

        results = self.multi_triangles.compute_intersect(single_ray)[0]

        self.assertTrue(results["check"])
        np.testing.assert_array_equal(results["intersect"], [0, 0, 0])
        np.testing.assert_array_equal(results["normal"], [0, 0, 1])
        self.assertEqual(results["albedo"], 1.25)
        self.assertEqual(results["facet"], 0)

        single_ray = g_rays.Rays([0, 0, 1], [0, 0, 1])

        results = self.multi_triangles.compute_intersect(single_ray)[0]

        self.assertFalse(results["check"])
        np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["albedo"], np.nan)
        self.assertEqual(results["facet"], -1)

        single_ray = g_rays.Rays([100, 0, 1], [0, 0, -1])

        results = self.multi_triangles.compute_intersect(single_ray)[0]

        self.assertFalse(results["check"])
        np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["albedo"], np.nan)
        self.assertEqual(results["facet"], -1)

        single_ray = g_rays.Rays([-100, 0, 1], [0, 0, -1])

        results = self.multi_triangles.compute_intersect(single_ray)[0]

        self.assertFalse(results["check"])
        np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["albedo"], np.nan)
        self.assertEqual(results["facet"], -1)

        single_ray = g_rays.Rays([-1, 0, 0], [1, 0, 0])

        results = self.multi_triangles.compute_intersect(single_ray)[0]

        self.assertTrue(results["check"])
        np.testing.assert_array_equal(results["intersect"], [0.5, 0, 0])
        np.testing.assert_array_equal(results["normal"], self.multi_normals[1])
        self.assertEqual(results["albedo"], 1.5)
        self.assertEqual(results["facet"], 1)

    def test_single_trace(self):
        single_ray = g_rays.Rays([0, 0, 1], [0, 0, -1])

        results = self.single_triangle.trace(single_ray)[0]

        self.assertTrue(results["check"])
        np.testing.assert_array_equal(results["intersect"], [0, 0, 0])
        np.testing.assert_array_equal(results["normal"], [0, 0, 1])
        self.assertEqual(results["albedo"], 1.25)
        self.assertEqual(results["facet"], 0)

        single_ray = g_rays.Rays([1, 1, 1], [-1, -1, -1])

        results = self.single_triangle.trace(single_ray)[0]

        self.assertTrue(results["check"])
        np.testing.assert_array_equal(results["intersect"], [0, 0, 0])
        np.testing.assert_array_equal(results["normal"], [0, 0, 1])
        self.assertEqual(results["albedo"], 1.25)
        self.assertEqual(results["facet"], 0)

        single_ray = g_rays.Rays([0, 0, 1], [0, 0, 1])

        results = self.single_triangle.trace(single_ray)[0]

        self.assertFalse(results["check"])
        np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["albedo"], np.nan)
        self.assertEqual(results["facet"], -1)

        single_ray = g_rays.Rays([1, 0, 0], [-1, 0, 0])

        results = self.single_triangle.trace(single_ray)[0]

        self.assertFalse(results["check"])
        np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["albedo"], np.nan)
        self.assertEqual(results["facet"], -1)

        single_ray = g_rays.Rays([-100, 0, 1], [0, 0, -1])

        results = self.single_triangle.trace(single_ray)[0]

        self.assertFalse(results["check"])
        np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["albedo"], np.nan)
        self.assertEqual(results["facet"], -1)

        single_ray = g_rays.Rays([100, 0, 1], [0, 0, -1])

        results = self.single_triangle.trace(single_ray)[0]

        self.assertFalse(results["check"])
        np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["albedo"], np.nan)
        self.assertEqual(results["facet"], -1)

        multi_ray = g_rays.Rays([[0, 1], [0, 1], [1, 1]], [[0, -1], [0, -1], [-1, -1]])

        results = self.single_triangle.trace(multi_ray)

        self.assertTrue(results["check"].all())
        np.testing.assert_array_equal(results["intersect"], [[0, 0, 0]] * 2)
        np.testing.assert_array_equal(results["normal"], [[0, 0, 1]] * 2)
        np.testing.assert_array_equal(results["albedo"], [1.25] * 2)
        np.testing.assert_array_equal(results["facet"], [0] * 2)

    def test_multi_trace(self):
        single_ray = g_rays.Rays([0, 0, 1], [0, 0, -1])

        results = self.multi_triangles.trace(single_ray)[0]

        self.assertTrue(results["check"])
        np.testing.assert_array_equal(results["intersect"], [0, 0, 0])
        np.testing.assert_array_equal(results["normal"], [0, 0, 1])
        self.assertEqual(results["albedo"], 1.25)
        self.assertEqual(results["facet"], 0)

        single_ray = g_rays.Rays([1, 1, 1], [-1, -1, -1])

        results = self.multi_triangles.trace(single_ray)[0]

        self.assertTrue(results["check"])
        np.testing.assert_array_equal(results["intersect"], [0, 0, 0])
        np.testing.assert_array_equal(results["normal"], [0, 0, 1])
        self.assertEqual(results["albedo"], 1.25)
        self.assertEqual(results["facet"], 0)

        single_ray = g_rays.Rays([0, 0, 1], [0, 0, 1])

        results = self.multi_triangles.trace(single_ray)[0]

        self.assertFalse(results["check"])
        np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["albedo"], np.nan)
        self.assertEqual(results["facet"], -1)

        single_ray = g_rays.Rays([100, 0, 1], [0, 0, -1])

        results = self.multi_triangles.trace(single_ray)[0]

        self.assertFalse(results["check"])
        np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["albedo"], np.nan)
        self.assertEqual(results["facet"], -1)

        single_ray = g_rays.Rays([-100, 0, 1], [0, 0, -1])

        results = self.multi_triangles.trace(single_ray)[0]

        self.assertFalse(results["check"])
        np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results["albedo"], np.nan)
        self.assertEqual(results["facet"], -1)

        single_ray = g_rays.Rays([-1, 0, 0], [1, 0, 0])

        results = self.multi_triangles.trace(single_ray)[0]

        self.assertTrue(results["check"])
        np.testing.assert_array_equal(results["intersect"], [0.5, 0, 0])
        np.testing.assert_array_equal(results["normal"], self.multi_normals[1])
        self.assertEqual(results["albedo"], 1.5)
        self.assertEqual(results["facet"], 1)

        multi_ray = g_rays.Rays([[0, 0, 100],
                                 [0, 0, 100],
                                 [1, -1, 100]],
                                [[0, 0, 1],
                                [0, 0, 1],
                                [-1, 1, 1]])

        results = self.multi_triangles.trace(multi_ray)

        self.assertTrue(results[0]["check"])
        np.testing.assert_array_equal(results[0]["intersect"], [0, 0, 0])
        np.testing.assert_array_equal(results[0]["normal"], self.multi_normals[0])
        self.assertEqual(results[0]["albedo"], 1.25)
        self.assertEqual(results[0]["facet"], 0)

        self.assertTrue(results[1]["check"])
        np.testing.assert_array_equal(results[1]["intersect"], [0, 0, 0])
        np.testing.assert_array_equal(results[1]["normal"], self.multi_normals[0])
        self.assertEqual(results[1]["albedo"], 1.25)
        self.assertEqual(results[1]["facet"], 0)

        self.assertFalse(results[2]["check"])
        np.testing.assert_array_equal(results[2]["intersect"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results[2]["normal"], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(results[2]["albedo"], np.nan)
        self.assertEqual(results[2]["facet"], -1)


class TestAxisAlignedBoundingBox(TestCase):
    def setUp(self):
        self.min_sides = np.array([-1, -2, -3])
        self.max_sides = np.array([3, 2, 1])

        self.aabb = g_shapes.AxisAlignedBoundingBox(self.min_sides, self.max_sides)

        self.verts = min_max_to_bounding_box(self.min_sides, self.max_sides)

        self.rotation = at.Rotation([np.pi, np.pi / 2, np.pi / 4])

    def test_creation(self):
        for seq_type in [list, np.array, tuple]:
            with self.subTest(seq_type=seq_type):
                box = g_shapes.AxisAlignedBoundingBox(seq_type(self.min_sides), seq_type(self.max_sides))

                np.testing.assert_array_equal(box.min_sides.flatten(), self.min_sides, 'min_sides')
                np.testing.assert_array_equal(box.max_sides.flatten(), self.max_sides, 'max_sides')
                np.testing.assert_array_equal(box.vertices, self.verts, 'vertices')

    def test_equality(self):
        other = copy.deepcopy(self.aabb)

        self.assertEqual(self.aabb, other)

        other.rotate(self.rotation)

        self.assertNotEqual(self.aabb, other)

        other2 = copy.deepcopy(self.aabb)

        other2.rotate(self.rotation)

        self.assertEqual(other2, other)

        other2.rotate(self.rotation)

        self.assertNotEqual(other2, other)

        other.rotate(self.rotation)

        other.translate([4, 5, 6])

        self.assertNotEqual(other, other2)

        other2.translate([4, 5, 6])

        self.assertEqual(other, other2)

    def test_rotate(self):
        box_copy = copy.deepcopy(self.aabb)

        box_copy.rotate(self.rotation)

        np.testing.assert_array_almost_equal(box_copy.vertices, np.matmul(self.rotation.matrix,
                                                                          self.verts),
                                             err_msg="Rotation rotation vertices")

        np.testing.assert_array_almost_equal(box_copy._rotation.quaternion, self.rotation.inv().quaternion)

        box_copy = copy.deepcopy(self.aabb)

        box_copy.rotate(self.rotation.matrix)

        np.testing.assert_array_almost_equal(box_copy.vertices, np.matmul(self.rotation.matrix,
                                                                          self.verts),
                                             err_msg="Matrix rotation vertices")

        np.testing.assert_array_almost_equal(box_copy._rotation.quaternion, self.rotation.inv().quaternion)

    def test_translate(self):
        box_copy = copy.deepcopy(self.aabb)

        box_copy.translate([5, 4, 3])

        np.testing.assert_array_equal(box_copy.min_sides,
                                      self.aabb.min_sides + np.array([5, 4, 3]))
        np.testing.assert_array_equal(box_copy.max_sides,
                                      self.aabb.max_sides + np.array([5, 4, 3]))
        np.testing.assert_array_equal(box_copy.vertices,
                                      self.verts + np.array([5, 4, 3]).reshape(3, 1))

        box_copy = copy.deepcopy(self.aabb)

        box_copy.translate([[5, 4, 3]])

        np.testing.assert_array_equal(box_copy.min_sides,
                                      self.aabb.min_sides + np.array([5, 4, 3]))
        np.testing.assert_array_equal(box_copy.max_sides,
                                      self.aabb.max_sides + np.array([5, 4, 3]))
        np.testing.assert_array_equal(box_copy.vertices,
                                      self.verts + np.array([5, 4, 3]).reshape(3, 1))

    def test_rotate_translate(self):
        box_copy = copy.deepcopy(self.aabb)

        box_copy.translate([5, 4, 3])

        transverts = self.verts + np.array([5, 4, 3]).reshape(3, 1)

        np.testing.assert_array_equal(box_copy.min_sides,
                                      self.aabb.min_sides + np.array([5, 4, 3]))
        np.testing.assert_array_equal(box_copy.max_sides,
                                      self.aabb.max_sides + np.array([5, 4, 3]))
        np.testing.assert_array_equal(box_copy.vertices, transverts)

        box_copy.rotate(self.rotation)
        np.testing.assert_array_almost_equal(box_copy.vertices, np.matmul(self.rotation.matrix,
                                                                          transverts))

        np.testing.assert_array_almost_equal(box_copy._rotation.quaternion, self.rotation.inv().quaternion)

        box_copy = copy.deepcopy(self.aabb)

        box_copy.rotate(self.rotation)

        rotverts = self.rotation.matrix @ self.verts

        np.testing.assert_array_almost_equal(box_copy.vertices, rotverts)

        np.testing.assert_array_almost_equal(box_copy._rotation.quaternion, self.rotation.inv().quaternion)

        box_copy.translate([5, 4, 3])

        np.testing.assert_array_equal(box_copy.min_sides,
                                      self.aabb.min_sides +
                                      self.rotation.inv().matrix @ np.array([5, 4, 3]))
        np.testing.assert_array_equal(box_copy.max_sides,
                                      self.aabb.max_sides +
                                      self.rotation.inv().matrix @ np.array([5, 4, 3]))
        np.testing.assert_array_almost_equal(box_copy.vertices,
                                             rotverts + np.array([5, 4, 3]).reshape(3, 1))

    def test_trace(self):
        with self.subTest(numrays=1):
            single_ray = g_rays.Rays([0, 0, 10], [0, 0, -1])

            results = self.aabb.trace(single_ray)

            self.assertTrue(results)

            single_ray = g_rays.Rays([10, 10, 10], [-1, -1, -1])

            results = self.aabb.compute_intersect(single_ray)

            self.assertTrue(results)

            single_ray = g_rays.Rays([0, 0, 10], [0, 0, 1])

            results = self.aabb.compute_intersect(single_ray)

            self.assertFalse(results)

            single_ray = g_rays.Rays([-1, 10, 0], [0, -1, 0])

            results = self.aabb.compute_intersect(single_ray)

            self.assertFalse(results)

            single_ray = g_rays.Rays([-1, 10, 1], [0, -1, 0])

            results = self.aabb.compute_intersect(single_ray)

            self.assertFalse(results)

            single_ray = g_rays.Rays([-10, -20, -30], [1, 2, 3])

            results = self.aabb.compute_intersect(single_ray)

            self.assertTrue(results)

            single_ray = g_rays.Rays([0, 0, 0], [1, 0, 0])

            results = self.aabb.compute_intersect(single_ray)

            self.assertTrue(results)

            single_ray = g_rays.Rays([100, 0, 1], [0, 0, -1])

            results = self.aabb.compute_intersect(single_ray)

            self.assertFalse(results)

            single_ray = g_rays.Rays([-100, 0, 1], [0, 0, -1])

            results = self.aabb.compute_intersect(single_ray)

            self.assertFalse(results)

        with self.subTest(numrays=2):
            multi_ray = g_rays.Rays(np.asarray([[0, 0, 10], [10, 10, 10]]).T,
                                    np.asarray([[0, 0, -1], [-1, -1, -1]]).T)

            results = cast(np.ndarray, self.aabb.trace(multi_ray))

            self.assertTrue(results.all())

            multi_ray = g_rays.Rays(np.asarray([[0, 0, 10], [10, 10, 10]]).T,
                                    np.asarray([[0, 0, -1], [1, 1, 1]]).T)

            results = self.aabb.trace(multi_ray)

            np.testing.assert_array_equal(results, [True, False])

    def test_rotate_trace(self):
        box_copy = copy.deepcopy(self.aabb)

        box_copy.rotate([np.pi / 2, 0, 0])

        with self.subTest(numrays=1):
            single_ray = g_rays.Rays([0, 0, 10], [0, 0, -1])

            results = box_copy.trace(single_ray)

            self.assertTrue(results)

            single_ray = g_rays.Rays([10, 10, 10], [-1, -1, -1])

            results = box_copy.compute_intersect(single_ray)

            self.assertTrue(results)

            single_ray = g_rays.Rays([0, 0, 10], [0, 0, 1])

            results = box_copy.compute_intersect(single_ray)

            self.assertFalse(results)

            single_ray = g_rays.Rays([-1, 10, 0], [0, -1, 0])

            results = box_copy.compute_intersect(single_ray)

            self.assertFalse(results)

            single_ray = g_rays.Rays([0, 10, 1], [0, -1, 0])

            results = box_copy.compute_intersect(single_ray)

            self.assertTrue(results)

            single_ray = g_rays.Rays([0, 10, 1.9], [0, -1, 0])

            results = box_copy.compute_intersect(single_ray)

            self.assertTrue(results)

            single_ray = g_rays.Rays([0, 10, 2], [0, -1, 0])

            results = box_copy.compute_intersect(single_ray)

            self.assertFalse(results)

            single_ray = g_rays.Rays([-10, -20, -30], [1, 2, 3])

            results = box_copy.compute_intersect(single_ray)

            self.assertTrue(results)

            single_ray = g_rays.Rays([0, 0, 0], [1, 0, 0])

            results = box_copy.compute_intersect(single_ray)

            self.assertTrue(results)

            single_ray = g_rays.Rays([100, 0, 1], [0, 0, -1])

            results = box_copy.compute_intersect(single_ray)

            self.assertFalse(results)

            single_ray = g_rays.Rays([-100, 0, 1], [0, 0, -1])

            results = box_copy.compute_intersect(single_ray)

            self.assertFalse(results)

        with self.subTest(numrays=2):
            multi_ray = g_rays.Rays(np.asarray([[0, 0, 10], [10, 10, 10]]).T,
                                    np.asarray([[0, 0, -1], [-1, -1, -1]]).T)

            results = cast(np.ndarray, box_copy.trace(multi_ray))

            self.assertTrue(results.all())

            multi_ray = g_rays.Rays(np.asarray([[0, 0, 10], [10, 10, 10]]).T,
                                    np.asarray([[0, 0, -1], [1, 1, 1]]).T)

            results = box_copy.trace(multi_ray)

            np.testing.assert_array_equal(results, [True, False])


class TestEllipsoid(TestCase):
    def setUp(self):
        self.origin = np.array([0, 0, 0]).astype(np.float64)
        self.off_center = np.array([20, 20, 20]).astype(np.float64)

        self.orientation = at.Rotation([np.pi, np.pi / 2, np.pi / 4])

        self.principal_axes_sphere = np.array([5, 5, 5]).astype(np.float64)
        self.principal_axes_ellipse = np.array([10, 5, 1]).astype(np.float64)

        self.ellipsoid_matrix_sphere = np.diag(np.power(self.principal_axes_sphere, -2))

        self.ellipsoid_matrix_ellipse = np.matmul(self.orientation.matrix,
                                                  np.matmul(np.diag(np.power(self.principal_axes_ellipse, -2)),
                                                            self.orientation.matrix.T))

    def test_creation(self):
        for seq_type in [list, np.array, tuple]:
            with self.subTest(seq_type=seq_type, input=['origin', 'princ axes', 'orientation'], case='sphere'):
                shape = g_shapes.Ellipsoid(self.origin, principal_axes=self.principal_axes_sphere,
                                           orientation=self.orientation.matrix)

                np.testing.assert_array_almost_equal(shape.center, self.origin)
                np.testing.assert_array_almost_equal(shape.principal_axes, self.principal_axes_sphere)
                np.testing.assert_array_almost_equal(shape.orientation, self.orientation.matrix)
                np.testing.assert_array_almost_equal(shape.ellipsoid_matrix, self.ellipsoid_matrix_sphere)

            with self.subTest(seq_type=seq_type, input=['origin', 'princ axes'], case='sphere'):
                shape = g_shapes.Ellipsoid(self.origin, principal_axes=self.principal_axes_sphere)

                np.testing.assert_array_almost_equal(shape.center, self.origin)
                np.testing.assert_array_almost_equal(shape.principal_axes, self.principal_axes_sphere)
                np.testing.assert_array_almost_equal(shape.orientation, np.eye(3))
                np.testing.assert_array_almost_equal(shape.ellipsoid_matrix, self.ellipsoid_matrix_sphere)

            with self.subTest(seq_type=seq_type, input=['origin', 'ellispoid mat'], case='sphere'):
                shape = g_shapes.Ellipsoid(self.origin, ellipsoid_matrix=self.ellipsoid_matrix_sphere)

                np.testing.assert_array_almost_equal(shape.center, self.origin)
                np.testing.assert_array_almost_equal(shape.principal_axes, self.principal_axes_sphere)
                np.testing.assert_array_almost_equal(shape.orientation, np.eye(3))
                np.testing.assert_array_almost_equal(shape.ellipsoid_matrix, self.ellipsoid_matrix_sphere)

            with self.subTest(seq_type=seq_type, input=['origin', 'ellispoid mat', 'princ axes', 'orientation'],
                              case='sphere'):
                shape = g_shapes.Ellipsoid(self.origin, ellipsoid_matrix=self.ellipsoid_matrix_sphere,
                                           principal_axes=self.principal_axes_sphere,
                                           orientation=self.orientation.matrix)

                np.testing.assert_array_almost_equal(shape.center, self.origin)
                np.testing.assert_array_almost_equal(shape.principal_axes, self.principal_axes_sphere)
                np.testing.assert_array_almost_equal(shape.orientation, self.orientation.matrix)
                np.testing.assert_array_almost_equal(shape.ellipsoid_matrix, self.ellipsoid_matrix_sphere)

            with self.subTest(seq_type=seq_type, input=['origin', 'princ axes', 'orientation'], case='ellipse'):
                shape = g_shapes.Ellipsoid(self.origin, principal_axes=self.principal_axes_ellipse,
                                           orientation=self.orientation.matrix)

                np.testing.assert_array_almost_equal(shape.center, self.origin)
                np.testing.assert_array_almost_equal(shape.principal_axes, self.principal_axes_ellipse)
                np.testing.assert_array_almost_equal(shape.orientation, self.orientation.matrix)
                np.testing.assert_array_almost_equal(shape.ellipsoid_matrix, self.ellipsoid_matrix_ellipse)

            with self.subTest(seq_type=seq_type, input=['origin', 'princ axes'], case='ellipse'):
                shape = g_shapes.Ellipsoid(self.origin, principal_axes=self.principal_axes_ellipse)

                np.testing.assert_array_almost_equal(shape.center, self.origin)
                np.testing.assert_array_almost_equal(shape.principal_axes, self.principal_axes_ellipse)
                np.testing.assert_array_almost_equal(shape.orientation, np.eye(3))
                np.testing.assert_array_almost_equal(shape.ellipsoid_matrix,
                                                     np.diag(np.power(self.principal_axes_ellipse, -2)))

            with self.subTest(seq_type=seq_type, input=['origin', 'ellispoid mat'], case='ellipse'):
                shape = g_shapes.Ellipsoid(self.origin, ellipsoid_matrix=self.ellipsoid_matrix_ellipse)

                np.testing.assert_array_almost_equal(shape.center, self.origin)
                np.testing.assert_array_almost_equal(shape.principal_axes, self.principal_axes_ellipse)
                np.testing.assert_array_almost_equal(np.sign(shape.orientation[0]) *
                                                     np.sign(self.orientation.matrix[0]) * shape.orientation,
                                                     self.orientation.matrix)
                np.testing.assert_array_almost_equal(shape.ellipsoid_matrix, self.ellipsoid_matrix_ellipse)

            with self.subTest(seq_type=seq_type, input=['origin', 'ellispoid mat', 'princ axes', 'orientation'],
                              case='ellipse'):
                shape = g_shapes.Ellipsoid(self.origin, ellipsoid_matrix=self.ellipsoid_matrix_ellipse,
                                           principal_axes=self.principal_axes_ellipse,
                                           orientation=self.orientation.matrix)

                np.testing.assert_array_almost_equal(shape.center, self.origin)
                np.testing.assert_array_almost_equal(shape.principal_axes, self.principal_axes_ellipse)
                np.testing.assert_array_almost_equal(shape.orientation, self.orientation.matrix)
                np.testing.assert_array_almost_equal(shape.ellipsoid_matrix, self.ellipsoid_matrix_ellipse)

    def test_bounding_box(self):
        # TODO: figure out this test..
        pass

    def test_compute_intersect(self):
        sphere = g_shapes.Ellipsoid(self.origin, ellipsoid_matrix=self.ellipsoid_matrix_sphere,
                                    principal_axes=self.principal_axes_sphere,
                                    orientation=self.orientation.matrix)

        with self.subTest(case='sphere', center='origin', numrays=1):
            ray = g_rays.Rays([100, 0, 0], [-1, 0, 0])

            results = sphere.compute_intersect(ray)

            self.assertTrue(results["check"])
            np.testing.assert_array_almost_equal(results["intersect"], [5, 0, 0])
            np.testing.assert_array_almost_equal(results["normal"], [1, 0, 0])
            self.assertEqual(results["albedo"], 1)

            ray = g_rays.Rays([0, 100, 0], [0, -1, 0])

            results = sphere.compute_intersect(ray)

            self.assertTrue(results["check"])
            np.testing.assert_array_almost_equal(results["intersect"], [0, 5, 0])
            np.testing.assert_array_almost_equal(results["normal"], [0, 1, 0])
            self.assertEqual(results["albedo"], 1)

            ray = g_rays.Rays([0, 0, 100], [0, 0, -1])

            results = sphere.compute_intersect(ray)

            self.assertTrue(results["check"])
            np.testing.assert_array_almost_equal(results["intersect"], [0, 0, 5])
            np.testing.assert_array_almost_equal(results["normal"], [0, 0, 1])
            self.assertEqual(results["albedo"], 1)

            ray = g_rays.Rays([100] * 3, [-1 / np.sqrt(3)] * 3)

            results = sphere.compute_intersect(ray)

            self.assertTrue(results["check"])
            np.testing.assert_array_almost_equal(results["intersect"], [5 / np.sqrt(3)] * 3)
            np.testing.assert_array_almost_equal(results["normal"], [1 / np.sqrt(3)] * 3)
            self.assertEqual(results["albedo"], 1)

            ray = g_rays.Rays([100] * 3, [1 / np.sqrt(3)] * 3)

            results = sphere.compute_intersect(ray)

            self.assertFalse(results["check"])
            np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
            np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
            np.testing.assert_array_equal(results["albedo"], np.nan)

            ray = g_rays.Rays([100] * 3, [1, 0, 0])

            results = sphere.compute_intersect(ray)

            self.assertFalse(results["check"])
            np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
            np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
            np.testing.assert_array_equal(results["albedo"], np.nan)

        with self.subTest(case='sphere', center='origin', numrays=2):
            rays = g_rays.Rays([[100] * 2, [0] * 2, [0] * 2], [[-1] * 2, [0] * 2, [0] * 2])

            results = sphere.compute_intersect(rays)

            self.assertTrue(results["check"].all())
            np.testing.assert_array_almost_equal(results["intersect"], [[5, 0, 0]] * 2)
            np.testing.assert_array_almost_equal(results["normal"], [[1, 0, 0]] * 2)
            np.testing.assert_array_equal(results["albedo"], [1, 1])

            rays = g_rays.Rays([[0] * 2, [100] * 2, [0] * 2], [[0] * 2, [-1] * 2, [0] * 2])

            results = sphere.compute_intersect(rays)

            self.assertTrue(results["check"].all())
            np.testing.assert_array_almost_equal(results["intersect"], [[0, 5, 0]] * 2)
            np.testing.assert_array_almost_equal(results["normal"], [[0, 1, 0]] * 2)
            np.testing.assert_array_equal(results["albedo"], [1, 1])

            rays = g_rays.Rays([[0] * 2, [0] * 2, [100] * 2], [[0] * 2, [0] * 2, [-1] * 2])

            results = sphere.compute_intersect(rays)

            self.assertTrue(results["check"].all())
            np.testing.assert_array_almost_equal(results["intersect"], [[0, 0, 5]] * 2)
            np.testing.assert_array_almost_equal(results["normal"], [[0, 0, 1]] * 2)
            np.testing.assert_array_equal(results["albedo"], [1, 1])

            rays = g_rays.Rays([[100] * 2] * 3, [[-1 / np.sqrt(3)] * 2] * 3)

            results = sphere.compute_intersect(rays)

            self.assertTrue(results["check"].all())
            np.testing.assert_array_almost_equal(results["intersect"], [[5 / np.sqrt(3)] * 3] * 2)
            np.testing.assert_array_almost_equal(results["normal"], [[1 / np.sqrt(3)] * 3] * 2)
            np.testing.assert_array_equal(results["albedo"], [1, 1])

            rays = g_rays.Rays([[100] * 2] * 3, [[1 / np.sqrt(3)] * 2] * 3)

            results = sphere.compute_intersect(rays)

            self.assertFalse(results["check"].any())
            np.testing.assert_array_equal(results["intersect"], [[np.nan, np.nan, np.nan]] * 2)
            np.testing.assert_array_equal(results["normal"], [[np.nan, np.nan, np.nan]] * 2)
            np.testing.assert_array_equal(results["albedo"], [1] * 2)  # due to albedo not actually implemented

            rays = g_rays.Rays([[100] * 2] * 3, [[1] * 2, [0] * 2, [0] * 2])

            results = sphere.compute_intersect(rays)

            self.assertFalse(results["check"].any())
            np.testing.assert_array_equal(results["intersect"], [[np.nan, np.nan, np.nan]] * 2)
            np.testing.assert_array_equal(results["normal"], [[np.nan, np.nan, np.nan]] * 2)
            np.testing.assert_array_equal(results["albedo"], [1] * 2)  # due to albedo not actually implemented

            rays = g_rays.Rays([[100] * 2] * 3, [[-1 / np.sqrt(3), 1 / np.sqrt(3)]] * 3)

            results = sphere.compute_intersect(rays)

            np.testing.assert_array_equal(results["check"], [True, False])
            np.testing.assert_array_almost_equal(results["intersect"], [[5 / np.sqrt(3)] * 3, [np.nan] * 3])
            np.testing.assert_array_almost_equal(results["normal"], [[1 / np.sqrt(3)] * 3, [np.nan] * 3])
            np.testing.assert_array_equal(results["albedo"], [1, 1])

        sphere = g_shapes.Ellipsoid(self.off_center, ellipsoid_matrix=self.ellipsoid_matrix_sphere,
                                    principal_axes=self.principal_axes_sphere,
                                    orientation=self.orientation.matrix)

        with self.subTest(case='sphere', center='offset', numrays=1):
            ray = g_rays.Rays([100] * 3, [-1 / np.sqrt(3)] * 3)

            results = sphere.compute_intersect(ray)

            self.assertTrue(results["check"])
            np.testing.assert_array_almost_equal(results["intersect"], [5 / np.sqrt(3) + 20] * 3)
            np.testing.assert_array_almost_equal(results["normal"], [1 / np.sqrt(3)] * 3)
            self.assertEqual(results["albedo"], 1)

            ray = g_rays.Rays([100] * 3, [1 / np.sqrt(3)] * 3)

            results = sphere.compute_intersect(ray)

            self.assertFalse(results["check"])
            np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
            np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
            np.testing.assert_array_equal(results["albedo"], np.nan)

            ray = g_rays.Rays([100] * 3, [1, 0, 0])

            results = sphere.compute_intersect(ray)

            self.assertFalse(results["check"])
            np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
            np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
            np.testing.assert_array_equal(results["albedo"], np.nan)

        with self.subTest(case='sphere', center='offset', numrays=2):
            rays = g_rays.Rays([[100] * 2] * 3, [[-1 / np.sqrt(3)] * 2] * 3)

            results = sphere.compute_intersect(rays)

            self.assertTrue(results["check"].all())
            np.testing.assert_array_almost_equal(results["intersect"], [[5 / np.sqrt(3) + 20] * 3] * 2)
            np.testing.assert_array_almost_equal(results["normal"], [[1 / np.sqrt(3)] * 3] * 2)
            np.testing.assert_array_equal(results["albedo"], [1, 1])

            rays = g_rays.Rays([[100] * 2] * 3, [[1 / np.sqrt(3)] * 2] * 3)

            results = sphere.compute_intersect(rays)

            self.assertFalse(results["check"].any())
            np.testing.assert_array_equal(results["intersect"], [[np.nan, np.nan, np.nan]] * 2)
            np.testing.assert_array_equal(results["normal"], [[np.nan, np.nan, np.nan]] * 2)
            np.testing.assert_array_equal(results["albedo"], [1] * 2)  # due to albedo not actually implemented

            rays = g_rays.Rays([[100] * 2] * 3, [[1] * 2, [0] * 2, [0] * 2])

            results = sphere.compute_intersect(rays)

            self.assertFalse(results["check"].any())
            np.testing.assert_array_equal(results["intersect"], [[np.nan, np.nan, np.nan]] * 2)
            np.testing.assert_array_equal(results["normal"], [[np.nan, np.nan, np.nan]] * 2)
            np.testing.assert_array_equal(results["albedo"], [1] * 2)  # due to albedo not actually implemented

            rays = g_rays.Rays([[100] * 2] * 3, [[-1 / np.sqrt(3), 1 / np.sqrt(3)]] * 3)

            results = sphere.compute_intersect(rays)

            np.testing.assert_array_equal(results["check"], [True, False])
            np.testing.assert_array_almost_equal(results["intersect"], [[5 / np.sqrt(3) + 20] * 3, [np.nan] * 3])
            np.testing.assert_array_almost_equal(results["normal"], [[1 / np.sqrt(3)] * 3, [np.nan] * 3])
            np.testing.assert_array_equal(results["albedo"], [1, 1])

        ellipse = g_shapes.Ellipsoid(self.origin, ellipsoid_matrix=self.ellipsoid_matrix_ellipse,
                                     principal_axes=self.principal_axes_ellipse,
                                     orientation=self.orientation.matrix)

        with self.subTest(case='ellipse', center='origin', numrays=1):
            for ax in range(3):
                ray = g_rays.Rays(100 * self.orientation.matrix[:, ax], -self.orientation.matrix[:, ax])

                results = ellipse.compute_intersect(ray)

                self.assertTrue(results["check"])

                np.testing.assert_array_almost_equal(results["intersect"],
                                                     self.orientation.matrix[:, ax] * self.principal_axes_ellipse[ax])
                np.testing.assert_array_almost_equal(results["normal"], self.orientation.matrix[:, ax])
                self.assertEqual(results["albedo"], 1)

            ray = g_rays.Rays([100] * 3, [-1 / np.sqrt(3)] * 3)

            results = ellipse.compute_intersect(ray)
            self.assertTrue(results["check"])

            int_point = [np.sqrt(1 / np.matmul(ray.direction, np.matmul(self.ellipsoid_matrix_ellipse,
                                                                        ray.direction))) * 1 / np.sqrt(3)] * 3
            np.testing.assert_array_almost_equal(results["intersect"], int_point)
            normal = np.matmul(self.ellipsoid_matrix_ellipse, int_point)
            normal /= np.linalg.norm(normal, axis=0)
            np.testing.assert_array_almost_equal(results["normal"], normal)
            self.assertEqual(results["albedo"], 1)

            ray = g_rays.Rays([100] * 3, [1 / np.sqrt(3)] * 3)

            results = ellipse.compute_intersect(ray)

            self.assertFalse(results["check"])
            np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
            np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
            np.testing.assert_array_equal(results["albedo"], np.nan)

            ray = g_rays.Rays([100] * 3, [1, 0, 0])

            results = ellipse.compute_intersect(ray)

            self.assertFalse(results["check"])
            np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
            np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
            np.testing.assert_array_equal(results["albedo"], np.nan)

        with self.subTest(case='ellipse', center='origin', numrays=3):
            rays = g_rays.Rays(100 * self.orientation.matrix, -self.orientation.matrix)

            results = ellipse.compute_intersect(rays)

            self.assertTrue(results["check"].all())

            np.testing.assert_array_almost_equal(results["intersect"],
                                                 (self.orientation.matrix * self.principal_axes_ellipse).T)
            np.testing.assert_array_almost_equal(results["normal"], self.orientation.matrix.T)
            np.testing.assert_array_equal(results["albedo"], [1] * 3)

            rays = g_rays.Rays([[100] * 2] * 3, [[-1 / np.sqrt(3)] * 2] * 3)

            results = ellipse.compute_intersect(rays)

            int_point = [np.sqrt(1 / np.matmul(rays[0].direction, np.matmul(self.ellipsoid_matrix_ellipse,
                                                                            rays[0].direction))) * 1 / np.sqrt(3)] * 3
            np.testing.assert_array_almost_equal(results["intersect"], [int_point] * 2)
            normal = np.matmul(self.ellipsoid_matrix_ellipse, int_point)
            normal /= np.linalg.norm(normal, axis=0)
            np.testing.assert_array_almost_equal(results["normal"], [normal] * 2)
            np.testing.assert_array_equal(results["albedo"], [1, 1])

            rays = g_rays.Rays([[100] * 2] * 3, [[1 / np.sqrt(3)] * 2] * 3)

            results = ellipse.compute_intersect(rays)

            self.assertFalse(results["check"].any())
            np.testing.assert_array_equal(results["intersect"], [[np.nan, np.nan, np.nan]] * 2)
            np.testing.assert_array_equal(results["normal"], [[np.nan, np.nan, np.nan]] * 2)
            np.testing.assert_array_equal(results["albedo"], [1] * 2)  # due to albedo not actually implemented

            rays = g_rays.Rays([[100] * 2] * 3, [[1] * 2, [0] * 2, [0] * 2])

            results = ellipse.compute_intersect(rays)

            self.assertFalse(results["check"].any())
            np.testing.assert_array_equal(results["intersect"], [[np.nan, np.nan, np.nan]] * 2)
            np.testing.assert_array_equal(results["normal"], [[np.nan, np.nan, np.nan]] * 2)
            np.testing.assert_array_equal(results["albedo"], [1] * 2)  # due to albedo not actually implemented

            rays = g_rays.Rays([[100] * 2] * 3, [[-1 / np.sqrt(3), 1 / np.sqrt(3)]] * 3)

            results = ellipse.compute_intersect(rays)

            np.testing.assert_array_equal(results["check"], [True, False])
            np.testing.assert_array_almost_equal(results["intersect"], [int_point, [np.nan] * 3])
            np.testing.assert_array_almost_equal(results["normal"], [normal, [np.nan] * 3])
            np.testing.assert_array_equal(results["albedo"], [1, 1])

        ellipse = g_shapes.Ellipsoid(self.off_center, ellipsoid_matrix=self.ellipsoid_matrix_ellipse,
                                     principal_axes=self.principal_axes_ellipse,
                                     orientation=self.orientation.matrix)

        with self.subTest(case='ellipse', center='offset', numrays=1):
            ray = g_rays.Rays([100] * 3, [-1 / np.sqrt(3)] * 3)

            results = ellipse.compute_intersect(ray)

            int_point = [np.sqrt(1 / np.matmul(ray.direction, np.matmul(self.ellipsoid_matrix_ellipse,
                                                                        ray.direction))) * 1 / np.sqrt(3) + 20] * 3
            np.testing.assert_array_almost_equal(results["intersect"], int_point)
            normal = np.matmul(self.ellipsoid_matrix_ellipse, np.array(int_point) - 20)
            normal /= np.linalg.norm(normal, axis=0)
            np.testing.assert_array_almost_equal(results["normal"], normal)
            self.assertEqual(results["albedo"], 1)

            ray = g_rays.Rays([100] * 3, [1 / np.sqrt(3)] * 3)

            results = ellipse.compute_intersect(ray)

            self.assertFalse(results["check"])
            np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
            np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
            np.testing.assert_array_equal(results["albedo"], np.nan)

            ray = g_rays.Rays([100] * 3, [1, 0, 0])

            results = ellipse.compute_intersect(ray)

            self.assertFalse(results["check"])
            np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
            np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
            np.testing.assert_array_equal(results["albedo"], np.nan)

        with self.subTest(case='ellipse', center='offset', numrays=2):
            rays = g_rays.Rays([[100] * 2] * 3, [[-1 / np.sqrt(3)] * 2] * 3)

            results = ellipse.compute_intersect(rays)

            int_point = [np.sqrt(1 / np.matmul(rays[0].direction,
                                               np.matmul(self.ellipsoid_matrix_ellipse,
                                                         rays[0].direction))) * 1 / np.sqrt(3) + 20] * 3
            np.testing.assert_array_almost_equal(results["intersect"], [int_point] * 2)
            normal = np.matmul(self.ellipsoid_matrix_ellipse, np.array(int_point) - 20)
            normal /= np.linalg.norm(normal, axis=0)
            np.testing.assert_array_almost_equal(results["normal"], [normal] * 2)
            np.testing.assert_array_equal(results["albedo"], [1, 1])

            rays = g_rays.Rays([[100] * 2] * 3, [[1 / np.sqrt(3)] * 2] * 3)

            results = ellipse.compute_intersect(rays)

            self.assertFalse(results["check"].any())
            np.testing.assert_array_equal(results["intersect"], [[np.nan, np.nan, np.nan]] * 2)
            np.testing.assert_array_equal(results["normal"], [[np.nan, np.nan, np.nan]] * 2)
            np.testing.assert_array_equal(results["albedo"], [1] * 2)  # due to albedo not actually implemented

            rays = g_rays.Rays([[100] * 2] * 3, [[1] * 2, [0] * 2, [0] * 2])

            results = ellipse.compute_intersect(rays)

            self.assertFalse(results["check"].any())
            np.testing.assert_array_equal(results["intersect"], [[np.nan, np.nan, np.nan]] * 2)
            np.testing.assert_array_equal(results["normal"], [[np.nan, np.nan, np.nan]] * 2)
            np.testing.assert_array_equal(results["albedo"], [1] * 2)  # due to albedo not actually implemented

            rays = g_rays.Rays([[100] * 2] * 3, [[-1 / np.sqrt(3), 1 / np.sqrt(3)]] * 3)

            results = ellipse.compute_intersect(rays)

            np.testing.assert_array_equal(results["check"], [True, False])
            np.testing.assert_array_almost_equal(results["intersect"], [int_point, [np.nan] * 3])
            np.testing.assert_array_almost_equal(results["normal"], [normal, [np.nan] * 3])
            np.testing.assert_array_equal(results["albedo"], [1, 1])

    def test_trace(self):
        sphere = g_shapes.Ellipsoid(self.origin, ellipsoid_matrix=self.ellipsoid_matrix_sphere,
                                    principal_axes=self.principal_axes_sphere,
                                    orientation=self.orientation.matrix)

        with self.subTest(case='sphere', center='origin', numrays=1):
            ray = g_rays.Rays([100, 0, 0], [-1, 0, 0])

            results = sphere.trace(ray)

            self.assertTrue(results["check"])
            np.testing.assert_array_almost_equal(results["intersect"], [5, 0, 0])
            np.testing.assert_array_almost_equal(results["normal"], [1, 0, 0])
            self.assertEqual(results["albedo"], 1)

            ray = g_rays.Rays([0, 100, 0], [0, -1, 0])

            results = sphere.trace(ray)

            self.assertTrue(results["check"])
            np.testing.assert_array_almost_equal(results["intersect"], [0, 5, 0])
            np.testing.assert_array_almost_equal(results["normal"], [0, 1, 0])
            self.assertEqual(results["albedo"], 1)

            ray = g_rays.Rays([0, 0, 100], [0, 0, -1])

            results = sphere.trace(ray)

            self.assertTrue(results["check"])
            np.testing.assert_array_almost_equal(results["intersect"], [0, 0, 5])
            np.testing.assert_array_almost_equal(results["normal"], [0, 0, 1])
            self.assertEqual(results["albedo"], 1)

            ray = g_rays.Rays([100] * 3, [-1 / np.sqrt(3)] * 3)

            results = sphere.trace(ray)

            self.assertTrue(results["check"])
            np.testing.assert_array_almost_equal(results["intersect"], [5 / np.sqrt(3)] * 3)
            np.testing.assert_array_almost_equal(results["normal"], [1 / np.sqrt(3)] * 3)
            self.assertEqual(results["albedo"], 1)

            ray = g_rays.Rays([100] * 3, [1 / np.sqrt(3)] * 3)

            results = sphere.trace(ray)

            self.assertFalse(results["check"])
            np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
            np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
            np.testing.assert_array_equal(results["albedo"], np.nan)

            ray = g_rays.Rays([100] * 3, [1, 0, 0])

            results = sphere.trace(ray)

            self.assertFalse(results["check"])
            np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
            np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
            np.testing.assert_array_equal(results["albedo"], np.nan)

        with self.subTest(case='sphere', center='origin', numrays=2):
            rays = g_rays.Rays([[100] * 2, [0] * 2, [0] * 2], [[-1] * 2, [0] * 2, [0] * 2])

            results = sphere.trace(rays)

            self.assertTrue(results["check"].all())
            np.testing.assert_array_almost_equal(results["intersect"], [[5, 0, 0]] * 2)
            np.testing.assert_array_almost_equal(results["normal"], [[1, 0, 0]] * 2)
            np.testing.assert_array_equal(results["albedo"], [1, 1])

            rays = g_rays.Rays([[0] * 2, [100] * 2, [0] * 2], [[0] * 2, [-1] * 2, [0] * 2])

            results = sphere.trace(rays)

            self.assertTrue(results["check"].all())
            np.testing.assert_array_almost_equal(results["intersect"], [[0, 5, 0]] * 2)
            np.testing.assert_array_almost_equal(results["normal"], [[0, 1, 0]] * 2)
            np.testing.assert_array_equal(results["albedo"], [1, 1])

            rays = g_rays.Rays([[0] * 2, [0] * 2, [100] * 2], [[0] * 2, [0] * 2, [-1] * 2])

            results = sphere.trace(rays)

            self.assertTrue(results["check"].all())
            np.testing.assert_array_almost_equal(results["intersect"], [[0, 0, 5]] * 2)
            np.testing.assert_array_almost_equal(results["normal"], [[0, 0, 1]] * 2)
            np.testing.assert_array_equal(results["albedo"], [1, 1])

            rays = g_rays.Rays([[100] * 2] * 3, [[-1 / np.sqrt(3)] * 2] * 3)

            results = sphere.trace(rays)

            self.assertTrue(results["check"].all())
            np.testing.assert_array_almost_equal(results["intersect"], [[5 / np.sqrt(3)] * 3] * 2)
            np.testing.assert_array_almost_equal(results["normal"], [[1 / np.sqrt(3)] * 3] * 2)
            np.testing.assert_array_equal(results["albedo"], [1, 1])

            rays = g_rays.Rays([[100] * 2] * 3, [[1 / np.sqrt(3)] * 2] * 3)

            results = sphere.trace(rays)

            self.assertFalse(results["check"].any())
            np.testing.assert_array_equal(results["intersect"], [[np.nan, np.nan, np.nan]] * 2)
            np.testing.assert_array_equal(results["normal"], [[np.nan, np.nan, np.nan]] * 2)
            np.testing.assert_array_equal(results["albedo"], [1] * 2)  # due to albedo not actually implemented

            rays = g_rays.Rays([[100] * 2] * 3, [[1] * 2, [0] * 2, [0] * 2])

            results = sphere.trace(rays)

            self.assertFalse(results["check"].any())
            np.testing.assert_array_equal(results["intersect"], [[np.nan, np.nan, np.nan]] * 2)
            np.testing.assert_array_equal(results["normal"], [[np.nan, np.nan, np.nan]] * 2)
            np.testing.assert_array_equal(results["albedo"], [1] * 2)  # due to albedo not actually implemented

            rays = g_rays.Rays([[100] * 2] * 3, [[-1 / np.sqrt(3), 1 / np.sqrt(3)]] * 3)

            results = sphere.trace(rays)

            np.testing.assert_array_equal(results["check"], [True, False])
            np.testing.assert_array_almost_equal(results["intersect"], [[5 / np.sqrt(3)] * 3, [np.nan] * 3])
            np.testing.assert_array_almost_equal(results["normal"], [[1 / np.sqrt(3)] * 3, [np.nan] * 3])
            np.testing.assert_array_equal(results["albedo"], [1, 1])

        sphere.translate(np.array([5, 0, 0]))

        with self.subTest(case='sphere', center='translated', numrays=2):
            rays = g_rays.Rays([[100] * 2, [0] * 2, [0] * 2], [[-1] * 2, [0] * 2, [0] * 2])

            results = sphere.trace(rays)

            self.assertTrue(results["check"].all())
            np.testing.assert_array_almost_equal(results["intersect"], [[10, 0, 0]] * 2)
            np.testing.assert_array_almost_equal(results["normal"], [[1, 0, 0]] * 2)
            np.testing.assert_array_equal(results["albedo"], [1, 1])

            rays = g_rays.Rays([[0] * 2, [100] * 2, [0] * 2], [[0] * 2, [-1] * 2, [0] * 2])

            results = sphere.trace(rays)

            self.assertTrue(results["check"].all())
            np.testing.assert_array_almost_equal(results["intersect"], [[0, 0, 0]] * 2)
            np.testing.assert_array_almost_equal(results["normal"], [[-1, 0, 0]] * 2)
            np.testing.assert_array_equal(results["albedo"], [1, 1])

            rays = g_rays.Rays([[0] * 2, [0] * 2, [100] * 2], [[0] * 2, [0] * 2, [-1] * 2])

            results = sphere.trace(rays)

            self.assertTrue(results["check"].all())
            np.testing.assert_array_almost_equal(results["intersect"], [[0, 0, 0]] * 2)
            np.testing.assert_array_almost_equal(results["normal"], [[-1, 0, 0]] * 2)
            np.testing.assert_array_equal(results["albedo"], [1, 1])

            rays = g_rays.Rays([[100] * 2] * 3, [[1 / np.sqrt(3)] * 2] * 3)

            results = sphere.trace(rays)

            self.assertFalse(results["check"].any())
            np.testing.assert_array_equal(results["intersect"], [[np.nan, np.nan, np.nan]] * 2)
            np.testing.assert_array_equal(results["normal"], [[np.nan, np.nan, np.nan]] * 2)
            np.testing.assert_array_equal(results["albedo"], [1] * 2)  # due to albedo not actually implemented

            rays = g_rays.Rays([[100] * 2] * 3, [[1] * 2, [0] * 2, [0] * 2])

            results = sphere.trace(rays)

            self.assertFalse(results["check"].any())
            np.testing.assert_array_equal(results["intersect"], [[np.nan, np.nan, np.nan]] * 2)
            np.testing.assert_array_equal(results["normal"], [[np.nan, np.nan, np.nan]] * 2)
            np.testing.assert_array_equal(results["albedo"], [1] * 2)  # due to albedo not actually implemented

            rays = g_rays.Rays([[100, 100], [0, 0], [0, 0]], [[-1, 1], [0, 0], [0, 0]])

            results = sphere.trace(rays)

            np.testing.assert_array_equal(results["check"], [True, False])
            np.testing.assert_array_almost_equal(results["intersect"], [[10, 0, 0], [np.nan] * 3])
            np.testing.assert_array_almost_equal(results["normal"], [[1, 0, 0], [np.nan] * 3])
            np.testing.assert_array_equal(results["albedo"], [1, 1])

        sphere = g_shapes.Ellipsoid(self.off_center, ellipsoid_matrix=self.ellipsoid_matrix_sphere,
                                    principal_axes=self.principal_axes_sphere,
                                    orientation=self.orientation.matrix)

        with self.subTest(case='sphere', center='offset', numrays=1):
            ray = g_rays.Rays([100] * 3, [-1 / np.sqrt(3)] * 3)

            results = sphere.trace(ray)

            self.assertTrue(results["check"])
            np.testing.assert_array_almost_equal(results["intersect"], [5 / np.sqrt(3) + 20] * 3)
            np.testing.assert_array_almost_equal(results["normal"], [1 / np.sqrt(3)] * 3)
            self.assertEqual(results["albedo"], 1)

            ray = g_rays.Rays([100] * 3, [1 / np.sqrt(3)] * 3)

            results = sphere.trace(ray)

            self.assertFalse(results["check"])
            np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
            np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
            np.testing.assert_array_equal(results["albedo"], np.nan)

            ray = g_rays.Rays([100] * 3, [1, 0, 0])

            results = sphere.trace(ray)

            self.assertFalse(results["check"])
            np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
            np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
            np.testing.assert_array_equal(results["albedo"], np.nan)

        with self.subTest(case='sphere', center='offset', numrays=2):
            rays = g_rays.Rays([[100] * 2] * 3, [[-1 / np.sqrt(3)] * 2] * 3)

            results = sphere.trace(rays)

            self.assertTrue(results["check"].all())
            np.testing.assert_array_almost_equal(results["intersect"], [[5 / np.sqrt(3) + 20] * 3] * 2)
            np.testing.assert_array_almost_equal(results["normal"], [[1 / np.sqrt(3)] * 3] * 2)
            np.testing.assert_array_equal(results["albedo"], [1, 1])

            rays = g_rays.Rays([[100] * 2] * 3, [[1 / np.sqrt(3)] * 2] * 3)

            results = sphere.trace(rays)

            self.assertFalse(results["check"].any())
            np.testing.assert_array_equal(results["intersect"], [[np.nan, np.nan, np.nan]] * 2)
            np.testing.assert_array_equal(results["normal"], [[np.nan, np.nan, np.nan]] * 2)
            np.testing.assert_array_equal(results["albedo"], [1] * 2)  # due to albedo not actually implemented

            rays = g_rays.Rays([[100] * 2] * 3, [[1] * 2, [0] * 2, [0] * 2])

            results = sphere.trace(rays)

            self.assertFalse(results["check"].any())
            np.testing.assert_array_equal(results["intersect"], [[np.nan, np.nan, np.nan]] * 2)
            np.testing.assert_array_equal(results["normal"], [[np.nan, np.nan, np.nan]] * 2)
            np.testing.assert_array_equal(results["albedo"], [1] * 2)  # due to albedo not actually implemented

            rays = g_rays.Rays([[100] * 2] * 3, [[-1 / np.sqrt(3), 1 / np.sqrt(3)]] * 3)

            results = sphere.trace(rays)

            np.testing.assert_array_equal(results["check"], [True, False])
            np.testing.assert_array_almost_equal(results["intersect"], [[5 / np.sqrt(3) + 20] * 3, [np.nan] * 3])
            np.testing.assert_array_almost_equal(results["normal"], [[1 / np.sqrt(3)] * 3, [np.nan] * 3])
            np.testing.assert_array_equal(results["albedo"], [1, 1])

        ellipse = g_shapes.Ellipsoid(self.origin, ellipsoid_matrix=self.ellipsoid_matrix_ellipse,
                                     principal_axes=self.principal_axes_ellipse,
                                     orientation=self.orientation.matrix)

        with self.subTest(case='ellipse', center='origin', numrays=1):
            for ax in range(3):
                ray = g_rays.Rays(100 * self.orientation.matrix[:, ax], -self.orientation.matrix[:, ax])

                results = ellipse.trace(ray)

                self.assertTrue(results["check"])

                np.testing.assert_array_almost_equal(results["intersect"],
                                                     self.orientation.matrix[:, ax] * self.principal_axes_ellipse[ax])
                np.testing.assert_array_almost_equal(results["normal"], self.orientation.matrix[:, ax])
                self.assertEqual(results["albedo"], 1)

            ray = g_rays.Rays([100] * 3, [-1 / np.sqrt(3)] * 3)

            results = ellipse.trace(ray)
            self.assertTrue(results["check"])

            int_point = [np.sqrt(1 / np.matmul(ray.direction, np.matmul(self.ellipsoid_matrix_ellipse,
                                                                        ray.direction))) * 1 / np.sqrt(3)] * 3
            np.testing.assert_array_almost_equal(results["intersect"], int_point)
            normal = np.matmul(self.ellipsoid_matrix_ellipse, int_point)
            normal /= np.linalg.norm(normal, axis=0)
            np.testing.assert_array_almost_equal(results["normal"], normal)
            self.assertEqual(results["albedo"], 1)

            ray = g_rays.Rays([100] * 3, [1 / np.sqrt(3)] * 3)

            results = ellipse.trace(ray)

            self.assertFalse(results["check"])
            np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
            np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
            np.testing.assert_array_equal(results["albedo"], np.nan)

            ray = g_rays.Rays([100] * 3, [1, 0, 0])

            results = ellipse.trace(ray)

            self.assertFalse(results["check"])
            np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
            np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
            np.testing.assert_array_equal(results["albedo"], np.nan)

        with self.subTest(case='ellipse', center='origin', numrays=3):
            rays = g_rays.Rays(100 * self.orientation.matrix, -self.orientation.matrix)

            results = ellipse.trace(rays)

            self.assertTrue(results["check"].all())

            np.testing.assert_array_almost_equal(results["intersect"],
                                                 (self.orientation.matrix * self.principal_axes_ellipse).T)
            np.testing.assert_array_almost_equal(results["normal"], self.orientation.matrix.T)
            np.testing.assert_array_equal(results["albedo"], [1] * 3)

            rays = g_rays.Rays([[100] * 2] * 3, [[-1 / np.sqrt(3)] * 2] * 3)

            results = ellipse.trace(rays)

            int_point = [np.sqrt(1 / np.matmul(rays[0].direction, np.matmul(self.ellipsoid_matrix_ellipse,
                                                                            rays[0].direction))) * 1 / np.sqrt(3)] * 3
            np.testing.assert_array_almost_equal(results["intersect"], [int_point] * 2)
            normal = np.matmul(self.ellipsoid_matrix_ellipse, int_point)
            normal /= np.linalg.norm(normal, axis=0)
            np.testing.assert_array_almost_equal(results["normal"], [normal] * 2)
            np.testing.assert_array_equal(results["albedo"], [1, 1])

            rays = g_rays.Rays([[100] * 2] * 3, [[1 / np.sqrt(3)] * 2] * 3)

            results = ellipse.trace(rays)

            self.assertFalse(results["check"].any())
            np.testing.assert_array_equal(results["intersect"], [[np.nan, np.nan, np.nan]] * 2)
            np.testing.assert_array_equal(results["normal"], [[np.nan, np.nan, np.nan]] * 2)
            np.testing.assert_array_equal(results["albedo"], [1] * 2)  # due to albedo not actually implemented

            rays = g_rays.Rays([[100] * 2] * 3, [[1] * 2, [0] * 2, [0] * 2])

            results = ellipse.trace(rays)

            self.assertFalse(results["check"].any())
            np.testing.assert_array_equal(results["intersect"], [[np.nan, np.nan, np.nan]] * 2)
            np.testing.assert_array_equal(results["normal"], [[np.nan, np.nan, np.nan]] * 2)
            np.testing.assert_array_equal(results["albedo"], [1] * 2)  # due to albedo not actually implemented

            rays = g_rays.Rays([[100] * 2] * 3, [[-1 / np.sqrt(3), 1 / np.sqrt(3)]] * 3)

            results = ellipse.trace(rays)

            np.testing.assert_array_equal(results["check"], [True, False])
            np.testing.assert_array_almost_equal(results["intersect"], [int_point, [np.nan] * 3])
            np.testing.assert_array_almost_equal(results["normal"], [normal, [np.nan] * 3])
            np.testing.assert_array_equal(results["albedo"], [1, 1])

        ellipse = g_shapes.Ellipsoid(self.off_center, ellipsoid_matrix=self.ellipsoid_matrix_ellipse,
                                     principal_axes=self.principal_axes_ellipse,
                                     orientation=self.orientation.matrix)

        with self.subTest(case='ellipse', center='offset', numrays=1):
            ray = g_rays.Rays([100] * 3, [-1 / np.sqrt(3)] * 3)

            results = ellipse.trace(ray)

            int_point = [np.sqrt(1 / np.matmul(ray.direction, np.matmul(self.ellipsoid_matrix_ellipse,
                                                                        ray.direction))) * 1 / np.sqrt(3) + 20] * 3
            np.testing.assert_array_almost_equal(results["intersect"], int_point)
            normal = np.matmul(self.ellipsoid_matrix_ellipse, np.array(int_point) - 20)
            normal /= np.linalg.norm(normal, axis=0)
            np.testing.assert_array_almost_equal(results["normal"], normal)
            self.assertEqual(results["albedo"], 1)

            ray = g_rays.Rays([100] * 3, [1 / np.sqrt(3)] * 3)

            results = ellipse.trace(ray)

            self.assertFalse(results["check"])
            np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
            np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
            np.testing.assert_array_equal(results["albedo"], np.nan)

            ray = g_rays.Rays([100] * 3, [1, 0, 0])

            results = ellipse.trace(ray)

            self.assertFalse(results["check"])
            np.testing.assert_array_equal(results["intersect"], [np.nan, np.nan, np.nan])
            np.testing.assert_array_equal(results["normal"], [np.nan, np.nan, np.nan])
            np.testing.assert_array_equal(results["albedo"], np.nan)

        with self.subTest(case='ellipse', center='offset', numrays=2):
            rays = g_rays.Rays([[100] * 2] * 3, [[-1 / np.sqrt(3)] * 2] * 3)

            results = ellipse.trace(rays)

            int_point = [np.sqrt(1 / np.matmul(rays[0].direction,
                                               np.matmul(self.ellipsoid_matrix_ellipse,
                                                         rays[0].direction))) * 1 / np.sqrt(3) + 20] * 3
            np.testing.assert_array_almost_equal(results["intersect"], [int_point] * 2)
            normal = np.matmul(self.ellipsoid_matrix_ellipse, np.array(int_point) - 20)
            normal /= np.linalg.norm(normal, axis=0)
            np.testing.assert_array_almost_equal(results["normal"], [normal] * 2)
            np.testing.assert_array_equal(results["albedo"], [1, 1])

            rays = g_rays.Rays([[100] * 2] * 3, [[1 / np.sqrt(3)] * 2] * 3)

            results = ellipse.trace(rays)

            self.assertFalse(results["check"].any())
            np.testing.assert_array_equal(results["intersect"], [[np.nan, np.nan, np.nan]] * 2)
            np.testing.assert_array_equal(results["normal"], [[np.nan, np.nan, np.nan]] * 2)
            np.testing.assert_array_equal(results["albedo"], [1] * 2)  # due to albedo not actually implemented

            rays = g_rays.Rays([[100] * 2] * 3, [[1] * 2, [0] * 2, [0] * 2])

            results = ellipse.trace(rays)

            self.assertFalse(results["check"].any())
            np.testing.assert_array_equal(results["intersect"], [[np.nan, np.nan, np.nan]] * 2)
            np.testing.assert_array_equal(results["normal"], [[np.nan, np.nan, np.nan]] * 2)
            np.testing.assert_array_equal(results["albedo"], [1] * 2)  # due to albedo not actually implemented

            rays = g_rays.Rays([[100] * 2] * 3, [[-1 / np.sqrt(3), 1 / np.sqrt(3)]] * 3)

            results = ellipse.trace(rays)

            np.testing.assert_array_equal(results["check"], [True, False])
            np.testing.assert_array_almost_equal(results["intersect"], [int_point, [np.nan] * 3])
            np.testing.assert_array_almost_equal(results["normal"], [normal, [np.nan] * 3])
            np.testing.assert_array_equal(results["albedo"], [1, 1])

    def test_normals(self):
        sphere = g_shapes.Ellipsoid(self.origin, ellipsoid_matrix=self.ellipsoid_matrix_sphere,
                                    principal_axes=self.principal_axes_sphere,
                                    orientation=self.orientation.matrix)

        with self.subTest(case='sphere', center='origin', num_locs=2):
            loc = np.array([0, 0, 5])

            normal = sphere.compute_normals(loc)

            np.testing.assert_array_almost_equal(normal, [0, 0, 1])

            loc = np.array([0, 5, 0])

            normal = sphere.compute_normals(loc)

            np.testing.assert_array_almost_equal(normal, [0, 1, 0])

            loc = np.array([5, 0, 0])

            normal = sphere.compute_normals(loc)

            np.testing.assert_array_almost_equal(normal, [1, 0, 0])

        sphere = g_shapes.Ellipsoid(self.off_center, ellipsoid_matrix=self.ellipsoid_matrix_sphere,
                                    principal_axes=self.principal_axes_sphere,
                                    orientation=self.orientation.matrix)

        with self.subTest(case='sphere', center='off set', num_locs=2):
            loc = np.array([0, 0, 5])

            normal = sphere.compute_normals(loc)

            np.testing.assert_array_almost_equal(normal, [0, 0, 1])

            loc = np.array([0, 5, 0])

            normal = sphere.compute_normals(loc)

            np.testing.assert_array_almost_equal(normal, [0, 1, 0])

            loc = np.array([5, 0, 0])

            normal = sphere.compute_normals(loc)

            np.testing.assert_array_almost_equal(normal, [1, 0, 0])

        ellipse = g_shapes.Ellipsoid(self.origin, ellipsoid_matrix=self.ellipsoid_matrix_ellipse,
                                     principal_axes=self.principal_axes_ellipse,
                                     orientation=self.orientation.matrix)

        with self.subTest(case='ellipse', center='origin', num_locs=2):
            loc = self.orientation.matrix[:, 0] * self.principal_axes_ellipse[0]

            normal = ellipse.compute_normals(loc)

            np.testing.assert_array_almost_equal(normal, self.orientation.matrix[:, 0])

            loc = self.orientation.matrix[:, 1] * self.principal_axes_ellipse[1]

            normal = ellipse.compute_normals(loc)

            np.testing.assert_array_almost_equal(normal, self.orientation.matrix[:, 1])

            loc = self.orientation.matrix[:, 2] * self.principal_axes_ellipse[2]

            normal = ellipse.compute_normals(loc)

            np.testing.assert_array_almost_equal(normal, self.orientation.matrix[:, 2])

        ellipse = g_shapes.Ellipsoid(self.off_center, ellipsoid_matrix=self.ellipsoid_matrix_ellipse,
                                     principal_axes=self.principal_axes_ellipse,
                                     orientation=self.orientation.matrix)

        with self.subTest(case='ellipse', center='off set', num_locs=2):
            loc = self.orientation.matrix[:, 0] * self.principal_axes_ellipse[0]

            normal = ellipse.compute_normals(loc)

            np.testing.assert_array_almost_equal(normal, self.orientation.matrix[:, 0])

            loc = self.orientation.matrix[:, 1] * self.principal_axes_ellipse[1]

            normal = ellipse.compute_normals(loc)

            np.testing.assert_array_almost_equal(normal, self.orientation.matrix[:, 1])

            loc = self.orientation.matrix[:, 2] * self.principal_axes_ellipse[2]

            normal = ellipse.compute_normals(loc)

            np.testing.assert_array_almost_equal(normal, self.orientation.matrix[:, 2])

    def test_rotate(self):
        rotation = at.Rotation([0.5, 0.2, -0.3])

        sphere = g_shapes.Ellipsoid([0, 0, 0], ellipsoid_matrix=self.ellipsoid_matrix_sphere,
                                    principal_axes=self.principal_axes_sphere, orientation=self.orientation.matrix)

        # TODO: consider checking the rotation of the bounding box

        with self.subTest(case='sphere', center='origin', type='array'):
            sphere_copy = copy.deepcopy(sphere)

            sphere_copy.rotate(rotation.matrix)

            np.testing.assert_array_almost_equal(sphere_copy.orientation,
                                                 np.matmul(rotation.matrix, sphere.orientation))

            np.testing.assert_array_almost_equal(sphere_copy.ellipsoid_matrix, sphere.ellipsoid_matrix)

            np.testing.assert_array_equal(sphere_copy.principal_axes, sphere.principal_axes)

        with self.subTest(case='sphere', center='origin', type='Rotation'):
            sphere_copy = copy.deepcopy(sphere)

            sphere_copy.rotate(rotation)

            np.testing.assert_array_almost_equal(sphere_copy.orientation,
                                                 np.matmul(rotation.matrix, sphere.orientation))

            np.testing.assert_array_almost_equal(sphere_copy.ellipsoid_matrix, sphere.ellipsoid_matrix)

            np.testing.assert_array_equal(sphere_copy.principal_axes, sphere.principal_axes)

        sphere = g_shapes.Ellipsoid(self.off_center, ellipsoid_matrix=self.ellipsoid_matrix_sphere,
                                    principal_axes=self.principal_axes_sphere, orientation=self.orientation.matrix)

        with self.subTest(case='sphere', center='offset', type='array'):
            sphere_copy = copy.deepcopy(sphere)

            sphere_copy.rotate(rotation.matrix)

            np.testing.assert_array_almost_equal(sphere_copy.orientation,
                                                 np.matmul(rotation.matrix, sphere.orientation))

            np.testing.assert_array_almost_equal(sphere_copy.ellipsoid_matrix, sphere.ellipsoid_matrix)

            np.testing.assert_array_equal(sphere_copy.principal_axes, sphere.principal_axes)

        with self.subTest(case='sphere', center='offset', type='Rotation'):
            sphere_copy = copy.deepcopy(sphere)

            sphere_copy.rotate(rotation)

            np.testing.assert_array_almost_equal(sphere_copy.orientation,
                                                 np.matmul(rotation.matrix, sphere.orientation))

            np.testing.assert_array_almost_equal(sphere_copy.ellipsoid_matrix, sphere.ellipsoid_matrix)

            np.testing.assert_array_equal(sphere_copy.principal_axes, sphere.principal_axes)

        ellipse = g_shapes.Ellipsoid([0, 0, 0], ellipsoid_matrix=self.ellipsoid_matrix_ellipse,
                                     principal_axes=self.principal_axes_ellipse, orientation=self.orientation.matrix)

        with self.subTest(case='ellipse', center='origin', type='array'):
            ellipse_copy = copy.deepcopy(ellipse)

            ellipse_copy.rotate(rotation)

            np.testing.assert_array_almost_equal(ellipse_copy.orientation,
                                                 np.matmul(rotation.matrix, ellipse.orientation))

            np.testing.assert_array_almost_equal(ellipse_copy.ellipsoid_matrix, np.matmul(rotation.matrix,
                                                                                          np.matmul(
                                                                                              ellipse.ellipsoid_matrix,
                                                                                              rotation.matrix.T)))

            np.testing.assert_array_equal(ellipse_copy.principal_axes, ellipse.principal_axes)

        with self.subTest(case='ellipse', center='origin', type='Rotation'):
            ellipse_copy = copy.deepcopy(ellipse)

            ellipse_copy.rotate(rotation)

            np.testing.assert_array_almost_equal(ellipse_copy.orientation,
                                                 np.matmul(rotation.matrix, ellipse.orientation))

            np.testing.assert_array_almost_equal(ellipse_copy.ellipsoid_matrix, np.matmul(rotation.matrix,
                                                                                          np.matmul(
                                                                                              ellipse.ellipsoid_matrix,
                                                                                              rotation.matrix.T)))

            np.testing.assert_array_equal(ellipse_copy.principal_axes, ellipse.principal_axes)

        ellipse = g_shapes.Ellipsoid(self.off_center, ellipsoid_matrix=self.ellipsoid_matrix_sphere,
                                     principal_axes=self.principal_axes_sphere, orientation=self.orientation.matrix)

        with self.subTest(case='ellipse', center='offset', type='array'):
            ellipse_copy = copy.deepcopy(ellipse)

            ellipse_copy.rotate(rotation)

            np.testing.assert_array_almost_equal(ellipse_copy.orientation,
                                                 np.matmul(rotation.matrix, ellipse.orientation))

            np.testing.assert_array_almost_equal(ellipse_copy.ellipsoid_matrix, np.matmul(rotation.matrix,
                                                                                          np.matmul(
                                                                                              ellipse.ellipsoid_matrix,
                                                                                              rotation.matrix.T)))

            np.testing.assert_array_equal(ellipse_copy.principal_axes, ellipse.principal_axes)

        with self.subTest(case='ellipse', center='offset', type='Rotation'):
            ellipse_copy = copy.deepcopy(ellipse)

            ellipse_copy.rotate(rotation)

            np.testing.assert_array_almost_equal(ellipse_copy.orientation,
                                                 np.matmul(rotation.matrix, ellipse.orientation))

            np.testing.assert_array_almost_equal(ellipse_copy.ellipsoid_matrix, np.matmul(rotation.matrix,
                                                                                          np.matmul(
                                                                                              ellipse.ellipsoid_matrix,
                                                                                              rotation.matrix.T)))

            np.testing.assert_array_equal(ellipse_copy.principal_axes, ellipse.principal_axes)

    def test_translate(self):
        translation_3axis = np.array([0.5, 0.2, -0.3])

        sphere = g_shapes.Ellipsoid([0, 0, 0], ellipsoid_matrix=self.ellipsoid_matrix_sphere,
                                    principal_axes=self.principal_axes_sphere, orientation=self.orientation.matrix)

        # TODO: check translation of bounding box?

        with self.subTest(case='sphere', center='origin', trans_type='3 axis'):
            sphere_copy = copy.deepcopy(sphere)

            sphere_copy.translate(translation_3axis)

            np.testing.assert_array_equal(sphere.center + translation_3axis, sphere_copy.center)

        sphere = g_shapes.Ellipsoid(self.off_center, ellipsoid_matrix=self.ellipsoid_matrix_sphere,
                                    principal_axes=self.principal_axes_sphere, orientation=self.orientation.matrix)

        with self.subTest(case='sphere', center='offset', trans_type='3 axis'):
            sphere_copy = copy.deepcopy(sphere)

            sphere_copy.translate(translation_3axis)

            np.testing.assert_array_equal(sphere.center + translation_3axis, sphere_copy.center)

        ellipse = g_shapes.Ellipsoid([0, 0, 0], ellipsoid_matrix=self.ellipsoid_matrix_ellipse,
                                     principal_axes=self.principal_axes_ellipse, orientation=self.orientation.matrix)

        with self.subTest(case='ellipse', center='origin', trans_type='3 axis'):
            ellipse_copy = copy.deepcopy(ellipse)

            ellipse_copy.translate(translation_3axis)

            np.testing.assert_array_equal(ellipse.center + translation_3axis, ellipse_copy.center)

        ellipse = g_shapes.Ellipsoid(self.off_center, ellipsoid_matrix=self.ellipsoid_matrix_ellipse,
                                     principal_axes=self.principal_axes_ellipse, orientation=self.orientation.matrix)

        with self.subTest(case='ellipse', center='offset', trans_type='3 axis'):
            ellipse_copy = copy.deepcopy(ellipse)

            ellipse_copy.translate(translation_3axis)

            np.testing.assert_array_equal(ellipse.center + translation_3axis, ellipse_copy.center)

    def test_find_limbs(self):

        sphere = g_shapes.Ellipsoid([0, 0, 0], ellipsoid_matrix=self.ellipsoid_matrix_sphere,
                                    principal_axes=self.principal_axes_sphere, orientation=self.orientation.matrix)

        ellipse = g_shapes.Ellipsoid([0, 0, 10], ellipsoid_matrix=self.ellipsoid_matrix_ellipse,
                                     principal_axes=self.principal_axes_ellipse, orientation=self.orientation.matrix)

        position = np.array([.1, .2, 50.])

        angles = np.linspace(0, 2 * np.pi, 90)

        scan_center_dir = position / np.linalg.norm(position) + [0.01, -0.01, 0.02]

        scan_dirs = np.vstack([np.cos(angles), np.sin(angles), np.zeros(angles.size)])

        def assert_limb(shape, limbs, position, center_dir, scan_vector):
            cxs = np.cross(center_dir.T, scan_vector.T)

            alimb = shape.ellipsoid_matrix @ limbs

            np.testing.assert_almost_equal((limbs * alimb).sum(axis=0), 1)
            np.testing.assert_almost_equal((shape.center - position).T @ alimb, -1)
            np.testing.assert_almost_equal((cxs.T * limbs).sum(axis=0), -cxs @ (shape.center-position))

        with self.subTest(case='sphere'):
            limbs = sphere.find_limbs(scan_center_dir, scan_dirs, position)

            assert_limb(sphere, limbs-(sphere.center - position).reshape(-1, 1), position, scan_center_dir, scan_dirs)

        with self.subTest(case='ellipse'):
            limbs = ellipse.find_limbs(scan_center_dir, scan_dirs, position)

            assert_limb(ellipse, limbs-(ellipse.center-position).reshape(-1, 1), position, scan_center_dir, scan_dirs)

    def test_compute_limb_jacobian(self):

        def num_deriv(shape, scan_center_dir, scan_dirs, position, state='center', delta=1e-6):

            if state == "center":

                scan_center_dir_pert = scan_center_dir + [delta, 0, 0]

                limb_pert_x_f = shape.find_limbs(scan_center_dir_pert, scan_dirs, position)

                scan_center_dir_pert = scan_center_dir + [0, delta, 0]

                limb_pert_y_f = shape.find_limbs(scan_center_dir_pert, scan_dirs, position)

                scan_center_dir_pert = scan_center_dir + [0, 0, delta]

                limb_pert_z_f = shape.find_limbs(scan_center_dir_pert, scan_dirs, position)

                scan_center_dir_pert = scan_center_dir - [delta, 0, 0]

                limb_pert_x_b = shape.find_limbs(scan_center_dir_pert, scan_dirs, position)

                scan_center_dir_pert = scan_center_dir - [0, delta, 0]

                limb_pert_y_b = shape.find_limbs(scan_center_dir_pert, scan_dirs, position)

                scan_center_dir_pert = scan_center_dir - [0, 0, delta]

                limb_pert_z_b = shape.find_limbs(scan_center_dir_pert, scan_dirs, position)

                return np.array([(limb_pert_x_f - limb_pert_x_b) / (2 * delta),
                                 (limb_pert_y_f - limb_pert_y_b) / (2 * delta),
                                 (limb_pert_z_f - limb_pert_z_b) / (2 * delta)]).swapaxes(0, -1)

            elif state == "relative position":

                position_pert = position + [delta, 0, 0]

                limb_pert_x_f = shape.find_limbs(scan_center_dir, scan_dirs, position_pert)

                position_pert = position + [0, delta, 0]

                limb_pert_y_f = shape.find_limbs(scan_center_dir, scan_dirs, position_pert)

                position_pert = position + [0, 0, delta]

                limb_pert_z_f = shape.find_limbs(scan_center_dir, scan_dirs, position_pert)

                position_pert = position - [delta, 0, 0]

                limb_pert_x_b = shape.find_limbs(scan_center_dir, scan_dirs, position_pert)

                position_pert = position - [0, delta, 0]

                limb_pert_y_b = shape.find_limbs(scan_center_dir, scan_dirs, position_pert)

                position_pert = position - [0, 0, delta]

                limb_pert_z_b = shape.find_limbs(scan_center_dir, scan_dirs, position_pert)

                return np.array([(limb_pert_x_f - limb_pert_x_b) / (2 * delta),
                                 (limb_pert_y_f - limb_pert_y_b) / (2 * delta),
                                 (limb_pert_z_f - limb_pert_z_b) / (2 * delta)]).swapaxes(0, -1)

        sphere = g_shapes.Ellipsoid([0, 0, 0], ellipsoid_matrix=self.ellipsoid_matrix_sphere,
                                    principal_axes=self.principal_axes_sphere, orientation=self.orientation.matrix)

        ellipse = g_shapes.Ellipsoid([0, 0, 0], ellipsoid_matrix=self.ellipsoid_matrix_ellipse,
                                     principal_axes=self.principal_axes_ellipse, orientation=self.orientation.matrix)

        position = np.array([.1, .2, 50.])

        angles = np.linspace(0, 2 * np.pi, 90)

        scan_center_dir = position / np.linalg.norm(position) + [0.01, -0.01, 0.02]

        scan_dirs = np.vstack([np.cos(angles), np.sin(angles), np.zeros(angles.size)])

        slimbs = sphere.find_limbs(scan_center_dir, scan_dirs, position)
        elimbs = ellipse.find_limbs(scan_center_dir, scan_dirs, position)

        with self.subTest(case='sphere'):
            state = 'relative position'

            jac_ana = sphere.compute_limb_jacobian(scan_center_dir, scan_dirs, slimbs, position)

            jac_num = num_deriv(sphere, scan_center_dir, scan_dirs, position, state=state, delta=1e-4)
            
            assert jac_ana is not None
            assert jac_num is not None

            # negative numeric since the pert is actually to camera position, no center like it should
            np.testing.assert_allclose(jac_ana, -jac_num, atol=1e-9, rtol=1e-5)

        with self.subTest(case='ellipse'):
            state = 'relative position'

            jac_ana = ellipse.compute_limb_jacobian(scan_center_dir, scan_dirs, elimbs, position)

            jac_num = num_deriv(ellipse, scan_center_dir, scan_dirs, position, state=state, delta=1e-4)
            
            assert jac_ana is not None
            assert jac_num is not None

            # negative numeric since the pert is actually to camera position, no center like it should
            np.testing.assert_allclose(jac_ana, -jac_num, atol=1e-8, rtol=1e-4)
