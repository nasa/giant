from unittest import TestCase, skip
import copy

import numpy as np

from giant import rotations as at
from giant.ray_tracer import kdtree, shapes, rays


class TestKDTree(TestCase):

    def setUp(self):

        self.max_depth = 4

        tri1 = np.array([[-5, -4, -4.5],
                         [0, 0, 1],
                         [0, 0, 0]])

        tri2 = tri1+np.array([[2.5, 0, 0]]).T

        tri3 = tri2+np.array([[2.5, 0, 0]]).T

        tri4 = tri3+np.array([[2.5, 0, 0]]).T

        self.triangles = shapes.Triangle64(np.hstack([tri1, tri2, tri3, tri4]).T, 1,
                                                          np.arange(12).reshape(-1, 3))

        self.shapes = self.triangles

        self.stacked_tries = shapes.Triangle64(np.hstack([tri1, tri2,
                                                        tri1+[[0], [0], [2.5]],
                                                        tri2 + [[0], [0], [2.5]]]).T, 1,
                                         np.arange(12).reshape(-1, 3))


    def test_creation(self):

        tree = kdtree.KDTree(self.shapes, max_depth=self.max_depth)

        self.assertEqual(tree.max_depth, self.max_depth)
        self.assertEqual(tree.surface, self.shapes)

    def test_build(self):

        tree = kdtree.KDTree(self.shapes, max_depth=self.max_depth)

        tree.build(force=True, print_progress=False)

        facets = np.arange(12).reshape(-1, 3)
        tris = [shapes.Triangle64(self.triangles.vertices, self.triangles.albedos, face)
                for face in facets]

        for tri in tris:
            tri.bounding_box = None

        node20 = kdtree.KDNode(tris[0])

        node21 = kdtree.KDNode(tris[1])

        node22 = kdtree.KDNode(tris[2])

        node23 = kdtree.KDNode(tris[3])

        node10 = kdtree.KDNode()
        node10.bounding_box = shapes.AxisAlignedBoundingBox([-5, 0, 0], [-1.5, 1, 0])
        node10.left = node20
        node10.right = node21

        node11 = kdtree.KDNode()
        node11.bounding_box = shapes.AxisAlignedBoundingBox([0., 0, 0], [3.5, 1, 0])
        node11.left = node22
        node11.right = node23

        node00 = kdtree.KDNode()
        node00.bounding_box = self.triangles.bounding_box
        node00.left = node10
        node00.right = node11
        node00.order = 2

        self.assertEqual(node00, tree.root)

    def test_trace(self):

        with self.subTest(stacked=False):
            tree = kdtree.KDTree(self.shapes, max_depth=self.max_depth)

            tree.build(force=True, print_progress=False)

            starts = np.array([[-4.5, -2, 0.5, 3],
                               [0.5, 0.5, 0.5, 0.5],
                               [1, 1, 1, 1]])
            directions = np.array([[0, 0, 0, 0],
                                   [0, 0, 0, 0],
                                   [-1, -1, -1, -1]], dtype=np.float64)

            rays_test = rays.Rays(starts, directions)

            ints = tree.trace(rays_test)

            nodes = [tree.root.left.left, tree.root.left.right, tree.root.right.left, tree.root.right.right]

            with self.subTest(rotation=None, translation=None):

                for ind, int_check in enumerate(ints):

                    with self.subTest(ignore=False, ind=ind):
                        self.assertTrue(int_check["check"])

                        np.testing.assert_array_equal(int_check["intersect"], starts[:, ind]-[0, 0, 1])

                        np.testing.assert_array_equal(int_check["normal"], self.triangles.normals[ind])

                        self.assertEqual(int_check["albedo"], 1.0)

                        self.assertEqual(int_check["facet"], 0+nodes[ind].id*(10**(tree.root.order+1)))

                ignore_ind = 2

                rays_test.ignore = [nodes[ignore_ind].id*(10**(tree.root.order+1))]*rays_test.num_rays

                ints = tree.trace(rays_test)

                for ind, int_check in enumerate(ints):

                    with self.subTest(ignore=True, ind=ind):

                        if ind != ignore_ind:
                            # int_check = int_check[0]

                            self.assertTrue(int_check["check"])

                            np.testing.assert_array_equal(int_check["intersect"], starts[:, ind]-[0, 0, 1])

                            np.testing.assert_array_equal(int_check["normal"], self.triangles.normals[ind])

                            self.assertEqual(int_check["albedo"], 1.0)

                            self.assertEqual(int_check["facet"], 0+nodes[ind].id*(10**(tree.root.order+1)))

                        else:

                            self.assertFalse(int_check["check"])

                            self.assertTrue(np.isnan(int_check["intersect"]).all())
                            self.assertTrue(np.isnan(int_check["normal"]).all())
                            self.assertTrue(np.isnan(int_check["albedo"]))
                            self.assertEqual(int_check["facet"], -1)

            rotation = at.Rotation([0, 0, -np.pi / 2])
            rays_test.ignore = None

            with self.subTest(rotation=rotation, translation=None):

                tc = copy.deepcopy(tree)

                tc.rotate(rotation)

                ints = tc.trace(rays_test)

                self.assertFalse(ints["check"].any())

                starts2 = np.array([[0.5, 0.5, 0.5, 0.5],
                                    [4.5, 2, -0.5, -3],
                                    [1, 1, 1, 1]])
                directions2 = np.array([[0, 0, 0, 0],
                                        [0, 0, 0, 0],
                                        [-1, -1, -1, -1]], dtype=np.float64)

                rays_test2 = rays.Rays(starts2, directions2)

                ints = tc.trace(rays_test2)

                for ind, int_check in enumerate(ints):
                    # int_check = int_check[0]
                    self.assertTrue(int_check["check"])

                    np.testing.assert_array_almost_equal(int_check["intersect"], starts2[:, ind]-[0, 0, 1])

                    np.testing.assert_array_equal(int_check["normal"], rotation.matrix@self.triangles.normals[ind])

                    self.assertEqual(int_check["albedo"], 1.0)

                    self.assertEqual(int_check["facet"], 0+nodes[ind].id*(10**(tc.root.order+1)))

            translation = [0, 0, -0.5]

            with self.subTest(rotation=None, translation=translation):

                tc = copy.deepcopy(tree)

                tc.translate(translation)

                ints = tc.trace(rays_test)

                for ind, int_check in enumerate(ints):
                    # int_check = int_check[0]
                    self.assertTrue(int_check["check"])

                    np.testing.assert_array_almost_equal(int_check["intersect"], starts[:, ind]-[0, 0, 1.5])

                    np.testing.assert_array_almost_equal(int_check["normal"], self.triangles.normals[ind])

                    self.assertEqual(int_check["albedo"], 1.0)

                    self.assertEqual(int_check["facet"], 0+nodes[ind].id*(10**(tc.root.order+1)))

            with self.subTest(rotation=rotation, translation=translation):

                tc = copy.deepcopy(tree)

                tc.rotate(rotation)
                tc.translate(translation)

                ints = tc.trace(rays_test)

                self.assertFalse(ints["check"].any())

                starts2 = np.array([[0.5, 0.5, 0.5, 0.5],
                                    [4.5, 2, -0.5, -3],
                                    [1, 1, 1, 1]])
                directions2 = np.array([[0, 0, 0, 0],
                                        [0, 0, 0, 0],
                                        [-1, -1, -1, -1]], dtype=np.float64)

                rays_test2 = rays.Rays(starts2, directions2)

                ints = tc.trace(rays_test2)

                for ind, int_check in enumerate(ints):
                    # int_check = int_check[0]
                    self.assertTrue(int_check["check"])

                    np.testing.assert_array_almost_equal(int_check["intersect"], starts2[:, ind]-[0, 0, 1.5])

                    np.testing.assert_array_equal(int_check["normal"], rotation.matrix@self.triangles.normals[ind])

                    self.assertEqual(int_check["albedo"], 1.0)

                    self.assertEqual(int_check["facet"], 0+nodes[ind].id*(10**(tc.root.order+1)))

            rotation = at.Rotation([np.pi / 2, 0, 0])

            with self.subTest(rotation=rotation, translation=None):

                tc = copy.deepcopy(tree)

                tc.rotate(rotation)

                ints = tc.trace(rays_test)

                self.assertFalse(ints["check"].any())

                starts2 = np.array([[-4.5, -2, 0.5, 3],
                                    [1, 1, 1, 1],
                                    [0.5, 0.5, 0.5, 0.5]])
                directions2 = np.array([[0, 0, 0, 0],
                                        [-1, -1, -1, -1],
                                        [0, 0, 0, 0]], dtype=np.float64)

                rays_test2 = rays.Rays(starts2, directions2)

                ints = tc.trace(rays_test2)

                for ind, int_check in enumerate(ints):
                    # int_check = int_check[0]
                    self.assertTrue(int_check["check"])

                    np.testing.assert_array_almost_equal(int_check["intersect"], starts2[:, ind]-[0, 1, 0])

                    np.testing.assert_array_equal(int_check["normal"], rotation.matrix@self.triangles.normals[ind])

                    self.assertEqual(int_check["albedo"], 1.0)

                    self.assertEqual(int_check["facet"], 0+nodes[ind].id*(10**(tc.root.order+1)))

            translation = [2.5, 0, 0]

            with self.subTest(rotation=None, translation=translation):

                tc = copy.deepcopy(tree)

                tc.translate(translation)

                ints = tc.trace(rays_test)

                self.assertFalse(ints["check"][0])

                for ind, int_check in enumerate(ints[1:]):
                    ind += 1
                    # int_check = int_check[0]
                    self.assertTrue(int_check["check"])

                    np.testing.assert_array_almost_equal(int_check["intersect"], starts[:, ind]-[0, 0, 1])

                    np.testing.assert_array_almost_equal(int_check["normal"], self.triangles.normals[ind-1])

                    self.assertEqual(int_check["albedo"], 1.0)

                    self.assertEqual(int_check["facet"], 0+nodes[ind-1].id*(10**(tc.root.order+1)))

            translation = [0, -0.5, 0]

            with self.subTest(rotation=rotation, translation=translation):

                with self.subTest(order='rt'):
                    tc = copy.deepcopy(tree)

                    tc.rotate(rotation)
                    tc.translate(translation)

                    ints = tc.trace(rays_test)

                    self.assertFalse(ints["check"].any())

                    starts2 = np.array([[-4.5, -2, 0.5, 3],
                                        [1, 1, 1, 1],
                                        [0.5, 0.5, 0.5, 0.5]])
                    directions2 = np.array([[0, 0, 0, 0],
                                            [-1, -1, -1, -1],
                                            [0, 0, 0, 0]], dtype=np.float64)

                    rays_test2 = rays.Rays(starts2, directions2)

                    ints = tc.trace(rays_test2)

                    for ind, int_check in enumerate(ints):
                        # int_check = int_check[0]
                        self.assertTrue(int_check["check"])

                        np.testing.assert_array_almost_equal(int_check["intersect"], starts2[:, ind]-[0, 1.5, 0])

                        np.testing.assert_array_equal(int_check["normal"], rotation.matrix@self.triangles.normals[ind])

                        self.assertEqual(int_check["albedo"], 1.0)

                        self.assertEqual(int_check["facet"], 0+nodes[ind].id*(10**(tc.root.order+1)))

                with self.subTest(order='tr'):
                    tc = copy.deepcopy(tree)

                    tc.translate(translation)
                    tc.rotate(rotation)

                    ints = tc.trace(rays_test)

                    self.assertFalse(ints["check"].any())

                    starts2 = np.array([[-4.5, -2, 0.5, 3],
                                        [1, 1, 1, 1],
                                        [0, 0, 0, 0]])
                    directions2 = np.array([[0, 0, 0, 0],
                                            [-1, -1, -1, -1],
                                            [0, 0, 0, 0]], dtype=np.float64)

                    rays_test2 = rays.Rays(starts2, directions2)

                    ints = tc.trace(rays_test2)

                    for ind, int_check in enumerate(ints):
                        # int_check = int_check[0]
                        self.assertTrue(int_check["check"])

                        np.testing.assert_array_almost_equal(int_check["intersect"], starts2[:, ind]-[0, 1, 0])

                        np.testing.assert_array_equal(int_check["normal"], rotation.matrix@self.triangles.normals[ind])

                        self.assertEqual(int_check["albedo"], 1.0)

                        self.assertEqual(int_check["facet"], 0+nodes[ind].id*(10**(tc.root.order+1)))

        with self.subTest(stacked=True):
            tree = kdtree.KDTree(self.stacked_tries, max_depth=self.max_depth)

            tree.build(force=True, print_progress=False)

            starts = np.array([[-4.5, -2, -4.5, -2],
                               [0.5, 0.5, 0.5, 0.5],
                               [1, 1, 5, 5]])
            directions = np.array([[0, 0, 0, 0],
                                   [0, 0, 0, 0],
                                   [-1, -1, -1, -1]], dtype=np.float64)

            rays_test = rays.Rays(starts, directions)

            ints = tree.trace(rays_test)

            nodes = [tree.root.left.left, tree.root.right.left, tree.root.left.right, tree.root.right.right]

            for ind, int_check in enumerate(ints):

                with self.subTest(ignore=False, ind=ind):
                    self.assertTrue(int_check["check"])

                    if ind < 2:
                        np.testing.assert_array_equal(int_check["intersect"], starts[:, ind]-[0, 0, 1])
                    else:
                        np.testing.assert_array_equal(int_check["intersect"], starts[:, ind]-[0, 0, 2.5])

                    np.testing.assert_array_equal(int_check["normal"], self.triangles.normals[ind])

                    self.assertEqual(int_check["albedo"], 1.0)

                    self.assertEqual(int_check["facet"], 0+nodes[ind].id*(10**(tree.root.order+1)))

            ignore_ind = 2

            rays_test.ignore = [nodes[ignore_ind].id*(10**(tree.root.order+1))]*rays_test.num_rays

            ints = tree.trace(rays_test)

            for ind, int_check in enumerate(ints):

                with self.subTest(ignore=True, ind=ind):

                    if ind != ignore_ind:
                        # int_check = int_check[0]

                        self.assertTrue(int_check["check"])

                        if ind < 2:
                            np.testing.assert_array_equal(int_check["intersect"], starts[:, ind]-[0, 0, 1])
                        else:
                            np.testing.assert_array_equal(int_check["intersect"], starts[:, ind]-[0, 0, 2.5])

                        np.testing.assert_array_equal(int_check["normal"], self.triangles.normals[ind])

                        self.assertEqual(int_check["albedo"], 1.0)

                        self.assertEqual(int_check["facet"], 0+nodes[ind].id*(10**(tree.root.order+1)))

                    else:

                        self.assertTrue(int_check["check"])

                        np.testing.assert_array_equal(int_check["intersect"], starts[:, ind]-[0, 0, 5])

                        np.testing.assert_array_equal(int_check["normal"], self.triangles.normals[ind])

                        self.assertEqual(int_check["albedo"], 1.0)

                        self.assertEqual(int_check["facet"], 0+nodes[0].id*(10**(tree.root.order+1)))


class TestKDNode(TestCase):

    def setUp(self):

        tri1 = np.array([[-5, -4, -4.5],
                         [0, 0, 1],
                         [0, 0, 0]])

        tri2 = tri1+np.array([[2.5, 0, 0]]).T

        tri3 = tri2+np.array([[2.5, 0, 0]]).T

        tri4 = tri3+np.array([[2.5, 0, 0]]).T

        self.triangles = shapes.Triangle64(np.hstack([tri1, tri2, tri3, tri4]).T, 1, np.arange(12).reshape(-1, 3))

    def test_creation(self):

        node = kdtree.KDNode(surface=self.triangles)

        self.assertEqual(node.surface, self.triangles)

        self.assertEqual(node.bounding_box, self.triangles.bounding_box)

        self.assertIsNone(node.left)
        self.assertIsNone(node.right)

    def test_compute_bounding_box(self):

        node = kdtree.KDNode()

        node.surface = self.triangles
        node.has_surface = True

        node.compute_bounding_box()

        self.assertEqual(node.bounding_box, self.triangles.bounding_box)

    def test_split(self):

        node = kdtree.KDNode(surface=self.triangles)

        node.split(force=True, print_progress=False)

        left_tris = kdtree.KDNode(shapes.Triangle64(self.triangles.vertices, 1,  np.arange(6).reshape(3, -1), compute_bounding_box=False))
        right_tris = kdtree.KDNode(shapes.Triangle64(self.triangles.vertices, 1, np.arange(6, 12).reshape(3, -1), compute_bounding_box=False))

        self.assertEqual(node.left, left_tris)
        self.assertEqual(node.right, right_tris)

    def test_trace(self):

        # TODO: figure out how to implement this
        pass

