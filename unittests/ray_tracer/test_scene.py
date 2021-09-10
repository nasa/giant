import copy
import datetime
from unittest import TestCase

import numpy as np

from giant import rotations as at
from giant.ray_tracer import scene, shapes, rays, INTERSECT_DTYPE

import os


LOCALDIR = os.path.dirname(os.path.realpath(__file__))


class TestSceneObj(TestCase):

    def setUp(self):

        tri1 = np.array([[-5, -4, -4.5],
                         [0, 0, 1],
                         [0, 0, 0]])

        tri2 = tri1+np.array([[2.5, 0, 0]]).T

        tri3 = tri2+np.array([[2.5, 0, 0]]).T

        tri4 = tri3+np.array([[2.5, 0, 0]]).T

        self.triangles = shapes.Triangle64(np.hstack([tri1, tri2, tri3, tri4]).T, 1, np.arange(12).reshape(-1, 3))

    def check_parameters(self, obj, model, position, orientation):

        self.assertTrue(obj.shape == model)

        np.testing.assert_array_equal(obj.position, position.ravel())

        self.assertEqual(obj.orientation, orientation)

    def test_creation(self):

        sobj = scene.SceneObject(self.triangles)

        self.check_parameters(sobj, self.triangles, np.zeros(3, dtype=np.float64),
                              at.Rotation(np.eye(3)))

        sobj = scene.SceneObject(self.triangles, current_position=np.ones(3),
                                 current_orientation=at.Rotation([1, 2, 3]))

        self.check_parameters(sobj, self.triangles, np.ones(3), at.Rotation([1, 2, 3]))

    def test_change_position(self):

        translation = [1, 2, 3]

        tri = copy.deepcopy(self.triangles)

        tri.translate([4, 5, 5])

        sobj = scene.SceneObject(tri, current_position=[4, 5, 5])

        tri2 = copy.deepcopy(self.triangles)

        tri2.translate(translation)

        for intype in (list, tuple, np.array):

            test_obj = copy.deepcopy(sobj)

            with self.subTest(intype=intype):

                test_obj.change_position(intype(translation))

                self.check_parameters(test_obj, tri2, np.array(translation), at.Rotation(np.eye(3)))

    def test_change_orientation(self):

        rotation = [1, 2, 3]

        tri = copy.deepcopy(self.triangles)

        tri.rotate([4, 5, 5])

        sobj = scene.SceneObject(tri, current_orientation=[4, 5, 5])

        tri2 = copy.deepcopy(self.triangles)

        tri2.rotate(rotation)

        for intype in (list, tuple, np.array):

            test_obj = copy.deepcopy(sobj)

            with self.subTest(intype=intype):

                test_obj.change_orientation(intype(rotation))

                self.check_parameters(test_obj, tri2, np.zeros(3), at.Rotation(rotation))

    def test_translate(self):

        translation = [1, 2, 3]

        sobj = scene.SceneObject(copy.deepcopy(self.triangles), current_position=[2, 3, 4])

        tri2 = copy.deepcopy(self.triangles)

        tri2.translate(translation)

        for intype in (list, tuple, np.array):

            test_obj = copy.deepcopy(sobj)

            with self.subTest(intype=intype):

                test_obj.translate(intype(translation))

                self.check_parameters(test_obj, tri2, np.array(translation)+np.array([2, 3, 4]), np.eye(3))

    def test_rotate(self):

        rotation = [1, 2, 3]

        sobj = scene.SceneObject(copy.deepcopy(self.triangles), current_orientation=[3, 2, 1])

        tri2 = copy.deepcopy(self.triangles)

        tri2.rotate(rotation)

        for intype in (list, tuple, np.array):

            test_obj = copy.deepcopy(sobj)

            with self.subTest(intype=intype):

                test_obj.rotate(intype(rotation))

                self.check_parameters(test_obj, tri2, np.zeros(3), at.Rotation(rotation) * at.Rotation([3, 2, 1]))


# noinspection PyArgumentList
class TestAutoSceneObj(TestCase):

    def setUp(self):

        tri1 = np.array([[-5, -4, -4.5],
                         [0, 0, 1],
                         [0, 0, 0]])

        tri2 = tri1+np.array([[2.5, 0, 0]]).T

        tri3 = tri2+np.array([[2.5, 0, 0]]).T

        tri4 = tri3+np.array([[2.5, 0, 0]]).T

        self.triangles = shapes.Triangle64(np.hstack([tri1, tri2, tri3, tri4]).T, 1, np.arange(12).reshape(-1, 3))

        def pos_fun(time):

            rng = np.random.RandomState(int(time.toordinal()))

            return rng.randn(3)

        self.pos_fun = pos_fun

        def frame_fun(time):

            rng = np.random.RandomState(int(time.toordinal()))

            return at.Rotation(rng.randn(3))

        self.frame_fun = frame_fun

    def check_parameters(self, obj, model, position, orientation):

        self.assertTrue(obj.shape == model)

        np.testing.assert_array_equal(obj.position, position.ravel())

        self.assertEqual(obj.orientation, orientation)

        self.assertEqual(obj.orientation_function, self.frame_fun)
        self.assertEqual(obj.position_function, self.pos_fun)

    def test_creation(self):

        asobj = scene.SceneObject(self.triangles, position_function=self.pos_fun,
                                  orientation_function=self.frame_fun)

        self.check_parameters(asobj, self.triangles, np.zeros(3, dtype=np.float64),
                              at.Rotation(np.eye(3)))

        asobj = scene.SceneObject(self.triangles, position_function=self.pos_fun,
                                  orientation_function=self.frame_fun, current_position=np.ones(3),
                                  current_orientation=at.Rotation([1, 2, 3]))

        self.check_parameters(asobj, self.triangles, np.ones(3), at.Rotation([1, 2, 3]))

    def test_change_position(self):

        translation = [1, 2, 3]

        tri = copy.deepcopy(self.triangles)

        tri.translate([4, 5, 5])

        asobj = scene.SceneObject(tri, position_function=self.pos_fun,
                                  orientation_function=self.frame_fun, current_position=[4, 5, 5])

        tri2 = copy.deepcopy(self.triangles)

        tri2.translate(translation)

        for intype in (list, tuple, np.array):

            test_obj = copy.deepcopy(asobj)

            with self.subTest(intype=intype):

                test_obj.change_position(intype(translation))

                self.check_parameters(test_obj, tri2, np.array(translation), at.Rotation(np.eye(3)))

    def test_change_orientation(self):

        rotation = [1, 2, 3]

        tri = copy.deepcopy(self.triangles)

        tri.rotate([4, 5, 5])

        asobj = scene.SceneObject(tri, position_function=self.pos_fun,
                                  orientation_function=self.frame_fun, current_orientation=[4, 5, 5])

        tri2 = copy.deepcopy(self.triangles)

        tri2.rotate(rotation)

        for intype in (list, tuple, np.array):

            test_obj = copy.deepcopy(asobj)

            with self.subTest(intype=intype):

                test_obj.change_orientation(intype(rotation))

                self.check_parameters(test_obj, tri2, np.zeros(3), at.Rotation(rotation))

    def test_translate(self):

        translation = [1, 2, 3]

        asobj = scene.SceneObject(copy.deepcopy(self.triangles), position_function=self.pos_fun,
                                      orientation_function=self.frame_fun,
                                      current_position=[2, 3, 4])

        tri2 = copy.deepcopy(self.triangles)

        tri2.translate(translation)

        for intype in (list, tuple, np.array):

            test_obj = copy.deepcopy(asobj)

            with self.subTest(intype=intype):

                test_obj.translate(intype(translation))

                self.check_parameters(test_obj, tri2, np.array(translation)+np.array([2, 3, 4]), np.eye(3))

    def test_rotate(self):

        rotation = [1, 2, 3]

        asobj = scene.SceneObject(copy.deepcopy(self.triangles), position_function=self.pos_fun,
                                  orientation_function=self.frame_fun,
                                  current_orientation=[3, 2, 1])

        tri2 = copy.deepcopy(self.triangles)

        tri2.rotate(rotation)

        for intype in (list, tuple, np.array):

            test_obj = copy.deepcopy(asobj)

            with self.subTest(intype=intype):

                test_obj.rotate(intype(rotation))

                self.check_parameters(test_obj, tri2, np.zeros(3), at.Rotation(rotation) * at.Rotation([3, 2, 1]))

    def test_place(self):

        asobj = scene.SceneObject(copy.deepcopy(self.triangles), position_function=self.pos_fun,
                                  orientation_function=self.frame_fun)

        date = datetime.datetime(2018, 3, 4, 0, 0, 0)

        asobj.place(date)

        tri2 = copy.deepcopy(self.triangles)

        tri2.rotate(self.frame_fun(date))

        tri2.translate(self.pos_fun(date))

        self.check_parameters(asobj, tri2, self.pos_fun(date), self.frame_fun(date))

        asobj.translate([1, 2, 3])
        asobj.rotate([4, 5, 6])

        date2 = datetime.datetime(2019, 3, 4, 5, 6, 3)

        asobj.place(date2)

        tri2 = copy.deepcopy(self.triangles)

        tri2.rotate(self.frame_fun(date2))

        tri2.translate(self.pos_fun(date2))

        self.check_parameters(asobj, tri2, self.pos_fun(date2), self.frame_fun(date2))


class TestScene(TestCase):

    def test_creation(self):

        tri1 = np.array([[-5, -4, -4.5],
                         [0, 0, 1],
                         [0, 0, 0]])

        tri2 = tri1+np.array([[2.5, 0, 0]]).T

        tri3 = tri2+np.array([[2.5, 0, 0]]).T

        tri4 = tri3+np.array([[2.5, 0, 0]]).T

        triangles = shapes.Triangle64(np.hstack([tri1, tri2, tri3, tri4]).T, 1, np.arange(12).reshape(-1, 3))

        triangles2 = copy.deepcopy(triangles)

        triangles2.translate([0, 0, 3.5])

        sc = scene.Scene([scene.SceneObject(triangles), scene.SceneObject(triangles2)])

        self.assertEqual(sc.target_objs[0].shape, triangles)
        self.assertEqual(sc.target_objs[1].shape, triangles2)

    def test_trace(self):
        tri1 = np.array([[-5, -4, -4.5],
                         [0, 0, 1],
                         [0, 0, 0]])

        tri2 = tri1+np.array([[2.5, 0, 0]]).T

        tri3 = tri2+np.array([[2.5, 0, 0]]).T

        tri4 = tri3+np.array([[2.5, 0, 0]]).T

        triangles = shapes.Triangle64(np.hstack([tri1, tri2, tri3, tri4]).T,1, np.arange(12).reshape(-1, 3))

        triangles2 = copy.deepcopy(triangles)

        triangles2.translate([0, 0, 3.5])

        sc = scene.Scene([scene.SceneObject(triangles), scene.SceneObject(triangles2)])

        ray = rays.Rays([[0, 0, 0],
                         [0, 0, 0],
                         [4, -4, 2]],
                        [[0, 0, 0],
                        [0, 0, 0],
                        [-1, 1, -1]])

        res = sc.trace(ray)

        res_true = np.array([(True, 0.5, np.array([0., 0., 3.5]), np.array([0., 0., 1.]), 1., 12)],
                            dtype=INTERSECT_DTYPE)
        np.testing.assert_array_equal(res[0], res_true)

        res_true = np.array([(True, 4.0, np.array([0., 0., 0]), np.array([0., 0., 1.]), 1., 2)], dtype=INTERSECT_DTYPE)
        np.testing.assert_array_equal(res[1], res_true)

        res_true = np.array([(True, 2.0, np.array([0., 0., 0]), np.array([0., 0., 1.]), 1., 2)], dtype=INTERSECT_DTYPE)
        np.testing.assert_array_equal(res[2], res_true)

    def test_get_illumination_inputs(self):

        tri1 = np.array([[-5, -4, -4.5],
                         [0, 0, 1],
                         [0, 0, 0]])

        tri2 = tri1+np.array([[2.5, 0, 0]]).T

        tri3 = tri2+np.array([[2.5, 0, 0]]).T

        tri4 = tri3+np.array([[2.5, 0, 0]]).T

        triangles = shapes.Triangle64(np.hstack([tri1, tri2, tri3, tri4]).T,1, np.arange(12).reshape(-1, 3))

        triangles2 = copy.deepcopy(triangles)

        triangles2.translate([0, 0, 3.5])

        light_obj = scene.SceneObject(shapes.Point(np.array([0, 0, 10])))

        sc = scene.Scene(target_objs=[scene.SceneObject(triangles), scene.SceneObject(triangles2)], light_obj=light_obj)

        ray = rays.Rays([[0, 0, 0],
                         [0, 0, 0],
                         [4, -4, 2]],
                        [[0, 0, 0],
                        [0, 0, 0],
                        [-1, 1, 1]])

        results = sc.get_illumination_inputs(ray)

        np.testing.assert_array_equal(results["incidence"], [[0, 0, -1], [np.nan]*3, [0, 0, -1]])
        np.testing.assert_array_equal(results["exidence"], [[0, 0, 1], [np.nan]*3, [0, 0, -1]])
        np.testing.assert_array_equal(results["normal"], [[0, 0, 1], [np.nan]*3, [0, 0, 1]])
        np.testing.assert_array_equal(results["visible"], [True, False, True])

    def test_get_first(self):

        trace_rays = rays.Rays([[0] * 50, [0] * 50, [0] * 50], [[-1] * 50, [0] * 50, [0] * 50])

        dtype = np.dtype([('check', bool), ('intersect', np.float64, (3,)), ('normal', np.float64, (3,)),
                          ('albedo', np.float64), ('facet', int)])

        res_temp = [np.array([(True, np.array([2., 3., 4.]), np.array([4., 5., 6.]), 8., 1)]*50, dtype=dtype),
                    np.array([(True, np.array([1., 2., 3.]), np.array([4., 5., 6.]), 8., 1)]*50, dtype=dtype)]

        res_temp = np.vstack(res_temp)

        res2 = scene.Scene.get_first(res_temp, trace_rays)

        np.testing.assert_array_equal(res_temp[1], res2)

        res_temp = [np.array([(True, np.array([2., 3., 4.]), np.array([4., 5., 6.]), 8., 1)]*50, dtype=dtype),
                    np.array([(True, np.array([10., 20., 30.]), np.array([4., 5., 6.]), 8, 1)]*50, dtype=dtype)]

        res_temp = np.vstack(res_temp)

        res2 = scene.Scene.get_first(res_temp, trace_rays)

        np.testing.assert_array_equal(res_temp[0], res2)

        res_temp = [np.array([(True, np.array([2., 3., 4.]), np.array([4., 5., 6.]), 8., 1)]*50, dtype=dtype),
                    np.array([(True, np.array([1., 2., 3.]), np.array([4., 5., 6.]), 8., 1)]*50, dtype=dtype),
                    np.array([(False, None, None, None, -1)]*50, dtype=dtype)]

        res_temp = np.vstack(res_temp)

        res2 = scene.Scene.get_first(res_temp, trace_rays)

        np.testing.assert_array_equal(np.array(list(res_temp[1]), dtype=dtype), np.array(list(res2), dtype=dtype))

        res_temp = [np.array([(False, None, None, None, -1)]*50, dtype=dtype),
                    np.array([(True, np.array([2., 3., 4.]), np.array([4., 5., 6.]), 8., 1)]*50, dtype=dtype),
                    np.array([(True, np.array([10., 20., 30.]), np.array([4., 5., 6.]), 8., 1)]*50, dtype=dtype)]

        res_temp = np.vstack(res_temp)

        res2 = scene.Scene.get_first(res_temp, trace_rays)

        np.testing.assert_array_equal(res_temp[1], res2)

        trace_rays = rays.Rays([[0] * 3, [0] * 3, [0] * 3], [[-1] * 3, [0] * 3, [0] * 3])

        res_temp = [np.array([(False, None, None, None, -1),
                              (True, np.array([2., 3., 4.]), np.array([4., 5, 6.]), 4., 1),
                              (True, np.array([7., 8., 9.]), np.array([4., 5, 6.]), 4., 100)], dtype=dtype),
                    np.array([(True, np.array([1., 2., 3.]), np.array([4., 5, 6.]), 4., 100),
                              (True, np.array([2., 3., 4.]), np.array([4., 5, 6.]), 4., 1),
                              (False, None, None, None, -1)], dtype=dtype),
                    np.array([(True, np.array([2., 3., 4.]), np.array([4., 5, 6.]), 4., 1),
                              (False, None, None, None, -1),
                              (True, np.array([7., 8., 9.]), np.array([4., 5, 6.]), 4., 100)], dtype=dtype)]

        res_temp = np.vstack(res_temp)

        res2 = scene.Scene.get_first(res_temp, trace_rays)

        result = np.array([(True, np.array([1., 2., 3.]), np.array([4., 5, 6.]), 4., 100),
                           (True, np.array([2., 3., 4.]), np.array([4., 5, 6.]), 4., 1),
                           (True, np.array([7., 8., 9.]), np.array([4., 5, 6.]), 4., 100)], dtype=dtype)

        np.testing.assert_array_equal(res2, result)


class TestCorrectLightTime(TestCase):
    
    def setUp(self):
        
        try:
            import spiceypy as spice
            self.spice = spice
            self.hasspice = True
            spice.furnsh(os.path.join(LOCALDIR, '..', 'test_data', 'naif0012.tls'))
            spice.furnsh(os.path.join(LOCALDIR, '..', 'test_data', 'de424.bsp'))
            
            import giant.utilities.spice_interface as sputil
            
            self.sputil = sputil
            
        except ImportError:
            self.spice = None
            self.hasspice = False
            
    def tearDown(self):
        
        if self.hasspice:
            self.spice.kclear()
            
    def test_correct_lighttime(self):
        
        if self.hasspice:
            
            date = datetime.datetime(year=2018, month=10, day=12, hour=10)
            
            pos_fun = self.sputil.et_callable_to_datetime_callable(
                self.sputil.create_callable_position('Moon', 'J2000', 'None', 'SSB')
            ) 
            
            camera_location, _ = self.spice.spkpos('Earth', self.sputil.datetime_to_et(date), 'J2000', 'None', 'SSB')
            
            apparent_location = scene.correct_light_time(pos_fun, camera_location, date)
            
            true_apparent_location, _ = self.spice.spkpos('Moon', self.sputil.datetime_to_et(date),
                                                          'J2000', 'CN', 'Earth') 
            
            np.testing.assert_allclose(apparent_location.flatten(), true_apparent_location.flatten())


class TestCorrectStellarAberration(TestCase):

    def setUp(self):

        try:
            import spiceypy as spice
            self.spice = spice
            self.hasspice = True
            spice.furnsh(os.path.join(LOCALDIR, '..', 'test_data', 'naif0012.tls'))
            spice.furnsh(os.path.join(LOCALDIR, '..', 'test_data', 'de424.bsp'))

            import giant.utilities.spice_interface as sputil

            self.sputil = sputil

        except ImportError:
            self.spice = None
            self.hasspice = False

    def tearDown(self):

        if self.hasspice:
            self.spice.kclear()

    def test_correct_lighttime(self):

        if self.hasspice:

            date = datetime.datetime(year=2018, month=10, day=12, hour=10)
            
            lt_corrected_apparent_pos, _ = self.spice.spkpos('Moon', self.sputil.datetime_to_et(date),
                                                             'J2000', 'CN', 'Earth')

            camera_state, _ = self.spice.spkezr('Earth', self.sputil.datetime_to_et(date), 'J2000', 'None', 'SSB')

            apparent_location = scene.correct_stellar_aberration(lt_corrected_apparent_pos, camera_state[3:])

            true_apparent_location, _ = self.spice.spkpos('Moon', self.sputil.datetime_to_et(date),
                                                          'J2000', 'CN+S', 'Earth')

            np.testing.assert_allclose(apparent_location.flatten(), true_apparent_location.flatten())

            lt_corrected_apparent_pos2, _ = self.spice.spkpos('Sun', self.sputil.datetime_to_et(date),
                                                              'J2000', 'CN', 'Earth')
            
            double_array = np.array([lt_corrected_apparent_pos, lt_corrected_apparent_pos2]).T

            camera_state, _ = self.spice.spkezr('Earth', self.sputil.datetime_to_et(date), 'J2000', 'None', 'SSB')

            apparent_location = scene.correct_stellar_aberration(double_array, camera_state[3:])

            true_apparent_location2, _ = self.spice.spkpos('Sun', self.sputil.datetime_to_et(date),
                                                           'J2000', 'CN+S', 'Earth')

            np.testing.assert_allclose(apparent_location.flatten(), 
                                       np.array([true_apparent_location, true_apparent_location2]).T.flatten())  
