from unittest import TestCase

from giant.camera import Camera
from giant.camera_models import PinholeModel
from giant.image import OpNavImage

from giant.relative_opnav.relnav_class import RelativeOpNav

from giant.ray_tracer.illumination import McEwenIllumination
from giant.ray_tracer.scene import Scene, SceneObject

from giant.ray_tracer.rays import Rays

import giant.ray_tracer.shapes as g_shapes

from giant.point_spread_functions.gaussians import Gaussian

import numpy as np

from datetime import datetime

try:
    import spiceypy as spice

    HASSPICE = True

except ImportError:
    spice = None
    HASSPICE = False

from giant import rotations as at

import os

LOCALDIR = os.path.dirname(os.path.realpath(__file__))


class MyTestCamera(Camera):
    def preprocessor(self, image):
        return image


class TestCallable:
    def __call__(self, image):
        return image


class TestAttitudeFunction:
    def __call__(self):
        return at.Rotation([0, 0, 0, 1])


class TestRelNav(TestCase):

    # set up camera, scene, etc.
    def setUp(self):

        def camera_frame_fun(time):
            return at.Rotation(np.array([0, 0, 0]))

        self.camera_frame_fun = camera_frame_fun

        def target_pos_fun(time):

            if time == datetime(2017, 2, 1, 0, 0, 0):
                return np.array([[0], [0], [5000]]).ravel()
            elif time == datetime(2017, 2, 2, 0, 0, 0):
                return np.array([[0], [0], [25000 / 7]]).ravel()
            else:
                return np.array([0, 0, 0])

        self.target_pos_fun = target_pos_fun

        def target_frame_fun(time):
            if time == datetime(2017, 2, 1, 0, 0, 0):
                return at.Rotation(np.array([0, 0, 0]))
            elif time == datetime(2017, 2, 2, 0, 0, 0):
                return at.Rotation(np.array([0, 0, 0]))
            else:
                return np.array([0, 0, 0])

        self.target_frame_fun = target_frame_fun

        def light_pos_fun(time):

            if time == datetime(2017, 2, 1, 0, 0, 0):
                return np.array([[0], [0], [-100]]).ravel()
            elif time == datetime(2017, 2, 2, 0, 0, 0):
                return np.array([[0], [0], [-101]]).ravel()
            else:
                return np.array([[0], [0], [0]]).ravel()

        self.light_pos_fun = light_pos_fun

        def light_frame_fun(time):
            if time == datetime(2017, 2, 1, 0, 0, 0):
                return at.Rotation(np.array([0, 0, -100]))
            elif time == datetime(2017, 2, 2, 0, 0, 0):
                return at.Rotation(np.array([0, 0, -100.00001]))
            else:
                return np.array([0, 0, 0])

        self.light_frame_fun = light_frame_fun

        def load_image(a, b, n, r):
            y, x = np.ogrid[-a:n - a, -b:n - b]
            mask = x * x + y * y <= r * r
            image = np.ones((n, n))
            image[mask] = 255
            return image

        self.load_image = load_image

        self.images = [OpNavImage(self.load_image(49, 49, 100, 1), observation_date=datetime(2017, 2, 1, 0, 0, 0),
                                  temperature=20, exposure_type='long'),
                       OpNavImage(self.load_image(49, 49, 100, 3), observation_date=datetime(2017, 2, 2, 0, 0, 0),
                                  temperature=20, exposure_type='long')]

        self.cmodel = PinholeModel(kx=500, ky=500, px=49, py=49, focal_length=10, n_rows=5000, n_cols=5000)

        self.camera = MyTestCamera(images=self.images, model=self.cmodel, name="AwesomeCam",
                                   spacecraft_name="AwesomeSpacecraft", frame="AwesomeFrame",
                                   parse_data=False, psf=Gaussian(), attitude_function=self.camera_frame_fun,
                                   start_date=datetime(2017, 2, 1, 0, 0, 0, 0),
                                   end_date=datetime(2017, 2, 2, 0, 0, 0, 0),
                                   default_image_class=OpNavImage, metadata_only=False)

        # Define shapes
        self.point = g_shapes.Point([0, 0, -100])
        self.ellipse = g_shapes.Ellipsoid(np.array([0, 0, 1000]),
                                          principal_axes=np.array([50, 50, 50]).astype(np.float64),
                                          orientation=at.Rotation([0, 0, 0]).matrix)

        # Define target
        self.target_obj = SceneObject(self.ellipse, position_function=self.target_pos_fun,
                                      orientation_function=self.target_frame_fun)

        # Define Sun
        self.sun_obj = SceneObject(self.point, position_function=self.light_pos_fun,
                                   orientation_function=self.light_frame_fun)

        # Define scene
        self.opnav_scene = Scene(target_objs=[self.target_obj], light_obj=self.sun_obj)

        # Define Rays object
        start = np.hstack(4 * [np.array([[0], [0], [-100]])])
        direction = np.hstack(4 * [np.array([[0], [0], [10000]])])
        self.opnav_scene.rays = Rays(start, direction)

        # Define Relnav object
        self.relnav = RelativeOpNav(self.camera, self.opnav_scene, xcorr_kwargs={"grid_size": 3, "denoise_image": True},
                                    brdf=McEwenIllumination(), auto_corrections=None)

        # # Define ImageProcessing object
        # self.image_processing = gimp.ImageProcessing()

        # # Definite XCorrCenterFinding object
        # self.xcorr = XCorrCenterFinding(scene=self.opnav_scene, camera=self.camera,
        #                                 image_processing=self.image_processing,
        #                                 brdf=McEwenIllumination, rays=self.rays)

    def test_scene_property(self):

        relnav = self.relnav

        self.assertIsInstance(relnav.scene, Scene)

        self.assertIsInstance(relnav.scene.light_obj, SceneObject)

        self.assertIsInstance(relnav.scene.target_objs, list)
        self.assertIsInstance(relnav.scene.target_objs[0], SceneObject)

        self.assertIsInstance(relnav.scene.obscuring_objs, list)
        self.assertEqual(len(relnav.scene.obscuring_objs), 0)

    def test_scene_setter(self):

        # Define relnav
        relnav = self.relnav

        # Check current scene
        self.assertEqual(len(relnav.scene.target_objs), 1)
        self.assertIsInstance(relnav.scene.target_objs[0], SceneObject)
        self.assertIsInstance(relnav.scene.light_obj, SceneObject)
        self.assertEqual(len(relnav.scene.obscuring_objs), 0)
        self.assertIsInstance(self.relnav.scene, Scene)

        # Set scene to have multiple target objects in Scene object
        new_opnav_scene = Scene(target_objs=[self.target_obj, self.target_obj, self.target_obj],
                                    light_obj=self.sun_obj)
        self.relnav.scene = new_opnav_scene
        self.assertEqual(len(relnav.scene.target_objs), 3)
        self.assertIsInstance(self.relnav.scene, Scene)

        # Set scene to Scene object
        new_target = SceneObject(g_shapes.Point([2, 2, 2]), current_position=np.array([0, 0, 0]),
                                 current_orientation=np.eye(3))
        new_sun = SceneObject(self.ellipse, current_position=np.array([0, 0, -1]), current_orientation=np.eye(3))
        another_opnav_scene = Scene(target_objs=[new_target, new_target], light_obj=new_sun)
        self.relnav.scene = another_opnav_scene
        self.assertIsInstance(self.relnav.scene, Scene)
        self.assertIsInstance(self.relnav.scene.target_objs[0], SceneObject)
        self.assertIsInstance(self.relnav.scene.light_obj, SceneObject)

    def test_brdf_property(self):
        pass

    def test_brdf_setter(self):
        pass

    def test_auto_estimate(self):
        pass

    def test_unresolved_estimate(self):
        pass

    def test_xcorr_estimate(self):

        self.images = [OpNavImage(self.load_image(49, 49, 100, 5), observation_date=datetime(2017, 2, 1, 0, 0, 0),
                                  temperature=20, exposure_type='long'),
                       OpNavImage(self.load_image(49, 49, 100, 7), observation_date=datetime(2017, 2, 2, 0, 0, 0),
                                  temperature=20, exposure_type='long')]

        self.cmodel = PinholeModel(kx=500, ky=500, px=49, py=49, focal_length=10, n_rows=5000, n_cols=5000)

        self.camera = MyTestCamera(images=self.images, model=self.cmodel, name="AwesomeCam",
                                   spacecraft_name="AwesomeSpacecraft", frame="AwesomeFrame",
                                   parse_data=False, psf=Gaussian(), attitude_function=self.camera_frame_fun,
                                   start_date=datetime(2017, 2, 1, 0, 0, 0, 0),
                                   end_date=datetime(2017, 2, 2, 0, 0, 0, 0),
                                   default_image_class=OpNavImage, metadata_only=False)

        # Define shapes
        self.point = g_shapes.Point([-100, -100, -100])
        self.ellipse = g_shapes.Ellipsoid(np.array([0, 0, 0]), principal_axes=np.array([5, 5, 5]).astype(np.float64),
                                          orientation=at.Rotation([0, 0, 0]).matrix)

        # Define target
        target_obj = SceneObject(self.ellipse, position_function=self.target_pos_fun,
                                 orientation_function=self.target_frame_fun)

        # Define Sun
        sun_obj = SceneObject(self.point, position_function=self.light_pos_fun,
                              orientation_function=self.light_frame_fun)

        # Define scene
        opnav_scene = Scene(target_objs=[target_obj], light_obj=sun_obj)

        # Run xcorr
        relnav = RelativeOpNav(self.camera, opnav_scene, xcorr_kwargs={"grid_size": 3, "denoise_image": True},
                               brdf=McEwenIllumination(), auto_corrections=None)

        relnav.save_templates = True
        relnav.cross_correlation.search_region = 100
        relnav.cross_correlation_estimate()

        # Compare residuals
        residuals = relnav.center_finding_results['measured'] - relnav.center_finding_results['predicted']

        np.testing.assert_array_less(residuals, 1)
