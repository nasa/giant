from typing import cast, Any
from unittest import TestCase

import giant.rotations as at

from giant.camera import Camera
from giant.camera_models import PinholeModel

from giant.image import OpNavImage

from giant.image_processing import pixel_level_peak_finder_2d

import giant.image_processing as gimp

import giant.ray_tracer.shapes as g_shapes
from giant.ray_tracer.scene import Scene, SceneObject
from giant.ray_tracer.illumination import McEwenIllumination
from giant.ray_tracer.rays import Rays

from giant.point_spread_functions.gaussians import Gaussian

from giant.relative_opnav.relnav_class import RelativeOpNav
from giant.relative_opnav import XCorrCenterFinding, XCorrCenterFindingOptions
from giant.image_processing import quadric_peak_finder_2d

import numpy as np
from datetime import datetime


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
                return np.array([[0], [0], [5000]])
            elif time == datetime(2017, 2, 2, 0, 0, 0):
                return np.array([[0], [0], [25000 / 7]]).reshape((3, 1))
            else:
                return np.array([0, 0, 0]).reshape((3, 1))

        self.target_pos_fun = target_pos_fun

        def target_frame_fun(time):
            if time == datetime(2017, 2, 1, 0, 0, 0):
                return at.Rotation(np.array([0, 0, 0]))
            elif time == datetime(2017, 2, 2, 0, 0, 0):
                return at.Rotation(np.array([0, 0, 0]))
            else:
                return at.Rotation(np.array([0, 0, 0]))

        self.target_frame_fun = target_frame_fun

        def light_pos_fun(time):

            if time == datetime(2017, 2, 1, 0, 0, 0):
                return np.array([[0], [0], [-100]])
            elif time == datetime(2017, 2, 2, 0, 0, 0):
                return np.array([[0], [0], [-101]])
            else:
                return np.array([[0], [0], [0]])

        self.light_pos_fun = light_pos_fun

        def light_frame_fun(time):
            if time == datetime(2017, 2, 1, 0, 0, 0):
                return at.Rotation(np.array([0, 0, -100]))
            elif time == datetime(2017, 2, 2, 0, 0, 0):
                return at.Rotation(np.array([0, 0, -100.00001]))
            else:
                return at.Rotation()

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
        self.ellipse = g_shapes.Ellipsoid(np.array([0, 0, 5000]),
                                          principal_axes=np.array([50, 50, 50]).astype(np.float64),
                                          orientation=at.Rotation([0, 0, 0]).matrix)

        # Define target
        self.target_obj = SceneObject(self.ellipse, position_function=self.target_pos_fun,
                                      orientation_function=self.target_frame_fun,
                                      current_position=[0, 0, 5000])

        # Define Sun
        self.sun_obj = SceneObject(self.point, position_function=self.light_pos_fun,
                                  orientation_function=self.light_frame_fun,
                                  current_position=[0, 0, -100])

        # Define scene
        self.opnav_scene = Scene(target_objs=[self.target_obj], light_obj=self.sun_obj)

        # Define Relnav object
        self.relnav = RelativeOpNav(self.camera, self.opnav_scene, xcorr_kwargs={"grid_size": 3, "denoise_image": True},
                                    brdf=McEwenIllumination(), auto_corrections=None)

        # Define Rays object
        start = np.zeros(3)
        direction = np.array(1000 * [[0, 0, 1]]).T
        self.rays = Rays(start, direction)

        # Definite XCorrCenterFinding object
        self.xcorr = XCorrCenterFinding(scene=self.opnav_scene, camera=self.camera,
                                        options=XCorrCenterFindingOptions(rays=self.rays))

    def test_init(self):

        # Test xcorr
        self.assertIsInstance(self.xcorr, XCorrCenterFinding)

        # Test Scene
        self.assertIsInstance(self.xcorr.scene, Scene)

        # Test Camera
        self.assertIsInstance(self.xcorr.camera, Camera)


        # Test Rays.start
        self.assertIsNotNone(self.xcorr.rays)
        rays = cast(Rays, self.xcorr.rays)
        np.testing.assert_equal(rays.start, np.zeros((3, self.rays.num_rays)))

        # Test Rays.direction
        self.assertTrue((rays.direction == [[0], [0], [1]]).all())

    def test_pixel_level_peak_finder(self):

        # load an image
        image = OpNavImage(self.load_image(49, 49, 100, 1), observation_date=datetime(2017, 2, 1, 0, 0, 0),
                           temperature=20, exposure_type='long')

        # uncomment to show image
        # plt.imshow(image.astype(np.float32), cmap=plt.cm.binary)
        # plt.show()

        # set blur to False
        vals_noblur = pixel_level_peak_finder_2d(image, blur=False)

        np.testing.assert_array_almost_equal(vals_noblur, np.array([49, 48]))

        # set blur to True
        vals_blur = pixel_level_peak_finder_2d(image, blur=True)

        np.testing.assert_array_almost_equal(vals_blur, np.array([49, 49]))

    def test_parabolic_peak_finder(self):

        # run parabolic peak finder with blur
        vals = quadric_peak_finder_2d(self.images[0], fit_size=1, blur=True, shift_limit=3)
        np.testing.assert_array_almost_equal(vals, np.array([49, 49]))

        # run parabolic peak finder without blur
        vals_2 = quadric_peak_finder_2d(self.images[0], fit_size=4, blur=False, shift_limit=1)
        np.testing.assert_array_almost_equal(vals_2, np.array([49, 48.48734177]))

    def test_render(self):

        illums, uv = self.xcorr.render(0, self.target_obj, temperature=0)

        np.testing.assert_array_almost_equal(illums, [1]*self.rays.num_rays)
        np.testing.assert_array_almost_equal(uv, np.array([[49]*self.rays.num_rays, [49]*self.rays.num_rays]))

    def test_compute_rays(self):

        (rays, pixels), (ul, lr) = self.xcorr.compute_rays(self.target_obj, temperature=0)

        np.testing.assert_array_almost_equal(self.camera.model.project_onto_image(rays.direction, temperature=0),
                                             pixels)

        eul, elr = self.target_obj.get_bounding_pixels(self.camera.model, temperature=0)

        np.testing.assert_equal(ul, np.floor(eul))
        np.testing.assert_equal(lr, np.ceil(elr))

    def test_estimate(self):

        self.xcorr.rays = None
        self.xcorr.estimate(self.images[0])

        self.assertIsNotNone(self.xcorr.details)
        details = cast(list[dict[str, Any]], self.xcorr.details)
        self.assertIn('Failed', details[0])
