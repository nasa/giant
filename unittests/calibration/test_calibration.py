from unittest import TestCase

import numpy as np

import pandas as pd

from giant.image import OpNavImage
from giant.calibration.calibration_class import Calibration
from giant.calibration import estimators as est
from giant.camera_models import PinholeModel
from giant.stellar_opnav import estimators as sopnavest
from giant.camera import Camera
from giant import rotations as at

import copy

from datetime import datetime


class MyTestCamera(Camera):
    def preprocessor(self, image):
        return image


class BogusCatalogue:
    pass


class BogusCatalogueWithData():

    def query_catalogue(self, *args, **kwargs):
        data = pd.DataFrame({'ra': [1., 2., 3.], 'dec': [-1., -2., -3.], 'distance': [100., 200., 300.],
                             'ra_proper_motion': [90., 30., 60.], 'dec_proper_motion': [90., 30., 30.],
                             'mag': [1., 1., 1.],
                             'ra_sigma': [0.1, 0.1, 0.1], 'dec_sigma': [0.1, 0.1, 0.1],
                             'distance_sigma': [0.1, 0.1, 0.1],
                             'ra_pm_sigma': [0.1, 0.1, 0.1], 'dec_pm_sigma': [0.1, 0.1, 0.1]})

        return data


# class BaseFrameFunction:
#     def __call__(self, observation_date):
#         return at.Rotation([0, 0, 0, 1])

class TestCalibration(TestCase):

    def setUp(self):
        def base_frame_function(time):
            return at.Rotation(np.array([0, 0, 0]))

        self.base_frame_function = base_frame_function

    @staticmethod
    def load_image(a, b, n, r):
        y, x = np.ogrid[-a:n - a, -b:n - b]
        mask = x * x + y * y <= r * r
        image = np.ones((n, n))
        image[mask] = 255
        return image

    def load_images(self):
        return OpNavImage(self.load_image(49, 49, 100, 5), observation_date=datetime(2017, 2, 1, 0, 0, 0),
                          temperature=20, exposure_type='long')

    @staticmethod
    def load_cmodel():
        return PinholeModel(kx=500, ky=500, px=49, py=49, focal_length=10, n_rows=5000, n_cols=5000)

    def load_camera(self):
        images = self.load_images()
        cmodel = self.load_cmodel()

        return MyTestCamera(images=images, model=cmodel)

    def load_camera_with_data(self):
        images = [np.array([np.arange(0, 99, 1), np.arange(-99, 0, 1), np.zeros((1, 100))]) for _ in range(10)]
        cmodel = self.load_cmodel()

        return MyTestCamera(images=images, model=cmodel)

    def load_calibration(self):
        cam = self.load_camera()

        cal = Calibration(cam, image_processing_kwargs=None, star_id_kwargs={'catalogue': BogusCatalogue()},
                          attitude_estimator=sopnavest.DavenportQMethod(),
                          calibration_estimator=est.LMAEstimator(),
                          static_alignment_estimator=est.StaticAlignmentEstimator())

        return cal

    def load_calibration_with_data(self):
        cam = self.load_camera_with_data()
        cal = Calibration(cam, image_processing_kwargs=None, star_id_kwargs={'catalogue': BogusCatalogueWithData()},
                          attitude_estimator=sopnavest.DavenportQMethod(),
                          calibration_estimator=est.LMAEstimator(),
                          static_alignment_estimator=est.StaticAlignmentEstimator())

        return cal

    def test__init__(self):
        cal = self.load_calibration()

        self.assertIsInstance(cal._calibration_est, est.LMAEstimator)
        self.assertIsInstance(cal._static_alignment_est, est.StaticAlignmentEstimator)
        self.assertIsNone(cal._initial_calibration_est_kwargs)
        self.assertIsNone(cal._initial_static_alignment_est_kwargs)

    def test_estimate_calibration(self):
        images = [OpNavImage(self.load_image(49, 49, 100, 5), observation_date=datetime(2017, 2, 1, 0, 0, 0),
                             temperature=20, exposure_type='long'),
                  OpNavImage(self.load_image(49, 49, 100, 5), observation_date=datetime(2017, 2, 2, 0, 0, 0),
                             temperature=20, exposure_type='long')]

        cmodel = PinholeModel(kx=500, ky=500, px=49, py=49, focal_length=10, n_rows=5000, n_cols=5000)

        camera = MyTestCamera(images=images, model=cmodel)

        cal = Calibration(camera, image_processing_kwargs=None, star_id_kwargs={'catalogue': BogusCatalogue()})

        cal._calibration_est = est.LMAEstimator(model=PinholeModel(kx=500, ky=500, px=49, py=49, focal_length=10,
                                                                   n_rows=5000, n_cols=5000,
                                                                   misalignment=[[1e-12, -2e-14, 3e-10],
                                                                                 [2e-15, 1e-13, 3e-10]],
                                                                   estimation_parameters=['multiple misalignments']),
                                                measurements=np.hstack([np.arange(0, 6).reshape((2, 3)),
                                                                        np.arange(0, 8).reshape((2, 4))]),
                                                camera_frame_directions=[np.arange(0, 9).reshape((3, 3)),
                                                                         np.arange(0, 12).reshape((3, 4))],
                                                measurement_covariance=np.random.rand(14 * 14).reshape((14, 14)),
                                                temperatures=[20, 20])

        cal._matched_extracted_image_points = [np.arange(0, 6).reshape((2, 3), order='F'),
                                               np.arange(0, 8).reshape(2, 4)]
        cal._matched_catalogue_unit_vectors_camera = [np.arange(0, 9).reshape(3, 3), np.arange(0, 12).reshape(3, 4)]

        cal_copy = copy.deepcopy(cal)

        # Check non-weighted estimation
        cal.estimate_calibration()

        self.assertFalse(cal._calibration_est.weighted_estimation)
        self.assertEqual(cal_copy.model.kx, cal.model.kx)
        self.assertNotEqual(cal_copy.model.ky, cal.model.ky)
        self.assertNotEqual(cal_copy.model.kx, cal.model.px)
        self.assertNotEqual(cal_copy.model.ky, cal.model.py)

        # Check weighted estimation
        cal._matched_weights_picture = [np.arange(1, 7), np.arange(1, 9)]

    def test_estimate_alignment(self):
        images = [OpNavImage(self.load_image(49, 49, 100, 5), observation_date=datetime(2017, 2, 1, 0, 0, 0),
                             temperature=20, exposure_type='long'),
                  OpNavImage(self.load_image(49, 49, 100, 5), observation_date=datetime(2017, 2, 2, 0, 0, 0),
                             temperature=20, exposure_type='long')]

        cmodel = PinholeModel(kx=500, ky=500, px=49, py=49, focal_length=10, n_rows=5000, n_cols=5000)

        camera = MyTestCamera(images=images, model=cmodel)

        cal = Calibration(camera, image_processing_kwargs=None, star_id_kwargs={'catalogue': BogusCatalogue()})

        cal._matched_catalogue_unit_vectors_inertial = [np.arange(1, 10).reshape(3, 3), np.arange(1, 13).reshape(3, 4)]

        cal_copy = copy.deepcopy(cal)

        cal._matched_extracted_image_points = [np.arange(0, 6).reshape((2, 3), order='F'),
                                               np.arange(0, 8).reshape(2, 4)]
        cal._matched_catalogue_unit_vectors_camera = [np.arange(0, 9).reshape(3, 3), np.arange(0, 12).reshape(3, 4)]

        cal.alignment_base_frame_func = self.base_frame_function
        cal.estimate_static_alignment()

        self.assertIsNone(cal_copy.static_alignment)
        self.assertIsInstance(cal.static_alignment, at.Rotation)
