from unittest import TestCase

from typing import cast

import numpy as np

import pandas as pd

from giant.image import OpNavImage
from giant.calibration.calibration_class import Calibration, CalibrationOptions
from giant.stellar_opnav.star_identification import StarIDOptions
from giant.calibration import estimators as est
from giant.camera_models import PinholeModel
from giant.stellar_opnav import estimators as sopnavest
from giant.camera import Camera
from giant import rotations as at

import copy

from datetime import datetime

import warnings


class MyTestCamera(Camera):
    def preprocessor(self, image):
        return image


class BogusCatalog:
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
        self.default_options = CalibrationOptions(star_id_options=StarIDOptions(catalog=BogusCatalog()), # pyright: ignore[reportArgumentType]
                                                  geometric_estimator_type=est.geometric.GeometricEstimatorImplementations.LMA)


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
        
        cal = Calibration(cam, options=self.default_options)

        return cal

    def load_calibration_with_data(self):
        cam = self.load_camera_with_data()
        cal = Calibration(cam, options=self.default_options)

        return cal

    def test__init__(self):
        cal = self.load_calibration()

        self.assertIsInstance(cal.geometric_estimator, est.LMAEstimator)
        self.assertIsInstance(cal.attitude_estimator, sopnavest.DavenportQMethod)

    def test_estimate_calibration(self):
        images = [OpNavImage(self.load_image(49, 49, 100, 5), observation_date=datetime(2017, 2, 1, 0, 0, 0),
                             temperature=20, exposure_type='long'),
                  OpNavImage(self.load_image(49, 49, 100, 5), observation_date=datetime(2017, 2, 2, 0, 0, 0),
                             temperature=20, exposure_type='long')]

        cmodel = PinholeModel(kx=500, ky=500, px=49, py=49, focal_length=10, n_rows=5000, n_cols=5000)

        camera = MyTestCamera(images=images, model=cmodel)

        cal = Calibration(camera, options=CalibrationOptions(star_id_options=StarIDOptions(catalog=BogusCatalog()))) # pyright: ignore[reportArgumentType]

        cal.geometric_estimator = est.LMAEstimator(model=PinholeModel(kx=500, ky=500, px=49, py=49, focal_length=10,
                                                   n_rows=5000, n_cols=5000,
                                                   misalignment=[[1e-12, -2e-14, 3e-10],
                                                                 [2e-15, 1e-13, 3e-10]],
                                                   estimation_parameters=['multiple misalignments']))

        cal.geometric_estimator.measurements = np.hstack([np.arange(0, 6).reshape((2, 3)),
                                                          np.arange(0, 8).reshape((2, 4))]).astype(np.float64)
        cal.geometric_estimator.camera_frame_directions = [np.arange(0, 9).reshape((3, 3)).astype(np.float64),
                                                           np.arange(0, 12).reshape((3, 4)).astype(np.float64)]
        cal.geometric_estimator.measurement_covariance = np.random.rand(14 * 14).reshape((14, 14))
        cal.geometric_estimator.temperatures=[20, 20]

        cal._matched_extracted_image_points = [np.arange(0, 6).reshape((2, 3), order='F').astype(np.float64),
                                               np.arange(0, 8).reshape(2, 4).astype(np.float64)]
        cal._matched_catalog_unit_vectors_camera = [np.arange(0, 9).reshape(3, 3).astype(np.float64), 
                                                    np.arange(0, 12).reshape(3, 4).astype(np.float64)]

        cal_copy = copy.deepcopy(cal)

        # Check non-weighted estimation
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cal.estimate_geometric_calibration()

        self.assertFalse(cal.geometric_estimator.weighted_estimation)
        copy_model = cast(PinholeModel, cal_copy.model)
        cal_model = cast(PinholeModel, cal.model)
        self.assertEqual(copy_model.kx, cal_model.kx)
        self.assertNotEqual(copy_model.ky, cal_model.ky)
        self.assertNotEqual(copy_model.kx, cal_model.px)
        self.assertNotEqual(copy_model.ky, cal_model.py)

        # Check weighted estimation
        cal._matched_weights_picture = [np.arange(1, 7).astype(np.float64), 
                                        np.arange(1, 9).astype(np.float64)]

    def test_estimate_alignment(self):
        images = [OpNavImage(self.load_image(49, 49, 100, 5), observation_date=datetime(2017, 2, 1, 0, 0, 0),
                             temperature=20, exposure_type='long'),
                  OpNavImage(self.load_image(49, 49, 100, 5), observation_date=datetime(2017, 2, 2, 0, 0, 0),
                             temperature=20, exposure_type='long')]

        cmodel = PinholeModel(kx=500, ky=500, px=49, py=49, focal_length=10, n_rows=5000, n_cols=5000)

        camera = MyTestCamera(images=images, model=cmodel)

        cal = Calibration(camera, options=self.default_options)

        cal._matched_catalog_unit_vectors_inertial = [np.arange(1, 10).reshape(3, 3).astype(np.float64), 
                                                      np.arange(1, 13).reshape(3, 4).astype(np.float64)]

        cal._matched_extracted_image_points = [np.arange(0, 6).reshape((2, 3), order='F').astype(np.float64),
                                               np.arange(0, 8).reshape(2, 4).astype(np.float64)]
        cal._matched_catalog_unit_vectors_camera = [np.arange(0, 9).reshape(3, 3).astype(np.float64), 
                                                    np.arange(0, 12).reshape(3, 4).astype(np.float64)]

        cal.alignment_base_frame_func = self.base_frame_function
        res = cal.estimate_static_alignment()

        self.assertIsInstance(res, at.Rotation)
