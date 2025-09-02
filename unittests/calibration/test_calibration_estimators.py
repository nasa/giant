from unittest import TestCase

from typing import cast

from giant.image import OpNavImage
from giant.point_spread_functions.gaussians import Gaussian
from giant.calibration import estimators as est
from giant.camera_models import PinholeModel
from giant.camera import Camera
from giant import rotations as at

from datetime import datetime

import numpy as np


# Contains camera
class MyTestCamera(Camera):
    def preprocessor(self, image):
        return image


# Contains nothing
class BogusCatalogue:
    pass


# Test attitude function
class TestAttitudeFunction:
    def __call__(self):
        return at.Rotation([0, 0, 0, 1])


class TestCalibrationEstimator(TestCase):

    @staticmethod
    def load_image(a, b, n, r):
        y, x = np.ogrid[-a:n - a, -b:n - b]
        mask = x * x + y * y <= r * r
        image = np.ones((n, n))
        image[mask] = 255
        return image

    def load_images(self):
        return [
            OpNavImage(self.load_image(49, 49, 100, 5), observation_date=datetime(2017, 2, 1, 0, 0, 0), temperature=20,
                       exposure_type='long'),
            OpNavImage(self.load_image(49, 49, 100, 7), observation_date=datetime(2017, 2, 2, 0, 0, 0), temperature=20,
                       exposure_type='long')]

    @staticmethod
    def load_cmodel():
        return PinholeModel(kx=500, ky=500, px=49, py=49, focal_length=10, n_rows=5000, n_cols=5000,
                            a1=1e-5, a2=1e-6, misalignment=[[1e-12, -2e-14, 3e-10], [2e-15, 1e-13, 3e-10]],
                            estimation_parameters=['multiple misalignment'])

    @staticmethod
    def camera_frame_fun(date):
        return at.Rotation(np.array([0, 0, 0]))

    def load_camera(self):

        images = self.load_images()
        cmodel = self.load_cmodel()

        return MyTestCamera(images=images, model=cmodel, name="AwesomeCam",
                            spacecraft_name="AwesomeSpacecraft", frame="AwesomeFrame",
                            parse_data=False, psf=Gaussian(sigma_x=1, sigma_y=1),
                            attitude_function=self.camera_frame_fun,
                            start_date=datetime(2017, 2, 1, 0, 0, 0, 0), end_date=datetime(2017, 2, 2, 0, 0, 0, 0),
                            default_image_class=OpNavImage, metadata_only=False)

    @staticmethod
    def load_calibration_LMA_estimator():

        # image 1: contains 3 (x,y) measurement pairs
        # image 2: contains 4 (x,y) measurement pairs

        calest = est.LMAEstimator(model=PinholeModel(kx=500, ky=500, px=49, py=49, focal_length=10, n_rows=5000,
                                                     n_cols=5000))
        
        calest.measurements = np.hstack([np.arange(0, 6).reshape((2, 3)), np.arange(0, 8).reshape((2, 4))])
        calest.camera_frame_directions = [np.arange(0, 9).reshape((3, 3)).astype(np.float64), 
                                          np.arange(0, 12).reshape((3, 4)).astype(np.float64)]
        calest.measurement_covariance = np.random.rand(14 * 14).reshape((14, 14))
        calest.temperatures = [20, 20]

        return calest

    def test_LMA_Estimator(self):

        # Generate a "truth" camera model
        cmodel_truth = PinholeModel(kx=500, ky=500, px=49, py=49, focal_length=10, n_rows=50, n_cols=50)

        # Database directions
        dd_1 = np.vstack([np.array([0, 0, 4985.]), np.array([1, 2, 4999.]), np.array([-10, 10, 4998.])]).transpose()
        dd_2 = np.vstack([np.array([0, 0, 5005.]), np.array([1, 3, 4997.]), np.array([30, -5, 4999.]),
                          np.array([100, -120.5, 4998])]).transpose()
        dd = [dd_1.astype(np.float64), dd_2.astype(np.float64)]

        # Measurements
        meas_truth = np.zeros((2, 7))

        c_idx = 0
        for dd_set in dd:
            for idx in range(0, dd_set.shape[1]):
                direction = dd_set[:, idx].reshape((3, 1))
                meas_vec = cmodel_truth.project_onto_image(direction, temperature=20-c_idx*10)
                meas_truth[:, c_idx] = meas_vec.flatten()
                c_idx = c_idx + 1

        # Define a second camera model
        cmodel = PinholeModel(kx=600, ky=400, px=49, py=49, focal_length=10, n_rows=50, n_cols=50)

        # # Define a calibration estimation object
        calest = est.LMAEstimator(model=cmodel)  # calibration estimate based on new cmodel
        
        calest.measurements = meas_truth
        calest.camera_frame_directions = dd
        calest.temperatures = [20, 10]

        prefit_residuals = calest.compute_residuals()

        # Run estimator
        calest.estimate()

        # Compare prefit and postfit residuals for calibration object
        self.assertIsNotNone(calest.postfit_residuals)
        self.assertLessEqual(float(np.linalg.norm(cast(np.ndarray, calest.postfit_residuals))), 
                             float(np.linalg.norm(prefit_residuals)))

        self.assertTrue(calest.successful)
