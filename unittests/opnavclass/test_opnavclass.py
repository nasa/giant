from unittest import TestCase

from giant.opnav_class import OpNav
from giant.camera import Camera
from giant.camera_models import PinholeModel
from giant.image import OpNavImage
from giant.utilities.spice_interface import et_callable_to_datetime_callable, create_callable_orientation
from giant.point_spread_functions import Moment
from giant.rotations import Rotation
from giant.point_spread_functions.gaussians import Gaussian
import cv2

import numpy as np

from datetime import datetime, timedelta


class MyTestCamera(Camera):
    def preprocessor(self, image):
        return image


class TestAttitudeFunction:
    def __call__(self, time) -> Rotation:
        return Rotation()


class TestOpNavClass(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.image = np.abs(np.random.randn(1000, 1000))

        x, y = np.meshgrid(np.arange(20, 30), np.arange(50, 60))  # form the underlying grid
        z = 20000 * np.exp(-(np.power(x - 25, 2) / (2 * (3 ** 2)) +
                             (np.power(y - 55, 2) / (2 * (1 ** 2)))))  # generate the height data

        cls.image[y, x] += z

        x, y = np.meshgrid(np.arange(500, 530), np.arange(800, 820))  # form the underlying grid
        z = 20000 * np.exp(-(np.power(x - 520.2, 2) / (2 * (2 ** 2)) + (np.power(y - 810.9, 2) / (2 * (2 ** 2)))))
        # generate the height data

        cls.image[y, x] += z

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

        for row, col, illum_val in zip(rows.flatten(), cols.flatten(), illum_vals.flatten()):
            cls.image[int(np.round(row)), int(np.round(col))] += illum_val

        cls.radius = radius
        cls.center = np.array(center)

    @staticmethod
    def load_images():
        return [OpNavImage(np.arange(0 + x, 100 + x).reshape(10, 10),
                           observation_date=datetime(2019, 5, 4, 0, 0, 0, 0) + timedelta(days=x),
                           temperature=20 + 0.1 * x, exposure_type='long') for x in range(0, 10)]

    @staticmethod
    def load_cmodel():
        return PinholeModel(kx=500, ky=500, px=2500, py=3500, focal_length=10, n_rows=5000, n_cols=5000)

    def load_camera(self):
        images = self.load_images()
        cmodel = self.load_cmodel()

        return MyTestCamera(images=images, model=cmodel, name="AwesomeCam", spacecraft_name="AwesomeSpacecraft",
                            frame="AwesomeFrame", parse_data=False, psf=Gaussian(),
                            attitude_function=TestAttitudeFunction(),
                            start_date=datetime(2019, 5, 4, 0, 0, 0, 0), end_date=datetime(2019, 5, 5, 0, 0, 0, 0),
                            default_image_class=OpNavImage, metadata_only=False)

    def load_opnav(self):
        cam = self.load_camera()

        return OpNav(cam)

    @staticmethod
    def load_test_image(n, j, use_opnav_image=True):

        # opnav_inst = self.load_opnav()

        # make a test image based on cubic function,highest values at middle of image (4,4)
        test_image = np.empty((n, n), dtype=np.float32)  # the default dtype is float, so set dtype if it isn't float
        mat = np.hstack((np.arange(j / 2, j)[::-1], np.arange(j / 2, j)))

        for idx, line in enumerate(mat):
            x = (n - 1) * np.ones(n, dtype=np.float32)
            y = mat
            test_image[idx] = (n - line) ** 2 * np.subtract(x, y)

        if use_opnav_image:
            return OpNavImage(test_image)
        else:
            return test_image

    # DONE
    def test_init(self):

        opnav_inst = self.load_opnav()

        self.assertIsInstance(opnav_inst, OpNav)

        self.assertIsInstance(opnav_inst.camera, MyTestCamera)

    # DONE
    def test_camera_property(self):

        opnav_inst = self.load_opnav()

        self.assertEqual(opnav_inst.camera, opnav_inst._camera)

        class MyOtherTestCamera(Camera):
            def preprocessor(self, image):
                return image

        self.assertIsInstance(opnav_inst.camera, MyTestCamera)

        opnav_inst.camera = MyOtherTestCamera()

        self.assertIsInstance(opnav_inst.camera, MyOtherTestCamera)

    def test_add_images(self):

        opnav_inst = self.load_opnav()

        opnav_inst.add_images(OpNavImage(np.arange(0, 100).reshape(10, 10), observation_date=datetime(2019, 5, 3, 0, 0, 0, 0)))

        self.assertEqual(opnav_inst.camera.images[10].observation_date, datetime(2019, 5, 3, 0, 0, 0, 0))
