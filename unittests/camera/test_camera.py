from unittest import TestCase

from giant.camera import Camera
from giant.camera_models import PinholeModel, BrownModel
from giant.image import OpNavImage
from giant.utilities.spice_interface import et_callable_to_datetime_callable, create_callable_orientation

import numpy as np

from datetime import datetime, timedelta

class MyTestCamera(Camera):
    def preprocessor(self, image):
        return image

class TestCallable:
    def __call__(self):
        return


class TestAttitudeFunction:
    def __call__(self):
        return

np.random.seed(10)

class TestCamera(TestCase):

    def load_images(self):
        return [OpNavImage(np.arange(0+x, 100+x).reshape(10, 10), observation_date=datetime(2019, 5, 4, 0, 0, 0, 0)
                             + timedelta(days=x),temperature=20+0.1*x, exposure_type='long') for x in range(0, 10)]

    def load_cmodel(self):
        return PinholeModel(kx=500, ky=500, px=2500, py=3500, focal_length=10, n_rows=5000, n_cols=5000)

    def test_attitude_function(self):
        return et_callable_to_datetime_callable(create_callable_orientation('J2000', 'MyTestCamera'))

    def load_camera(self):
        images = self.load_images()
        cmodel = self.load_cmodel()

        return MyTestCamera(images=images, model=cmodel, spacecraft_name="AwesomeSpacecraft",
                frame="AwesomeFrame", parse_data=False, psf=TestCallable(), name='AwesomeCam',
                attitude_function=TestAttitudeFunction(),
                start_date=datetime(2019, 5, 4, 0, 0, 0, 0),end_date=datetime(2019, 5, 5, 0, 0, 0, 0),
                default_image_class=OpNavImage, metadata_only=False)

    # DONE
    def test_init(self):

        cam = self.load_camera()

        self.assertIsInstance(cam, MyTestCamera)
        self.assertEqual(len(cam.images), 10)
        self.assertIsInstance(cam.model, PinholeModel)
        self.assertEqual(cam.name, "AwesomeCam")
        self.assertEqual(cam.frame, "AwesomeFrame")
        self.assertEqual(cam.start_date, datetime(2019, 5, 4, 0, 0, 0, 0))
        self.assertEqual(cam.end_date, datetime(2019, 5, 5, 0, 0, 0, 0))
        self.assertIsInstance(cam.psf, TestCallable)

    # DONE
    def test_iter(self):

        cam = self.load_camera()

        cam._image_mask[5], cam._image_mask[9] = True, True # cam._image_mask[0] and cam._image_mask[1] are True by default

        indices = [0,1,5,9]
        for ind, value in enumerate(iter(cam)):
            self.assertEqual(value[0], indices[ind])
            self.assertIsInstance(value[1], OpNavImage)

    # DONE
    def test_images_property(self):

        cam = self.load_camera()

        self.assertEqual(cam.images, cam._images)

        for ind, image in enumerate(cam.images):
            self.assertIsInstance(image, OpNavImage)

    # DONE
    def test_images_setter(self):

        cam = self.load_camera()

        self.assertRaises(AttributeError, setattr, cam.images, "images", [1, 2, 3])

    # DONE
    def test_image_mask_property(self):

        cam = self.load_camera()

        np.testing.assert_array_equal(cam.image_mask, [True, True, False, False, False, False, False, False, False, False])

    # DONE
    def test_image_mask_setter(self):

        cam = self.load_camera()

        # Test Sequence
        # set image masks to varying values
        cam.image_mask = [False, True, False, True, False, True, False, True, False, True]
        np.testing.assert_array_equal(cam._image_mask, [False, True, False, True, False, True, False, True, False, True])

        # set all image masks to True
        cam.image_mask = [True]
        np.testing.assert_array_equal(cam._image_mask, 10*[True])

        # Test bool
        cam.image_mask = False
        np.testing.assert_array_equal(cam._image_mask, 10*[False])

        # Test None - sets image_mask to True
        cam.image_mask = None
        np.testing.assert_array_equal(cam._image_mask, 10*[True])

    # DONE
    def test_psf_setter(self):

        cam = self.load_camera()
        self.assertIsInstance(cam.psf, TestCallable)

    # DONE
    def test_attitude_function_property(self):

        cam = self.load_camera()
        self.assertIsInstance(cam.attitude_function, TestAttitudeFunction)

    # DONE
    def test_attitude_function_setter(self):

        cam = self.load_camera()

        cam.attitude_function = TestAttitudeFunction()
        self.assertIsInstance(cam.attitude_function, TestAttitudeFunction)

        cam.attitude_function = None
        self.assertIsNone(cam.attitude_function)

    # DONE
    def test_model_property(self):

        cam = self.load_camera()

        self.assertEqual(cam.model, cam._model)
        self.assertIsInstance(cam.model, PinholeModel)
        self.assertIsInstance(cam._model, PinholeModel)

    # DONE
    def test_model_setter(self):

        cam = self.load_camera()

        cam.model = BrownModel(kx=5, ky=10, px=100, py=500)

        self.assertIsInstance(cam.model, BrownModel)
        self.assertIsInstance(cam._model, BrownModel)

    # DONE
    def test_short_on(self):

        # load camera
        cam = self.load_camera()

        # set image_mask to False
        cam.image_mask = [False]

        # set exposure_type of last image to 'short'
        for ind, image in enumerate(cam._images):
            if ind == 9:
                image.exposure_type = 'short'
            else:
                image.exposure_type = 'long'

        # Run command to change image_mask for short exposure images only
        cam.short_on()

        # Check image_mask value over all indices (last image is short)
        for ind,  image in enumerate(cam._images):
            if ind == 9:
                self.assertEqual(cam.image_mask[ind], True)
            else:
                self.assertEqual(cam.image_mask[ind], False)

    # DONE
    def test_short_off(self):

        # load camera
        cam = self.load_camera()

        # set image_mask to True
        cam.image_mask = [True]

        # set exposure_type of last image to 'short'
        for ind, image in enumerate(cam._images):
            if ind == 9:
                image.exposure_type = 'short'
            else:
                image.exposure_type = 'long'

        # Run command to change image_mask for short exposure images only
        cam.short_off()

        # Check image_mask value over all indices (last image is short)
        for ind,  image in enumerate(cam._images):
            if ind != 9:
                self.assertEqual(cam.image_mask[ind], True)
            else:
                self.assertEqual(cam.image_mask[ind], False)

    # DONE
    def test_long_on(self):

        # load camera
        cam = self.load_camera()

        # set image_mask to False
        cam.image_mask = [False]

        # set exposure_type of last image to 'long'
        for ind, image in enumerate(cam._images):
            if ind == 9:
                image.exposure_type = 'long'
            else:
                image.exposure_type = 'short'

        # Run command to change image_mask for short exposure images only
        cam.long_on()

        # Check image_mask value over all indices (last image is long)
        for ind,  image in enumerate(cam._images):
            if ind == 9:
                self.assertEqual(cam.image_mask[ind], True)
            else:
                self.assertEqual(cam.image_mask[ind], False)

    # DONE
    def test_long_off(self):

        # load camera
        cam = self.load_camera()

        # set image_mask to True
        cam.image_mask = [True]

        # set exposure_type of last image to 'long'
        for ind, image in enumerate(cam._images):
            if ind == 9:
                image.exposure_type = 'long'
            else:
                image.exposure_type = 'short'

        # Run command to change image_mask for short exposure images only
        cam.long_off()

        # Check image_mask value over all indices (last image is short)
        for ind,  image in enumerate(cam._images):
            if ind != 9:
                self.assertEqual(cam.image_mask[ind], True)
            else:
                self.assertEqual(cam.image_mask[ind], False)

    # DONE
    def test_all_on(self):

        # load camera
        cam = self.load_camera()

        # check current image mask
        self.assertEqual(cam.image_mask, 2*[True] + 8*[False])

        # turn all on
        cam.all_on()

        # check image mask for all True
        self.assertEqual(cam.image_mask, 10*[True])

    # DONE
    def test_all_off(self):

        # load camera
        cam = self.load_camera()

        # check current image mask
        self.assertEqual(cam.image_mask, 2 * [True] + 8 * [False])

        # turn all on
        cam.all_off()

        # check image mask for all False
        self.assertEqual(cam.image_mask, 10 * [False])

    # DONE
    def test_only_short_on(self):

        # load camera
        cam = self.load_camera()

        # set image_mask to False
        cam.image_mask = [False]

        self.assertEqual(cam.image_mask, 10*[False])

        # set exposure_type of first and last image to 'short'
        for ind, image in enumerate(cam._images):
            # if ind != 0 or ind != 9:
            if ind not in [0, 9]:
                image.exposure_type = 'long'
            else:
                image.exposure_type = 'short'

        # apply only short on
        cam.only_short_on()

        # check if first and last image mask are now True
        self.assertTrue(cam.image_mask[0])
        self.assertTrue(cam.image_mask[9])

    # DONE
    def test_only_long_on(self):

        # load camera
        cam = self.load_camera()

        # set image_mask to False
        cam.image_mask = [False]

        self.assertEqual(cam.image_mask, 10 * [False])

        # set exposure_type of first and last image to 'long'
        for ind, image in enumerate(cam._images):
            # if ind != 0 or ind != 9:
            if ind not in [0, 9]:
                image.exposure_type = 'short'
            else:
                image.exposure_type = 'long'

        # apply only long on
        cam.only_long_on()

        # check if first and last image mask are now True
        self.assertTrue(cam.image_mask[0])
        self.assertTrue(cam.image_mask[9])

    # DONE
    def test_apply_date_range(self):

        # load camera
        cam = self.load_camera()

        # set image_mask to True
        cam.image_mask = [True]

        # select a new start/end observation_date for camera
        cam.start_date = datetime(2019, 5, 7, 0, 0, 0, 0)
        cam.end_date = datetime(2019, 5, 10, 0, 0, 0, 0)

        # apply the observation_date range
        cam.apply_date_range()


        for ind, mask in enumerate(cam.image_mask):
            if ind in range(3, 7):
                self.assertEqual(mask, True)
            else:
                self.assertEqual(mask, False)

    # DONE
    def test_sort_by_date(self):

        # Load camera with image observation_date range 2019-05-04 through 2019-05-13
        cam = self.load_camera()

        cam.add_images(OpNavImage(np.arange(0, 100).reshape(10, 10), observation_date=datetime(2019, 5, 3, 0, 0, 0, 0)))

        # Added image is the last in the set
        for ind, image in enumerate(cam._images):
            if ind == 10:
                self.assertEqual(image.observation_date, datetime(2019, 5, 3, 0, 0, 0, 0))

        cam.sort_by_date()

        # Added image is now the first in the set after sorting
        for ind, image in enumerate(cam._images):
            if ind == 0:
                self.assertEqual(image.observation_date, datetime(2019, 5, 3, 0, 0, 0, 0))

    # DONE -- could test other ways (ndarray)
    def test_add_images(self):

        cam = self.load_camera()

        cam.add_images(OpNavImage(np.arange(0, 100).reshape(10, 10), observation_date=datetime(2019, 5, 3, 0, 0, 0, 0)))

        self.assertEqual(cam._images[10].observation_date, datetime(2019, 5, 3, 0, 0, 0, 0))

    # DONE
    def test_remove_images(self):

        cam = self.load_camera()

        self.assertEqual(len(cam.images), 10)

        cam.remove_images(5) # Remove a single image

        self.assertEqual(len(cam.images), 9)

    # DONE
    def test_image_check(self):

        cam = self.load_camera()

        # OpNavImage does not produce warnings
        datum = OpNavImage(np.arange(0, 100).reshape(10, 10), observation_date=datetime(2019, 5, 3, 0, 0, 0, 0))
        image = cam.image_check(datum, parse_data=True, metadata_only=False)
        self.assertIsInstance(image, OpNavImage)

        # List should produce a warning
        datum = [np.arange(0, 100).reshape(10, 10)]
        image = cam.image_check(datum)
        self.assertWarns(Warning)

        # Int should produce a warning
        datum = 1
        image = cam.image_check(datum)
        self.assertWarns(Warning)

