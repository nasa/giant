"""
test_imageprocessing
====================

Tests the methods and classes contained in the imageprocessing submodule of GIANT.

Test Cases
__________
"""

from unittest import TestCase
import numpy as np
import giant.image_processing as gimp

from copy import deepcopy


class TestLocalMaxima(TestCase):
    def test_local_maxima(self):
        im = [[0, 1, 2, 20, 1],
              [5, 2, 1, 3, 1],
              [0, 1, 2, 10, 1],
              [1, 2, 10, -2, -5],
              [50, 2, -1, -2, 30]]

        desired = [[False, False, False, True, False],
                   [True, False, False, False, False],
                   [False, False, False, True, False],
                   [False, False, True, False, False],
                   [True, False, False, False, True]]

        np.testing.assert_array_equal(gimp.local_maxima(im), desired)


class TestCv2Correlator(TestCase):
    def test_cv2correlator(self):
        # TODO: test a couple other coefficients as well
        img = np.random.randn(30, 30)
        temp = img[20:27, 15:27]

        cor_surf = gimp.cv2_correlator_2d(img, temp)

        temp_middle = np.floor(np.array(temp.shape) / 2)

        temp_point = np.array([0, 0])  # look for the upper left corner

        img_loc = np.unravel_index(cor_surf.argmax(), cor_surf.shape) - temp_middle + temp_point

        np.testing.assert_array_equal([20, 15], img_loc)

        self.assertAlmostEqual(cor_surf.max(), 1, places=4)


class TestFFTCorrelator(TestCase):
    def test_fftcorrelator(self):
        # TODO: test a couple other coefficients as well
        img = np.random.randn(30, 30)
        temp = img[20:27, 15:27]

        cor_surf = gimp.fft_correlator_2d(img, temp)

        temp_middle = np.floor(np.array(temp.shape) / 2)

        temp_point = np.array([0, 0])  # look for the upper left corner

        img_loc = np.unravel_index(cor_surf.argmax(), cor_surf.shape) - temp_middle + temp_point

        np.testing.assert_array_equal([20, 15], img_loc)

        self.assertAlmostEqual(cor_surf.max(), 1, places=4)


class TestSpatialCorrelator(TestCase):
    def test_spatialcorrelator(self):
        # TODO: test a couple other coefficients as well
        img = np.random.randn(30, 30)
        temp = img[20:27, 15:27]

        cor_surf = gimp.spatial_correlator_2d(img, temp)

        temp_middle = np.floor(np.array(temp.shape) / 2)

        temp_point = np.array([0, 0])  # look for the upper left corner

        img_loc = np.unravel_index(cor_surf.argmax(), cor_surf.shape) - temp_middle + temp_point

        np.testing.assert_array_equal([20, 15], img_loc)

        self.assertAlmostEqual(cor_surf.max(), 1, places=4)


class TestOtsu(TestCase):

    def test_otsu(self):

        image = np.zeros((200, 200), dtype=np.uint8)

        s1 = image.copy().astype(bool)
        s2 = image.copy().astype(bool)
        s3 = image.copy().astype(bool)

        s1[100:150, 100:150] = True
        image[s1] = 25

        s2[50:70, 50:70] = True
        image[s2] = 125

        s3[10:50, 150:170] = True
        image[s3] = 250

        image = gimp.cv2.GaussianBlur(image, (5, 5), 1)

        thresh, labeled = gimp.otsu(image, 4)

        # todo figure out how to test this


class TestImageProcessing(TestCase):
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

        cls.ip = gimp.ImageProcessing()

        cls.radius = radius
        cls.center = np.array(center)

    def test_flatten_image_and_get_noise_level(self):
        test = np.arange(100).reshape(10, 10)

        flat, noise = self.ip.flatten_image_and_get_noise_level(test)

        true_flat = np.array([[-2, -2, -2, -2, -2, -2, -2, -2, -1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 2, 2, 2, 2, 2, 2, 2, 2]])

        np.testing.assert_array_equal(true_flat, flat)

        # 0.5 is the maximum standard deviation that can be computed given the test image
        self.assertLessEqual(noise, 0.5)

    def test_corners_to_roi(self):
        im = np.random.randn(500, 600)
        corn_row = [5.5, 3, 6.5, 8.9]
        corn_col = [4.3, 2.7, 3.3, 7.8]
        roi = im[gimp.ImageProcessing.corners_to_roi(corn_row, corn_col)]
        np.testing.assert_array_equal(roi, im[3:10, 2:9])

    def test_find_poi_in_roi(self):
        self.ip.poi_min_size = 5
        self.ip.poi_max_size = 200
        with self.subTest(region=None):
            poi = self.ip.find_poi_in_roi(self.image)

            np.testing.assert_array_equal(poi, [[25, 55], [520, 811]])

        with self.subTest(region=(slice(30, 70), slice(5, 50))):
            region = self.ip.corners_to_roi([30, 70], [5, 50])

            poi = self.ip.find_poi_in_roi(self.image, region=region)

            np.testing.assert_array_equal(poi, [[25, 55]])

        with self.subTest(region=(slice(790, 840), slice(500, 550))):
            region = self.ip.corners_to_roi([790, 840], [500, 550])

            poi = self.ip.find_poi_in_roi(self.image, region=region)

            np.testing.assert_array_equal(poi, [[520, 811]])

        with self.subTest(region=(slice(200, 400), slice(300, 450))):
            region = self.ip.corners_to_roi([200, 400], [300, 450])

            poi = self.ip.find_poi_in_roi(self.image, region=region)

            self.assertFalse(poi)

    def test_refine_locations(self):
        refined_locs, _ = self.ip.refine_locations(self.image, np.array([[25, 55], [520, 810]]))

        np.testing.assert_array_almost_equal(refined_locs.T, [[25, 55], [520.2, 810.9]], decimal=2)

    def test_locate_subpixel_poi_in_roi(self):
        self.ip.poi_min_size = 5
        self.ip.poi_max_size = 200

        with self.subTest(region=None):
            refined_locs, _ = self.ip.locate_subpixel_poi_in_roi(self.image)

            np.testing.assert_array_almost_equal(refined_locs[::-1].T, [[55, 25], [810.9, 520.2]], decimal=2)

        with self.subTest(region=(slice(30, 70), slice(5, 50))):
            region = self.ip.corners_to_roi([30, 70], [5, 50])

            refined_locs, _ = self.ip.locate_subpixel_poi_in_roi(self.image, region=region)

            np.testing.assert_array_almost_equal(refined_locs[::-1].T, [[55, 25]], decimal=2)

        with self.subTest(region=(slice(790, 840), slice(500, 550))):
            region = self.ip.corners_to_roi([790, 840], [500, 550])

            refined_locs, _ = self.ip.locate_subpixel_poi_in_roi(self.image, region=region)

            np.testing.assert_array_almost_equal(refined_locs[::-1].T, [[810.9, 520.2]], decimal=2)

        with self.subTest(region=(slice(200, 400), slice(300, 450))):
            region = self.ip.corners_to_roi([200, 400], [300, 450])

            refined_locs, _ = self.ip.locate_subpixel_poi_in_roi(self.image, region=region)

            self.assertEqual(refined_locs.size, 0)

    def test_pae_edges(self):
        self.ip.pae_threshold = 10000
        edges = self.ip.pae_edges(self.image)

        radius_est = np.sqrt(((edges - self.center[::-1].reshape(2, 1)) ** 2).sum(axis=0))

        radius_err = self.radius - radius_est

        self.assertTrue((radius_err < 1).all())

        # TODO: need to figure out why these are so high.
        self.assertLess(radius_err.std(), 0.3)

        self.assertLess(np.abs(radius_err.mean()), 0.01)





    # def test_ip_centroiding_property(self):
    #
    #     opnav_inst = self.load_opnav()
    #
    #     self.assertTrue(opnav_inst._image_processing.centroiding, centroid_gaussian)
    #
    # # DONE
    # def test_ip_centroiding_setter(self):
    #
    #     opnav_inst = self.load_opnav()
    #
    #     opnav_inst.ip_centroiding = gaussian2d
    #
    #     self.assertTrue(opnav_inst._image_processing.centroiding, gaussian2d)
    #
    # # DONE
    # def test_ip_save_psf_property(self):
    #
    #     opnav_inst = self.load_opnav()
    #
    #     self.assertFalse(opnav_inst._image_processing.save_psf)
    #
    # # DONE
    # def test_ip_save_psf_setter(self):
    #
    #     opnav_inst = self.load_opnav()
    #
    #     self.assertFalse(opnav_inst._image_processing.save_psf)
    #
    #     opnav_inst._image_processing.save_psf = True
    #
    #     self.assertTrue(opnav_inst._image_processing.save_psf)
    #
    # # DONE
    # def test_ip_image_denoising_property(self):
    #
    #     opnav_inst = self.load_opnav()
    #
    #     self.assertTrue(opnav_inst._image_processing.image_denoising, cv2.GaussianBlur)
    #
    # # DONE
    # def test_ip_image_denoising_setter(self):
    #
    #     opnav_inst = self.load_opnav()
    #
    #     self.assertTrue(opnav_inst._image_processing.image_denoising, cv2.GaussianBlur)
    #
    #     opnav_inst._image_processing.image_denoising =  cv2.blur
    #
    #     self.assertTrue(opnav_inst._image_processing.image_denoising, cv2.blur)
    #
    # # DONE
    # def test_ip_denoising_args_property(self):
    #
    #     opnav_inst = self.load_opnav()
    #
    #     self.assertEqual(opnav_inst._image_processing.denoising_args, ((3, 3), 0))
    #
    # # DONE
    # def test_ip_denoising_args_setter(self):
    #
    #     opnav_inst = self.load_opnav()
    #
    #     self.assertEqual(opnav_inst._image_processing.denoising_args, ((3, 3), 0))
    #
    #     opnav_inst._image_processing.denoising_args = ((1, 1), 0)
    #
    #     self.assertEqual(opnav_inst._image_processing.denoising_args, ((1, 1), 0))
    #
    # # DONE
    # def test_ip_denoising_kwargs_property(self):
    #
    #     opnav_inst = self.load_opnav()
    #
    #     self.assertEqual(opnav_inst._image_processing.denoising_kwargs, {})
    #
    # # DONE
    # def test_ip_denoising_kwargs_setter(self):
    #
    #     opnav_inst = self.load_opnav()
    #
    #     self.assertEqual(opnav_inst._image_processing.denoising_kwargs, {})
    #
    #     opnav_inst._image_processing.denoising_kwargs = {'denoise': True} # check denoise_kwargs
    #
    #     self.assertEqual(opnav_inst._image_processing.denoising_kwargs, {'denoise': True})
    #
    # # DONE
    # def test_ip_denoise_flag_property(self):
    #
    #     opnav_inst = self.load_opnav()
    #
    #     self.assertFalse(opnav_inst._image_processing.denoise_flag)
    #
    # # DONE
    # def test_ip_denoise_flag_setter(self):
    #
    #     opnav_inst = self.load_opnav()
    #
    #     self.assertFalse(opnav_inst._image_processing.denoise_flag)
    #
    #     opnav_inst._image_processing.denoise_flag = True
    #
    #     self.assertTrue(opnav_inst._image_processing.denoise_flag)
    #
    # # DONE
    # def test_ip_correlator_property(self):
    #
    #     opnav_inst = self.load_opnav()
    #
    #     self.assertTrue(opnav_inst._image_processing.correlator, fft_correlator_2d)
    #
    # # DONE
    # def test_ip_correlator_setter(self):
    #
    #     opnav_inst = self.load_opnav()
    #
    #     self.assertTrue(opnav_inst._image_processing.correlator, fft_correlator_2d)
    #
    #     opnav_inst._image_processing.correlator = spatial_correlator_2d
    #
    #     self.assertTrue(opnav_inst._image_processing.correlator, spatial_correlator_2d)
    #
    # # DONE
    # def test_ip_correlator_kwargs_property(self):
    #
    #     opnav_inst = self.load_opnav()
    #
    #     self.assertEqual(opnav_inst._image_processing.correlator_kwargs, {})
    #
    # # DONE
    # def test_ip_correlator_kwargs_setter(self):
    #
    #     opnav_inst = self.load_opnav()
    #
    #     self.assertEqual(opnav_inst._image_processing.correlator_kwargs, {})
    #
    #     opnav_inst._image_processing.correlator_kwargs = {'correlator': True}
    #
    #     self.assertEqual(opnav_inst._image_processing.correlator_kwargs, {'correlator': True})
    #
    # # DONE
    # def test_ip_pae_threshold_property(self):
    #
    #     opnav_inst = self.load_opnav()
    #
    #     self.assertEqual(opnav_inst._image_processing.pae_threshold, 40)
    #
    # # DONE
    # def test_ip_pae_threshold_setter(self):
    #
    #     opnav_inst = self.load_opnav()
    #
    #     self.assertEqual(opnav_inst._image_processing.pae_threshold, 40)
    #
    #     opnav_inst._image_processing.pae_threshold = 50
    #
    #     self.assertEqual(opnav_inst._image_processing.pae_threshold, 50)
    #
    # # DONE
    # def test_ip_pae_order_property(self):
    #
    #     opnav_inst = self.load_opnav()
    #
    #     self.assertEqual(opnav_inst._image_processing.pae_order, 2)
    #
    # # DONE
    # def test_ip_pae_order_setter(self):
    #
    #     opnav_inst = self.load_opnav()
    #
    #     self.assertEqual(opnav_inst._image_processing.pae_order, 2)
    #
    #     opnav_inst._image_processing.pae_order = 3
    #
    #     self.assertEqual(opnav_inst._image_processing.pae_order, 3)
    #
    # # DONE
    # def test_ip_centroid_size_property(self):
    #
    #     opnav_inst = self.load_opnav()
    #
    #     self.assertEqual(opnav_inst._image_processing.centroid_size, 1)
    #
    # # DONE
    # def test_ip_centroid_size_setter(self):
    #
    #     opnav_inst = self.load_opnav()
    #
    #     self.assertEqual(opnav_inst._image_processing.centroid_size, 1)
    #
    #     opnav_inst._image_processing.centroid_size = 2
    #
    #     self.assertEqual(opnav_inst._image_processing.centroid_size, 2)
    #
    # # DONE
    # def test_ip_poi_threshold_property(self):
    #
    #     opnav_inst = self.load_opnav()
    #
    #     self.assertEqual(opnav_inst._image_processing.poi_threshold, 8)
    #
    # # DONE
    # def test_ip_poi_threshold_setter(self):
    #
    #     opnav_inst = self.load_opnav()
    #
    #     self.assertEqual(opnav_inst._image_processing.poi_threshold, 8)
    #
    #     opnav_inst._image_processing.poi_threshold = 10
    #
    #     self.assertEqual(opnav_inst._image_processing.poi_threshold, 10)
    #
    # # DONE
    # def test_ip_poi_max_size_property(self):
    #
    #     opnav_inst = self.load_opnav()
    #
    #     self.assertEqual(opnav_inst._image_processing.poi_max_size, 50)
    #
    # # DONE
    # def test_ip_poi_max_size_setter(self):
    #
    #     opnav_inst = self.load_opnav()
    #
    #     self.assertEqual(opnav_inst._image_processing.poi_max_size, 50)
    #
    #     opnav_inst._image_processing.poi_max_size = 70
    #
    #     self.assertEqual(opnav_inst._image_processing.poi_max_size, 70)
    #
    # # DONE
    # def test_ip_poi_min_size_property(self):
    #
    #     opnav_inst = self.load_opnav()
    #
    #     self.assertEqual(opnav_inst._image_processing.poi_min_size, 2)
    #
    # # DONE
    # def test_ip_poi_min_size_setter(self):
    #
    #     opnav_inst = self.load_opnav()
    #
    #     self.assertEqual(opnav_inst._image_processing.poi_min_size, 2)
    #
    #     opnav_inst._image_processing.poi_min_size = 4
    #
    #     self.assertEqual(opnav_inst._image_processing.poi_min_size, 4)
    #
    # # DONE
    # def test_ip_reject_saturation_property(self):
    #
    #     opnav_inst = self.load_opnav()
    #
    #     self.assertTrue(opnav_inst._image_processing.reject_saturation)
    #
    # # DONE
    # def test_ip_reject_saturation_setter(self):
    #
    #     opnav_inst = self.load_opnav()
    #
    #     self.assertTrue(opnav_inst._image_processing.reject_saturation)
    #
    #     opnav_inst._image_processing.reject_saturation = False
    #
    #     self.assertFalse(opnav_inst._image_processing.reject_saturation)
    #
    # # DONE
    # def test_ip_flatten_image_and_get_noise_level(self):
    #
    #     opnav_inst = self.load_opnav()
    #
    #     test_image = np.arange(100).reshape(10, 10)
    #
    #     true_flat_image = np.array([[-2, -2, -2, -2, -2, -2, -2, -2, -1, 0],
    #                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                                 [0, 1, 2, 2, 2, 2, 2, 2, 2, 2]])
    #
    #     flat_image, noise = opnav_inst.ip_flatten_image_and_get_noise_level(test_image)
    #
    #     self.assertRaises(AssertionError, np.testing.assert_array_equal, test_image, true_flat_image)
    #
    #     self.assertRaises(AssertionError, np.testing.assert_array_equal, test_image, flat_image)
    #
    #     np.testing.assert_array_equal(true_flat_image, flat_image)
    #
    #     self.assertLessEqual(noise, 0.5)
    #
    # # DONE
    # def test_ip_corners_to_roi(self):
    #
    #     opnav_inst = self.load_opnav()
    #
    #     im = np.random.randn(500, 600)
    #
    #     corn_row = [5.5, 3, 6.5, 8.9]
    #
    #     corn_col = [4.3, 2.7, 3.3, 7.8]
    #
    #     roi = im[opnav_inst.ip_corners_to_roi(corn_row, corn_col)]
    #
    #     np.testing.assert_array_equal(roi, im[3:10, 2:9])
    #
    # # DONE
    # def test_ip_find_poi_in_roi(self):
    #     # Note: uses self.image defined by classmethod
    #
    #     opnav_inst = self.load_opnav()
    #
    #     opnav_inst.ip_poi_min_size = 5
    #
    #     opnav_inst.ip_poi_max_size = 200
    #
    #     # image = np.abs(np.random.randn(1000, 1000))
    #
    #     with self.subTest(region=None):
    #
    #         poi = opnav_inst.ip_find_poi_in_roi(self.image) # uses image defined by classmethod!
    #
    #         np.testing.assert_array_equal(poi, [[25, 55], [520, 811]])
    #
    #     with self.subTest(region=(slice(30, 70), slice(5, 50))):
    #
    #         region = opnav_inst.ip_corners_to_roi([30, 70], [5, 50])
    #
    #         refined_locs, _ = opnav_inst.ip_locate_subpixel_poi_in_roi(self.image, region=region)
    #
    #         np.testing.assert_array_almost_equal(refined_locs[::-1].T, [[55, 25]], decimal=2)
    #
    #     with self.subTest(region=(slice(790, 840), slice(500, 550))):
    #
    #         region = opnav_inst.ip_corners_to_roi([790, 840], [500, 550])
    #
    #         refined_locs, _ = opnav_inst.ip_locate_subpixel_poi_in_roi(self.image, region=region)
    #
    #         np.testing.assert_array_almost_equal(refined_locs[::-1].T, [[810.9, 520.2]], decimal=2)
    #
    #     with self.subTest(region=(slice(200, 400), slice(300, 450))):
    #
    #         region = opnav_inst.ip_corners_to_roi([200, 400], [300, 450])
    #
    #         refined_locs, _ = opnav_inst.ip_locate_subpixel_poi_in_roi(self.image, region=region)
    #
    #         self.assertFalse(refined_locs)
    #
    # # DONE
    # def test_pae_edges(self):
    #     # Note: uses self.image defined by classmethod
    #
    #     opnav_inst = self.load_opnav()
    #
    #     opnav_inst.ip_pae_threshold = 10000
    #
    #     edges = opnav_inst.ip_pae_edges(self.image)
    #
    #     radius_est = np.sqrt(((edges - self.center[::-1].reshape(2, 1)) ** 2).sum(axis=0))
    #
    #     radius_err = self.radius - radius_est
    #
    #     self.assertTrue((radius_err < 1).all())
    #
    #     self.assertLess(radius_err.std(), 0.3)
    #
    #     self.assertLess(np.abs(radius_err.mean()), 0.01)
    #
