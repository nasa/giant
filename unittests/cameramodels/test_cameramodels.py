from unittest import TestCase

from tempfile import TemporaryDirectory

from pathlib import Path

from giant.camera_models import PinholeModel, OwenModel, BrownModel, OpenCVModel, save, load

import numpy as np

import giant.rotations as at

import lxml.etree as etree


class TestPinholeModel(TestCase):
    def setUp(self):

        self.Class = PinholeModel

    def test___init__(self):

        model = self.Class(intrinsic_matrix=np.array([[1, 2, 3], [4, 5, 6]]), focal_length=10.5, field_of_view=20.5,
                           use_a_priori=True, misalignment=[np.zeros(3), np.ones(3)],
                           estimation_parameters='basic intrinsic',
                           a1=1, a2=2, a3=3)

        np.testing.assert_array_equal(model.intrinsic_matrix, [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(model.focal_length, 10.5)
        self.assertEqual(model.field_of_view, 20.5)
        self.assertTrue(model.use_a_priori)
        self.assertEqual(model.estimation_parameters, ['basic intrinsic'])
        self.assertEqual(model.a1, 1)
        self.assertEqual(model.a2, 2)
        self.assertEqual(model.a3, 3)

        model = self.Class(kx=1, ky=2, px=4, py=5, focal_length=10.5, field_of_view=20.5,
                           use_a_priori=True, misalignment=[np.zeros(3), np.ones(3)],
                           estimation_parameters=['focal_length', 'px'])

        np.testing.assert_array_equal(model.intrinsic_matrix, [[1, 0, 4], [0, 2, 5]])
        self.assertEqual(model.focal_length, 10.5)
        self.assertEqual(model.field_of_view, 20.5)
        self.assertTrue(model.use_a_priori)
        self.assertEqual(model.estimation_parameters, ['focal_length', 'px'])

    def test_estimation_parameters(self):

        model = self.Class()

        model.estimation_parameters = 'kx'

        self.assertEqual(model.estimation_parameters, ['kx'])

        model.estimate_multiple_misalignments = False

        model.estimation_parameters = ['px', 'py', 'Multiple misalignments']

        self.assertEqual(model.estimation_parameters, ['px', 'py', 'multiple misalignments'])

        self.assertTrue(model.estimate_multiple_misalignments)

    def test_kx(self):

        model = self.Class(intrinsic_matrix=np.array([[1, 0, 0], [0, 0, 0]]))

        self.assertEqual(model.kx, 1)

        model.kx = 100

        self.assertEqual(model.kx, 100)

        self.assertEqual(model.intrinsic_matrix[0, 0], 100)

    def test_ky(self):

        model = self.Class(intrinsic_matrix=np.array([[0, 0, 0], [0, 3, 0]]))

        self.assertEqual(model.ky, 3)

        model.ky = 100

        self.assertEqual(model.ky, 100)

        self.assertEqual(model.intrinsic_matrix[1, 1], 100)

    def test_px(self):

        model = self.Class(intrinsic_matrix=np.array([[0, 0, 20], [0, 3, 0]]))

        self.assertEqual(model.px, 20)

        model.px = 100

        self.assertEqual(model.px, 100)

        self.assertEqual(model.intrinsic_matrix[0, 2], 100)

    def test_py(self):

        model = self.Class(intrinsic_matrix=np.array([[0, 0, 0], [0, 0, 10]]))

        self.assertEqual(model.py, 10)

        model.py = 100

        self.assertEqual(model.py, 100)

        self.assertEqual(model.intrinsic_matrix[1, 2], 100)

    def test_a1(self):

        model = self.Class(temperature_coefficients=np.array([10, 0, 0]))

        self.assertEqual(model.a1, 10)

        model.a1 = 100

        self.assertEqual(model.a1, 100)

        self.assertEqual(model.temperature_coefficients[0], 100)

    def test_a2(self):

        model = self.Class(temperature_coefficients=np.array([0, 10, 0]))

        self.assertEqual(model.a2, 10)

        model.a2 = 100

        self.assertEqual(model.a2, 100)

        self.assertEqual(model.temperature_coefficients[1], 100)

    def test_a3(self):

        model = self.Class(temperature_coefficients=np.array([0, 0, 10]))

        self.assertEqual(model.a3, 10)

        model.a3 = 100

        self.assertEqual(model.a3, 100)

        self.assertEqual(model.temperature_coefficients[2], 100)

    def test_intrinsic_matrix_inv(self):

        model = self.Class(kx=5, ky=10, px=100, py=-5)

        np.testing.assert_array_almost_equal(model.intrinsic_matrix @ np.vstack([model.intrinsic_matrix_inv,
                                                                                 [0, 0, 1]]),
                                             [[1, 0, 0], [0, 1, 0]])

        np.testing.assert_array_almost_equal(model.intrinsic_matrix_inv @ np.vstack([model.intrinsic_matrix,
                                                                                     [0, 0, 1]]),
                                             [[1, 0, 0], [0, 1, 0]])

    def test_get_temperature_scale(self):

        model = self.Class(temperature_coefficients=np.array([1, 2, 3.]))

        self.assertEqual(model.get_temperature_scale(1), 7)

        np.testing.assert_array_equal(model.get_temperature_scale([1, 2]), [7, 35])

        np.testing.assert_array_equal(model.get_temperature_scale([-1, 2.]), [-1, 35])

    def test_apply_distortion(self):
        inputs = [[0, 0], [1, 0], [-1, 0], [1.5, 0], [-1.5, 0], [[1.5], [0]], [[1.5, -1], [0, 0]],
                  [0, 1], [0, -1], [0, 1.5], [0, -1.5], [[0], [1.5]], [[0, 0], [1.5, -1]], [1, 1]]

        model = self.Class()

        for inp in inputs:
            gnom_dist = model.apply_distortion(np.array(inp))

            np.testing.assert_array_almost_equal(gnom_dist, inp)

    def test_get_projections(self):

        points = [[0, 0, 1], [-0.1, 0.2, 2.2], [[-0.1], [0.2], [2.2]], [[-0.1, 0], [0.2, 0], [2.2, 1]]]

        model = self.Class(focal_length=8.7, kx=500, ky=500.5, px=1500, py=1500.5, a1=1, a2=2, a3=3)

        with self.subTest(misalignment=None):

            for point in points:
                gnom, _, pix = model.get_projections(point)

                gnom_true = model.focal_length * np.array(point[:2]) / point[2]

                pix_true = (np.matmul(model.intrinsic_matrix[:, :2], gnom_true).T + model.intrinsic_matrix[:, 2]).T

                np.testing.assert_array_equal(gnom, gnom_true)
                np.testing.assert_array_equal(pix, pix_true)

        with self.subTest(temperature=1):

            for point in points:
                gnom, _, pix = model.get_projections(point, temperature=1)

                gnom_true = model.focal_length * np.array(point[:2]) / point[2]

                gnom_true *= model.get_temperature_scale(1)

                pix_true = (np.matmul(model.intrinsic_matrix[:, :2], gnom_true).T + model.intrinsic_matrix[:, 2]).T

                np.testing.assert_array_equal(gnom, gnom_true)
                np.testing.assert_array_equal(pix, pix_true)

        with self.subTest(temperature=-10.5):

            for point in points:
                gnom, _, pix = model.get_projections(point, temperature=-10.5)

                gnom_true = model.focal_length * np.array(point[:2]) / point[2]

                gnom_true *= model.get_temperature_scale(-10.5)

                pix_true = (np.matmul(model.intrinsic_matrix[:, :2], gnom_true).T + model.intrinsic_matrix[:, 2]).T

                np.testing.assert_array_equal(gnom, gnom_true)
                np.testing.assert_array_equal(pix, pix_true)

        model = self.Class(focal_length=8.7, kx=500, ky=500.5, px=1500, py=1500.5,
                           misalignment=np.array([0, 0, np.pi]))

        with self.subTest(misalignment=[0, 0, np.pi]):

            for point in points:
                gnom, _, pix = model.get_projections(point)

                gnom_true = -model.focal_length * np.array(point[:2]) / point[2]

                pix_true = (np.matmul(model.intrinsic_matrix[:, :2], gnom_true).T + model.intrinsic_matrix[:, 2]).T

                np.testing.assert_array_almost_equal(gnom, gnom_true)
                np.testing.assert_array_almost_equal(pix, pix_true)

        model = self.Class(focal_length=8.7, kx=500, ky=500.5, px=1500, py=1500.5,
                           misalignment=np.array([np.pi, 0, 0]))

        with self.subTest(misalignment=[np.pi, 0, 0]):

            for point in points:
                gnom, _, pix = model.get_projections(point)

                gnom_true = model.focal_length * np.array(point[:2]) / point[2]
                gnom_true[0] *= -1

                pix_true = (np.matmul(model.intrinsic_matrix[:, :2], gnom_true).T + model.intrinsic_matrix[:, 2]).T

                np.testing.assert_array_almost_equal(gnom, gnom_true)
                np.testing.assert_array_almost_equal(pix, pix_true)

        model = self.Class(focal_length=8.7, kx=500, ky=500.5, px=1500, py=1500.5,
                           misalignment=np.array([0, np.pi, 0]))

        with self.subTest(misalignment=[0, np.pi, 0]):

            for point in points:
                gnom, _, pix = model.get_projections(point)

                gnom_true = model.focal_length * np.array(point[:2]) / point[2]
                gnom_true[1] *= -1

                pix_true = (np.matmul(model.intrinsic_matrix[:, :2], gnom_true).T + model.intrinsic_matrix[:, 2]).T

                np.testing.assert_array_almost_equal(gnom, gnom_true)
                np.testing.assert_array_almost_equal(pix, pix_true)

        model = self.Class(focal_length=8.7, kx=500, ky=500.5, px=1500, py=1500.5,
                           misalignment=np.array([1, 0.2, 0.3]))

        with self.subTest(misalignment=[1, 0.2, 0.3]):

            rot_mat = at.rotvec_to_rotmat([1, 0.2, 0.3]).squeeze()

            for point in points:
                point_new = rot_mat @ point

                gnom, _, pix = model.get_projections(point)

                gnom_true = model.focal_length * np.array(point_new[:2]) / np.array(point_new[2])

                pix_true = (np.matmul(model.intrinsic_matrix[:, :2], gnom_true).T + model.intrinsic_matrix[:, 2]).T

                np.testing.assert_array_almost_equal(gnom, gnom_true)
                np.testing.assert_array_almost_equal(pix, pix_true)

        model = self.Class(focal_length=8.7, kx=500, ky=500.5, px=1500, py=1500.5,
                           misalignment=[[1, 0.2, 0.3], [0, 0, np.pi]])

        model.estimate_multiple_misalignments = True

        with self.subTest(misalignment=[[1, 0.2, 0.3], [0, 0, np.pi]]):

            for point in points:
                rot_mat = at.rotvec_to_rotmat([1, 0.2, 0.3]).squeeze()

                point_new = rot_mat @ point

                gnom, _, pix = model.get_projections(point, image=0)

                gnom_true = model.focal_length * np.array(point_new[:2]) / np.array(point_new[2])

                pix_true = (np.matmul(model.intrinsic_matrix[:, :2], gnom_true).T + model.intrinsic_matrix[:, 2]).T

                np.testing.assert_array_almost_equal(gnom, gnom_true)
                np.testing.assert_array_almost_equal(pix, pix_true)

                gnom, _, pix = model.get_projections(point, image=1)

                gnom_true = -model.focal_length * np.array(point[:2]) / point[2]

                pix_true = (np.matmul(model.intrinsic_matrix[:, :2], gnom_true).T + model.intrinsic_matrix[:, 2]).T

                np.testing.assert_array_almost_equal(gnom, gnom_true)
                np.testing.assert_array_almost_equal(pix, pix_true)

    def test_project_onto_image(self):

        points = [[0, 0, 1], [-0.1, 0.2, 2.2], [[-0.1], [0.2], [2.2]], [[-0.1, 0], [0.2, 0], [2.2, 1]]]

        model = self.Class(focal_length=8.7, kx=500, ky=500.5, px=1500, py=1500.5, a1=-1e-3, a2=1e-6, a3=-7e-8)

        with self.subTest(misalignment=None):

            for point in points:
                pix = model.project_onto_image(point)

                gnom_true = model.focal_length * np.array(point[:2]) / point[2]

                pix_true = (np.matmul(model.intrinsic_matrix[:, :2], gnom_true).T + model.intrinsic_matrix[:, 2]).T

                np.testing.assert_array_equal(pix, pix_true)

        with self.subTest(temperature=1):

            for point in points:
                pix = model.project_onto_image(point, temperature=1)

                gnom_true = model.focal_length * np.array(point[:2]) / point[2]

                gnom_true *= model.get_temperature_scale(1)

                pix_true = (np.matmul(model.intrinsic_matrix[:, :2], gnom_true).T + model.intrinsic_matrix[:, 2]).T

                np.testing.assert_array_equal(pix, pix_true)

        with self.subTest(temperature=-10.5):

            for point in points:
                pix = model.project_onto_image(point, temperature=-10.5)

                gnom_true = model.focal_length * np.array(point[:2]) / point[2]

                gnom_true *= model.get_temperature_scale(-10.5)

                pix_true = (np.matmul(model.intrinsic_matrix[:, :2], gnom_true).T + model.intrinsic_matrix[:, 2]).T

                np.testing.assert_array_equal(pix, pix_true)

        model = self.Class(focal_length=8.7, kx=500, ky=500.5, px=1500, py=1500.5,
                           misalignment=np.array([0, 0, np.pi]))

        with self.subTest(misalignment=[0, 0, np.pi]):

            for point in points:
                pix = model.project_onto_image(point)

                gnom_true = -model.focal_length * np.array(point[:2]) / point[2]

                pix_true = (np.matmul(model.intrinsic_matrix[:, :2], gnom_true).T + model.intrinsic_matrix[:, 2]).T

                np.testing.assert_array_almost_equal(pix, pix_true)

        model = self.Class(focal_length=8.7, kx=500, ky=500.5, px=1500, py=1500.5,
                           misalignment=np.array([np.pi, 0, 0]))

        with self.subTest(misalignment=[np.pi, 0, 0]):

            for point in points:
                pix = model.project_onto_image(point)

                gnom_true = model.focal_length * np.array(point[:2]) / point[2]
                gnom_true[0] *= -1

                pix_true = (np.matmul(model.intrinsic_matrix[:, :2], gnom_true).T + model.intrinsic_matrix[:, 2]).T

                np.testing.assert_array_almost_equal(pix, pix_true)

        model = self.Class(focal_length=8.7, kx=500, ky=500.5, px=1500, py=1500.5,
                           misalignment=np.array([0., np.pi, 0.]))

        with self.subTest(misalignment=[0, np.pi, 0]):

            for point in points:
                pix = model.project_onto_image(point)

                gnom_true = model.focal_length * np.array(point[:2]) / point[2]
                gnom_true[1] *= -1

                pix_true = (np.matmul(model.intrinsic_matrix[:, :2], gnom_true).T + model.intrinsic_matrix[:, 2]).T

                np.testing.assert_array_almost_equal(pix, pix_true)

        model = self.Class(focal_length=8.7, kx=500, ky=500.5, px=1500, py=1500.5,
                           misalignment=np.array([1, 0.2, 0.3]))

        with self.subTest(misalignment=[1, 0.2, 0.3]):

            rot_mat = at.rotvec_to_rotmat([1, 0.2, 0.3]).squeeze()

            for point in points:
                point_new = rot_mat @ point

                pix = model.project_onto_image(point)

                gnom_true = model.focal_length * np.array(point_new[:2]) / np.array(point_new[2])

                pix_true = (np.matmul(model.intrinsic_matrix[:, :2], gnom_true).T + model.intrinsic_matrix[:, 2]).T

                np.testing.assert_array_almost_equal(pix, pix_true)

        model = self.Class(focal_length=8.7, kx=500, ky=500.5, px=1500, py=1500.5,
                           misalignment=[[1, 0.2, 0.3], [0, 0, np.pi]])

        model.estimate_multiple_misalignments = True

        with self.subTest(misalignment=[[1, 0.2, 0.3], [0, 0, np.pi]]):

            for point in points:
                rot_mat = at.rotvec_to_rotmat([1, 0.2, 0.3]).squeeze()

                point_new = rot_mat @ point

                pix = model.project_onto_image(point, image=0)

                gnom_true = model.focal_length * np.array(point_new[:2]) / np.array(point_new[2])

                pix_true = (np.matmul(model.intrinsic_matrix[:, :2], gnom_true).T + model.intrinsic_matrix[:, 2]).T

                np.testing.assert_array_almost_equal(pix, pix_true)

                pix = model.project_onto_image(point, image=1)

                gnom_true = -model.focal_length * np.array(point[:2]) / point[2]

                pix_true = (np.matmul(model.intrinsic_matrix[:, :2], gnom_true).T + model.intrinsic_matrix[:, 2]).T

                np.testing.assert_array_almost_equal(pix, pix_true)

    def test_compute_pixel_jacobian(self):

        def num_deriv(uvec, cmodel, delta=1e-8, image=0, temperature=0) -> np.ndarray:

            uvec = np.array(uvec).reshape(3, -1)

            pix_true = cmodel.project_onto_image(uvec, image=image, temperature=temperature)

            uvec_pert = uvec + [[delta], [0], [0]]

            pix_pert_x_f = cmodel.project_onto_image(uvec_pert, image=image, temperature=temperature)

            uvec_pert = uvec + [[0], [delta], [0]]

            pix_pert_y_f = cmodel.project_onto_image(uvec_pert, image=image, temperature=temperature)

            uvec_pert = uvec + [[0], [0], [delta]]

            pix_pert_z_f = cmodel.project_onto_image(uvec_pert, image=image, temperature=temperature)

            uvec_pert = uvec - [[delta], [0], [0]]

            pix_pert_x_b = cmodel.project_onto_image(uvec_pert, image=image, temperature=temperature)

            uvec_pert = uvec - [[0], [delta], [0]]

            pix_pert_y_b = cmodel.project_onto_image(uvec_pert, image=image, temperature=temperature)

            uvec_pert = uvec - [[0], [0], [delta]]

            pix_pert_z_b = cmodel.project_onto_image(uvec_pert, image=image, temperature=temperature)

            return np.array([(pix_pert_x_f-pix_pert_x_b)/(2*delta),
                             (pix_pert_y_f-pix_pert_y_b)/(2*delta),
                             (pix_pert_z_f-pix_pert_z_b)/(2*delta)]).swapaxes(0, -1)

        model_param = {"focal_length": 100.75, "kx": 30, "ky": 40,
                       "px": 4005.23, "py": 4005.23,
                       "misalignment": [[1e-8, 1e-9, 1e-10], [-1e-8, 2e-9, -1e-11], [2e-10, -5e-12, 1e-9]],
                       "a1": 1e-6, "a2": 2e-7, "a3": 3e-8}

        inputs = [np.array([[0.5, 0, 1]]).T,
                  np.array([[0, 0.5, 1], [0.5, 0.5, 1], [-0.5, 0, 1]]).T,
                  np.array([[0.1, -0.5, 1], [-0.5, -0.5, 1], [5, 10, 1000.23], [1, 2, 1200.23]]).T]

        temperatures = [0, 1, -1, 10.5, -10.5]

        model = self.Class(**model_param)
        model.estimate_multiple_misalignments = True

        for temp in temperatures:

            for input in inputs:

                for image in range(3):

                    with self.subTest(image=image, temp=temp, input=input):

                        jac_ana = model.compute_pixel_jacobian(input, image=image, temperature=temp)

                        jac_num = num_deriv(input, model, image=image, temperature=temp, delta=1e-2)

                        np.testing.assert_allclose(jac_ana, jac_num, rtol=1e-3, atol=1e-10)

    def test__compute_dcamera_point_dgnomic(self):

        def num_deriv(gnomic_locations, cmodel, delta=1e-6) -> np.ndarray:

            def g2u(g):

                v = np.vstack([g, cmodel.focal_length*np.ones(g.shape[-1])])

                return v/np.linalg.norm(v, axis=0, keepdims=True)

            gnomic_locations = np.asarray(gnomic_locations).reshape(2, -1)

            gnom_pert = gnomic_locations + [[delta], [0]]

            cam_loc_pert_x_f = g2u(gnom_pert)

            gnom_pert = gnomic_locations + [[0], [delta]]

            cam_loc_pert_y_f = g2u(gnom_pert)

            gnom_pert = gnomic_locations - [[delta], [0]]

            cam_loc_pert_x_b = g2u(gnom_pert)

            gnom_pert = gnomic_locations - [[0], [delta]]

            cam_loc_pert_y_b = g2u(gnom_pert)

            return np.array([(cam_loc_pert_x_f -cam_loc_pert_x_b)/(2*delta),
                             (cam_loc_pert_y_f -cam_loc_pert_y_b)/(2*delta)]).swapaxes(0, -1)

        model_param = {"focal_length": 100.75, "kx": 300, "ky": 400,
                       "px": 1005.23, "py": 1005.23,
                       "misalignment": [[1e-8, 1e-9, 1e-10], [-1e-8, 2e-9, -1e-11], [2e-10, -5e-12, 1e-9]],
                       "a1": 1e-6, "a2": 2e-7, "a3": 3e-8}

        inputs = [np.array([[0, 0]]).T,
                  np.array([[0, 2000], [2000, 0], [2000, 2000]]).T,
                  np.array([[1000, 1000], [1000, 2000], [2000, 1000], [0, 1000], [1000, 0]]).T]

        model = self.Class(**model_param)
        model.estimate_multiple_misalignments = True

        for input in inputs:

            with self.subTest(input=input):
                jac_ana = []
                for gnom in input.T:
                    jac_ana.append(
                        model._compute_dcamera_point_dgnomic(gnom, np.sqrt(np.sum(gnom*gnom) + model.focal_length**2)))

                jac_ana = np.array(jac_ana)

                jac_num = num_deriv(input, model)

                np.testing.assert_almost_equal(jac_ana, jac_num)

    def test__compute_dgnomic_ddist_gnomic(self):

        def num_deriv(dist_gnomic_locations, cmodel, delta=1e-6) -> np.ndarray:

            def dg2g(dg):

                gnomic_guess = dg.copy()

                # perform the fpa
                for _ in np.arange(20):

                    # get the distorted location assuming the current guess is correct
                    gnomic_guess_distorted = cmodel.apply_distortion(gnomic_guess)

                    # subtract off the residual distortion from the gnomic guess
                    gnomic_guess += dg - gnomic_guess_distorted

                    # check for convergence
                    if np.all(np.linalg.norm(gnomic_guess_distorted - dg, axis=0) <= 1e-15):
                        break

                return gnomic_guess

            dist_gnomic_locations = np.asarray(dist_gnomic_locations).reshape(2, -1)

            dist_gnom_pert = dist_gnomic_locations + [[delta], [0]]

            gnom_loc_pert_x_f = dg2g(dist_gnom_pert)

            dist_gnom_pert = dist_gnomic_locations + [[0], [delta]]

            gnom_loc_pert_y_f = dg2g(dist_gnom_pert)

            dist_gnom_pert = dist_gnomic_locations - [[delta], [0]]

            gnom_loc_pert_x_b = dg2g(dist_gnom_pert)

            dist_gnom_pert = dist_gnomic_locations - [[0], [delta]]

            gnom_loc_pert_y_b = dg2g(dist_gnom_pert)

            return np.array([(gnom_loc_pert_x_f - gnom_loc_pert_x_b)/(2*delta),
                             (gnom_loc_pert_y_f - gnom_loc_pert_y_b)/(2*delta)]).swapaxes(0, -1)

        model_param = {"focal_length": 100.75, "kx": 300, "ky": 400,
                       "px": 1005.23, "py": 1005.23,
                       "misalignment": [[1e-8, 1e-9, 1e-10], [-1e-8, 2e-9, -1e-11], [2e-10, -5e-12, 1e-9]],
                       "a1": 1e-6, "a2": 2e-7, "a3": 3e-8}

        inputs = [np.array([[0, 0]]).T,
                  np.array([[0, 0.1], [0.1, 0], [0.1, 0.1]]).T,
                  np.array([[-0.1, 0], [0, -0.1], [-0.1, -0.1], [0.1, -0.1], [-0.1, 0.1]]).T]

        model = self.Class(**model_param)
        model.estimate_multiple_misalignments = True

        for input in inputs:

            with self.subTest(input=input):
                jac_ana = []
                for dist_gnom in input.T:
                    jac_ana.append(model._compute_dgnomic_ddist_gnomic(dist_gnom))

                jac_ana = np.array(jac_ana)

                jac_num = num_deriv(input, model)

                np.testing.assert_almost_equal(jac_ana, jac_num)

    def test_compute_unit_vector_jacobian(self):

        def num_deriv(pixels, cmodel, delta=1e-6, image=0, temperature=0) -> np.ndarray:

            pixels = np.array(pixels).reshape(2, -1)

            pix_pert = pixels + [[delta], [0]]

            uvec_pert_x_f = cmodel.pixels_to_unit(pix_pert, image=image, temperature=temperature)

            pix_pert = pixels + [[0], [delta]]

            uvec_pert_y_f = cmodel.pixels_to_unit(pix_pert, image=image, temperature=temperature)

            pix_pert = pixels - [[delta], [0]]

            uvec_pert_x_b = cmodel.pixels_to_unit(pix_pert, image=image, temperature=temperature)

            pix_pert = pixels - [[0], [delta]]

            uvec_pert_y_b = cmodel.pixels_to_unit(pix_pert, image=image, temperature=temperature)

            return np.array([(uvec_pert_x_f-uvec_pert_x_b)/(2*delta),
                             (uvec_pert_y_f-uvec_pert_y_b)/(2*delta)]).swapaxes(0, -1)

        model_param = {"focal_length": 100.75, "kx": 300, "ky": 400,
                       "px": 1005.23, "py": 1005.23,
                       "misalignment": [[1e-8, 1e-9, 1e-10], [-1e-8, 2e-9, -1e-11], [2e-10, -5e-12, 1e-9]],
                       "a1": 1e-6, "a2": 2e-7, "a3": 3e-8}

        inputs = [np.array([[0, 0]]).T,
                  np.array([[0, 2000], [2000, 0], [2000, 2000]]).T,
                  np.array([[1000, 1000], [1000, 2000], [2000, 1000], [0, 1000], [1000, 0]]).T]

        temperatures = [0, 1, -1, 10.5, -10.5]

        model = self.Class(**model_param)
        model.estimate_multiple_misalignments = True

        for temp in temperatures:

            for input in inputs:

                for image in range(3):

                    with self.subTest(image=image, temp=temp, input=input):

                        jac_ana = model.compute_unit_vector_jacobian(input, image=image, temperature=temp)

                        jac_num = num_deriv(input, model, image=image, temperature=temp, delta=1e-2)

                        np.testing.assert_allclose(jac_ana, jac_num, rtol=1e-3, atol=1e-10)

    def test__compute_dcamera_point_dmisalignment(self):

        def num_deriv(loc, dtheta, delta=1e-10) -> np.ndarray:
            mis_pert = at.rotvec_to_rotmat(np.array(dtheta) + [delta, 0, 0]).squeeze()
            point_pert_x_f = mis_pert @ loc
            mis_pert = at.rotvec_to_rotmat(np.array(dtheta) + [0, delta, 0]).squeeze()
            point_pert_y_f = mis_pert @ loc
            mis_pert = at.rotvec_to_rotmat(np.array(dtheta) + [0, 0, delta]).squeeze()
            point_pert_z_f = mis_pert @ loc

            mis_pert = at.rotvec_to_rotmat(np.array(dtheta) - [delta, 0, 0]).squeeze()
            point_pert_x_b = mis_pert @ loc
            mis_pert = at.rotvec_to_rotmat(np.array(dtheta) - [0, delta, 0]).squeeze()
            point_pert_y_b = mis_pert @ loc
            mis_pert = at.rotvec_to_rotmat(np.array(dtheta) - [0, 0, delta]).squeeze()
            point_pert_z_b = mis_pert @ loc

            return np.array([(point_pert_x_f - point_pert_x_b) / (2 * delta),
                             (point_pert_y_f - point_pert_y_b) / (2 * delta),
                             (point_pert_z_f - point_pert_z_b) / (2 * delta)]).T

        inputs = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [np.sqrt(3), np.sqrt(3), np.sqrt(3)],
                  [-1, 0, 0], [0, -1, 0], [0, 0, -1], [-np.sqrt(3), -np.sqrt(3), -np.sqrt(3)],
                  [1, 0, 100], [0, 0.5, 1]]

        misalignment = [[1e-8, 0, 0], [0, 1e-8, 0], [0, 0, 1e-8], [1e-9, 1e-9, 1e-9],
                        [-1e-8, 0, 0], [0, -1e-8, 0], [0, 0, -1e-8], [-1e-9, -1e-9, -1e-9],
                        [1e-9, 2.3e-9, -0.5e-9]]

        for mis in misalignment:

            with self.subTest(misalignment=mis):

                for inp in inputs:
                    num = num_deriv(inp, mis)

                    # noinspection PyTypeChecker
                    ana = self.Class._compute_dcamera_point_dmisalignment(inp)

                    np.testing.assert_allclose(num, ana, atol=1e-10, rtol=1e-4)

    def test__compute_dpixel_ddistorted_gnomic(self):

        def num_deriv(loc, cmodel, delta=1e-8, temperature=0) -> np.ndarray:
            loc_pert = np.array(loc) + [delta, 0]
            loc_pert *= cmodel.get_temperature_scale(temperature)
            pix_pert_x_f = cmodel.intrinsic_matrix[:, :2] @ loc_pert + cmodel.intrinsic_matrix[:, 2]
            loc_pert = np.array(loc) + [0, delta]
            loc_pert *= cmodel.get_temperature_scale(temperature)
            pix_pert_y_f = cmodel.intrinsic_matrix[:, :2] @ loc_pert + cmodel.intrinsic_matrix[:, 2]

            loc_pert = np.array(loc) - [delta, 0]
            loc_pert *= cmodel.get_temperature_scale(temperature)
            pix_pert_x_b = cmodel.intrinsic_matrix[:, :2] @ loc_pert + cmodel.intrinsic_matrix[:, 2]
            loc_pert = np.array(loc) - [0, delta]
            loc_pert *= cmodel.get_temperature_scale(temperature)
            pix_pert_y_b = cmodel.intrinsic_matrix[:, :2] @ loc_pert + cmodel.intrinsic_matrix[:, 2]

            return np.array(
                [(pix_pert_x_f - pix_pert_x_b) / (2 * delta), (pix_pert_y_f - pix_pert_y_b) / (2 * delta)]).T

        intrins_coefs = [{"kx": 1.5, "ky": 0, "px": 0, "py": 0, 'a1': 0, 'a2': 0, 'a3': 0},
                         {"kx": 0, "ky": 1.5, "px": 0, "py": 0, 'a1': 0, 'a2': 0, 'a3': 0},
                         {"kx": 0, "ky": 0, "px": 1.5, "py": 0, 'a1': 0, 'a2': 0, 'a3': 0},
                         {"kx": 0, "ky": 0, "px": 0, "py": 1.5, 'a1': 0, 'a2': 0, 'a3': 0},
                         {"kx": 0, "ky": 0, "px": 0, "py": 0, 'a1': 1.5, 'a2': 0, 'a3': 0},
                         {"kx": 0, "ky": 0, "px": 0, "py": 0, 'a1': 0, 'a2': 1.5, 'a3': 0},
                         {"kx": 0, "ky": 0, "px": 0, "py": 0, 'a1': 0, 'a2': 0, 'a3': 1.5},
                         {"kx": 1.5, "ky": 1.5, "px": 1.5, "py": 1.5, 'a1': 1.5, 'a2': 1.5, 'a3': 1.5}]

        inputs = [[1e-6, 1e-6], [1, 0], [-1, 0], [1.5, 0], [-1.5, 0], [0, 1], [0, -1], [0, 1.5], [0, -1.5], [1, 1]]

        temperatures = [0, 1, -1, 10.5, -10.5]

        for intrins_coef in intrins_coefs:

            model = self.Class(**intrins_coef)

            for temp in temperatures:

                with self.subTest(temp=temp, **intrins_coef):

                    for inp in inputs:
                        num = num_deriv(inp, model, temperature=temp)

                        ana = model._compute_dpixel_ddistorted_gnomic(temperature=temp)

                        np.testing.assert_allclose(num, ana, atol=1e-10)

    def test__compute_dgnomic_dcamera_point(self):
        def num_deriv(loc, cmodel, delta=1e-8) -> np.ndarray:
            loc_pert = np.array(loc) + [delta, 0, 0]
            gnom_pert_x_f = cmodel.get_projections(loc_pert)[0]
            loc_pert = np.array(loc) + [0, delta, 0]
            gnom_pert_y_f = cmodel.get_projections(loc_pert)[0]
            loc_pert = np.array(loc) + [0, 0, delta]
            gnom_pert_z_f = cmodel.get_projections(loc_pert)[0]

            loc_pert = np.array(loc) - [delta, 0, 0]
            gnom_pert_x_b = cmodel.get_projections(loc_pert)[0]
            loc_pert = np.array(loc) - [0, delta, 0]
            gnom_pert_y_b = cmodel.get_projections(loc_pert)[0]
            loc_pert = np.array(loc) - [0, 0, delta]
            gnom_pert_z_b = cmodel.get_projections(loc_pert)[0]

            return np.array([(gnom_pert_x_f - gnom_pert_x_b) / (2 * delta),
                             (gnom_pert_y_f - gnom_pert_y_b) / (2 * delta),
                             (gnom_pert_z_f - gnom_pert_z_b) / (2 * delta)]).T

        intrins_coefs = [{"focal_length": 1},
                         {"focal_length": 2.5},
                         {"focal_length": 1000},
                         {"focal_length": 0.1}]

        inputs = [[0, 0, 1], [0.5, 0, 1], [0, 0.5, 1], [0.5, 0.5, 1], [-0.5, 0, 1], [0, -0.5, 1], [-0.5, -0.5, 1],
                  [5, 10, 1000.23], [0.5, 1e-14, 1]]

        for intrins_coef in intrins_coefs:

            model = self.Class(**intrins_coef)

            with self.subTest(**intrins_coef):

                for inp in inputs:
                    num = num_deriv(inp, model)

                    ana = model._compute_dgnomic_dcamera_point(np.array(inp))

                    np.testing.assert_allclose(num, ana, atol=1e-9, rtol=1e-5)

    def test__compute_dgnomic_dfocal_length(self):

        def num_deriv(loc, cmodel, delta=1e-8) -> np.ndarray:

            model_pert = cmodel.copy()
            model_pert.focal_length += delta

            gnom_pert_f = model_pert.get_projections(loc)[0]

            model_pert = cmodel.copy()
            model_pert.focal_length -= delta

            gnom_pert_b = model_pert.get_projections(loc)[0]

            # noinspection PyTypeChecker
            return np.asarray((gnom_pert_f - gnom_pert_b) / (2 * delta))

        intrins_coefs = [{"focal_length": 1},
                         {"focal_length": 2.5},
                         {"focal_length": 1000},
                         {"focal_length": 0.1}]

        inputs = [[0, 0, 1], [0.5, 0, 1], [0, 0.5, 1], [0.5, 0.5, 1], [-0.5, 0, 1], [0, -0.5, 1], [-0.5, -0.5, 1]]

        for intrins_coef in intrins_coefs:

            model = self.Class(**intrins_coef)

            with self.subTest(**intrins_coef):

                for inp in inputs:
                    num = num_deriv(inp, model)

                    ana = model._compute_dgnomic_dfocal_length(np.array(inp))

                    np.testing.assert_allclose(num, ana, atol=1e-10, rtol=1e-5)

    def test__compute_dpixel_dintrinsic(self):
        def num_deriv(loc, cmodel, delta=1e-6) -> np.ndarray:

            model_pert = cmodel.copy()
            model_pert.kx += delta
            pix_pert_kx_f = model_pert.intrinsic_matrix[:, :2] @ loc + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.ky += delta
            pix_pert_ky_f = model_pert.intrinsic_matrix[:, :2] @ loc + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.px += delta
            pix_pert_px_f = model_pert.intrinsic_matrix[:, :2] @ loc + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.py += delta
            pix_pert_py_f = model_pert.intrinsic_matrix[:, :2] @ loc + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.kx -= delta
            pix_pert_kx_b = model_pert.intrinsic_matrix[:, :2] @ loc + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.ky -= delta
            pix_pert_ky_b = model_pert.intrinsic_matrix[:, :2] @ loc + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.px -= delta
            pix_pert_px_b = model_pert.intrinsic_matrix[:, :2] @ loc + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.py -= delta
            pix_pert_py_b = model_pert.intrinsic_matrix[:, :2] @ loc + model_pert.intrinsic_matrix[:, 2]

            return np.array([(pix_pert_kx_f - pix_pert_kx_b) / (2 * delta),
                             (pix_pert_ky_f - pix_pert_ky_b) / (2 * delta),
                             (pix_pert_px_f - pix_pert_px_b) / (2 * delta),
                             (pix_pert_py_f - pix_pert_py_b) / (2 * delta)]).T

        intrins_coefs = [{"kx": 1.5, "ky": 0, "px": 0, "py": 0},
                         {"kx": 0, "ky": 1.5, "px": 0, "py": 0},
                         {"kx": 0, "ky": 0, "px": 1.5, "py": 0},
                         {"kx": 0, "ky": 0, "px": 0, "py": 1.5},
                         {"kx": 1.5, "ky": 1.5, "px": 1.5, "py": 1.5}]

        inputs = [[1e-6, 1e-6], [1, 0], [-1, 0], [1.5, 0], [-1.5, 0], [0, 1], [0, -1], [0, 1.5], [0, -1.5], [1, 1]]

        for intrins_coef in intrins_coefs:

            model = self.Class(**intrins_coef)

            with self.subTest(**intrins_coef):

                for inp in inputs:
                    num = num_deriv(inp, model)

                    ana = model._compute_dpixel_dintrinsic(np.array(inp))

                    np.testing.assert_allclose(num, ana, atol=1e-10)

    def test__compute_dpixel_dtemperature_coeffs(self):
        def num_deriv(loc, cmodel, delta=1e-6, temperature=0) -> np.ndarray:

            loc = np.array(loc)

            model_pert = cmodel.copy()
            model_pert.a1 += delta
            loc_copy = loc * model_pert.get_temperature_scale(temperature)
            pix_pert_a1_f = model_pert.intrinsic_matrix[:, :2] @ loc_copy + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.a2 += delta
            loc_copy = loc * model_pert.get_temperature_scale(temperature)
            pix_pert_a2_f = model_pert.intrinsic_matrix[:, :2] @ loc_copy + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.a3 += delta
            loc_copy = loc * model_pert.get_temperature_scale(temperature)
            pix_pert_a3_f = model_pert.intrinsic_matrix[:, :2] @ loc_copy + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.a1 -= delta
            loc_copy = loc * model_pert.get_temperature_scale(temperature)
            pix_pert_a1_b = model_pert.intrinsic_matrix[:, :2] @ loc_copy + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.a2 -= delta
            loc_copy = loc * model_pert.get_temperature_scale(temperature)
            pix_pert_a2_b = model_pert.intrinsic_matrix[:, :2] @ loc_copy + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.a3 -= delta
            loc_copy = loc * model_pert.get_temperature_scale(temperature)
            pix_pert_a3_b = model_pert.intrinsic_matrix[:, :2] @ loc_copy + model_pert.intrinsic_matrix[:, 2]

            return np.array([(pix_pert_a1_f - pix_pert_a1_b) / (2 * delta),
                             (pix_pert_a2_f - pix_pert_a2_b) / (2 * delta),
                             (pix_pert_a3_f - pix_pert_a3_b) / (2 * delta)]).T

        intrins_coefs = [{"kx": 1.5, "ky": 0, "px": 0, "py": 0, 'a1': 0, 'a2': 0, 'a3': 0},
                         {"kx": 0, "ky": 1.5, "px": 0, "py": 0, 'a1': 0, 'a2': 0, 'a3': 0},
                         {"kx": 0, "ky": 0, "px": 1.5, "py": 0, 'a1': 0, 'a2': 0, 'a3': 0},
                         {"kx": 0, "ky": 0, "px": 0, "py": 1.5, 'a1': 0, 'a2': 0, 'a3': 0},
                         {"kx": 0, "ky": 0, "px": 0, "py": 0, 'a1': 1.5, 'a2': 0, 'a3': 0},
                         {"kx": 0, "ky": 0, "px": 0, "py": 0, 'a1': 0, 'a2': 1.5, 'a3': 0},
                         {"kx": 0, "ky": 0, "px": 0, "py": 0, 'a1': 0, 'a2': 0, 'a3': 1.5},
                         {"kx": 1.5, "ky": 1.5, "px": 1.5, "py": 1.5, 'a1': 1.5, 'a2': 1.5, 'a3': 1.5}]

        inputs = [[1e-6, 1e-6], [1, 0], [-1, 0], [1.5, 0], [-1.5, 0], [0, 1], [0, -1], [0, 1.5], [0, -1.5], [1, 1]]

        temperatures = [0, 1, -1, -10.5, 10.5]

        for intrins_coef in intrins_coefs:

            model = self.Class(**intrins_coef)

            for temp in temperatures:

                with self.subTest(temp=temp, **intrins_coef):

                    for inp in inputs:
                        num = num_deriv(inp, model, temperature=temp)

                        ana = model._compute_dpixel_dtemperature_coeffs(inp, temperature=temp)

                        np.testing.assert_allclose(num, ana, atol=1e-10)

    def test_get_jacobian_row(self):
        def num_deriv(loc, cmodel, delta=1e-8, image=0, temperature=0) -> np.ndarray:
            model_pert = cmodel.copy()
            model_pert.focal_length += delta
            pix_pert_f_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.focal_length -= delta
            pix_pert_f_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.kx += delta
            pix_pert_kx_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.ky += delta
            pix_pert_ky_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.px += delta
            pix_pert_px_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.py += delta
            pix_pert_py_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.kx -= delta
            pix_pert_kx_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.ky -= delta
            pix_pert_ky_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.px -= delta
            pix_pert_px_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.py -= delta
            pix_pert_py_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.a1 += delta
            pix_pert_a1_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.a2 += delta
            pix_pert_a2_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.a3 += delta
            pix_pert_a3_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.a1 -= delta
            pix_pert_a1_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.a2 -= delta
            pix_pert_a2_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.a3 -= delta
            pix_pert_a3_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            delta_misalignment = 1e-6
            model_pert = cmodel.copy()
            model_pert.misalignment[image][0] += delta_misalignment
            pix_pert_mx_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][1] += delta_misalignment
            pix_pert_my_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][2] += delta_misalignment
            pix_pert_mz_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][0] -= delta_misalignment
            pix_pert_mx_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][1] -= delta_misalignment
            pix_pert_my_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][2] -= delta_misalignment
            pix_pert_mz_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            return np.vstack([(pix_pert_f_f - pix_pert_f_b) / (delta * 2),
                              (pix_pert_kx_f - pix_pert_kx_b) / (delta * 2),
                              (pix_pert_ky_f - pix_pert_ky_b) / (delta * 2),
                              (pix_pert_px_f - pix_pert_px_b) / (delta * 2),
                              (pix_pert_py_f - pix_pert_py_b) / (delta * 2),
                              (pix_pert_a1_f - pix_pert_a1_b) / (delta * 2),
                              (pix_pert_a2_f - pix_pert_a2_b) / (delta * 2),
                              (pix_pert_a3_f - pix_pert_a3_b) / (delta * 2),
                              np.zeros((image * 3, 2)),
                              (pix_pert_mx_f - pix_pert_mx_b) / (delta_misalignment * 2),
                              (pix_pert_my_f - pix_pert_my_b) / (delta_misalignment * 2),
                              (pix_pert_mz_f - pix_pert_mz_b) / (delta_misalignment * 2)]).T

        # TODO: investigate why this fails with slightly larger misalignments and temperature coefficients
        model_param = {"focal_length": 100.75, "kx": 30, "ky": 40, "px": 4005.23, "py": 4005.23,
                       "a1": 1e-4, "a2": 2e-7, "a3": 3e-8,
                       "misalignment": [[2e-15, -1.2e-14, 5e-16], [-1e-14, 2e-14, -1e-15]]}

        inputs = [[0.5, 0, 1], [0, 0.5, 1], [0.5, 0.5, 1], [-0.5, 0, 1], [0, -0.5, 1], [-0.5, -0.5, 1],
                  [5, 10, 1000.23], [[10], [-22], [1200.23]]]

        temperatures = [0, -1, 1, -10.5, 10.5]

        model = self.Class(**model_param)
        model.estimate_multiple_misalignments = True

        for inp in inputs:

            for temp in temperatures:
                with self.subTest(temp=temp, inp=inp):
                    num = num_deriv(inp, model, delta=1, temperature=temp)
                    ana = model._get_jacobian_row(np.array(inp), 0, 1, temperature=temp)

                    np.testing.assert_allclose(ana, num, rtol=1e-2, atol=1e-10)

                    num = num_deriv(inp, model, delta=1, image=1, temperature=temp)
                    ana = model._get_jacobian_row(np.array(inp), 1, 2, temperature=temp)

                    np.testing.assert_allclose(ana, num, atol=1e-10, rtol=1e-2)

    def test_compute_jacobian(self):

        def num_deriv(loc, cmodel, delta=1e-8, image=0, nimages=1, temperature=0) -> np.ndarray:
            model_pert = cmodel.copy()
            model_pert.focal_length += delta
            pix_pert_f_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.focal_length -= delta
            pix_pert_f_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.kx += delta
            pix_pert_kx_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.ky += delta
            pix_pert_ky_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.px += delta
            pix_pert_px_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.py += delta
            pix_pert_py_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.kx -= delta
            pix_pert_kx_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.ky -= delta
            pix_pert_ky_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.px -= delta
            pix_pert_px_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.py -= delta
            pix_pert_py_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.a1 += delta
            pix_pert_a1_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.a2 += delta
            pix_pert_a2_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.a3 += delta
            pix_pert_a3_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.a1 -= delta
            pix_pert_a1_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.a2 -= delta
            pix_pert_a2_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.a3 -= delta
            pix_pert_a3_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            delta_misalignment = 1e-6
            model_pert = cmodel.copy()
            model_pert.misalignment[image][0] += delta_misalignment
            pix_pert_mx_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][1] += delta_misalignment
            pix_pert_my_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][2] += delta_misalignment
            pix_pert_mz_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][0] -= delta_misalignment
            pix_pert_mx_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][1] -= delta_misalignment
            pix_pert_my_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][2] -= delta_misalignment
            pix_pert_mz_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            return np.vstack([(pix_pert_f_f - pix_pert_f_b) / (delta * 2),
                              (pix_pert_kx_f - pix_pert_kx_b) / (delta * 2),
                              (pix_pert_ky_f - pix_pert_ky_b) / (delta * 2),
                              (pix_pert_px_f - pix_pert_px_b) / (delta * 2),
                              (pix_pert_py_f - pix_pert_py_b) / (delta * 2),
                              (pix_pert_a1_f - pix_pert_a1_b) / (delta * 2),
                              (pix_pert_a2_f - pix_pert_a2_b) / (delta * 2),
                              (pix_pert_a3_f - pix_pert_a3_b) / (delta * 2),
                              np.zeros((image * 3, 2)),
                              (pix_pert_mx_f - pix_pert_mx_b) / (delta_misalignment * 2),
                              (pix_pert_my_f - pix_pert_my_b) / (delta_misalignment * 2),
                              (pix_pert_mz_f - pix_pert_mz_b) / (delta_misalignment * 2),
                              np.zeros(((nimages - image - 1) * 3, 2))]).T

        model_param = {"focal_length": 100.75, "kx": 30, "ky": 40, "px": 4005.23, "py": 4005.23,
                       "a1": 1e-5, "a2": 1e-6, "a3": 1e-7,
                       "misalignment": [[0, 0, 1e-15], [0, 2e-15, 0], [3e-15, 0, 0]]}

        inputs = [np.array([[0.5, 0, 1]]).T,
                  np.array([[0, 0.5, 1], [0.5, 0.5, 1], [-0.5, 0, 1000]]).T,
                  np.array([[0.1, -0.5, 1], [-0.5, -0.5, 1], [5, 10, 1000.23], [1, 2, 1200.23]]).T]

        temperatures = [0, 1, -1, 10.5, -10.5]
        model = self.Class(**model_param, estimation_parameters=['intrinsic',
                                                                 'temperature dependence',
                                                                 'multiple misalignments'])

        for temp in temperatures:

            with self.subTest(temp=temp):
                model.use_a_priori = False

                jac_ana = model.compute_jacobian(inputs, temperature=temp)

                jac_num = []

                numim = len(inputs)

                for ind, inp in enumerate(inputs):
                    for vec in inp.T:
                        jac_num.append(num_deriv(vec.T, model, delta=1, image=ind, nimages=numim, temperature=temp))

                np.testing.assert_allclose(jac_ana, np.vstack(jac_num), rtol=1e-2, atol=1e-10)

                model.use_a_priori = True

                jac_ana = model.compute_jacobian(inputs, temperature=temp)

                jac_num = []

                numim = len(inputs)

                for ind, inp in enumerate(inputs):
                    for vec in inp.T:
                        jac_num.append(num_deriv(vec.T, model, delta=1, image=ind, nimages=numim, temperature=temp))

                jac_num = np.vstack(jac_num)

                jac_num = np.pad(jac_num, [(0, jac_num.shape[1]), (0, 0)], 'constant', constant_values=0)

                jac_num[-jac_num.shape[1]:] = np.eye(jac_num.shape[1])

                np.testing.assert_allclose(jac_ana, jac_num, rtol=1e-2, atol=1e-10)

    def test_remove_jacobian_columns(self):

        jac = np.arange(30).reshape(1, -1)

        model = self.Class()

        for est_param, vals in model.element_dict.items():
            model.estimation_parameters = [est_param]

            expected = jac[0, vals]

            np.testing.assert_array_equal(model._remove_jacobian_columns(jac), [expected])

    def test_apply_update(self):
        model_param = {"focal_length": 0, "kx": 0, "ky": 0,
                       "px": 0, "py": 0, "a1": 0, "a2": 0, "a3": 0,
                       "misalignment": [[0, 0, 0], [0, 0, 0]]}

        model = self.Class(**model_param, estimation_parameters=['intrinsic',
                                                                 'temperature dependence',
                                                                 'multiple misalignments'])

        update_vec = np.arange(14).astype(np.float64)

        model.apply_update(update_vec)

        keys = list(model_param.keys())

        keys.remove('misalignment')

        for key in keys:
            self.assertEqual(getattr(model, key), update_vec[model.element_dict[key][0]])

        for ind, vec in enumerate(update_vec[8:].reshape(-1, 3)):
            np.testing.assert_array_almost_equal(at.Rotation(vec).quaternion, at.Rotation(model.misalignment[ind]).quaternion)

    def test_pixels_to_gnomic(self):

        gnomic = [[1, 0], [0, 1], [-1, 0], [0, -1],
                  [0.5, 0], [0, 0.5], [-0.5, 0], [0, -0.5],
                  [0.5, 0.5], [-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5],
                  [[1, 0, 0.5], [0, 1.5, -0.5]]]

        model = self.Class(kx=2000, ky=-3000.2, px=1025, py=937.567,
                           a1=1e-3, a2=2e-6, a3=-5.5e-8)

        temperatures = [0, 1, -1, 10.5, -10.5]

        for gnoms in gnomic:

            for temp in temperatures:
                with self.subTest(gnoms=gnoms, temp=temp):
                    dis_gnoms = np.asarray(model.apply_distortion(gnoms)).astype(float)

                    dis_gnoms *= model.get_temperature_scale(temp)

                    pixels = ((model.intrinsic_matrix[:, :2] @ dis_gnoms).T + model.intrinsic_matrix[:, 2]).T

                    gnoms_solved = model.pixels_to_gnomic(pixels, temperature=temp)

                    np.testing.assert_allclose(gnoms_solved, gnoms)

    def test_undistort_pixels(self):

        intrins_param = {"kx": 3000, "ky": 4000, "px": 4005.23, 'py': 2000.33, 'a1': 1e-6, 'a2': 1e-5, 'a3': 2e-5}

        pinhole = [[0, 0], [1, 0], [-1, 0], [1.5, 0], [-1.5, 0], [[1.5], [0]], [[1.5, -1], [0, 0]],
                   [0, 1], [0, -1], [0, 1.5], [0, -1.5], [[0], [1.5]], [[0, 0], [1.5, -1]], [1, 1]]

        model = self.Class(**intrins_param)

        temperatures = [0, 1, -1, 10.5, -10.5]

        for gnom in pinhole:

            gnom = np.asarray(gnom).astype(float)

            for temp in temperatures:
                with self.subTest(gnom=gnom, temp=temp):
                    mm_dist = model.apply_distortion(np.array(gnom))

                    temp_scale = model.get_temperature_scale(temp)

                    mm_dist *= temp_scale

                    pix_dist = ((model.intrinsic_matrix[:, :2] @ mm_dist).T + model.intrinsic_matrix[:, 2]).T

                    pix_undist = model.undistort_pixels(pix_dist, temperature=temp)

                    gnom *= temp_scale

                    pix_pinhole = ((model.intrinsic_matrix[:, :2] @ gnom).T + model.intrinsic_matrix[:, 2]).T

                    np.testing.assert_allclose(pix_undist, pix_pinhole, atol=1e-13)

    def test_pixels_to_unit(self):
        intrins_param = {"focal_length": 32.7, "kx": 3000, "ky": 4000,
                         "px": 4005.23, 'py': 2000.33, 'a1': 1e-5, 'a2': -1e-10, 'a3': 2e-4,
                         'misalignment': [[1e-10, 2e-13, -3e-12], [4e-8, -5.3e-9, 9e-15]]}

        camera_vecs = [[0, 0, 1], [0.01, 0, 1], [-0.01, 0, 1], [0, 0.01, 1], [0, -0.01, 1], [0.01, 0.01, 1],
                       [-0.01, -0.01, 1], [[0.01, -0.01], [-0.01, 0.01], [1, 1]]]

        temperatures = [0, 1, -1, 10.5, -10.5]

        model = self.Class(**intrins_param)

        # TODO: consider adjusting so this isn't needed
        model.estimate_multiple_misalignments = True

        for vec in camera_vecs:

            for image in [0, 1]:

                for temp in temperatures:
                    with self.subTest(vec=vec, image=image, temp=temp):
                        pixel_loc = model.project_onto_image(vec, image=image, temperature=temp)

                        unit_vec = model.pixels_to_unit(pixel_loc, image=image, temperature=temp)

                        unit_true = np.array(vec).astype(np.float64)

                        unit_true /= np.linalg.norm(unit_true, axis=0, keepdims=True)

                        np.testing.assert_allclose(unit_vec, unit_true, atol=1e-13)

    def test_overwrite(self):

        model1 = self.Class(field_of_view=10, intrinsic_matrix=np.array([[1, 0, 3], [0, 5, 6]]), focal_length=60,
                            misalignment=[[1, 2, 3], [4, 5, 6]], use_a_priori=False,
                            estimation_parameters=['multiple misalignments'])

        model2 = self.Class(field_of_view=20, intrinsic_matrix=np.array([[11, 0, 13], [0, 15, 16]]), focal_length=160,
                            misalignment=[[11, 12, 13], [14, 15, 16]], use_a_priori=True,
                            estimation_parameters=['single misalignment'])

        modeltest = model1.copy()

        modeltest.overwrite(model2)

        self.assertEqual(model2.field_of_view, modeltest.field_of_view)
        self.assertEqual(model2.use_a_priori, modeltest.use_a_priori)
        self.assertEqual(model2.estimate_multiple_misalignments, modeltest.estimate_multiple_misalignments)
        np.testing.assert_array_equal(model2.intrinsic_matrix, modeltest.intrinsic_matrix)
        np.testing.assert_array_equal(model2.misalignment, modeltest.misalignment)
        np.testing.assert_array_equal(model2.estimation_parameters, modeltest.estimation_parameters)

        modeltest = model2.copy()

        modeltest.overwrite(model1)

        self.assertEqual(model1.field_of_view, modeltest.field_of_view)
        self.assertEqual(model1.use_a_priori, modeltest.use_a_priori)
        self.assertEqual(model1.estimate_multiple_misalignments, modeltest.estimate_multiple_misalignments)
        np.testing.assert_array_equal(model1.intrinsic_matrix, modeltest.intrinsic_matrix)
        np.testing.assert_array_equal(model1.misalignment, modeltest.misalignment)
        np.testing.assert_array_equal(model1.estimation_parameters, modeltest.estimation_parameters)

    def test_distort_pixels(self):

        model = self.Class(kx=1000, ky=-950.5, px=4500, py=139.32, a1=1e-3, a2=1e-4, a3=1e-5)

        pixels = [[0, 1], [1, 0], [-1, 0], [0, -1], [9000., 200.2],
                  [[4500, 100, 10.98], [0, 139.23, 200.3]]]

        temperatures = [0, 1, -1, 10.5, -10.5]

        for pix in pixels:

            for temp in temperatures:
                with self.subTest(pix=pix, temp=temp):
                    undist_pix = model.undistort_pixels(pix, temperature=temp)
                    dist_pix = model.distort_pixels(undist_pix, temperature=temp)

                    np.testing.assert_allclose(dist_pix, pix)

    def test_distortion_map(self):

        model = self.Class(kx=100, ky=-985.234, px=1000, py=1095)

        rows, cols, dist = model.distortion_map((2000, 250), step=10)

        # noinspection PyTypeChecker
        np.testing.assert_allclose(dist, 0, atol=1e-10)

        rl, cl = np.arange(0, 2000, 10), np.arange(0, 250, 10)

        rs, cs = np.meshgrid(rl, cl, indexing='ij')

        np.testing.assert_array_equal(rows, rs)
        np.testing.assert_array_equal(cols, cs)

    def test_undistort_image(self):
        # not sure how best to do this test...
        pass

    def test_copy(self):

        model = self.Class()

        model_copy = model.copy()

        model.kx = 1000

        model.ky = 999

        model.px = 100
        model.py = -20

        model.a1 = 5
        model.a2 = 6
        model.a3 = 7

        model._focal_length = 11231

        model.field_of_view = 1231231

        model.use_a_priori = True

        model.estimation_parameters = ['a1', 'kx', 'ky']

        model.estimate_multiple_misalignments = True

        model.misalignment = np.array([1231241, 123124, .12])

        self.assertNotEqual(model.kx, model_copy.kx)
        self.assertNotEqual(model.ky, model_copy.ky)
        self.assertNotEqual(model.px, model_copy.px)
        self.assertNotEqual(model.py, model_copy.py)
        self.assertNotEqual(model.a1, model_copy.a1)
        self.assertNotEqual(model.a2, model_copy.a2)
        self.assertNotEqual(model.a3, model_copy.a3)
        self.assertNotEqual(model.focal_length, model_copy.focal_length)
        self.assertNotEqual(model.field_of_view, model_copy.field_of_view)
        self.assertNotEqual(model.use_a_priori, model_copy.use_a_priori)
        self.assertNotEqual(model.estimate_multiple_misalignments, model_copy.estimate_multiple_misalignments)
        self.assertNotEqual(model.estimation_parameters, model_copy.estimation_parameters)
        self.assertTrue((model.misalignment != model_copy.misalignment).all())

    def test_to_from_elem(self):

        element = etree.Element(self.Class.__name__)

        model = self.Class(focal_length=20, field_of_view=5, use_a_priori=True,
                           misalignment=np.array([1, 2, 3]), kx=2, ky=200, px=50, py=300,
                           a1=37, a2=1, a3=-1230,
                           estimation_parameters=['a1', 'multiple misalignments'], n_rows=20, n_cols=30)

        model_copy = model.copy()

        with self.subTest(misalignment=True):
            element = model.to_elem(element, misalignment=True)

            self.assertEqual(model, model_copy)

            model_new = self.Class.from_elem(element)

            self.assertEqual(model, model_new)

        with self.subTest(misalignment=False):
            element = model.to_elem(element, misalignment=False)

            self.assertEqual(model, model_copy)

            model_new = self.Class.from_elem(element)

            model.estimation_parameters[-1] = 'single misalignment'

            model.estimate_multiple_misalignments = False

            model.misalignment = np.zeros(3)

            self.assertEqual(model, model_new)


class TestOwenModel(TestPinholeModel):
    def setUp(self):

        self.Class = OwenModel

    def test___init__(self):

        model = self.Class(intrinsic_matrix=np.array([[1, 2, 3], [4, 5, 6]]), focal_length=10.5, field_of_view=20.5,
                           use_a_priori=True, misalignment=[np.zeros(3), np.ones(3)],
                           distortion_coefficients=np.array([1, 2, 3, 4, 5, 6]),
                           estimation_parameters='basic intrinsic',
                           a1=1, a2=2, a3=3)

        np.testing.assert_array_equal(model.intrinsic_matrix, [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(model.focal_length, 10.5)
        self.assertEqual(model.field_of_view, 20.5)
        self.assertTrue(model.use_a_priori)
        self.assertEqual(model.estimation_parameters, ['basic intrinsic'])
        self.assertEqual(model.a1, 1)
        self.assertEqual(model.a2, 2)
        self.assertEqual(model.a3, 3)
        np.testing.assert_array_equal(model.distortion_coefficients, np.arange(1, 7))

        model = self.Class(kx=1, ky=2, px=4, py=5, focal_length=10.5, field_of_view=20.5,
                           use_a_priori=True, misalignment=[np.zeros(3), np.ones(3)], kxy=80, kyx=90,
                           estimation_parameters=['focal_length', 'px'], n_rows=500, n_cols=600,
                           e1=1, radial2=2, pinwheel2=3, e4=4, tangential_x=6, e5=5)

        np.testing.assert_array_equal(model.intrinsic_matrix, [[1, 80, 4], [90, 2, 5]])
        self.assertEqual(model.focal_length, 10.5)
        self.assertEqual(model.field_of_view, 20.5)
        self.assertTrue(model.use_a_priori)
        self.assertEqual(model.estimation_parameters, ['focal_length', 'px'])
        self.assertEqual(model.n_rows, 500)
        self.assertEqual(model.n_cols, 600)

        np.testing.assert_array_equal(model.distortion_coefficients, [2, 4, 5, 6, 1, 3])

    def test_kxy(self):

        model = self.Class(intrinsic_matrix=np.array([[0, 1, 0], [0, 0, 0]]))

        self.assertEqual(model.kxy, 1)

        model.kxy = 100

        self.assertEqual(model.kxy, 100)

        self.assertEqual(model.intrinsic_matrix[0, 1], 100)

    def test_kyx(self):

        model = self.Class(intrinsic_matrix=np.array([[0, 0, 0], [3, 0, 0]]))

        self.assertEqual(model.kyx, 3)

        model.kyx = 100

        self.assertEqual(model.kyx, 100)

        self.assertEqual(model.intrinsic_matrix[1, 0], 100)

    def test_e1(self):

        model = self.Class(distortion_coefficients=np.array([0, 0, 0, 0, 1, 0]))

        self.assertEqual(model.e1, 1)

        model.e1 = 100

        self.assertEqual(model.e1, 100)

        self.assertEqual(model.distortion_coefficients[4], 100)

    def test_e2(self):

        model = self.Class(distortion_coefficients=np.array([1, 0, 0, 0, 0, 0]))

        self.assertEqual(model.e2, 1)

        model.e2 = 100

        self.assertEqual(model.e2, 100)

        self.assertEqual(model.distortion_coefficients[0], 100)

    def test_e3(self):

        model = self.Class(distortion_coefficients=np.array([0, 0, 0, 0, 0, 1]))

        self.assertEqual(model.e3, 1)

        model.e3 = 100

        self.assertEqual(model.e3, 100)

        self.assertEqual(model.distortion_coefficients[5], 100)

    def test_e4(self):

        model = self.Class(distortion_coefficients=np.array([0, 1, 0, 0, 0, 0]))

        self.assertEqual(model.e4, 1)

        model.e4 = 100

        self.assertEqual(model.e4, 100)

        self.assertEqual(model.distortion_coefficients[1], 100)

    def test_e5(self):

        model = self.Class(distortion_coefficients=np.array([0, 0, 1, 0, 0, 0]))

        self.assertEqual(model.e5, 1)

        model.e5 = 100

        self.assertEqual(model.e5, 100)

        self.assertEqual(model.distortion_coefficients[2], 100)

    def test_e6(self):

        model = self.Class(distortion_coefficients=np.array([0, 0, 0, 1, 0, 0]))

        self.assertEqual(model.e6, 1)

        model.e6 = 100

        self.assertEqual(model.e6, 100)

        self.assertEqual(model.distortion_coefficients[3], 100)

    def test_pinwheel1(self):

        model = self.Class(distortion_coefficients=np.array([0, 0, 0, 0, 1, 0]))

        self.assertEqual(model.pinwheel1, 1)

        model.pinwheel1 = 100

        self.assertEqual(model.pinwheel1, 100)

        self.assertEqual(model.distortion_coefficients[4], 100)

    def test_radial2(self):

        model = self.Class(distortion_coefficients=np.array([1, 0, 0, 0, 0, 0]))

        self.assertEqual(model.radial2, 1)

        model.radial2 = 100

        self.assertEqual(model.radial2, 100)

        self.assertEqual(model.distortion_coefficients[0], 100)

    def test_pinwheel2(self):

        model = self.Class(distortion_coefficients=np.array([0, 0, 0, 0, 0, 1]))

        self.assertEqual(model.pinwheel2, 1)

        model.pinwheel2 = 100

        self.assertEqual(model.pinwheel2, 100)

        self.assertEqual(model.distortion_coefficients[5], 100)

    def test_radial4(self):

        model = self.Class(distortion_coefficients=np.array([0, 1, 0, 0, 0, 0]))

        self.assertEqual(model.radial4, 1)

        model.radial4 = 100

        self.assertEqual(model.radial4, 100)

        self.assertEqual(model.distortion_coefficients[1], 100)

    def test_tangential_y(self):

        model = self.Class(distortion_coefficients=np.array([0, 0, 1, 0, 0, 0]))

        self.assertEqual(model.tangential_y, 1)

        model.tangential_y = 100

        self.assertEqual(model.tangential_y, 100)

        self.assertEqual(model.distortion_coefficients[2], 100)

    def test_tangential_x(self):

        model = self.Class(distortion_coefficients=np.array([0, 0, 0, 1, 0, 0]))

        self.assertEqual(model.tangential_x, 1)

        model.tangential_x = 100

        self.assertEqual(model.tangential_x, 100)

        self.assertEqual(model.distortion_coefficients[3], 100)

    def test_apply_distortion(self):
        dist_coefs = [{"radial2": 1.5, "radial4": 0, "tangential_x": 0, "tangential_y": 0,
                       "pinwheel1": 0, "pinwheel2": 0},
                      {"radial2": 0, "radial4": 1.5, "tangential_x": 0, "tangential_y": 0,
                       "pinwheel1": 0, "pinwheel2": 0},
                      {"radial2": 0, "radial4": 0, "tangential_x": 1.5, "tangential_y": 0,
                       "pinwheel1": 0, "pinwheel2": 0},
                      {"radial2": 0, "radial4": 0, "tangential_x": 0, "tangential_y": 1.5,
                       "pinwheel1": 0, "pinwheel2": 0},
                      {"radial2": 0, "radial4": 0, "tangential_x": 0, "tangential_y": 0,
                       "pinwheel1": 1.5, "pinwheel2": 0},
                      {"radial2": 0, "radial4": 0, "tangential_x": 0, "tangential_y": 0,
                       "pinwheel1": 0, "pinwheel2": 1.5}]
        inputs = [[0, 0], [1, 0], [-1, 0], [1.5, 0], [-1.5, 0], [[1.5], [0]], [[1.5, -1], [0, 0]],
                  [0, 1], [0, -1], [0, 1.5], [0, -1.5], [[0], [1.5]], [[0, 0], [1.5, -1]], [1, 1]]

        solus = [[[0, 0], [2.5, 0], [-2.5, 0], [1.5 + 1.5 ** 4, 0], [-(1.5 + 1.5 ** 4), 0],
                  [[(1.5 + 1.5 ** 4)], [0]], [[(1.5 + 1.5 ** 4), -2.5], [0, 0]],
                  [0, 2.5], [0, -2.5], [0, (1.5 + 1.5 ** 4)], [0, -(1.5 + 1.5 ** 4)], [[0], [(1.5 + 1.5 ** 4)]],
                  [[0, 0], [(1.5 + 1.5 ** 4), -2.5]], [1 + 2 * 1.5, 1 + 2 * 1.5]],
                 [[0, 0], [2.5, 0], [-2.5, 0], [(1.5 + 1.5 ** 6), 0], [-(1.5 + 1.5 ** 6), 0],
                  [[(1.5 + 1.5 ** 6)], [0]], [[(1.5 + 1.5 ** 6), -2.5], [0, 0]],
                  [0, 2.5], [0, -2.5], [0, (1.5 + 1.5 ** 6)], [0, -(1.5 + 1.5 ** 6)], [[0], [(1.5 + 1.5 ** 6)]],
                  [[0, 0], [(1.5 + 1.5 ** 6), -2.5]], [1 + 4 * 1.5, 1 + 4 * 1.5]],
                 [[0, 0], [2.5, 0], [0.5, 0], [(1.5 + 1.5 ** 3), 0], [-1.5 + 1.5 ** 3, 0],
                  [[(1.5 + 1.5 ** 3)], [0]], [[(1.5 + 1.5 ** 3), 0.5], [0, 0]],
                  [0, 1], [0, -1], [0, 1.5], [0, -1.5], [[0], [1.5]], [[0, 0], [1.5, -1]], [2.5, 2.5]],
                 [[0, 0], [1, 0], [-1, 0], [1.5, 0], [-1.5, 0], [[1.5], [0]], [[1.5, -1], [0, 0]],
                  [0, 2.5], [0, 0.5], [0, 1.5 + 1.5 ** 3], [0, -1.5 + 1.5 ** 3], [[0], [1.5 + 1.5 ** 3]],
                  [[0, 0], [1.5 + 1.5 ** 3, 0.5]], [2.5, 2.5]],
                 [[0, 0], [1, 1.5], [-1, -1.5], [1.5, 1.5 ** 3], [-1.5, -1.5 ** 3],
                  [[1.5], [1.5 ** 3]], [[1.5, -1], [1.5 ** 3, -1.5]],
                  [-1.5, 1], [1.5, -1], [-1.5 ** 3, 1.5], [1.5 ** 3, -1.5], [[-1.5 ** 3], [1.5]],
                  [[-1.5 ** 3, 1.5], [1.5, -1]],
                  [1 - np.sqrt(2) * 1.5, 1 + np.sqrt(2) * 1.5]],
                 [[0, 0], [1, 1.5], [-1, -1.5], [1.5, 1.5 ** 5], [-1.5, -1.5 ** 5],
                  [[1.5], [1.5 ** 5]], [[1.5, -1], [1.5 ** 5, -1.5]],
                  [-1.5, 1], [1.5, -1], [-1.5 ** 5, 1.5], [1.5 ** 5, -1.5], [[-1.5 ** 5], [1.5]],
                  [[-1.5 ** 5, 1.5], [1.5, -1]],
                  [1 - 2 * np.sqrt(2) * 1.5, 1 + 2 * np.sqrt(2) * 1.5]]]

        for dist, sols in zip(dist_coefs, solus):

            with self.subTest(**dist):

                model = self.Class(**dist)

                for inp, solu in zip(inputs, sols):
                    gnom_dist = model.apply_distortion(np.array(inp))

                    np.testing.assert_array_almost_equal(gnom_dist, solu)

    def test_get_projections(self):

        points = [[0, 0, 1], [-0.1, 0.2, 2.2], [[-0.1], [0.2], [2.2]], [[-0.1, 0], [0.2, 0], [2.2, 1]]]

        model = self.Class(focal_length=8.7, kx=500, ky=500.5, kxy=1.5, kyx=-1.5, px=1500, py=1500.5,
                           radial2=1e-3, radial4=-2.2e-5, tangential_y=1e-3, tangential_x=1e-6,
                           pinwheel1=1e-6, pinwheel2=-2.23e-8, a1=1e-1, a2=1e-6, a3=-3e-7)

        temps = [0, 1, -1, 10, -10]

        for temp in temps:
            with self.subTest(misalignment=None, temp=temp):

                for point in points:
                    pin, dist, pix = model.get_projections(point, temperature=temp)

                    pin_true = model.focal_length * np.array(point[:2]) / point[2]

                    dist_true = model.apply_distortion(pin_true)

                    dist_true *= model.get_temperature_scale(temp)

                    pix_true = (np.matmul(model.intrinsic_matrix[:, :2], dist_true).T + model.intrinsic_matrix[:, 2]).T

                    np.testing.assert_array_equal(pin, pin_true)
                    np.testing.assert_array_equal(dist, dist_true)
                    np.testing.assert_array_equal(pix, pix_true)

        model = self.Class(focal_length=8.7, kx=500, ky=500.5, kxy=1.5, kyx=-1.5, px=1500, py=1500.5,
                           radial2=1e-3, radial4=-2.2e-5, tangential_y=1e-3, tangential_x=1e-6,
                           pinwheel1=1e-6, pinwheel2=-2.23 - 8, misalignment=np.array([0, 0, np.pi]),
                           field_of_view=100)

        with self.subTest(misalignment=[0, 0, np.pi]):

            for point in points:
                pin, dist, pix = model.get_projections(point)

                pin_true = -model.focal_length * np.array(point[:2]) / point[2]

                dist_true = model.apply_distortion(pin_true)

                pix_true = (np.matmul(model.intrinsic_matrix[:, :2], dist_true).T + model.intrinsic_matrix[:, 2]).T

                np.testing.assert_array_almost_equal(pin, pin_true)
                np.testing.assert_array_almost_equal(dist, dist_true)
                np.testing.assert_array_almost_equal(pix, pix_true)

        model = self.Class(focal_length=8.7, kx=500, ky=500.5, kxy=1.5, kyx=-1.5, px=1500, py=1500.5,
                           radial2=1e-3, radial4=-2.2e-5, tangential_y=1e-3, tangential_x=1e-6,
                           pinwheel1=1e-6, pinwheel2=-2.23 - 8, misalignment=[np.pi, 0, 0],
                           field_of_view=100)

        with self.subTest(misalignment=[np.pi, 0, 0]):

            for point in points:
                pin, dist, pix = model.get_projections(point)

                pin_true = model.focal_length * np.array(point[:2]) / point[2]
                pin_true[0] *= -1

                dist_true = model.apply_distortion(pin_true)

                pix_true = (np.matmul(model.intrinsic_matrix[:, :2], dist_true).T + model.intrinsic_matrix[:, 2]).T

                np.testing.assert_array_almost_equal(pin, pin_true)
                np.testing.assert_array_almost_equal(dist, dist_true)
                np.testing.assert_array_almost_equal(pix, pix_true)

        model = self.Class(focal_length=8.7, kx=500, ky=500.5, kxy=1.5, kyx=-1.5, px=1500, py=1500.5,
                           radial2=1e-3, radial4=-2.2e-5, tangential_y=1e-3, tangential_x=1e-6,
                           pinwheel1=1e-6, pinwheel2=-2.23 - 8, misalignment=[0, np.pi, 0])

        with self.subTest(misalignment=[0, np.pi, 0]):

            for point in points:
                pin, dist, pix = model.get_projections(point)

                pin_true = model.focal_length * np.array(point[:2]) / point[2]
                pin_true[1] *= -1

                dist_true = model.apply_distortion(pin_true)

                pix_true = (np.matmul(model.intrinsic_matrix[:, :2], dist_true).T + model.intrinsic_matrix[:, 2]).T

                np.testing.assert_array_almost_equal(pin, pin_true)
                np.testing.assert_array_almost_equal(dist, dist_true)
                np.testing.assert_array_almost_equal(pix, pix_true)

        model = self.Class(focal_length=8.7, kx=500, ky=500.5, kxy=1.5, kyx=-1.5, px=1500, py=1500.5,
                           radial2=1e-3, radial4=-2.2e-5, tangential_y=1e-3, tangential_x=1e-6,
                           pinwheel1=1e-6, pinwheel2=-2.23 - 8, misalignment=[1, 0.2, 0.3])

        with self.subTest(misalignment=[1, 0.2, 0.3]):

            rot_mat = at.rotvec_to_rotmat([1, 0.2, 0.3]).squeeze()

            for point in points:
                point_new = rot_mat @ point

                pin, dist, pix = model.get_projections(point)

                pin_true = model.focal_length * np.array(point_new[:2]) / np.array(point_new[2])

                dist_true = model.apply_distortion(pin_true)

                pix_true = (np.matmul(model.intrinsic_matrix[:, :2], dist_true).T + model.intrinsic_matrix[:, 2]).T

                np.testing.assert_array_almost_equal(pin, pin_true)
                np.testing.assert_array_almost_equal(dist, dist_true)
                np.testing.assert_array_almost_equal(pix, pix_true)

        model = self.Class(focal_length=8.7, kx=500, ky=500.5, kxy=1.5, kyx=-1.5, px=1500, py=1500.5,
                           radial2=1e-3, radial4=-2.2e-5, tangential_y=1e-3, tangential_x=1e-6,
                           pinwheel1=1e-6, pinwheel2=-2.23 - 8, misalignment=[[1, 0.2, 0.3], [0, 0, np.pi]])

        model.estimate_multiple_misalignments = True

        with self.subTest(misalignment=[[1, 0.2, 0.3], [0, 0, np.pi]]):

            rot_mat = at.rotvec_to_rotmat([1, 0.2, 0.3]).squeeze()

            for point in points:
                point_new = rot_mat @ point

                pin, dist, pix = model.get_projections(point, image=0)

                pin_true = model.focal_length * np.array(point_new[:2]) / np.array(point_new[2])

                dist_true = model.apply_distortion(pin_true)

                pix_true = (np.matmul(model.intrinsic_matrix[:, :2], dist_true).T + model.intrinsic_matrix[:, 2]).T

                np.testing.assert_array_almost_equal(pin, pin_true)
                np.testing.assert_array_almost_equal(dist, dist_true)
                np.testing.assert_array_almost_equal(pix, pix_true)

                pin, dist, pix = model.get_projections(point, image=1)

                pin_true = -model.focal_length * np.array(point[:2]) / point[2]

                dist_true = model.apply_distortion(pin_true)

                pix_true = (np.matmul(model.intrinsic_matrix[:, :2], dist_true).T + model.intrinsic_matrix[:, 2]).T

                np.testing.assert_array_almost_equal(pin, pin_true)
                np.testing.assert_array_almost_equal(dist, dist_true)
                np.testing.assert_array_almost_equal(pix, pix_true)

    def test_project_onto_image(self):

        points = [[0, 0, 1], [-0.1, 0.2, 2.2], [[-0.1], [0.2], [2.2]], [[-0.1, 0], [0.2, 0], [2.2, 1]]]

        model = self.Class(focal_length=8.7, kx=500, ky=500.5, kxy=1.5, kyx=-1.5, px=1500, py=1500.5,
                           radial2=1e-3, radial4=-2.2e-5, tangential_y=1e-3, tangential_x=1e-6,
                           pinwheel1=1e-6, pinwheel2=-2.23 - 8, a1=1, a2=2, a3=-3, field_of_view=100)

        temps = [0, 1, -1, 10, -10]

        for temp in temps:

            with self.subTest(misalignment=None, temp=temp):

                for point in points:
                    _, __, pix = model.get_projections(point, temperature=temp)

                    pix_proj = model.project_onto_image(point, temperature=temp)

                    np.testing.assert_array_equal(pix, pix_proj)

        model = self.Class(focal_length=8.7, kx=500, ky=500.5, kxy=1.5, kyx=-1.5, px=1500, py=1500.5,
                           radial2=1e-3, radial4=-2.2e-5, tangential_y=1e-3, tangential_x=1e-6,
                           pinwheel1=1e-6, pinwheel2=-2.23 - 8, misalignment=[[1, 0.2, 0.3], [0, 0, np.pi]],
                           field_of_view=100)

        model.estimate_multiple_misalignments = True

        with self.subTest(misalignment=[[1, 0.2, 0.3], [0, 0, np.pi]]):

            for point in points:
                _, __, pix = model.get_projections(point, image=0)

                pix_proj = model.project_onto_image(point, image=0)

                np.testing.assert_array_equal(pix, pix_proj)

                _, __, pix = model.get_projections(point, image=1)

                pix_proj = model.project_onto_image(point, image=1)

                np.testing.assert_array_equal(pix, pix_proj)

    def test_compute_pixel_jacobian(self):

        def num_deriv(uvec, cmodel, delta=1e-8, image=0, temperature=0) -> np.ndarray:

            uvec = np.array(uvec).reshape(3, -1)

            pix_true = cmodel.project_onto_image(uvec, image=image, temperature=temperature)

            uvec_pert = uvec + [[delta], [0], [0]]

            pix_pert_x_f = cmodel.project_onto_image(uvec_pert, image=image, temperature=temperature)

            uvec_pert = uvec + [[0], [delta], [0]]

            pix_pert_y_f = cmodel.project_onto_image(uvec_pert, image=image, temperature=temperature)

            uvec_pert = uvec + [[0], [0], [delta]]

            pix_pert_z_f = cmodel.project_onto_image(uvec_pert, image=image, temperature=temperature)

            uvec_pert = uvec - [[delta], [0], [0]]

            pix_pert_x_b = cmodel.project_onto_image(uvec_pert, image=image, temperature=temperature)

            uvec_pert = uvec - [[0], [delta], [0]]

            pix_pert_y_b = cmodel.project_onto_image(uvec_pert, image=image, temperature=temperature)

            uvec_pert = uvec - [[0], [0], [delta]]

            pix_pert_z_b = cmodel.project_onto_image(uvec_pert, image=image, temperature=temperature)

            return np.array([(pix_pert_x_f-pix_pert_x_b)/(2*delta),
                             (pix_pert_y_f-pix_pert_y_b)/(2*delta),
                             (pix_pert_z_f-pix_pert_z_b)/(2*delta)]).swapaxes(0, -1)

        model_param = {"focal_length": 100.75, "radial2": 1.5e-5, "radial4": 1.5e-5, "tangential_x": 1.5e-5,
                       "tangential_y": 1.5e-5, "pinwheel1": 1.5e-5, "pinwheel2": 1.5e-5, "kx": 30, "ky": 40,
                       "kxy": 0.5, "kyx": -0.8, "px": 4005.23, "py": 4005.23,
                       "misalignment": [[1e-8, 1e-9, 1e-10], [-1e-8, 2e-9, -1e-11], [2e-10, -5e-12, 1e-9]],
                       "a1": 1e-6, "a2": 2e-7, "a3": 3e-8, "field_of_view": 100}

        inputs = [np.array([[0.5, 0, 1]]).T,
                  np.array([[0, 0.5, 1], [0.5, 0.5, 1], [-0.5, 0, 1]]).T,
                  np.array([[0.1, -0.5, 1], [-0.5, -0.5, 1], [5, 10, 1000.23], [1, 2, 1200.23]]).T]

        temperatures = [0, 1, -1, 10.5, -10.5]

        model = self.Class(**model_param)
        model.estimate_multiple_misalignments = True

        for temp in temperatures:

            for input in inputs:

                for image in range(3):

                    with self.subTest(image=image, temp=temp, input=input):

                        jac_ana = model.compute_pixel_jacobian(input, image=image, temperature=temp)

                        jac_num = num_deriv(input, model, image=image, temperature=temp, delta=1e-6)

                        np.testing.assert_allclose(jac_ana, jac_num, rtol=1e-3, atol=1e-10)

    def test__compute_dcamera_point_dgnomic(self):

        def num_deriv(gnomic_locations, cmodel, delta=1e-6) -> np.ndarray:

            def g2u(g):

                v = np.vstack([g, cmodel.focal_length*np.ones(g.shape[-1])])

                return v/np.linalg.norm(v, axis=0, keepdims=True)

            gnomic_locations = np.asarray(gnomic_locations).reshape(2, -1)

            gnom_pert = gnomic_locations + [[delta], [0]]

            cam_loc_pert_x_f = g2u(gnom_pert)

            gnom_pert = gnomic_locations + [[0], [delta]]

            cam_loc_pert_y_f = g2u(gnom_pert)

            gnom_pert = gnomic_locations - [[delta], [0]]

            cam_loc_pert_x_b = g2u(gnom_pert)

            gnom_pert = gnomic_locations - [[0], [delta]]

            cam_loc_pert_y_b = g2u(gnom_pert)

            return np.array([(cam_loc_pert_x_f -cam_loc_pert_x_b)/(2*delta),
                             (cam_loc_pert_y_f -cam_loc_pert_y_b)/(2*delta)]).swapaxes(0, -1)

        model_param = {"focal_length": 100.75, "radial2": 1.5e-8, "radial4": 1.5e-8, "tangential_x": 1.5e-8,
                       "tangential_y": 1.5e-8, "pinwheel1": 1.5e-8, "pinwheel2": 1.5e-8, "kx": 300, "ky": 400,
                       "kxy": 0.5, "kyx": -0.8, "px": 1005.23, "py": 1005.23,
                       "misalignment": [[1e-8, 1e-9, 1e-10], [-1e-8, 2e-9, -1e-11], [2e-10, -5e-12, 1e-9]],
                       "a1": 1e-6, "a2": 2e-7, "a3": 3e-8}

        inputs = [np.array([[0, 0]]).T,
                  np.array([[0, 2000], [2000, 0], [2000, 2000]]).T,
                  np.array([[1000, 1000], [1000, 2000], [2000, 1000], [0, 1000], [1000, 0]]).T]

        model = self.Class(**model_param)
        model.estimate_multiple_misalignments = True

        for input in inputs:

            with self.subTest(input=input):
                jac_ana = []
                for gnom in input.T:
                    jac_ana.append(
                        model._compute_dcamera_point_dgnomic(gnom, np.sqrt(np.sum(gnom*gnom) + model.focal_length**2)))

                jac_ana = np.array(jac_ana)

                jac_num = num_deriv(input, model)

                np.testing.assert_almost_equal(jac_ana, jac_num)

    def test__compute_dgnomic_ddist_gnomic(self):

        def num_deriv(dist_gnomic_locations, cmodel, delta=1e-6) -> np.ndarray:

            def dg2g(dg):

                gnomic_guess = dg.copy()

                # perform the fpa
                for _ in np.arange(20):

                    # get the distorted location assuming the current guess is correct
                    gnomic_guess_distorted = cmodel.apply_distortion(gnomic_guess)

                    # subtract off the residual distortion from the gnomic guess
                    gnomic_guess += dg - gnomic_guess_distorted

                    # check for convergence
                    if np.all(np.linalg.norm(gnomic_guess_distorted - dg, axis=0) <= 1e-15):
                        break

                return gnomic_guess

            dist_gnomic_locations = np.asarray(dist_gnomic_locations).reshape(2, -1)

            dist_gnom_pert = dist_gnomic_locations + [[delta], [0]]

            gnom_loc_pert_x_f = dg2g(dist_gnom_pert)

            dist_gnom_pert = dist_gnomic_locations + [[0], [delta]]

            gnom_loc_pert_y_f = dg2g(dist_gnom_pert)

            dist_gnom_pert = dist_gnomic_locations - [[delta], [0]]

            gnom_loc_pert_x_b = dg2g(dist_gnom_pert)

            dist_gnom_pert = dist_gnomic_locations - [[0], [delta]]

            gnom_loc_pert_y_b = dg2g(dist_gnom_pert)

            return np.array([(gnom_loc_pert_x_f - gnom_loc_pert_x_b)/(2*delta),
                             (gnom_loc_pert_y_f - gnom_loc_pert_y_b)/(2*delta)]).swapaxes(0, -1)

        model_param = {"focal_length": 100.75, "radial2": 1.5e-8, "radial4": 1.5e-8, "tangential_x": 1.5e-8,
                       "tangential_y": 1.5e-8, "pinwheel1": 1.5e-8, "pinwheel2": 1.5e-8, "kx": 300, "ky": 400,
                       "kxy": 0.5, "kyx": -0.8, "px": 1005.23, "py": 1005.23,
                       "misalignment": [[1e-8, 1e-9, 1e-10], [-1e-8, 2e-9, -1e-11], [2e-10, -5e-12, 1e-9]],
                       "a1": 1e-6, "a2": 2e-7, "a3": 3e-8, "field_of_view": 100}

        inputs = [np.array([[0, 0]]).T,
                  np.array([[0, 0.1], [0.1, 0], [0.1, 0.1]]).T,
                  np.array([[-0.1, 0], [0, -0.1], [-0.1, -0.1], [0.1, -0.1], [-0.1, 0.1]]).T]

        model = self.Class(**model_param)
        model.estimate_multiple_misalignments = True

        for input in inputs:

            with self.subTest(input=input):
                jac_ana = []
                for dist_gnom in input.T:
                    jac_ana.append(model._compute_dgnomic_ddist_gnomic(dist_gnom))

                jac_ana = np.array(jac_ana)

                jac_num = num_deriv(input, model)

                np.testing.assert_almost_equal(jac_ana, jac_num)

    def test_compute_unit_vector_jacobian(self):

        def num_deriv(pixels, cmodel, delta=1e-6, image=0, temperature=0) -> np.ndarray:

            pixels = np.array(pixels).reshape(2, -1)

            pix_pert = pixels + [[delta], [0]]

            uvec_pert_x_f = cmodel.pixels_to_unit(pix_pert, image=image, temperature=temperature)

            pix_pert = pixels + [[0], [delta]]

            uvec_pert_y_f = cmodel.pixels_to_unit(pix_pert, image=image, temperature=temperature)

            pix_pert = pixels - [[delta], [0]]

            uvec_pert_x_b = cmodel.pixels_to_unit(pix_pert, image=image, temperature=temperature)

            pix_pert = pixels - [[0], [delta]]

            uvec_pert_y_b = cmodel.pixels_to_unit(pix_pert, image=image, temperature=temperature)

            return np.array([(uvec_pert_x_f-uvec_pert_x_b)/(2*delta),
                             (uvec_pert_y_f-uvec_pert_y_b)/(2*delta)]).swapaxes(0, -1)

        model_param = {"focal_length": 100.75, "radial2": 1.5e-8, "radial4": 1.5e-8, "tangential_x": 1.5e-8,
                       "tangential_y": 1.5e-8, "pinwheel1": 1.5e-8, "pinwheel2": 1.5e-8, "kx": 300, "ky": 400,
                       "kxy": 0.5, "kyx": -0.8, "px": 1005.23, "py": 1005.23,
                       "misalignment": [[1e-8, 1e-9, 1e-10], [-1e-8, 2e-9, -1e-11], [2e-10, -5e-12, 1e-9]],
                       "a1": 1e-6, "a2": 2e-7, "a3": 3e-8}

        inputs = [np.array([[0, 0]]).T,
                  np.array([[0, 2000], [2000, 0], [2000, 2000]]).T,
                  np.array([[1000, 1000], [1000, 2000], [2000, 1000], [0, 1000], [1000, 0]]).T]

        temperatures = [0, 1, -1, 10.5, -10.5]

        model = self.Class(**model_param)
        model.estimate_multiple_misalignments = True

        for temp in temperatures:

            for input in inputs:

                for image in range(3):

                    with self.subTest(image=image, temp=temp, input=input):

                        jac_ana = model.compute_unit_vector_jacobian(input, image=image, temperature=temp)

                        jac_num = num_deriv(input, model, image=image, temperature=temp, delta=1e-2)

                        np.testing.assert_allclose(jac_ana, jac_num, rtol=1e-3, atol=1e-10)

    def test__compute_ddistortion_dgnomic(self):

        def num_deriv(loc, cmodel, delta=1e-8) -> np.ndarray:
            loc_pert = np.array(loc) + [delta, 0]
            dist_pert_x_f = cmodel.apply_distortion(loc_pert) - loc_pert
            loc_pert = np.array(loc) + [0, delta]
            dist_pert_y_f = cmodel.apply_distortion(loc_pert) - loc_pert

            loc_pert = np.array(loc) - [delta, 0]
            dist_pert_x_b = cmodel.apply_distortion(loc_pert) - loc_pert
            loc_pert = np.array(loc) - [0, delta]
            dist_pert_y_b = cmodel.apply_distortion(loc_pert) - loc_pert

            return np.array(
                [(dist_pert_x_f - dist_pert_x_b) / (2 * delta), (dist_pert_y_f - dist_pert_y_b) / (2 * delta)]).T

        dist_coefs = [{"radial2": 1.5, "radial4": 0, "tangential_x": 0, "tangential_y": 0,
                       "pinwheel1": 0, "pinwheel2": 0},
                      {"radial2": 0, "radial4": 1.5, "tangential_x": 0, "tangential_y": 0,
                       "pinwheel1": 0, "pinwheel2": 0},
                      {"radial2": 0, "radial4": 0, "tangential_x": 1.5, "tangential_y": 0,
                       "pinwheel1": 0, "pinwheel2": 0},
                      {"radial2": 0, "radial4": 0, "tangential_x": 0, "tangential_y": 1.5,
                       "pinwheel1": 0, "pinwheel2": 0},
                      {"radial2": 0, "radial4": 0, "tangential_x": 0, "tangential_y": 0,
                       "pinwheel1": 1.5, "pinwheel2": 0},
                      {"radial2": 0, "radial4": 0, "tangential_x": 0, "tangential_y": 0,
                       "pinwheel1": 0, "pinwheel2": 1.5},
                      {"e1": -1.5, "e2": -1.5, "e3": -1.5, "e4": -1.5, "e5": -1.5, "e6": -1.5}]

        inputs = [[1e-6, 1e-6], [1, 0], [-1, 0], [1.5, 0], [-1.5, 0], [0, 1], [0, -1], [0, 1.5], [0, -1.5], [1, 1]]

        for dist_coef in dist_coefs:

            model = self.Class(**dist_coef)

            with self.subTest(**dist_coef):

                for inp in inputs:
                    r = np.sqrt(inp[0] ** 2 + inp[1] ** 2)
                    r2 = r ** 2
                    r3 = r ** 3
                    r4 = r ** 4

                    num = num_deriv(inp, model)

                    ana = model._compute_ddistortion_dgnomic(np.array(inp), r, r2, r3, r4)

                    np.testing.assert_allclose(num, ana, atol=1e-10)

    def test__compute_dpixel_ddistorted_gnomic(self):

        def num_deriv(loc, cmodel, delta=1e-8, temperature=0) -> np.ndarray:
            loc_pert = np.array(loc) + [delta, 0]
            loc_pert *= cmodel.get_temperature_scale(temperature)
            pix_pert_x_f = cmodel.intrinsic_matrix[:, :2] @ loc_pert + cmodel.intrinsic_matrix[:, 2]
            loc_pert = np.array(loc) + [0, delta]
            loc_pert *= cmodel.get_temperature_scale(temperature)
            pix_pert_y_f = cmodel.intrinsic_matrix[:, :2] @ loc_pert + cmodel.intrinsic_matrix[:, 2]

            loc_pert = np.array(loc) - [delta, 0]
            loc_pert *= cmodel.get_temperature_scale(temperature)
            pix_pert_x_b = cmodel.intrinsic_matrix[:, :2] @ loc_pert + cmodel.intrinsic_matrix[:, 2]
            loc_pert = np.array(loc) - [0, delta]
            loc_pert *= cmodel.get_temperature_scale(temperature)
            pix_pert_y_b = cmodel.intrinsic_matrix[:, :2] @ loc_pert + cmodel.intrinsic_matrix[:, 2]

            return np.array(
                [(pix_pert_x_f - pix_pert_x_b) / (2 * delta), (pix_pert_y_f - pix_pert_y_b) / (2 * delta)]).T

        intrins_coefs = [{"kx": 1.5, "kxy": 0, "ky": 0, "kyx": 0, "px": 0, "py": 0},
                         {"kx": 0, "kxy": 1.5, "ky": 0, "kyx": 0, "px": 0, "py": 0},
                         {"kx": 0, "kxy": 0, "ky": 1.5, "kyx": 0, "px": 0, "py": 0},
                         {"kx": 0, "kxy": 0, "ky": 0, "kyx": 1.5, "px": 0, "py": 0},
                         {"kx": 0, "kxy": 0, "ky": 0, "kyx": 0, "px": 1.5, "py": 0},
                         {"kx": 0, "kxy": 0, "ky": 0, "kyx": 0, "px": 0, "py": 1.5},
                         {"kx": 0, "kxy": 0, "ky": 0, "kyx": 0, "px": 0, "py": 0, "a1": 1.5, "a2": 0, "a3": 0},
                         {"kx": 0, "kxy": 0, "ky": 0, "kyx": 0, "px": 0, "py": 0, "a1": 0, "a2": 1.5, "a3": 0},
                         {"kx": 0, "kxy": 0, "ky": 0, "kyx": 0, "px": 0, "py": 0, "a1": 0, "a2": 0, "a3": 1.5},
                         {"kx": 1.5, "kxy": 1.5, "ky": 1.5, "kyx": 1.5, "px": 1.5, "py": 1.5,
                          "a1": 1.5, "a2": 1.5, "a3": 1.5}]

        inputs = [[1e-6, 1e-6], [1, 0], [-1, 0], [1.5, 0], [-1.5, 0], [0, 1], [0, -1], [0, 1.5], [0, -1.5], [1, 1]]

        temps = [0, 1, -1, 10.5, -10.5]

        for temp in temps:
            for intrins_coef in intrins_coefs:

                model = self.Class(**intrins_coef)

                with self.subTest(**intrins_coef, temp=temp):

                    for inp in inputs:
                        num = num_deriv(inp, model, temperature=temp)

                        ana = model._compute_dpixel_ddistorted_gnomic(temperature=temp)

                        np.testing.assert_allclose(num, ana, atol=1e-10)

    def test__compute_dpixel_dintrinsic(self):
        def num_deriv(loc, cmodel, delta=1e-6) -> np.ndarray:

            model_pert = cmodel.copy()
            model_pert.kx += delta
            pix_pert_kx_f = model_pert.intrinsic_matrix[:, :2] @ loc + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.kxy += delta
            pix_pert_kxy_f = model_pert.intrinsic_matrix[:, :2] @ loc + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.kyx += delta
            pix_pert_kyx_f = model_pert.intrinsic_matrix[:, :2] @ loc + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.ky += delta
            pix_pert_ky_f = model_pert.intrinsic_matrix[:, :2] @ loc + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.px += delta
            pix_pert_px_f = model_pert.intrinsic_matrix[:, :2] @ loc + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.py += delta
            pix_pert_py_f = model_pert.intrinsic_matrix[:, :2] @ loc + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.kx -= delta
            pix_pert_kx_b = model_pert.intrinsic_matrix[:, :2] @ loc + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.kxy -= delta
            pix_pert_kxy_b = model_pert.intrinsic_matrix[:, :2] @ loc + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.kyx -= delta
            pix_pert_kyx_b = model_pert.intrinsic_matrix[:, :2] @ loc + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.ky -= delta
            pix_pert_ky_b = model_pert.intrinsic_matrix[:, :2] @ loc + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.px -= delta
            pix_pert_px_b = model_pert.intrinsic_matrix[:, :2] @ loc + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.py -= delta
            pix_pert_py_b = model_pert.intrinsic_matrix[:, :2] @ loc + model_pert.intrinsic_matrix[:, 2]

            return np.array([(pix_pert_kx_f - pix_pert_kx_b) / (2 * delta),
                             (pix_pert_kxy_f - pix_pert_kxy_b) / (2 * delta),
                             (pix_pert_kyx_f - pix_pert_kyx_b) / (2 * delta),
                             (pix_pert_ky_f - pix_pert_ky_b) / (2 * delta),
                             (pix_pert_px_f - pix_pert_px_b) / (2 * delta),
                             (pix_pert_py_f - pix_pert_py_b) / (2 * delta)]).T

        intrins_coefs = [{"kx": 1.5, "kxy": 0, "ky": 0, "kyx": 0, "px": 0, "py": 0},
                         {"kx": 0, "kxy": 1.5, "ky": 0, "kyx": 0, "px": 0, "py": 0},
                         {"kx": 0, "kxy": 0, "ky": 1.5, "kyx": 0, "px": 0, "py": 0},
                         {"kx": 0, "kxy": 0, "ky": 0, "kyx": 1.5, "px": 0, "py": 0},
                         {"kx": 0, "kxy": 0, "ky": 0, "kyx": 0, "px": 1.5, "py": 0},
                         {"kx": 0, "kxy": 0, "ky": 0, "kyx": 0, "px": 0, "py": 1.5},
                         {"kx": 1.5, "kxy": 1.5, "ky": 1.5, "kyx": 1.5, "px": 1.5, "py": 1.5}]

        inputs = [[1e-6, 1e-6], [1, 0], [-1, 0], [1.5, 0], [-1.5, 0], [0, 1], [0, -1], [0, 1.5], [0, -1.5], [1, 1]]

        for intrins_coef in intrins_coefs:

            model = self.Class(**intrins_coef)

            with self.subTest(**intrins_coef):

                for inp in inputs:
                    num = num_deriv(inp, model)

                    ana = model._compute_dpixel_dintrinsic(np.array(inp))

                    np.testing.assert_allclose(num, ana, atol=1e-10)

    def test__compute_ddistorted_gnomic_ddistortion(self):

        def num_deriv(loc, cmodel, delta=1e-8) -> np.ndarray:

            model_pert = cmodel.copy()
            model_pert.radial2 += delta
            loc_pert_r2_f = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.radial4 += delta
            loc_pert_r4_f = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.tangential_y += delta
            loc_pert_ty_f = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.tangential_x += delta
            loc_pert_tx_f = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.pinwheel1 += delta
            loc_pert_p1_f = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.pinwheel2 += delta
            loc_pert_p2_f = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.radial2 -= delta
            loc_pert_r2_b = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.radial4 -= delta
            loc_pert_r4_b = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.tangential_y -= delta
            loc_pert_ty_b = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.tangential_x -= delta
            loc_pert_tx_b = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.pinwheel1 -= delta
            loc_pert_p1_b = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.pinwheel2 -= delta
            loc_pert_p2_b = model_pert.apply_distortion(loc)

            return np.array([(loc_pert_r2_f - loc_pert_r2_b) / (2 * delta),
                             (loc_pert_r4_f - loc_pert_r4_b) / (2 * delta),
                             (loc_pert_ty_f - loc_pert_ty_b) / (2 * delta),
                             (loc_pert_tx_f - loc_pert_tx_b) / (2 * delta),
                             (loc_pert_p1_f - loc_pert_p1_b) / (2 * delta),
                             (loc_pert_p2_f - loc_pert_p2_b) / (2 * delta)]).T

        dist_coefs = [{"radial2": 1.5, "radial4": 0, "tangential_x": 0, "tangential_y": 0,
                       "pinwheel1": 0, "pinwheel2": 0},
                      {"radial2": 0, "radial4": 1.5, "tangential_x": 0, "tangential_y": 0,
                       "pinwheel1": 0, "pinwheel2": 0},
                      {"radial2": 0, "radial4": 0, "tangential_x": 1.5, "tangential_y": 0,
                       "pinwheel1": 0, "pinwheel2": 0},
                      {"radial2": 0, "radial4": 0, "tangential_x": 0, "tangential_y": 1.5,
                       "pinwheel1": 0, "pinwheel2": 0},
                      {"radial2": 0, "radial4": 0, "tangential_x": 0, "tangential_y": 0,
                       "pinwheel1": 1.5, "pinwheel2": 0},
                      {"radial2": 0, "radial4": 0, "tangential_x": 0, "tangential_y": 0,
                       "pinwheel1": 0, "pinwheel2": 1.5}]

        inputs = [[1e-6, 1e-6], [1, 0], [-1, 0], [1.5, 0], [-1.5, 0], [0, 1], [0, -1], [0, 1.5], [0, -1.5], [1, 1]]

        for dist_coef in dist_coefs:

            model = self.Class(**dist_coef)

            with self.subTest(**dist_coef):

                for inp in inputs:
                    r = np.sqrt(inp[0] ** 2 + inp[1] ** 2)
                    r2 = r ** 2
                    r3 = r ** 3
                    r4 = r ** 4

                    num = num_deriv(inp, model)

                    ana = model._compute_ddistorted_gnomic_ddistortion(np.array(inp), r, r2, r3, r4)

                    np.testing.assert_allclose(num, ana, atol=1e-10)

    def test_get_jacobian_row(self):

        def num_deriv(loc, cmodel, delta=1e-8, image=0, temperature=0) -> np.ndarray:
            model_pert = cmodel.copy()
            model_pert.focal_length += delta
            pix_pert_f_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.focal_length -= delta
            pix_pert_f_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.kx += delta
            pix_pert_kx_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.kxy += delta
            pix_pert_kxy_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.kyx += delta
            pix_pert_kyx_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.ky += delta
            pix_pert_ky_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.px += delta
            pix_pert_px_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.py += delta
            pix_pert_py_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.kx -= delta
            pix_pert_kx_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.kxy -= delta
            pix_pert_kxy_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.kyx -= delta
            pix_pert_kyx_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.ky -= delta
            pix_pert_ky_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.px -= delta
            pix_pert_px_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.py -= delta
            pix_pert_py_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.radial2 += delta
            pix_pert_r2_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.radial4 += delta
            pix_pert_r4_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.tangential_y += delta
            pix_pert_ty_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.tangential_x += delta
            pix_pert_tx_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.pinwheel1 += delta
            pix_pert_p1_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.pinwheel2 += delta
            pix_pert_p2_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.radial2 -= delta
            pix_pert_r2_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.radial4 -= delta
            pix_pert_r4_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.tangential_y -= delta
            pix_pert_ty_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.tangential_x -= delta
            pix_pert_tx_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.pinwheel1 -= delta
            pix_pert_p1_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.pinwheel2 -= delta
            pix_pert_p2_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.a1 += delta
            pix_pert_a1_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.a2 += delta
            pix_pert_a2_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.a3 += delta
            pix_pert_a3_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.a1 -= delta
            pix_pert_a1_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.a2 -= delta
            pix_pert_a2_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.a3 -= delta
            pix_pert_a3_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            delta_m = 1e-6
            model_pert = cmodel.copy()
            model_pert.misalignment[image][0] += delta_m
            pix_pert_mx_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][1] += delta_m
            pix_pert_my_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][2] += delta_m
            pix_pert_mz_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][0] -= delta_m
            pix_pert_mx_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][1] -= delta_m
            pix_pert_my_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][2] -= delta_m
            pix_pert_mz_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            return np.vstack([(pix_pert_f_f - pix_pert_f_b) / (delta * 2),
                              (pix_pert_kx_f - pix_pert_kx_b) / (delta * 2),
                              (pix_pert_kxy_f - pix_pert_kxy_b) / (delta * 2),
                              (pix_pert_kyx_f - pix_pert_kyx_b) / (delta * 2),
                              (pix_pert_ky_f - pix_pert_ky_b) / (delta * 2),
                              (pix_pert_px_f - pix_pert_px_b) / (delta * 2),
                              (pix_pert_py_f - pix_pert_py_b) / (delta * 2),
                              (pix_pert_r2_f - pix_pert_r2_b) / (delta * 2),
                              (pix_pert_r4_f - pix_pert_r4_b) / (delta * 2),
                              (pix_pert_ty_f - pix_pert_ty_b) / (delta * 2),
                              (pix_pert_tx_f - pix_pert_tx_b) / (delta * 2),
                              (pix_pert_p1_f - pix_pert_p1_b) / (delta * 2),
                              (pix_pert_p2_f - pix_pert_p2_b) / (delta * 2),
                              (pix_pert_a1_f - pix_pert_a1_b) / (delta * 2),
                              (pix_pert_a2_f - pix_pert_a2_b) / (delta * 2),
                              (pix_pert_a3_f - pix_pert_a3_b) / (delta * 2),
                              np.zeros((image * 3, 2)),
                              (pix_pert_mx_f - pix_pert_mx_b) / (delta_m * 2),
                              (pix_pert_my_f - pix_pert_my_b) / (delta_m * 2),
                              (pix_pert_mz_f - pix_pert_mz_b) / (delta_m * 2)]).T

        model_param = {"focal_length": 100.75, "radial2": 1.5e-5, "radial4": 1.5e-5, "tangential_x": 1.5e-5,
                       "tangential_y": 1.5e-5, "pinwheel1": 1.5e-5, "pinwheel2": 1.5e-5, "kx": 30, "ky": 40,
                       "kxy": 0.5, "kyx": -0.8, "px": 4005.23, "py": 4005.23,
                       "misalignment": [[1e-8, 1e-9, 1e-10], [-1e-8, 2e-9, -1e-11]],
                       "a1": 1e-6, "a2": 2e-7, "a3": 3e-8, "field_of_view": 100}

        inputs = [[0.1, 0, 1], [0, 0.1, 1], [0.1, 0.1, 1], [-0.1, 0, 1], [0, -0.1, 1], [-0.1, -0.1, 1],
                  [5, 10, 1000.23], [[1], [2], [1200.23]]]

        temps = [0, 1, -1, 10.5, -10.5]

        model = self.Class(**model_param)
        model.estimate_multiple_misalignments = True

        for temp in temps:
            for inp in inputs:
                with self.subTest(temp=temp, inp=inp):
                    num = num_deriv(inp, model, delta=1e-3, temperature=temp)
                    ana = model._get_jacobian_row(np.array(inp), 0, 1, temperature=temp)

                    np.testing.assert_allclose(ana, num, rtol=1e-3, atol=1e-10)

                    num = num_deriv(inp, model, delta=1e-3, image=1, temperature=temp)
                    ana = model._get_jacobian_row(np.array(inp), 1, 2, temperature=temp)

                    np.testing.assert_allclose(ana, num, atol=1e-10, rtol=1e-3)

    def test_compute_jacobian(self):

        def num_deriv(loc, cmodel, delta=1e-8, image=0, nimages=1, temperature=0) -> np.ndarray:

            model_pert = cmodel.copy()
            model_pert.focal_length += delta
            pix_pert_f_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.focal_length -= delta
            pix_pert_f_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.kx += delta
            pix_pert_kx_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.kxy += delta
            pix_pert_kxy_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.kyx += delta
            pix_pert_kyx_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.ky += delta
            pix_pert_ky_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.px += delta
            pix_pert_px_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.py += delta
            pix_pert_py_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.kx -= delta
            pix_pert_kx_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.kxy -= delta
            pix_pert_kxy_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.kyx -= delta
            pix_pert_kyx_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.ky -= delta
            pix_pert_ky_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.px -= delta
            pix_pert_px_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.py -= delta
            pix_pert_py_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.radial2 += delta
            pix_pert_r2_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.radial4 += delta
            pix_pert_r4_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.tangential_y += delta
            pix_pert_ty_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.tangential_x += delta
            pix_pert_tx_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.pinwheel1 += delta
            pix_pert_p1_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.pinwheel2 += delta
            pix_pert_p2_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.radial2 -= delta
            pix_pert_r2_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.radial4 -= delta
            pix_pert_r4_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.tangential_y -= delta
            pix_pert_ty_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.tangential_x -= delta
            pix_pert_tx_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.pinwheel1 -= delta
            pix_pert_p1_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.pinwheel2 -= delta
            pix_pert_p2_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.a1 += delta
            pix_pert_a1_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.a2 += delta
            pix_pert_a2_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.a3 += delta
            pix_pert_a3_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.a1 -= delta
            pix_pert_a1_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.a2 -= delta
            pix_pert_a2_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.a3 -= delta
            pix_pert_a3_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][0] += delta
            pix_pert_mx_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][1] += delta
            pix_pert_my_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][2] += delta
            pix_pert_mz_f = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][0] -= delta
            pix_pert_mx_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][1] -= delta
            pix_pert_my_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][2] -= delta
            pix_pert_mz_b = model_pert.project_onto_image(loc, image=image, temperature=temperature).ravel()

            return np.vstack([(pix_pert_f_f - pix_pert_f_b) / (delta * 2),
                              (pix_pert_kx_f - pix_pert_kx_b) / (delta * 2),
                              (pix_pert_kxy_f - pix_pert_kxy_b) / (delta * 2),
                              (pix_pert_kyx_f - pix_pert_kyx_b) / (delta * 2),
                              (pix_pert_ky_f - pix_pert_ky_b) / (delta * 2),
                              (pix_pert_px_f - pix_pert_px_b) / (delta * 2),
                              (pix_pert_py_f - pix_pert_py_b) / (delta * 2),
                              (pix_pert_r2_f - pix_pert_r2_b) / (delta * 2),
                              (pix_pert_r4_f - pix_pert_r4_b) / (delta * 2),
                              (pix_pert_ty_f - pix_pert_ty_b) / (delta * 2),
                              (pix_pert_tx_f - pix_pert_tx_b) / (delta * 2),
                              (pix_pert_p1_f - pix_pert_p1_b) / (delta * 2),
                              (pix_pert_p2_f - pix_pert_p2_b) / (delta * 2),
                              (pix_pert_a1_f - pix_pert_a1_b) / (delta * 2),
                              (pix_pert_a2_f - pix_pert_a2_b) / (delta * 2),
                              (pix_pert_a3_f - pix_pert_a3_b) / (delta * 2),
                              np.zeros((image * 3, 2)),
                              (pix_pert_mx_f - pix_pert_mx_b) / (delta * 2),
                              (pix_pert_my_f - pix_pert_my_b) / (delta * 2),
                              (pix_pert_mz_f - pix_pert_mz_b) / (delta * 2),
                              np.zeros(((nimages - image - 1) * 3, 2))]).T

        model_param = {"focal_length": 100.75, "radial2": 1.5e-5, "radial4": 1.5e-5, "tangential_x": 1.5e-5,
                       "tangential_y": 1.5e-5, "pinwheel1": 1.5e-5, "pinwheel2": 1.5e-5, "kx": 30, "ky": 40,
                       "kxy": 0.5, "kyx": -0.8, "px": 4005.23, "py": 4005.23,
                       "misalignment": [[1e-8, 1e-9, 1e-10], [-1e-8, 2e-9, -1e-11], [2e-10, -5e-12, 1e-9]],
                       "a1": 1e-6, "a2": 2e-7, "a3": 3e-8, "field_of_view": 100}

        inputs = [np.array([[0.5, 0, 1]]).T,
                  np.array([[0, 0.5, 1], [0.5, 0.5, 1], [-0.5, 0, 1]]).T,
                  np.array([[0.1, -0.5, 1], [-0.5, -0.5, 1], [5, 10, 1000.23], [1, 2, 1200.23]]).T]

        temperatures = [0, 1, -1, 10.5, -10.5, [1, -10, 10]]

        model = self.Class(**model_param, estimation_parameters=['intrinsic', 'temperature dependence',
                                                                 'multiple misalignments'])

        for temp in temperatures:

            with self.subTest(temp=temp):
                model.use_a_priori = False

                jac_ana = model.compute_jacobian(inputs, temperature=temp)

                jac_num = []

                numim = len(inputs)

                for ind, inp in enumerate(inputs):
                    if isinstance(temp, list):
                        templ = temp[ind]
                    else:
                        templ = temp

                    for vec in inp.T:
                        jac_num.append(num_deriv(vec.T, model, delta=1e-4, image=ind, nimages=numim, temperature=templ))

                np.testing.assert_allclose(jac_ana, np.vstack(jac_num), rtol=1e-3, atol=1e-10)

                model.use_a_priori = True

                jac_ana = model.compute_jacobian(inputs, temperature=temp)

                jac_num = []

                numim = len(inputs)

                for ind, inp in enumerate(inputs):
                    if isinstance(temp, list):
                        templ = temp[ind]
                    else:
                        templ = temp

                    for vec in inp.T:
                        jac_num.append(num_deriv(vec.T, model, delta=1e-4, image=ind, nimages=numim, temperature=templ))

                jac_num = np.vstack(jac_num)

                jac_num = np.pad(jac_num, [(0, jac_num.shape[1]), (0, 0)], 'constant', constant_values=0)

                jac_num[-jac_num.shape[1]:] = np.eye(jac_num.shape[1])

                np.testing.assert_allclose(jac_ana, jac_num, rtol=1e-3, atol=1e-10)

    def test_apply_update(self):
        model_param = {"focal_length": 0, "radial2": 0, "radial4": 0, "tangential_x": 0,
                       "tangential_y": 0, "pinwheel1": 0, "pinwheel2": 0, "kx": 0, "ky": 0,
                       "kxy": 0, "kyx": 0, "px": 0,
                       "misalignment": [[0, 0, 0], [0, 0, 0]],
                       "a1": 0, "a2": 0, "a3": 0}

        model = self.Class(**model_param, estimation_parameters=['intrinsic', "temperature dependence",
                                                                 'multiple misalignments'])

        update_vec = np.arange(22).astype(np.float64)

        model.apply_update(update_vec)

        keys = list(model_param.keys())

        keys.remove('misalignment')

        for key in keys:
            self.assertEqual(getattr(model, key), update_vec[model.element_dict[key][0]])

        for ind, vec in enumerate(update_vec[16:].reshape(-1, 3)):
            np.testing.assert_array_almost_equal(at.Rotation(vec).quaternion, at.Rotation(model.misalignment[ind]).quaternion)

    def test_pixels_to_gnomic(self):

        intrins_param = {"kx": 3000, "ky": 4000, "kxy": 0.5, "kyx": -0.8, "px": 4005.23, 'py': 2000.33,
                         'a1': 1e-6, 'a2': 1e-7, 'a3': 1e-8}

        dist_coefs = [{"radial2": 1.5e-3, "radial4": 0, "tangential_x": 0, "tangential_y": 0,
                       "pinwheel1": 0, "pinwheel2": 0},
                      {"radial2": 0, "radial4": 1.5e-3, "tangential_x": 0, "tangential_y": 0,
                       "pinwheel1": 0, "pinwheel2": 0},
                      {"radial2": 0, "radial4": 0, "tangential_x": 1.5e-3, "tangential_y": 0,
                       "pinwheel1": 0, "pinwheel2": 0},
                      {"radial2": 0, "radial4": 0, "tangential_x": 0, "tangential_y": 1.5e-3,
                       "pinwheel1": 0, "pinwheel2": 0},
                      {"radial2": 0, "radial4": 0, "tangential_x": 0, "tangential_y": 0,
                       "pinwheel1": 1.5e-3, "pinwheel2": 0},
                      {"radial2": 0, "radial4": 0, "tangential_x": 0, "tangential_y": 0,
                       "pinwheel1": 0, "pinwheel2": 1.5e-3},
                      {"radial2": 1.5e-3, "radial4": 1.5e-3, "tangential_x": 1.5e-3, "tangential_y": 1.5e-3,
                       "pinwheel1": 1.5e-3, "pinwheel2": 1.5e-3}]
        pinhole = [[0, 0], [1, 0], [-1, 0], [1.5, 0], [-1.5, 0], [[1.5], [0]], [[1.5, -1], [0, 0]],
                   [0, 1], [0, -1], [0, 1.5], [0, -1.5], [[0], [1.5]], [[0, 0], [1.5, -1]], [1, 1]]

        temperatures = [0, 1, -1, 10.5, -10.5]

        for temp in temperatures:
            for dist in dist_coefs:

                model = self.Class(**dist, **intrins_param)

                for gnoms in pinhole:
                    with self.subTest(**dist, temp=temp, gnoms=gnoms):
                        mm_dist = model.apply_distortion(np.array(gnoms))

                        mm_dist *= model.get_temperature_scale(temp)

                        pix_dist = ((model.intrinsic_matrix[:, :2] @ mm_dist).T + model.intrinsic_matrix[:, 2]).T

                        mm_undist = model.pixels_to_gnomic(pix_dist, temperature=temp)

                        np.testing.assert_allclose(mm_undist, gnoms, atol=1e-13)

    def test_undistort_pixels(self):

        intrins_param = {"kx": 3000, "ky": 4000, "kxy": 0.5, "kyx": -0.8, "px": 4005.23, 'py': 2000.33,
                         "a1": 1e-3, "a2": 1e-4, "a3": 1e-5}

        dist_coefs = [{"radial2": 1.5e-3, "radial4": 0, "tangential_x": 0, "tangential_y": 0,
                       "pinwheel1": 0, "pinwheel2": 0},
                      {"radial2": 0, "radial4": 1.5e-3, "tangential_x": 0, "tangential_y": 0,
                       "pinwheel1": 0, "pinwheel2": 0},
                      {"radial2": 0, "radial4": 0, "tangential_x": 1.5e-3, "tangential_y": 0,
                       "pinwheel1": 0, "pinwheel2": 0},
                      {"radial2": 0, "radial4": 0, "tangential_x": 0, "tangential_y": 1.5e-3,
                       "pinwheel1": 0, "pinwheel2": 0},
                      {"radial2": 0, "radial4": 0, "tangential_x": 0, "tangential_y": 0,
                       "pinwheel1": 1.5e-3, "pinwheel2": 0},
                      {"radial2": 0, "radial4": 0, "tangential_x": 0, "tangential_y": 0,
                       "pinwheel1": 0, "pinwheel2": 1.5e-3},
                      {"radial2": 1.5e-3, "radial4": 1.5e-3, "tangential_x": 1.5e-3, "tangential_y": 1.5e-3,
                       "pinwheel1": 1.5e-3, "pinwheel2": 1.5e-3}]

        pinhole = [[0, 0], [1, 0], [-1, 0], [1.5, 0], [-1.5, 0], [[1.5], [0]], [[1.5, -1], [0, 0]],
                   [0, 1], [0, -1], [0, 1.5], [0, -1.5], [[0], [1.5]], [[0, 0], [1.5, -1]], [1, 1]]

        temperatures = [0, 1, -1, 10.5, -10.5]

        for temp in temperatures:
            for dist in dist_coefs:

                model = self.Class(**dist, **intrins_param)

                for gnoms in pinhole:
                    with self.subTest(**dist, temp=temp, gnoms=gnoms):
                        gnoms = np.array(gnoms).astype(np.float64)

                        mm_dist = model.apply_distortion(np.array(gnoms))

                        mm_dist *= model.get_temperature_scale(temp)

                        pix_dist = ((model.intrinsic_matrix[:, :2] @ mm_dist).T + model.intrinsic_matrix[:, 2]).T

                        pix_undist = model.undistort_pixels(pix_dist, temperature=temp)

                        gnoms *= model.get_temperature_scale(temp)

                        pix_pinhole = ((model.intrinsic_matrix[:, :2] @ gnoms).T + model.intrinsic_matrix[:, 2]).T

                        np.testing.assert_allclose(pix_undist, pix_pinhole, atol=1e-13)

    def test_pixels_to_unit(self):
        intrins_param = {"focal_length": 32.7, "kx": 3000, "ky": 4000, "kxy": 0.5, "kyx": -0.8,
                         "px": 4005.23, 'py': 2000.33, "a1": 1e-6, "a2": 1e-7, "a3": -3e-5}

        dist_coefs = [{"radial2": 1.5e-3, "radial4": 0, "tangential_x": 0, "tangential_y": 0,
                       "pinwheel1": 0, "pinwheel2": 0},
                      {"radial2": 0, "radial4": 1.5e-3, "tangential_x": 0, "tangential_y": 0,
                       "pinwheel1": 0, "pinwheel2": 0},
                      {"radial2": 0, "radial4": 0, "tangential_x": 1.5e-3, "tangential_y": 0,
                       "pinwheel1": 0, "pinwheel2": 0},
                      {"radial2": 0, "radial4": 0, "tangential_x": 0, "tangential_y": 1.5e-3,
                       "pinwheel1": 0, "pinwheel2": 0},
                      {"radial2": 0, "radial4": 0, "tangential_x": 0, "tangential_y": 0,
                       "pinwheel1": 1.5e-3, "pinwheel2": 0},
                      {"radial2": 0, "radial4": 0, "tangential_x": 0, "tangential_y": 0,
                       "pinwheel1": 0, "pinwheel2": 1.5e-3},
                      {"radial2": 1.5e-3, "radial4": 1.5e-3, "tangential_x": 1.5e-3, "tangential_y": 1.5e-3,
                       "pinwheel1": 1.5e-3, "pinwheel2": 1.5e-3},
                      {"misalignment": np.array([1e-11, 2e-12, -1e-10])},
                      {"misalignment": np.array([[1e-11, 2e-12, -1e-10], [-1e-13, 1e-11, 2e-12]]),
                       "estimation_parameters": "multiple misalignments"}]

        camera_vecs = [[0, 0, 1], [0.01, 0, 1], [-0.01, 0, 1], [0, 0.01, 1], [0, -0.01, 1], [0.01, 0.01, 1],
                       [-0.01, -0.01, 1], [[0.01, -0.01], [-0.01, 0.01], [1, 1]]]

        temperatures = [0, 1, -1, 10.5, -10.5]

        for temp in temperatures:
            for dist in dist_coefs:

                model = self.Class(**dist, **intrins_param)

                for vec in camera_vecs:
                    with self.subTest(**dist, temp=temp, vec=vec):
                        pixel_loc = model.project_onto_image(vec, image=-1, temperature=temp)

                        unit_vec = model.pixels_to_unit(pixel_loc, image=-1, temperature=temp)

                        unit_true = np.array(vec).astype(np.float64)

                        unit_true /= np.linalg.norm(unit_true, axis=0, keepdims=True)

                        np.testing.assert_allclose(unit_vec, unit_true, atol=1e-13)

    def test_overwrite(self):

        model1 = self.Class(field_of_view=10, intrinsic_matrix=np.array([[1, 2, 3], [4, 5, 6]]),
                            distortion_coefficients=np.array([1, 2, 3, 4, 5, 6]), focal_length=60,
                            misalignment=[[1, 2, 3], [4, 5, 6]], use_a_priori=False,
                            estimation_parameters=['multiple misalignments'], a1=0, a2=3, a3=5)

        model2 = self.Class(field_of_view=20, intrinsic_matrix=np.array([[11, 12, 13], [14, 15, 16]]),
                            distortion_coefficients=np.array([11, 12, 13, 14, 15, 16]), focal_length=160,
                            misalignment=[[11, 12, 13], [14, 15, 16]], use_a_priori=True,
                            estimation_parameters=['single misalignment'], a1=-100, a2=-200, a3=-300)

        modeltest = model1.copy()

        modeltest.overwrite(model2)

        self.assertEqual(modeltest, model2)

        modeltest = model2.copy()

        modeltest.overwrite(model1)

        self.assertEqual(modeltest, model1)

    def test_intrinsic_matrix_inv(self):

        model = self.Class(kx=5, ky=10, kxy=20, kyx=-30.4, px=100, py=-5)

        np.testing.assert_array_almost_equal(
            model.intrinsic_matrix @ np.vstack([model.intrinsic_matrix_inv, [0, 0, 1]]),
            [[1, 0, 0], [0, 1, 0]])

        np.testing.assert_array_almost_equal(
            model.intrinsic_matrix_inv @ np.vstack([model.intrinsic_matrix, [0, 0, 1]]),
            [[1, 0, 0], [0, 1, 0]])

    def test_distort_pixels(self):

        model = self.Class(kx=1000, ky=-950.5, px=4500, py=139.32, a1=1e-3, a2=1e-4, a3=1e-5,
                           kxy=0.5, kyx=-8, radial2=1e-5, radial4=1e-5, pinwheel2=1e-7, pinwheel1=-1e-12,
                           tangential_x=1e-6, tangential_y=2e-12)

        pixels = [[0, 1], [1, 0], [-1, 0], [0, -1], [9000., 200.2],
                  [[4500, 100, 10.98], [0, 139.23, 200.3]]]

        temperatures = [0, 1, -1, 10.5, -10.5]

        for pix in pixels:

            for temp in temperatures:
                with self.subTest(pix=pix, temp=temp):
                    undist_pix = model.undistort_pixels(pix, temperature=temp)
                    dist_pix = model.distort_pixels(undist_pix, temperature=temp)

                    np.testing.assert_allclose(dist_pix, pix, atol=1e-10)

    def test_distortion_map(self):

        model = self.Class(kx=100, ky=-985.234, px=1000, py=1095, kxy=10, kyx=-5,
                           e1=1e-6, e2=1e-12, e3=-4e-10, e5=6e-7, e6=-1e-5, e4=1e-7,
                           a1=1e-6, a2=-1e-7, a3=4e-12)

        rows, cols, dist = model.distortion_map((2000, 250), step=10)

        rl, cl = np.arange(0, 2000, 10), np.arange(0, 250, 10)

        rs, cs = np.meshgrid(rl, cl, indexing='ij')

        np.testing.assert_array_equal(rows, rs)
        np.testing.assert_array_equal(cols, cs)

        distl = model.distort_pixels(np.vstack([cs.ravel(), rs.ravel()]).astype(np.float64))

        np.testing.assert_array_equal(distl - np.vstack([cs.ravel(), rs.ravel()]), dist)


class TestBrownModel(TestPinholeModel):
    def setUp(self):

        self.Class = BrownModel

    # Not supported for this model
    test__compute_dgnomic_dfocal_length = None # pyright: ignore[reportAssignmentType]

    def test___init__(self):
        model = self.Class(intrinsic_matrix=np.array([[1, 2, 3], [4, 5, 6]]), field_of_view=20.5,
                           use_a_priori=True, misalignment=[np.zeros(3), np.ones(3)],
                           distortion_coefficients=np.array([1, 2, 3, 4, 5]),
                           estimation_parameters='basic intrinsic',
                           a1=1, a2=2, a3=3)

        np.testing.assert_array_equal(model.intrinsic_matrix, [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(model.focal_length, 1)
        self.assertEqual(model.field_of_view, 20.5)
        self.assertTrue(model.use_a_priori)
        self.assertEqual(model.estimation_parameters, ['basic intrinsic'])
        self.assertEqual(model.a1, 1)
        self.assertEqual(model.a2, 2)
        self.assertEqual(model.a3, 3)
        np.testing.assert_array_equal(model.distortion_coefficients, np.arange(1, 6))

        model = self.Class(kx=1, fy=2, px=4, py=5, field_of_view=20.5,
                           use_a_priori=True, misalignment=[np.zeros(3), np.ones(3)], kxy=80,
                           estimation_parameters=['kx', 'px'], n_rows=500, n_cols=600,
                           radial2=1, radial4=2, k3=3, p1=4, tiptilt_x=5)

        np.testing.assert_array_equal(model.intrinsic_matrix, [[1, 80, 4], [0, 2, 5]])
        self.assertEqual(model.focal_length, 1)
        self.assertEqual(model.field_of_view, 20.5)
        self.assertTrue(model.use_a_priori)
        self.assertEqual(model.estimation_parameters, ['kx', 'px'])
        self.assertEqual(model.n_rows, 500)
        self.assertEqual(model.n_cols, 600)

        np.testing.assert_array_equal(model.distortion_coefficients, np.arange(1, 6))

    def test_fx(self):
        model = self.Class(intrinsic_matrix=np.array([[1, 0, 0], [0, 0, 0]]))

        self.assertEqual(model.kx, 1)

        model.kx = 100

        self.assertEqual(model.kx, 100)

        self.assertEqual(model.intrinsic_matrix[0, 0], 100)

    def test_fy(self):
        model = self.Class(intrinsic_matrix=np.array([[0, 0, 0], [0, 1, 0]]))

        self.assertEqual(model.ky, 1)

        model.ky = 100

        self.assertEqual(model.ky, 100)

        self.assertEqual(model.intrinsic_matrix[1, 1], 100)

    def test_kxy(self):

        model = self.Class(intrinsic_matrix=np.array([[0, 1, 0], [0, 0, 0]]))

        self.assertEqual(model.kxy, 1)

        model.kxy = 100

        self.assertEqual(model.kxy, 100)

        self.assertEqual(model.intrinsic_matrix[0, 1], 100)

    def test_alpha(self):
        model = self.Class(intrinsic_matrix=np.array([[0, 1, 0], [0, 0, 0]]))

        self.assertEqual(model.alpha, 1)

        model.alpha = 100

        self.assertEqual(model.alpha, 100)

        self.assertEqual(model.intrinsic_matrix[0, 1], 100)

    def test_k1(self):
        model = self.Class(distortion_coefficients=np.array([1, 0, 0, 0, 0]))

        self.assertEqual(model.k1, 1)

        model.k1 = 100

        self.assertEqual(model.k1, 100)

        self.assertEqual(model.distortion_coefficients[0], 100)

    def test_k2(self):
        model = self.Class(distortion_coefficients=np.array([0, 1, 0, 0, 0]))

        self.assertEqual(model.k2, 1)

        model.k2 = 100

        self.assertEqual(model.k2, 100)

        self.assertEqual(model.distortion_coefficients[1], 100)

    def test_k3(self):
        model = self.Class(distortion_coefficients=np.array([0, 0, 1, 0, 0]))

        self.assertEqual(model.k3, 1)

        model.k3 = 100

        self.assertEqual(model.k3, 100)

        self.assertEqual(model.distortion_coefficients[2], 100)

    def test_p1(self):
        model = self.Class(distortion_coefficients=np.array([0, 0, 0, 1, 0]))

        self.assertEqual(model.p1, 1)

        model.p1 = 100

        self.assertEqual(model.p1, 100)

        self.assertEqual(model.distortion_coefficients[3], 100)

    def test_p2(self):
        model = self.Class(distortion_coefficients=np.array([0, 0, 0, 0, 1]))

        self.assertEqual(model.p2, 1)

        model.p2 = 100

        self.assertEqual(model.p2, 100)

        self.assertEqual(model.distortion_coefficients[4], 100)

    def test_radial2(self):
        model = self.Class(distortion_coefficients=np.array([1, 0, 0, 0, 0]))

        self.assertEqual(model.radial2, 1)

        model.radial2 = 100

        self.assertEqual(model.radial2, 100)

        self.assertEqual(model.distortion_coefficients[0], 100)

    def test_radial4(self):
        model = self.Class(distortion_coefficients=np.array([0, 1, 0, 0, 0]))

        self.assertEqual(model.radial4, 1)

        model.radial4 = 100

        self.assertEqual(model.radial4, 100)

        self.assertEqual(model.distortion_coefficients[1], 100)

    def test_radial6(self):
        model = self.Class(distortion_coefficients=np.array([0, 0, 1, 0, 0]))

        self.assertEqual(model.radial6, 1)

        model.radial6 = 100

        self.assertEqual(model.radial6, 100)

        self.assertEqual(model.distortion_coefficients[2], 100)

    def test_tiptilt_y(self):
        model = self.Class(distortion_coefficients=np.array([0, 0, 0, 1, 0]))

        self.assertEqual(model.tiptilt_y, 1)

        model.tiptilt_y = 100

        self.assertEqual(model.tiptilt_y, 100)

        self.assertEqual(model.distortion_coefficients[3], 100)

    def test_tiptilt_x(self):
        model = self.Class(distortion_coefficients=np.array([0, 0, 0, 0, 1]))

        self.assertEqual(model.tiptilt_x, 1)

        model.tiptilt_x = 100

        self.assertEqual(model.tiptilt_x, 100)

        self.assertEqual(model.distortion_coefficients[4], 100)

    def test_apply_distortion(self):
        dist_coefs = [{"k1": 1.5, "k2": 0, "k3": 0, "p1": 0, "p2": 0},
                      {"k1": 0, "k2": 1.5, "k3": 0, "p1": 0, "p2": 0},
                      {"k1": 0, "k2": 0, "k3": 1.5, "p1": 0, "p2": 0},
                      {"k1": 0, "k2": 0, "k3": 0, "p1": 1.5, "p2": 0},
                      {"k1": 0, "k2": 0, "k3": 0, "p1": 0, "p2": 1.5}]

        inputs = [[0, 0], [1, 0], [-1, 0], [1.5, 0], [-1.5, 0], [[1.5], [0]], [[1.5, -1], [0, 0]],
                  [0, 1], [0, -1], [0, 1.5], [0, -1.5], [[0], [1.5]], [[0, 0], [1.5, -1]], [1, 1]]

        solus = [[[0, 0], [2.5, 0], [-2.5, 0], [1.5 + 1.5 ** 4, 0], [-(1.5 + 1.5 ** 4), 0],
                  [[(1.5 + 1.5 ** 4)], [0]], [[(1.5 + 1.5 ** 4), -2.5], [0, 0]],
                  [0, 2.5], [0, -2.5], [0, (1.5 + 1.5 ** 4)], [0, -(1.5 + 1.5 ** 4)], [[0], [(1.5 + 1.5 ** 4)]],
                  [[0, 0], [(1.5 + 1.5 ** 4), -2.5]], [1 + 2 * 1.5, 1 + 2 * 1.5]],
                 [[0, 0], [2.5, 0], [-2.5, 0], [(1.5 + 1.5 ** 6), 0], [-(1.5 + 1.5 ** 6), 0],
                  [[(1.5 + 1.5 ** 6)], [0]], [[(1.5 + 1.5 ** 6), -2.5], [0, 0]],
                  [0, 2.5], [0, -2.5], [0, (1.5 + 1.5 ** 6)], [0, -(1.5 + 1.5 ** 6)], [[0], [(1.5 + 1.5 ** 6)]],
                  [[0, 0], [(1.5 + 1.5 ** 6), -2.5]], [1 + 4 * 1.5, 1 + 4 * 1.5]],
                 [[0, 0], [2.5, 0], [-2.5, 0], [(1.5 + 1.5 ** 8), 0], [-(1.5 + 1.5 ** 8), 0],
                  [[(1.5 + 1.5 ** 8)], [0]], [[(1.5 + 1.5 ** 8), -2.5], [0, 0]],
                  [0, 2.5], [0, -2.5], [0, (1.5 + 1.5 ** 8)], [0, -(1.5 + 1.5 ** 8)], [[0], [(1.5 + 1.5 ** 8)]],
                  [[0, 0], [(1.5 + 1.5 ** 8), -2.5]], [1 + 8 * 1.5, 1 + 8 * 1.5]],
                 [[0, 0], [1, 1.5], [-1, 1.5], [1.5, 1.5 ** 3], [-1.5, 1.5 ** 3], [[1.5], [1.5 ** 3]],
                  [[1.5, -1], [1.5 ** 3, 1.5]],
                  [0, 1 + 3 * 1.5], [0, -1 + 3 * 1.5], [0, 1.5 + 3 * 1.5 ** 3], [0, -1.5 + 3 * 1.5 ** 3],
                  [[0], [1.5 + 3 * 1.5 ** 3]],
                  [[0, 0], [1.5 + 3 * 1.5 ** 3, -1 + 3 * 1.5]], [1 + 2 * 1.5, 1 + 4 * 1.5]],
                 [[0, 0], [1 + 3 * 1.5, 0], [-1 + 3 * 1.5, 0], [1.5 + 3 * 1.5 ** 3, 0], [-1.5 + 3 * 1.5 ** 3, 0],
                  [[1.5 + 3 * 1.5 ** 3], [0]], [[1.5 + 3 * 1.5 ** 3, -1 + 3 * 1.5], [0, 0]],
                  [1.5, 1], [1.5, -1], [1.5 ** 3, 1.5], [1.5 ** 3, -1.5], [[1.5 ** 3], [1.5]],
                  [[1.5 ** 3, 1.5], [1.5, -1]],
                  [1 + 4 * 1.5, 1 + 2 * 1.5]]]

        for dist, sols in zip(dist_coefs, solus):

            with self.subTest(**dist):

                model = self.Class(**dist)

                for inp, solu in zip(inputs, sols):
                    gnom_dist = model.apply_distortion(np.array(inp))

                    np.testing.assert_array_almost_equal(gnom_dist, solu)

    def test_get_projections(self):

        points = [[0, 0, 1], [-0.1, 0.2, 2.2], [[-0.1], [0.2], [2.2]], [[-0.1, 0], [0.2, 0], [2.2, 1]]]

        model = self.Class(fx=4050.5, fy=3050.25, alpha=1.5, px=1500, py=1500.5,
                           k1=0.5, k2=-0.3, k3=0.15, p1=1e-7, p2=1e-6, a1=1e-1, a2=-1e-6, a3=3e-7)

        temps = [0, 1, -1, 10, -10]

        for temp in temps:
            with self.subTest(misalignment=None, temp=temp):

                for point in points:
                    pin, dist, pix = model.get_projections(point, temperature=temp)

                    pin_true = np.array(point[:2]) / point[2]

                    dist_true = model.apply_distortion(pin_true)

                    dist_true *= model.get_temperature_scale(temp)

                    pix_true = (np.matmul(model.intrinsic_matrix[:, :2], dist_true).T + model.intrinsic_matrix[:, 2]).T

                    np.testing.assert_array_equal(pin, pin_true)
                    np.testing.assert_array_equal(dist, dist_true)
                    np.testing.assert_array_equal(pix, pix_true)

        model = self.Class(fx=4050.5, fy=3050.25, alpha=1.5, px=1500, py=1500.5,
                           k1=0.5, k2=-0.3, k3=0.15, p1=1e-7, p2=1e-6, misalignment=[0, 0, np.pi])

        with self.subTest(misalignment=[0, 0, np.pi]):

            for point in points:
                pin, dist, pix = model.get_projections(point)

                pin_true = -np.array(point[:2]) / point[2]

                dist_true = model.apply_distortion(pin_true)

                pix_true = (np.matmul(model.intrinsic_matrix[:, :2], dist_true).T + model.intrinsic_matrix[:, 2]).T

                np.testing.assert_array_almost_equal(pin, pin_true)
                np.testing.assert_array_almost_equal(dist, dist_true)
                np.testing.assert_array_almost_equal(pix, pix_true)

        model = self.Class(fx=4050.5, fy=3050.25, alpha=1.5, px=1500, py=1500.5,
                           k1=0.5, k2=-0.3, k3=0.15, p1=1e-7, p2=1e-6, misalignment=[np.pi, 0, 0])

        with self.subTest(misalignment=[np.pi, 0, 0]):

            for point in points:
                pin, dist, pix = model.get_projections(point)

                pin_true = np.array(point[:2]) / point[2]
                pin_true[0] *= -1

                dist_true = model.apply_distortion(pin_true)

                pix_true = (np.matmul(model.intrinsic_matrix[:, :2], dist_true).T + model.intrinsic_matrix[:, 2]).T

                np.testing.assert_array_almost_equal(pin, pin_true)
                np.testing.assert_array_almost_equal(dist, dist_true)
                np.testing.assert_array_almost_equal(pix, pix_true)

        model = self.Class(fx=4050.5, fy=3050.25, alpha=1.5, px=1500, py=1500.5,
                           k1=0.5, k2=-0.3, k3=0.15, p1=1e-7, p2=1e-6, misalignment=[0, np.pi, 0])

        with self.subTest(misalignment=[0, np.pi, 0]):

            for point in points:
                pin, dist, pix = model.get_projections(point)

                pin_true = np.array(point[:2]) / point[2]
                pin_true[1] *= -1

                dist_true = model.apply_distortion(pin_true)

                pix_true = (np.matmul(model.intrinsic_matrix[:, :2], dist_true).T + model.intrinsic_matrix[:, 2]).T

                np.testing.assert_array_almost_equal(pin, pin_true)
                np.testing.assert_array_almost_equal(dist, dist_true)
                np.testing.assert_array_almost_equal(pix, pix_true)

        model = self.Class(fx=4050.5, fy=3050.25, alpha=1.5, px=1500, py=1500.5,
                           k1=0.5, k2=-0.3, k3=0.15, p1=1e-7, p2=1e-6, misalignment=[1, 0.2, 0.3])

        with self.subTest(misalignment=[1, 0.2, 0.3]):

            rot_mat = at.rotvec_to_rotmat([1, 0.2, 0.3]).squeeze()

            for point in points:
                point_new = rot_mat @ point

                pin, dist, pix = model.get_projections(point)

                pin_true = np.array(point_new[:2]) / np.array(point_new[2])

                dist_true = model.apply_distortion(pin_true)

                pix_true = (np.matmul(model.intrinsic_matrix[:, :2], dist_true).T + model.intrinsic_matrix[:, 2]).T

                np.testing.assert_array_almost_equal(pin, pin_true)
                np.testing.assert_array_almost_equal(dist, dist_true)
                np.testing.assert_array_almost_equal(pix, pix_true)

        model = self.Class(fx=4050.5, fy=3050.25, alpha=1.5, px=1500, py=1500.5,
                           k1=0.5, k2=-0.3, k3=0.15, p1=1e-7, p2=1e-6, misalignment=[[1, 0.2, 0.3], [0, 0, np.pi]])

        model.estimate_multiple_misalignments = True

        with self.subTest(misalignment=[[1, 0.2, 0.3], [0, 0, np.pi]]):

            rot_mat = at.rotvec_to_rotmat([1, 0.2, 0.3]).squeeze()

            for point in points:
                point_new = rot_mat @ point

                pin, dist, pix = model.get_projections(point, image=0)

                pin_true = np.array(point_new[:2]) / np.array(point_new[2])

                dist_true = model.apply_distortion(pin_true)

                pix_true = (np.matmul(model.intrinsic_matrix[:, :2], dist_true).T + model.intrinsic_matrix[:, 2]).T

                np.testing.assert_array_almost_equal(pin, pin_true)
                np.testing.assert_array_almost_equal(dist, dist_true)
                np.testing.assert_array_almost_equal(pix, pix_true)

                pin, dist, pix = model.get_projections(point, image=1)

                pin_true = -np.array(point[:2]) / point[2]

                dist_true = model.apply_distortion(pin_true)

                pix_true = (np.matmul(model.intrinsic_matrix[:, :2], dist_true).T + model.intrinsic_matrix[:, 2]).T

                np.testing.assert_array_almost_equal(pin, pin_true)
                np.testing.assert_array_almost_equal(dist, dist_true)
                np.testing.assert_array_almost_equal(pix, pix_true)

    def test_project_onto_image(self):

        points = [[0, 0, 1], [-0.1, 0.2, 2.2], [[-0.1], [0.2], [2.2]], [[-0.1, 0], [0.2, 0], [2.2, 1]]]

        model = self.Class(fx=4050.5, fy=3050.25, alpha=1.5, px=1500, py=1500.5,
                           k1=0.5, k2=-0.3, k3=0.15, p1=1e-7, p2=1e-6, a1=1, a2=2, a3=-3)

        temps = [0, 1, -1, 10, -10]

        for temp in temps:

            with self.subTest(temp=temp, misalignment=None):

                for point in points:
                    _, __, pix = model.get_projections(point, temperature=temp)

                    pix_proj = model.project_onto_image(point, temperature=temp)

                    np.testing.assert_array_equal(pix, pix_proj)

        model = self.Class(fx=4050.5, fy=3050.25, alpha=1.5, px=1500, py=1500.5,
                           k1=0.5, k2=-0.3, k3=0.15, p1=1e-7, p2=1e-6, misalignment=[[1, 0.2, 0.3], [0, 0, np.pi]])

        model.estimate_multiple_misalignments = True

        with self.subTest(misalignment=[[1, 0.2, 0.3], [0, 0, np.pi]]):

            for point in points:
                _, __, pix = model.get_projections(point, image=0)

                pix_proj = model.project_onto_image(point, image=0)

                np.testing.assert_array_equal(pix, pix_proj)

                _, __, pix = model.get_projections(point, image=1)

                pix_proj = model.project_onto_image(point, image=1)

                np.testing.assert_array_equal(pix, pix_proj)

    def test_compute_pixel_jacobian(self):

        def num_deriv(uvec, cmodel, delta=1e-8, image=0, temperature=0) -> np.ndarray:

            uvec = np.array(uvec).reshape(3, -1)

            pix_true = cmodel.project_onto_image(uvec, image=image, temperature=temperature)

            uvec_pert = uvec + [[delta], [0], [0]]

            pix_pert_x_f = cmodel.project_onto_image(uvec_pert, image=image, temperature=temperature)

            uvec_pert = uvec + [[0], [delta], [0]]

            pix_pert_y_f = cmodel.project_onto_image(uvec_pert, image=image, temperature=temperature)

            uvec_pert = uvec + [[0], [0], [delta]]

            pix_pert_z_f = cmodel.project_onto_image(uvec_pert, image=image, temperature=temperature)

            uvec_pert = uvec - [[delta], [0], [0]]

            pix_pert_x_b = cmodel.project_onto_image(uvec_pert, image=image, temperature=temperature)

            uvec_pert = uvec - [[0], [delta], [0]]

            pix_pert_y_b = cmodel.project_onto_image(uvec_pert, image=image, temperature=temperature)

            uvec_pert = uvec - [[0], [0], [delta]]

            pix_pert_z_b = cmodel.project_onto_image(uvec_pert, image=image, temperature=temperature)

            return np.array([(pix_pert_x_f-pix_pert_x_b)/(2*delta),
                             (pix_pert_y_f-pix_pert_y_b)/(2*delta),
                             (pix_pert_z_f-pix_pert_z_b)/(2*delta)]).swapaxes(0, -1)

        inputs = [np.array([[0.5, 0, 1]]).T,
                  np.array([[0, 0.5, 1], [0.5, 0.5, 1], [-0.5, 0, 1]]).T,
                  np.array([[0.1, -0.5, 1], [-0.5, -0.5, 1], [5, 10, 1000.23], [1, 2, 1200.23]]).T]

        temperatures = [0, 1, -1, 10.5, -10.5]

        model = self.Class(fx=4050.5, fy=5050.25, alpha=1.5, px=1500, py=1500.5,
                           a1=0.15e-7, a2=-0.01e-8, a3=1e-9,
                           k1=0.5, k2=-0.3, k3=0.15, p1=1e-7, p2=1e-6,
                           misalignment=[[1e-9, -1e-9, 2e-9], [-1e-9, 2e-9, -1e-9]])

        model.estimate_multiple_misalignments = True

        for temp in temperatures:

            for input in inputs:

                for image in range(2):

                    with self.subTest(image=image, temp=temp, input=input):

                        jac_ana = model.compute_pixel_jacobian(input, image=image, temperature=temp)

                        jac_num = num_deriv(input, model, image=image, temperature=temp, delta=1e-2)

                        np.testing.assert_allclose(jac_ana, jac_num, rtol=1e-3, atol=1e-10)

    def test__compute_dcamera_point_dgnomic(self):

        def num_deriv(gnomic_locations, cmodel, delta=1e-6) -> np.ndarray:

            def g2u(g):

                v = np.vstack([g, cmodel.focal_length*np.ones(g.shape[-1])])

                return v/np.linalg.norm(v, axis=0, keepdims=True)

            gnomic_locations = np.asarray(gnomic_locations).reshape(2, -1)

            gnom_pert = gnomic_locations + [[delta], [0]]

            cam_loc_pert_x_f = g2u(gnom_pert)

            gnom_pert = gnomic_locations + [[0], [delta]]

            cam_loc_pert_y_f = g2u(gnom_pert)

            gnom_pert = gnomic_locations - [[delta], [0]]

            cam_loc_pert_x_b = g2u(gnom_pert)

            gnom_pert = gnomic_locations - [[0], [delta]]

            cam_loc_pert_y_b = g2u(gnom_pert)

            return np.array([(cam_loc_pert_x_f -cam_loc_pert_x_b)/(2*delta),
                             (cam_loc_pert_y_f -cam_loc_pert_y_b)/(2*delta)]).swapaxes(0, -1)

        inputs = [np.array([[0, 0]]).T,
                  np.array([[0, 2000], [2000, 0], [2000, 2000]]).T,
                  np.array([[1000, 1000], [1000, 2000], [2000, 1000], [0, 1000], [1000, 0]]).T]

        model = self.Class(fx=4050.5, fy=5050.25, alpha=1.5, px=1500, py=1500.5,
                           a1=0.15e-7, a2=-0.01e-8, a3=1e-9,
                           k1=0.5, k2=-0.3, k3=0.15, p1=1e-7, p2=1e-6,
                           misalignment=[[1e-9, -1e-9, 2e-9], [-1e-9, 2e-9, -1e-9]])

        model.estimate_multiple_misalignments = True

        for input in inputs:

            with self.subTest(input=input):
                jac_ana = []
                for gnom in input.T:
                    jac_ana.append(
                        model._compute_dcamera_point_dgnomic(gnom, np.sqrt(np.sum(gnom*gnom) + model.focal_length**2)))

                jac_ana = np.array(jac_ana)

                jac_num = num_deriv(input, model)

                np.testing.assert_almost_equal(jac_ana, jac_num)

    def test__compute_dgnomic_ddist_gnomic(self):

        def num_deriv(dist_gnomic_locations, cmodel, delta=1e-6) -> np.ndarray:

            def dg2g(dg):

                gnomic_guess = dg.copy()

                # perform the fpa
                for _ in np.arange(20):

                    # get the distorted location assuming the current guess is correct
                    gnomic_guess_distorted = cmodel.apply_distortion(gnomic_guess)

                    # subtract off the residual distortion from the gnomic guess
                    gnomic_guess += dg - gnomic_guess_distorted

                    # check for convergence
                    if np.all(np.linalg.norm(gnomic_guess_distorted - dg, axis=0) <= 1e-15):
                        break

                return gnomic_guess

            dist_gnomic_locations = np.asarray(dist_gnomic_locations).reshape(2, -1)

            dist_gnom_pert = dist_gnomic_locations + [[delta], [0]]

            gnom_loc_pert_x_f = dg2g(dist_gnom_pert)

            dist_gnom_pert = dist_gnomic_locations + [[0], [delta]]

            gnom_loc_pert_y_f = dg2g(dist_gnom_pert)

            dist_gnom_pert = dist_gnomic_locations - [[delta], [0]]

            gnom_loc_pert_x_b = dg2g(dist_gnom_pert)

            dist_gnom_pert = dist_gnomic_locations - [[0], [delta]]

            gnom_loc_pert_y_b = dg2g(dist_gnom_pert)

            return np.array([(gnom_loc_pert_x_f - gnom_loc_pert_x_b)/(2*delta),
                             (gnom_loc_pert_y_f - gnom_loc_pert_y_b)/(2*delta)]).swapaxes(0, -1)

        inputs = [np.array([[0, 0]]).T,
                  np.array([[0, 0.1], [0.1, 0], [0.1, 0.1]]).T,
                  np.array([[-0.1, 0], [0, -0.1], [-0.1, -0.1], [0.1, -0.1], [-0.1, 0.1]]).T]

        model = self.Class(fx=4050.5, fy=5050.25, alpha=1.5, px=1500, py=1500.5,
                           a1=0.15e-7, a2=-0.01e-8, a3=1e-9,
                           k1=0.5, k2=-0.3, k3=0.15, p1=1e-7, p2=1e-6,
                           misalignment=[[1e-9, -1e-9, 2e-9], [-1e-9, 2e-9, -1e-9]])

        model.estimate_multiple_misalignments = True

        for input in inputs:

            with self.subTest(input=input):
                jac_ana = []
                for dist_gnom in input.T:
                    jac_ana.append(model._compute_dgnomic_ddist_gnomic(dist_gnom))

                jac_ana = np.array(jac_ana)

                jac_num = num_deriv(input, model, delta=1e-8)

                np.testing.assert_allclose(jac_ana, jac_num, rtol=1e-1, atol=1e-10)

    def test_compute_unit_vector_jacobian(self):

        def num_deriv(pixels, cmodel, delta=1e-6, image=0, temperature=0) -> np.ndarray:

            pixels = np.array(pixels).reshape(2, -1)

            pix_pert = pixels + [[delta], [0]]

            uvec_pert_x_f = cmodel.pixels_to_unit(pix_pert, image=image, temperature=temperature)

            pix_pert = pixels + [[0], [delta]]

            uvec_pert_y_f = cmodel.pixels_to_unit(pix_pert, image=image, temperature=temperature)

            pix_pert = pixels - [[delta], [0]]

            uvec_pert_x_b = cmodel.pixels_to_unit(pix_pert, image=image, temperature=temperature)

            pix_pert = pixels - [[0], [delta]]

            uvec_pert_y_b = cmodel.pixels_to_unit(pix_pert, image=image, temperature=temperature)

            return np.array([(uvec_pert_x_f-uvec_pert_x_b)/(2*delta),
                             (uvec_pert_y_f-uvec_pert_y_b)/(2*delta)]).swapaxes(0, -1)

        inputs = [np.array([[0, 0]]).T,
                  np.array([[0, 2000], [2000, 0], [2000, 2000]]).T/10,
                  np.array([[1000, 1000], [1000, 2000], [2000, 1000], [0, 1000], [1000, 0]]).T/10]

        temperatures = [0, 1, -1, 10.5, -10.5]

        model = self.Class(fx=4050.5, fy=5050.25, alpha=1.5, px=100, py=100.5,
                           a1=0.15e-7, a2=-0.01e-8, a3=1e-9,
                           k1=0.05, k2=-0.03, k3=0.015, p1=1e-7, p2=1e-6,
                           misalignment=[[1e-9, -1e-9, 2e-9], [-1e-9, 2e-9, -1e-9]])

        model.estimate_multiple_misalignments = True

        for temp in temperatures:

            for input in inputs:

                for image in range(2):

                    with self.subTest(image=image, temp=temp, input=input):

                        jac_ana = model.compute_unit_vector_jacobian(input, image=image, temperature=temp)

                        jac_num = num_deriv(input, model, image=image, temperature=temp, delta=1e-2)

                        np.testing.assert_allclose(jac_ana, jac_num, rtol=1e-3, atol=1e-10)

    def test__compute_ddistorted_gnomic_dgnomic(self):

        def num_deriv(loc, cmodel, delta=1e-8) -> np.ndarray:
            loc_pert = np.array(loc) + [delta, 0]
            dist_pert_x_f = cmodel.apply_distortion(loc_pert)
            loc_pert = np.array(loc) + [0, delta]
            dist_pert_y_f = cmodel.apply_distortion(loc_pert)

            loc_pert = np.array(loc) - [delta, 0]
            dist_pert_x_b = cmodel.apply_distortion(loc_pert)
            loc_pert = np.array(loc) - [0, delta]
            dist_pert_y_b = cmodel.apply_distortion(loc_pert)

            return np.array(
                [(dist_pert_x_f - dist_pert_x_b) / (2 * delta), (dist_pert_y_f - dist_pert_y_b) / (2 * delta)]).T

        dist_coefs = [{"k1": 1.5, "k2": 0, "k3": 0, "p1": 0, "p2": 0},
                      {"k1": 0, "k2": 1.5, "k3": 0, "p1": 0, "p2": 0},
                      {"k1": 0, "k2": 0, "k3": 1.5, "p1": 0, "p2": 0},
                      {"k1": 0, "k2": 0, "k3": 0, "p1": 1.5, "p2": 0},
                      {"k1": 0, "k2": 0, "k3": 0, "p1": 0, "p2": 1.5},
                      {"k1": -1.5, "k2": -1.5, "k3": -1.5, "p1": -1.5, "p2": -1.5}]

        inputs = [[0, 0], [1, 0], [-1, 0], [1.5, 0], [-1.5, 0], [0, 1], [0, -1], [0, 1.5], [0, -1.5], [1, 1]]

        for dist_coef in dist_coefs:

            model = self.Class(**dist_coef)

            with self.subTest(**dist_coef):

                for inp in inputs:
                    r = np.sqrt(inp[0] ** 2 + inp[1] ** 2)
                    r2 = r ** 2
                    r4 = r ** 4
                    r6 = r ** 6

                    num = num_deriv(inp, model)

                    ana = model._compute_ddistorted_gnomic_dgnomic(np.array(inp), r2, r4, r6)

                    np.testing.assert_allclose(num, ana, atol=1e-14)

    def test__compute_dpixel_ddistorted_gnomic(self):

        def num_deriv(loc, cmodel, delta=1e-8, temperature=0) -> np.ndarray:
            loc_pert = np.array(loc) + [delta, 0]
            loc_pert *= cmodel.get_temperature_scale(temperature)
            pix_pert_x_f = cmodel.intrinsic_matrix[:, :2] @ loc_pert + cmodel.intrinsic_matrix[:, 2]
            loc_pert = np.array(loc) + [0, delta]
            loc_pert *= cmodel.get_temperature_scale(temperature)
            pix_pert_y_f = cmodel.intrinsic_matrix[:, :2] @ loc_pert + cmodel.intrinsic_matrix[:, 2]

            loc_pert = np.array(loc) - [delta, 0]
            loc_pert *= cmodel.get_temperature_scale(temperature)
            pix_pert_x_b = cmodel.intrinsic_matrix[:, :2] @ loc_pert + cmodel.intrinsic_matrix[:, 2]
            loc_pert = np.array(loc) - [0, delta]
            loc_pert *= cmodel.get_temperature_scale(temperature)
            pix_pert_y_b = cmodel.intrinsic_matrix[:, :2] @ loc_pert + cmodel.intrinsic_matrix[:, 2]

            return np.array(
                [(pix_pert_x_f - pix_pert_x_b) / (2 * delta), (pix_pert_y_f - pix_pert_y_b) / (2 * delta)]).T

        intrins_coefs = [{"fx": 1.5, "fy": 0, "alpha": 0, "px": 0, "py": 0},
                         {"fx": 0, "fy": 1.5, "alpha": 0, "px": 0, "py": 0},
                         {"fx": 0, "fy": 0, "alpha": 1.5, "px": 0, "py": 0},
                         {"fx": 0, "fy": 0, "alpha": 0, "px": 1.5, "py": 0},
                         {"fx": 0, "fy": 0, "alpha": 0, "px": 0, "py": 1.5},
                         {"fx": -1.5, "fy": -1.5, "alpha": -1.5, "px": 1.5, "py": 1.5}]

        inputs = [[0, 0], [1, 0], [-1, 0], [1.5, 0], [-1.5, 0], [0, 1], [0, -1], [0, 1.5], [0, -1.5], [1, 1]]

        temps = [0, 1, -1, 10.5, -10.5]

        for temp in temps:

            for intrins_coef in intrins_coefs:

                model = self.Class(**intrins_coef)

                with self.subTest(**intrins_coef, temp=temp):

                    for inp in inputs:
                        num = num_deriv(np.array(inp), model, temperature=temp)

                        ana = model._compute_dpixel_ddistorted_gnomic(temperature=temp)

                        np.testing.assert_allclose(num, ana, atol=1e-14)

    def test__compute_dpixel_dintrinsic(self):
        def num_deriv(loc, cmodel, delta=1e-6) -> np.ndarray:

            model_pert = cmodel.copy()
            model_pert.kx += delta
            pix_pert_kx_f = model_pert.intrinsic_matrix[:, :2] @ loc + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.kxy += delta
            pix_pert_kxy_f = model_pert.intrinsic_matrix[:, :2] @ loc + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.ky += delta
            pix_pert_ky_f = model_pert.intrinsic_matrix[:, :2] @ loc + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.px += delta
            pix_pert_px_f = model_pert.intrinsic_matrix[:, :2] @ loc + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.py += delta
            pix_pert_py_f = model_pert.intrinsic_matrix[:, :2] @ loc + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.kx -= delta
            pix_pert_kx_b = model_pert.intrinsic_matrix[:, :2] @ loc + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.kxy -= delta
            pix_pert_kxy_b = model_pert.intrinsic_matrix[:, :2] @ loc + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.ky -= delta
            pix_pert_ky_b = model_pert.intrinsic_matrix[:, :2] @ loc + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.px -= delta
            pix_pert_px_b = model_pert.intrinsic_matrix[:, :2] @ loc + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.py -= delta
            pix_pert_py_b = model_pert.intrinsic_matrix[:, :2] @ loc + model_pert.intrinsic_matrix[:, 2]

            return np.array([(pix_pert_kx_f - pix_pert_kx_b) / (2 * delta),
                             (pix_pert_ky_f - pix_pert_ky_b) / (2 * delta),
                             (pix_pert_kxy_f - pix_pert_kxy_b) / (2 * delta),
                             (pix_pert_px_f - pix_pert_px_b) / (2 * delta),
                             (pix_pert_py_f - pix_pert_py_b) / (2 * delta)]).T

        intrins_coefs = [{"fx": 1.5, "fy": 0, "alpha": 0, "px": 0, "py": 0},
                         {"fx": 0, "fy": 1.5, "alpha": 0, "px": 0, "py": 0},
                         {"fx": 0, "fy": 0, "alpha": 1.5, "px": 0, "py": 0},
                         {"fx": 0, "fy": 0, "alpha": 0, "px": 1.5, "py": 0},
                         {"fx": 0, "fy": 0, "alpha": 0, "px": 0, "py": 1.5},
                         {"fx": -1.5, "fy": -1.5, "alpha": -1.5, "px": 1.5, "py": 1.5}]

        inputs = [[1e-6, 1e-6], [1, 0], [-1, 0], [1.5, 0], [-1.5, 0], [0, 1], [0, -1], [0, 1.5], [0, -1.5], [1, 1]]

        for intrins_coef in intrins_coefs:

            model = self.Class(**intrins_coef)

            for inp in inputs:
                with self.subTest(**intrins_coef, inp=inp):
                    num = num_deriv(inp, model, delta=1e-5)

                    ana = model._compute_dpixel_dintrinsic(np.array(inp))

                    np.testing.assert_allclose(num, ana, atol=1e-14, rtol=1e-5)

    def test__compute_ddistorted_gnomic_ddistortion(self):

        def num_deriv(loc, cmodel, delta=1e-8) -> np.ndarray:

            model_pert = cmodel.copy()
            model_pert.k1 += delta
            loc_pert_k1_f = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.k2 += delta
            loc_pert_k2_f = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.k3 += delta
            loc_pert_k3_f = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.p1 += delta
            loc_pert_p1_f = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.p2 += delta
            loc_pert_p2_f = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.k1 -= delta
            loc_pert_k1_b = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.k2 -= delta
            loc_pert_k2_b = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.k3 -= delta
            loc_pert_k3_b = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.p1 -= delta
            loc_pert_p1_b = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.p2 -= delta
            loc_pert_p2_b = model_pert.apply_distortion(loc)

            return np.array([(loc_pert_k1_f - loc_pert_k1_b) / (2 * delta),
                             (loc_pert_k2_f - loc_pert_k2_b) / (2 * delta),
                             (loc_pert_k3_f - loc_pert_k3_b) / (2 * delta),
                             (loc_pert_p1_f - loc_pert_p1_b) / (2 * delta),
                             (loc_pert_p2_f - loc_pert_p2_b) / (2 * delta)]).T

        dist_coefs = [{"k1": 1.5, "k2": 0, "k3": 0, "p1": 0, "p2": 0},
                      {"k1": 0, "k2": 1.5, "k3": 0, "p1": 0, "p2": 0},
                      {"k1": 0, "k2": 0, "k3": 1.5, "p1": 0, "p2": 0},
                      {"k1": 0, "k2": 0, "k3": 0, "p1": 1.5, "p2": 0},
                      {"k1": 0, "k2": 0, "k3": 0, "p1": 0, "p2": 1.5}]

        inputs = [[0, 0], [1, 0], [-1, 0], [1.5, 0], [-1.5, 0], [0, 1], [0, -1], [0, 1.5], [0, -1.5], [1, 1]]

        for dist_coef in dist_coefs:

            model = self.Class(**dist_coef)

            with self.subTest(**dist_coef):

                for inp in inputs:
                    r = np.sqrt(inp[0] ** 2 + inp[1] ** 2)
                    r2 = r ** 2
                    r4 = r ** 4
                    r6 = r ** 6

                    num = num_deriv(np.array(inp), model)

                    ana = model._compute_ddistorted_gnomic_ddistortion(np.array(inp), r2, r4, r6)

                    np.testing.assert_allclose(num, ana, atol=1e-14)

    def test__compute_dgnomic_dcamera_point(self):
        def num_deriv(loc, cmodel, delta=1e-8) -> np.ndarray:
            loc_pert = np.array(loc) + [delta, 0, 0]
            gnom_pert_x_f = cmodel.get_projections(loc_pert)[0]
            loc_pert = np.array(loc) + [0, delta, 0]
            gnom_pert_y_f = cmodel.get_projections(loc_pert)[0]
            loc_pert = np.array(loc) + [0, 0, delta]
            gnom_pert_z_f = cmodel.get_projections(loc_pert)[0]

            loc_pert = np.array(loc) - [delta, 0, 0]
            gnom_pert_x_b = cmodel.get_projections(loc_pert)[0]
            loc_pert = np.array(loc) - [0, delta, 0]
            gnom_pert_y_b = cmodel.get_projections(loc_pert)[0]
            loc_pert = np.array(loc) - [0, 0, delta]
            gnom_pert_z_b = cmodel.get_projections(loc_pert)[0]

            return np.array([(gnom_pert_x_f - gnom_pert_x_b) / (2 * delta),
                             (gnom_pert_y_f - gnom_pert_y_b) / (2 * delta),
                             (gnom_pert_z_f - gnom_pert_z_b) / (2 * delta)]).T

        inputs = [[0, 0, 1], [0.5, 0, 1], [0, 0.5, 1], [0.5, 0.5, 1], [-0.5, 0, 1], [0, -0.5, 1], [-0.5, -0.5, 1],
                  [5, 10, 1000.23], [0.5, 1e-14, 1]]

        model = self.Class()

        for inp in inputs:
            num = num_deriv(inp, model)

            ana = model._compute_dgnomic_dcamera_point(np.array(inp))

            np.testing.assert_allclose(num, ana, atol=1e-9, rtol=1e-5)

    def test_get_jacobian_row(self):

        def num_deriv(loc, temp, cmodel, delta=1e-8, image=0) -> np.ndarray:
            model_pert = cmodel.copy()
            model_pert.fx += delta
            pix_pert_fx_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.fy += delta
            pix_pert_fy_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.alpha += delta
            pix_pert_skew_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.px += delta
            pix_pert_px_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.py += delta
            pix_pert_py_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.a1 += delta
            pix_pert_a1_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.a2 += delta
            pix_pert_a2_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.a3 += delta
            pix_pert_a3_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.fx -= delta
            pix_pert_fx_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.fy -= delta
            pix_pert_fy_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.alpha -= delta
            pix_pert_skew_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.px -= delta
            pix_pert_px_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.py -= delta
            pix_pert_py_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.k1 += delta
            pix_pert_k1_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.k2 += delta
            pix_pert_k2_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.k3 += delta
            pix_pert_k3_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.p1 += delta
            pix_pert_p1_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.p2 += delta
            pix_pert_p2_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.k1 -= delta
            pix_pert_k1_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.k2 -= delta
            pix_pert_k2_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.k3 -= delta
            pix_pert_k3_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.p1 -= delta
            pix_pert_p1_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.p2 -= delta
            pix_pert_p2_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][0] += delta
            pix_pert_mx_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][1] += delta
            pix_pert_my_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][2] += delta
            pix_pert_mz_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][0] -= delta
            pix_pert_mx_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][1] -= delta
            pix_pert_my_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][2] -= delta
            pix_pert_mz_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.a1 -= delta
            pix_pert_a1_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.a2 -= delta
            pix_pert_a2_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.a3 -= delta
            pix_pert_a3_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            return np.vstack([(pix_pert_fx_f - pix_pert_fx_b) / (delta * 2),
                              (pix_pert_fy_f - pix_pert_fy_b) / (delta * 2),
                              (pix_pert_skew_f - pix_pert_skew_b) / (delta * 2),
                              (pix_pert_px_f - pix_pert_px_b) / (delta * 2),
                              (pix_pert_py_f - pix_pert_py_b) / (delta * 2),
                              (pix_pert_k1_f - pix_pert_k1_b) / (delta * 2),
                              (pix_pert_k2_f - pix_pert_k2_b) / (delta * 2),
                              (pix_pert_k3_f - pix_pert_k3_b) / (delta * 2),
                              (pix_pert_p1_f - pix_pert_p1_b) / (delta * 2),
                              (pix_pert_p2_f - pix_pert_p2_b) / (delta * 2),
                              (pix_pert_a1_f - pix_pert_a1_b) / (delta * 2),
                              (pix_pert_a2_f - pix_pert_a2_b) / (delta * 2),
                              (pix_pert_a3_f - pix_pert_a3_b) / (delta * 2),
                              np.zeros((image * 3, 2)),
                              (pix_pert_mx_f - pix_pert_mx_b) / (delta * 2),
                              (pix_pert_my_f - pix_pert_my_b) / (delta * 2),
                              (pix_pert_mz_f - pix_pert_mz_b) / (delta * 2)]).T

        model = self.Class(fx=4050.5, fy=5050.25, alpha=1.5, px=1500, py=1500.5,
                           a1=0.15e-7, a2=-0.01e-8, a3=1e-9,
                           k1=0.5, k2=-0.3, k3=0.15, p1=1e-7, p2=1e-6, misalignment=[[1e-9, -1e-9, 2e-9],
                                                                                     [-1e-9, 2e-9, -1e-9]])

        model.estimate_multiple_misalignments = True

        inputs = [[0.5, 0, 1], [0, 0.5, 1], [0.5, 0.5, 1], [-0.5, 0, 1], [0, -0.5, 1], [-0.5, -0.5, 1],
                  [5, 10, 1000.23], [[1], [2], [1200.23]]]

        temps = [0, 1.5, -10]

        # TODO: investigate if this is actually correct
        for temperature in temps:
            for inp in inputs:
                with self.subTest(temperature=temperature, inp=inp):
                    num = num_deriv(inp, temperature, model, delta=1e-2)
                    ana = model._get_jacobian_row(np.array(inp), 0, 1, temperature=temperature)

                    np.testing.assert_allclose(ana, num, rtol=1e-1, atol=1e-10)

                    num = num_deriv(inp, temperature, model, delta=1e-2, image=1)
                    ana = model._get_jacobian_row(np.array(inp), 1, 2, temperature=temperature)

                    np.testing.assert_allclose(ana, num, atol=1e-10, rtol=1e-1)

    def test_compute_jacobian(self):

        def num_deriv(loc, temp, cmodel, delta=1e-8, image=0, nimages=2) -> np.ndarray:

            model_pert = cmodel.copy()
            model_pert.fx += delta
            pix_pert_fx_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.fy += delta
            pix_pert_fy_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.alpha += delta
            pix_pert_skew_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.px += delta
            pix_pert_px_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.py += delta
            pix_pert_py_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.a1 += delta
            pix_pert_a1_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.a2 += delta
            pix_pert_a2_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.a3 += delta
            pix_pert_a3_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.fx -= delta
            pix_pert_fx_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.fy -= delta
            pix_pert_fy_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.alpha -= delta
            pix_pert_skew_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.px -= delta
            pix_pert_px_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.py -= delta
            pix_pert_py_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.k1 += delta
            pix_pert_k1_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.k2 += delta
            pix_pert_k2_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.k3 += delta
            pix_pert_k3_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.p1 += delta
            pix_pert_p1_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.p2 += delta
            pix_pert_p2_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.k1 -= delta
            pix_pert_k1_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.k2 -= delta
            pix_pert_k2_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.k3 -= delta
            pix_pert_k3_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.p1 -= delta
            pix_pert_p1_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.p2 -= delta
            pix_pert_p2_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][0] += delta
            pix_pert_mx_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][1] += delta
            pix_pert_my_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][2] += delta
            pix_pert_mz_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][0] -= delta
            pix_pert_mx_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][1] -= delta
            pix_pert_my_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][2] -= delta
            pix_pert_mz_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.a1 -= delta
            pix_pert_a1_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.a2 -= delta
            pix_pert_a2_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.a3 -= delta
            pix_pert_a3_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            return np.vstack([(pix_pert_fx_f - pix_pert_fx_b) / (delta * 2),
                              (pix_pert_fy_f - pix_pert_fy_b) / (delta * 2),
                              (pix_pert_skew_f - pix_pert_skew_b) / (delta * 2),
                              (pix_pert_px_f - pix_pert_px_b) / (delta * 2),
                              (pix_pert_py_f - pix_pert_py_b) / (delta * 2),
                              (pix_pert_k1_f - pix_pert_k1_b) / (delta * 2),
                              (pix_pert_k2_f - pix_pert_k2_b) / (delta * 2),
                              (pix_pert_k3_f - pix_pert_k3_b) / (delta * 2),
                              (pix_pert_p1_f - pix_pert_p1_b) / (delta * 2),
                              (pix_pert_p2_f - pix_pert_p2_b) / (delta * 2),
                              (pix_pert_a1_f - pix_pert_a1_b) / (delta * 2),
                              (pix_pert_a2_f - pix_pert_a2_b) / (delta * 2),
                              (pix_pert_a3_f - pix_pert_a3_b) / (delta * 2),
                              np.zeros((image * 3, 2)),
                              (pix_pert_mx_f - pix_pert_mx_b) / (delta * 2),
                              (pix_pert_my_f - pix_pert_my_b) / (delta * 2),
                              (pix_pert_mz_f - pix_pert_mz_b) / (delta * 2),
                              np.zeros(((nimages - image - 1) * 3, 2))]).T

        model = self.Class(fx=4050.5, fy=5050.25, alpha=1.5, px=1500, py=1500.5,
                           k1=0.5, k2=-0.3, k3=0.15, p1=1e-7, p2=1e-6, misalignment=[[1e-9, -1e-9, 2e-9],
                                                                                     [-1e-9, 2e-9, -1e-9],
                                                                                     [1e-10, 2e-11, 3e-12]],
                           a1=0.15e-6, a2=-0.01e-7, a3=0.5e-8,
                           estimation_parameters=['intrinsic', 'temperature dependence', 'multiple misalignments'])

        inputs = [np.array([[0.5, 0, 1]]).T,
                  np.array([[0, 0.5, 1], [0.5, 0.5, 1], [-0.5, 0, 1]]).T,
                  np.array([[0.1, -0.5, 1], [-0.5, -0.5, 1], [5, 10, 1000.23], [1, 2, 1200.23]]).T]

        model.use_a_priori = False

        temps = [0, -20, 20.5]

        jac_ana = model.compute_jacobian(inputs, temperature=temps)

        jac_num = []

        numim = len(inputs)

        for ind, inp in enumerate(inputs):

            temperature = temps[ind]

            for vec in inp.T:
                jac_num.append(num_deriv(vec.T, temperature, model, delta=1e-3, image=ind, nimages=numim))

        np.testing.assert_allclose(jac_ana, np.vstack(jac_num), rtol=1e-1, atol=1e-9)

        model.use_a_priori = True

        jac_ana = model.compute_jacobian(inputs, temperature=temps)

        jac_num = []

        numim = len(inputs)

        for ind, inp in enumerate(inputs):

            temperature = temps[ind]

            for vec in inp.T:
                jac_num.append(num_deriv(vec.T, temperature, model, delta=1e-3, image=ind, nimages=numim))

        jac_num = np.vstack(jac_num)

        jac_num = np.pad(jac_num, [(0, jac_num.shape[1]), (0, 0)], 'constant', constant_values=0)

        jac_num[-jac_num.shape[1]:] = np.eye(jac_num.shape[1])

        np.testing.assert_allclose(jac_ana, jac_num, rtol=1e-1, atol=1e-9)

    def test_apply_update(self):
        model_param = {"fx": 0, "fy": 0, "alpha": 0, "k1": 0,
                       "k2": 0, "k3": 0, "p1": 0, "p2": 0, 'a1': 0, 'a2': 0, 'a3': 0,
                       "px": 0, "py": 0,
                       "misalignment": [[0, 0, 0], [0, 0, 0]]}

        model = self.Class(**model_param,
                           estimation_parameters=['intrinsic', 'temperature dependence', 'multiple misalignments'])

        update_vec = np.arange(19)

        model.apply_update(update_vec)

        keys = list(model_param.keys())

        keys.remove('misalignment')

        for key in keys:
            self.assertEqual(getattr(model, key), update_vec[model.element_dict[key][0]])

        for ind, vec in enumerate(update_vec[13:].reshape(-1, 3)):
            np.testing.assert_array_almost_equal(at.Rotation(vec).quaternion, at.Rotation(model.misalignment[ind]).quaternion)

    def test_pixels_to_gnomic(self):

        intrins_param = {"fx": 3000, "fy": 4000, "alpha": 0.5,
                         "px": 4005.23, 'py': 2000.33, 'a1': 1e-5, 'a2': 1e-6, 'a3': -1e-7}

        dist_coefs = [{"k1": 1.5e-1, "k2": 0, "k3": 0, "p1": 0, "p2": 0},
                      {"k1": 0, "k2": 1.5e-1, "k3": 0, "p1": 0, "p2": 0},
                      {"k1": 0, "k2": 0, "k3": 1.5e-1, "p1": 0, "p2": 0},
                      {"k1": 0, "k2": 0, "k3": 0, "p1": 1.5e-6, "p2": 0},
                      {"k1": 0, "k2": 0, "k3": 0, "p1": 0, "p2": 1.5e-6}]

        pinhole = [[0, 0], [0.1, 0], [-0.1, 0], [0.15, 0], [-0.15, 0], [[0.15], [0]], [[0.15, -0.1], [0, 0]],
                   [0, 0.1], [0, -0.1], [0, 0.15], [0, -0.15], [[0], [0.15]], [[0, 0], [0.15, -0.1]], [0.1, 0.1]]

        temperatures = [0, 1, -1, 10.5, -10.5]

        for temp in temperatures:
            for dist in dist_coefs:

                model = self.Class(**dist, **intrins_param)

                for fp_pinhole in pinhole:
                    with self.subTest(**dist, temp=temp, fp_pinhole=fp_pinhole):
                        fp_dist = model.apply_distortion(np.array(fp_pinhole))

                        fp_dist *= model.get_temperature_scale(temp)

                        pix_dist = ((model.intrinsic_matrix[:, :2] @ fp_dist).T + model.intrinsic_matrix[:, 2]).T

                        fp_undist = model.pixels_to_gnomic(pix_dist, temperature=temp)

                        np.testing.assert_allclose(fp_undist, fp_pinhole, atol=1e-13)

    def test_undistort_pixels(self):

        intrins_param = {"fx": 3000, "fy": 4000, "alpha": 0.5,
                         "px": 4005.23, 'py': 2000.33, 'a1': 1e-5, 'a2': 1e-6, 'a3': -1e-7}

        dist_coefs = [{"k1": 1.5e-1, "k2": 0, "k3": 0, "p1": 0, "p2": 0},
                      {"k1": 0, "k2": 1.5e-1, "k3": 0, "p1": 0, "p2": 0},
                      {"k1": 0, "k2": 0, "k3": 1.5e-1, "p1": 0, "p2": 0},
                      {"k1": 0, "k2": 0, "k3": 0, "p1": 1.5e-6, "p2": 0},
                      {"k1": 0, "k2": 0, "k3": 0, "p1": 0, "p2": 1.5e-6}]

        pinhole = [[0, 0], [0.1, 0], [-0.1, 0], [0.15, 0], [-0.15, 0], [[0.15], [0]], [[0.15, -0.1], [0, 0]],
                   [0, 0.1], [0, -0.1], [0, 0.15], [0, -0.15], [[0], [0.15]], [[0, 0], [0.15, -0.1]], [0.1, 0.1]]

        temperatures = [0, 1, -1, 10.5, -10.5]

        for temp in temperatures:
            for dist in dist_coefs:

                model = self.Class(**dist, **intrins_param)

                with self.subTest(**dist, temp=temp):

                    for fp_pinhole in pinhole:
                        fp_pinhole = np.array(fp_pinhole).astype(np.float64)
                        fp_dist = model.apply_distortion(fp_pinhole)

                        fp_dist *= model.get_temperature_scale(temp)

                        pix_dist = ((model.intrinsic_matrix[:, :2] @ fp_dist).T + model.intrinsic_matrix[:, 2]).T

                        pix_undist = model.undistort_pixels(pix_dist, temperature=temp)

                        fp_pinhole *= model.get_temperature_scale(temp)

                        pix_pinhole = ((model.intrinsic_matrix[:, :2] @ fp_pinhole).T + model.intrinsic_matrix[:, 2]).T

                        np.testing.assert_allclose(pix_undist, pix_pinhole, atol=1e-13)

    def test_pixels_to_unit(self):
        intrins_param = {"fx": 3000, "fy": 4000, "alpha": 0.5,
                         "px": 4005.23, 'py': 2000.33, 'a1': 1e-6, 'a2': -2e-7, 'a3': 4.5e-8}

        dist_coefs = [{"k1": 1.5e-1, "k2": 0, "k3": 0, "p1": 0, "p2": 0},
                      {"k1": 0, "k2": 1.5e-1, "k3": 0, "p1": 0, "p2": 0},
                      {"k1": 0, "k2": 0, "k3": 1.5e-1, "p1": 0, "p2": 0},
                      {"k1": 0, "k2": 0, "k3": 0, "p1": 1.5e-6, "p2": 0},
                      {"k1": 0, "k2": 0, "k3": 0, "p1": 0, "p2": 1.5e-6},
                      {"misalignment": np.array([1e-11, 2e-12, -1e-10])},
                      {"misalignment": np.array([[1e-11, 2e-12, -1e-10], [-1e-13, 1e-11, 2e-12]]),
                       "estimation_parameters": "multiple misalignments"}]

        camera_vecs = [[0, 0, 1], [0.1, 0, 1], [-0.1, 0, 1], [0, 0.1, 1], [0, -0.1, 1], [0.1, 0.1, 1],
                       [-0.1, -0.1, 1], [[0.1, -0.1], [-0.1, 0.1], [1, 1]]]

        temperatures = [0, 1, -1, 10.5, -10.5]

        for temp in temperatures:
            for dist in dist_coefs:

                model = self.Class(**dist, **intrins_param)

                with self.subTest(**dist, temp=temp):
                    for vec in camera_vecs:
                        pixel_loc = model.project_onto_image(vec, image=-1, temperature=temp)

                        unit_vec = model.pixels_to_unit(pixel_loc, image=-1, temperature=temp)

                        unit_true = np.array(vec).astype(np.float64)

                        unit_true /= np.linalg.norm(unit_true, axis=0, keepdims=True)

                        np.testing.assert_allclose(unit_vec, unit_true, atol=1e-13)

    def test_overwrite(self):

        model1 = self.Class(field_of_view=10, intrinsic_matrix=np.array([[1, 2, 3], [0, 5, 6]]),
                            distortion_coefficients=np.array([1, 2, 3, 4, 5]),
                            misalignment=[[1, 2, 3], [4, 5, 6]], use_a_priori=False,
                            estimation_parameters=['multiple misalignments'])

        model2 = self.Class(field_of_view=20, intrinsic_matrix=np.array([[11, 12, 13], [0, 15, 16]]),
                            distortion_coefficients=np.array([11, 12, 13, 14, 15]),
                            misalignment=[[11, 12, 13], [14, 15, 16]], use_a_priori=True,
                            estimation_parameters=['single misalignment'])

        modeltest = model1.copy()

        modeltest.overwrite(model2)

        self.assertEqual(model2.field_of_view, modeltest.field_of_view)
        self.assertEqual(model2.use_a_priori, modeltest.use_a_priori)
        self.assertEqual(model2.estimate_multiple_misalignments, modeltest.estimate_multiple_misalignments)
        np.testing.assert_array_equal(model2.intrinsic_matrix, modeltest.intrinsic_matrix)
        np.testing.assert_array_equal(model2.distortion_coefficients, modeltest.distortion_coefficients)
        np.testing.assert_array_equal(model2.misalignment, modeltest.misalignment)
        np.testing.assert_array_equal(model2.estimation_parameters, modeltest.estimation_parameters)

        modeltest = model2.copy()

        modeltest.overwrite(model1)

        self.assertEqual(model1.field_of_view, modeltest.field_of_view)
        self.assertEqual(model1.use_a_priori, modeltest.use_a_priori)
        self.assertEqual(model1.estimate_multiple_misalignments, modeltest.estimate_multiple_misalignments)
        np.testing.assert_array_equal(model1.intrinsic_matrix, modeltest.intrinsic_matrix)
        np.testing.assert_array_equal(model1.distortion_coefficients, modeltest.distortion_coefficients)
        np.testing.assert_array_equal(model1.misalignment, modeltest.misalignment)
        np.testing.assert_array_equal(model1.estimation_parameters, modeltest.estimation_parameters)

    def test_distort_pixels(self):

        model = self.Class(kx=1000, ky=-950.5, px=4500, py=139.32, a1=1e-3, a2=1e-4, a3=1e-5,
                           kxy=0.5, radial2=1e-5, radial4=1e-5, radial6=1e-7,
                           tiptilt_x=1e-6, tiptilt_y=2e-12)

        pixels = [[0, 1], [1, 0], [-1, 0], [0, -1], [9000., 200.2],
                  [[4500, 100, 10.98], [0, 139.23, 200.3]]]

        temperatures = [0, 1, -1, 10.5, -10.5]

        for pix in pixels:

            for temp in temperatures:
                with self.subTest(pix=pix, temp=temp):
                    undist_pix = model.undistort_pixels(pix, temperature=temp)
                    dist_pix = model.distort_pixels(undist_pix, temperature=temp)

                    np.testing.assert_allclose(dist_pix, pix, atol=1e-10)

    def test_to_from_elem(self):

        element = etree.Element(self.Class.__name__)

        model = self.Class(field_of_view=5, use_a_priori=True,
                           misalignment=[1, 2, 3], kx=2, ky=200, px=50, py=300, kxy=12123,
                           a1=37, a2=1, a3=-1230, k1=5, k2=10, k3=20, p1=-10, p2=35,
                           estimation_parameters=['kx', 'multiple misalignments'], n_rows=20, n_cols=30)

        model_copy = model.copy()

        with self.subTest(misalignment=True):
            element = model.to_elem(element, misalignment=True)

            self.assertEqual(model, model_copy)

            model_new = self.Class.from_elem(element)

            self.assertEqual(model, model_new)

        with self.subTest(misalignment=False):
            element = model.to_elem(element, misalignment=False)

            self.assertEqual(model, model_copy)

            model_new = self.Class.from_elem(element)

            model.estimation_parameters[-1] = 'single misalignment'

            model.estimate_multiple_misalignments = False

            model.misalignment = np.zeros(3)

            self.assertEqual(model, model_new)

    def test_distortion_map(self):

        model = self.Class(kx=100, ky=-985.234, px=1000, py=1095, kxy=10,
                           k1=1e-6, k2=1e-12, k3=-4e-10, p1=6e-7, p2=-1e-5,
                           a1=1e-6, a2=-1e-7, a3=4e-12)

        rows, cols, dist = model.distortion_map((2000, 250), step=10)

        rl, cl = np.arange(0, 2000, 10), np.arange(0, 250, 10)

        rs, cs = np.meshgrid(rl, cl, indexing='ij')

        np.testing.assert_array_equal(rows, rs)
        np.testing.assert_array_equal(cols, cs)

        distl = model.distort_pixels(np.vstack([cs.ravel(), rs.ravel()]))

        np.testing.assert_array_equal(distl - np.vstack([cs.ravel(), rs.ravel()]), dist)


class TestOpenCVModel(TestPinholeModel):
    def setUp(self):

        self.Class = OpenCVModel

    # Not supported for this model
    test__compute_dgnomic_dfocal_length = None # pyright: ignore[reportAssignmentType]

    def test___init__(self):
        model = self.Class(intrinsic_matrix=np.array([[1, 2, 3], [4, 5, 6]]), field_of_view=20.5,
                           use_a_priori=True, misalignment=[np.zeros(3), np.ones(3)],
                           distortion_coefficients=np.array([1, 2, 3, 4, 5]),
                           estimation_parameters='basic intrinsic',
                           a1=1, a2=2, a3=3)

        np.testing.assert_array_equal(model.intrinsic_matrix, [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(model.focal_length, 1)
        self.assertEqual(model.field_of_view, 20.5)
        self.assertTrue(model.use_a_priori)
        self.assertEqual(model.estimation_parameters, ['basic intrinsic'])
        self.assertEqual(model.a1, 1)
        self.assertEqual(model.a2, 2)
        self.assertEqual(model.a3, 3)
        np.testing.assert_array_equal(model.distortion_coefficients, np.arange(1, 6))

        model = self.Class(kx=1, fy=2, px=4, py=5, field_of_view=20.5,
                           use_a_priori=True, misalignment=[np.zeros(3), np.ones(3)], kxy=80,
                           estimation_parameters=['kx', 'px'], n_rows=500, n_cols=600,
                           radial2n=1, radial4n=2, k3=3, p1=4, tiptilt_x=5, radial2d=9, k5=100, k6=-90,
                           s1=400, thinprism_2=-500, s3=600, s4=5)

        np.testing.assert_array_equal(model.intrinsic_matrix, [[1, 80, 4], [0, 2, 5]])
        self.assertEqual(model.focal_length, 1)
        self.assertEqual(model.field_of_view, 20.5)
        self.assertTrue(model.use_a_priori)
        self.assertEqual(model.estimation_parameters, ['kx', 'px'])
        self.assertEqual(model.n_rows, 500)
        self.assertEqual(model.n_cols, 600)
        np.testing.assert_array_equal(model.distortion_coefficients, [1, 2, 3, 9, 100, -90, 4, 5, 400, -500, 600, 5])

    def test_fx(self):
        model = self.Class(intrinsic_matrix=np.array([[1, 0, 0], [0, 0, 0]]))

        self.assertEqual(model.kx, 1)

        model.kx = 100

        self.assertEqual(model.kx, 100)

        self.assertEqual(model.intrinsic_matrix[0, 0], 100)

    def test_fy(self):
        model = self.Class(intrinsic_matrix=np.array([[0, 0, 0], [0, 1, 0]]))

        self.assertEqual(model.ky, 1)

        model.ky = 100

        self.assertEqual(model.ky, 100)

        self.assertEqual(model.intrinsic_matrix[1, 1], 100)

    def test_kxy(self):

        model = self.Class(intrinsic_matrix=np.array([[0, 1, 0], [0, 0, 0]]))

        self.assertEqual(model.kxy, 1)

        model.kxy = 100

        self.assertEqual(model.kxy, 100)

        self.assertEqual(model.intrinsic_matrix[0, 1], 100)

    def test_alpha(self):
        model = self.Class(intrinsic_matrix=np.array([[0, 1, 0], [0, 0, 0]]))

        self.assertEqual(model.alpha, 1)

        model.alpha = 100

        self.assertEqual(model.alpha, 100)

        self.assertEqual(model.intrinsic_matrix[0, 1], 100)

    def test_k1(self):
        model = self.Class(distortion_coefficients=np.array([1, 0, 0, 0, 0]))

        self.assertEqual(model.k1, 1)

        model.k1 = 100

        self.assertEqual(model.k1, 100)

        self.assertEqual(model.distortion_coefficients[0], 100)

    def test_k2(self):
        model = self.Class(distortion_coefficients=np.array([0, 1, 0, 0, 0]))

        self.assertEqual(model.k2, 1)

        model.k2 = 100

        self.assertEqual(model.k2, 100)

        self.assertEqual(model.distortion_coefficients[1], 100)

    def test_k3(self):
        model = self.Class(distortion_coefficients=np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

        self.assertEqual(model.k3, 1)

        model.k3 = 100

        self.assertEqual(model.k3, 100)

        self.assertEqual(model.distortion_coefficients[2], 100)

    def test_k4(self):
        model = self.Class(distortion_coefficients=np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))

        self.assertEqual(model.k4, 1)

        model.k4 = 100

        self.assertEqual(model.k4, 100)

        self.assertEqual(model.distortion_coefficients[3], 100)

    def test_k5(self):
        model = self.Class(distortion_coefficients=np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]))

        self.assertEqual(model.k5, 1)

        model.k5 = 100

        self.assertEqual(model.k5, 100)

        self.assertEqual(model.distortion_coefficients[4], 100)

    def test_k6(self):
        model = self.Class(distortion_coefficients=np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]))

        self.assertEqual(model.k6, 1)

        model.k6 = 100

        self.assertEqual(model.k6, 100)

        self.assertEqual(model.distortion_coefficients[5], 100)

    def test_p1(self):
        model = self.Class(distortion_coefficients=np.array([0, 0, 0, 0, 0, 0, 1, 0]))

        self.assertEqual(model.p1, 1)

        model.p1 = 100

        self.assertEqual(model.p1, 100)

        self.assertEqual(model.distortion_coefficients[6], 100)

    def test_p2(self):
        model = self.Class(distortion_coefficients=np.array([0, 0, 0, 0, 0, 0, 0, 1]))

        self.assertEqual(model.p2, 1)

        model.p2 = 100

        self.assertEqual(model.p2, 100)

        self.assertEqual(model.distortion_coefficients[7], 100)

    def test_s1(self):
        model = self.Class(distortion_coefficients=np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]))

        self.assertEqual(model.s1, 1)

        model.s1 = 100

        self.assertEqual(model.s1, 100)

        self.assertEqual(model.distortion_coefficients[8], 100)

    def test_s2(self):
        model = self.Class(distortion_coefficients=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]))

        self.assertEqual(model.s2, 1)

        model.s2 = 100

        self.assertEqual(model.s2, 100)

        self.assertEqual(model.distortion_coefficients[9], 100)

    def test_s3(self):
        model = self.Class(distortion_coefficients=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]))

        self.assertEqual(model.s3, 1)

        model.s3 = 100

        self.assertEqual(model.s3, 100)

        self.assertEqual(model.distortion_coefficients[10], 100)

    def test_s4(self):
        model = self.Class(distortion_coefficients=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))

        self.assertEqual(model.s4, 1)

        model.s4 = 100

        self.assertEqual(model.s4, 100)

        self.assertEqual(model.distortion_coefficients[11], 100)

    def test_radial2n(self):
        model = self.Class(distortion_coefficients=np.array([1, 0, 0, 0, 0]))

        self.assertEqual(model.radial2n, 1)

        model.radial2n = 100

        self.assertEqual(model.radial2n, 100)

        self.assertEqual(model.distortion_coefficients[0], 100)

    def test_radial4n(self):
        model = self.Class(distortion_coefficients=np.array([0, 1, 0, 0, 0]))

        self.assertEqual(model.radial4n, 1)

        model.radial4n = 100

        self.assertEqual(model.radial4n, 100)

        self.assertEqual(model.distortion_coefficients[1], 100)

    def test_radial6n(self):
        model = self.Class(distortion_coefficients=np.array([0, 0, 1, 0, 0]))

        self.assertEqual(model.radial6n, 1)

        model.radial6n = 100

        self.assertEqual(model.radial6n, 100)

        self.assertEqual(model.distortion_coefficients[2], 100)

    def test_radial2d(self):
        model = self.Class(distortion_coefficients=np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))

        self.assertEqual(model.radial2d, 1)

        model.radial2d = 100

        self.assertEqual(model.radial2d, 100)

        self.assertEqual(model.distortion_coefficients[3], 100)

    def test_radial4d(self):
        model = self.Class(distortion_coefficients=np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]))

        self.assertEqual(model.radial4d, 1)

        model.radial4d = 100

        self.assertEqual(model.radial4d, 100)

        self.assertEqual(model.distortion_coefficients[4], 100)

    def test_radial6d(self):
        model = self.Class(distortion_coefficients=np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]))

        self.assertEqual(model.radial6d, 1)

        model.radial6d = 100

        self.assertEqual(model.radial6d, 100)

        self.assertEqual(model.distortion_coefficients[5], 100)

    def test_tiptilt_y(self):
        model = self.Class(distortion_coefficients=np.array([0, 0, 0, 0, 0, 0, 1, 0]))

        self.assertEqual(model.tiptilt_y, 1)

        model.tiptilt_y = 100

        self.assertEqual(model.tiptilt_y, 100)

        self.assertEqual(model.distortion_coefficients[6], 100)

    def test_tiptilt_x(self):
        model = self.Class(distortion_coefficients=np.array([0, 0, 0, 0, 0, 0, 0, 1]))

        self.assertEqual(model.tiptilt_x, 1)

        model.tiptilt_x = 100

        self.assertEqual(model.tiptilt_x, 100)

        self.assertEqual(model.distortion_coefficients[7], 100)

    def test_thinprism_1(self):
        model = self.Class(distortion_coefficients=np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]))

        self.assertEqual(model.thinprism_1, 1)

        model.thinprism_1 = 100

        self.assertEqual(model.thinprism_1, 100)

        self.assertEqual(model.distortion_coefficients[8], 100)

    def test_thinprism_2(self):
        model = self.Class(distortion_coefficients=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]))

        self.assertEqual(model.thinprism_2, 1)

        model.thinprism_2 = 100

        self.assertEqual(model.thinprism_2, 100)

        self.assertEqual(model.distortion_coefficients[9], 100)

    def test_thinprism_3(self):
        model = self.Class(distortion_coefficients=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]))

        self.assertEqual(model.thinprism_3, 1)

        model.thinprism_3 = 100

        self.assertEqual(model.thinprism_3, 100)

        self.assertEqual(model.distortion_coefficients[10], 100)

    def test_thinprism_4(self):
        model = self.Class(distortion_coefficients=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))

        self.assertEqual(model.thinprism_4, 1)

        model.thinprism_4 = 100

        self.assertEqual(model.thinprism_4, 100)

        self.assertEqual(model.distortion_coefficients[11], 100)

    def test_apply_distortion(self):
        dist_coefs = [{"k1": 1.5},
                      {"k2": 1.5},
                      {"k3": 1.5},
                      {"p1": 1.5},
                      {"p2": 1.5},
                      {"k4": 1.5},
                      {"k5": 1.5},
                      {"k6": 1.5},
                      {"s1": 1.5},
                      {"s2": 1.5},
                      {"s3": 1.5},
                      {"s4": 1.5}]

        inputs = [[0, 0], [1, 0], [-1, 0], [1.5, 0], [-1.5, 0], [[1.5], [0]], [[1.5, -1], [0, 0]],
                  [0, 1], [0, -1], [0, 1.5], [0, -1.5], [[0], [1.5]], [[0, 0], [1.5, -1]], [1, 1]]

        solus = [
            # k1
            [[0, 0], [2.5, 0], [-2.5, 0], [1.5 + 1.5 ** 4, 0], [-(1.5 + 1.5 ** 4), 0],
             [[(1.5 + 1.5 ** 4)], [0]], [[(1.5 + 1.5 ** 4), -2.5], [0, 0]],
             [0, 2.5], [0, -2.5], [0, (1.5 + 1.5 ** 4)], [0, -(1.5 + 1.5 ** 4)], [[0], [(1.5 + 1.5 ** 4)]],
             [[0, 0], [(1.5 + 1.5 ** 4), -2.5]], [1 + 2 * 1.5, 1 + 2 * 1.5]],
            # k2
            [[0, 0], [2.5, 0], [-2.5, 0], [(1.5 + 1.5 ** 6), 0], [-(1.5 + 1.5 ** 6), 0],
             [[(1.5 + 1.5 ** 6)], [0]], [[(1.5 + 1.5 ** 6), -2.5], [0, 0]],
             [0, 2.5], [0, -2.5], [0, (1.5 + 1.5 ** 6)], [0, -(1.5 + 1.5 ** 6)], [[0], [(1.5 + 1.5 ** 6)]],
             [[0, 0], [(1.5 + 1.5 ** 6), -2.5]], [1 + 4 * 1.5, 1 + 4 * 1.5]],
            # k3
            [[0, 0], [2.5, 0], [-2.5, 0], [(1.5 + 1.5 ** 8), 0], [-(1.5 + 1.5 ** 8), 0],
             [[(1.5 + 1.5 ** 8)], [0]], [[(1.5 + 1.5 ** 8), -2.5], [0, 0]],
             [0, 2.5], [0, -2.5], [0, (1.5 + 1.5 ** 8)], [0, -(1.5 + 1.5 ** 8)], [[0], [(1.5 + 1.5 ** 8)]],
             [[0, 0], [(1.5 + 1.5 ** 8), -2.5]], [1 + 8 * 1.5, 1 + 8 * 1.5]],
            # p1
            [[0, 0], [1, 1.5], [-1, 1.5], [1.5, 1.5 ** 3], [-1.5, 1.5 ** 3], [[1.5], [1.5 ** 3]],
             [[1.5, -1], [1.5 ** 3, 1.5]],
             [0, 1 + 3 * 1.5], [0, -1 + 3 * 1.5], [0, 1.5 + 3 * 1.5 ** 3], [0, -1.5 + 3 * 1.5 ** 3],
             [[0], [1.5 + 3 * 1.5 ** 3]],
             [[0, 0], [1.5 + 3 * 1.5 ** 3, -1 + 3 * 1.5]], [1 + 2 * 1.5, 1 + 4 * 1.5]],
            # p2
            [[0, 0], [1 + 3 * 1.5, 0], [-1 + 3 * 1.5, 0], [1.5 + 3 * 1.5 ** 3, 0], [-1.5 + 3 * 1.5 ** 3, 0],
             [[1.5 + 3 * 1.5 ** 3], [0]], [[1.5 + 3 * 1.5 ** 3, -1 + 3 * 1.5], [0, 0]],
             [1.5, 1], [1.5, -1], [1.5 ** 3, 1.5], [1.5 ** 3, -1.5], [[1.5 ** 3], [1.5]],
             [[1.5 ** 3, 1.5], [1.5, -1]],
             [1 + 4 * 1.5, 1 + 2 * 1.5]],
            # k4
            [[0, 0], [1 / 2.5, 0], [-1 / 2.5, 0], [1.5 / (1 + 1.5 ** 3), 0], [-1.5 / (1 + 1.5 ** 3), 0],
             [[1.5 / (1 + 1.5 ** 3)], [0]], [[1.5 / (1 + 1.5 ** 3), -1 / 2.5], [0, 0]],
             [0, 1 / 2.5], [0, -1 / 2.5], [0, 1.5 / (1 + 1.5 ** 3)], [0, -1.5 / (1 + 1.5 ** 3)],
             [[0], [1.5 / (1 + 1.5 ** 3)]], [[0, 0], [1.5 / (1 + 1.5 ** 3), -1 / 2.5]], [1 / 4, 1 / 4]],
            # k5
            [[0, 0], [1 / 2.5, 0], [-1 / 2.5, 0], [1.5 / (1 + 1.5 ** 5), 0], [-1.5 / (1 + 1.5 ** 5), 0],
             [[1.5 / (1 + 1.5 ** 5)], [0]], [[1.5 / (1 + 1.5 ** 5), -1 / 2.5], [0, 0]],
             [0, 1 / 2.5], [0, -1 / 2.5], [0, 1.5 / (1 + 1.5 ** 5)], [0, -1.5 / (1 + 1.5 ** 5)],
             [[0], [1.5 / (1 + 1.5 ** 5)]], [[0, 0], [1.5 / (1 + 1.5 ** 5), -1 / 2.5]], [1 / 7, 1 / 7]],
            # k6
            [[0, 0], [1 / 2.5, 0], [-1 / 2.5, 0], [1.5 / (1 + 1.5 ** 7), 0], [-1.5 / (1 + 1.5 ** 7), 0],
             [[1.5 / (1 + 1.5 ** 7)], [0]], [[1.5 / (1 + 1.5 ** 7), -1 / 2.5], [0, 0]],
             [0, 1 / 2.5], [0, -1 / 2.5], [0, 1.5 / (1 + 1.5 ** 7)], [0, -1.5 / (1 + 1.5 ** 7)],
             [[0], [1.5 / (1 + 1.5 ** 7)]], [[0, 0], [1.5 / (1 + 1.5 ** 7), -1 / 2.5]], [1 / 13, 1 / 13]],
            # s1
            [[0, 0], [1 + 1.5, 0], [-1 + 1.5, 0], [1.5 + 1.5 ** 3, 0], [-1.5 + 1.5 ** 3, 0],
             [[1.5 + 1.5 ** 3], [0]], [[1.5 + 1.5 ** 3, -1 + 1.5], [0, 0]],
             [1.5, 1], [1.5, -1], [1.5 ** 3, 1.5], [1.5 ** 3, -1.5],
             [[1.5 ** 3], [1.5]], [[1.5 ** 3, 1.5], [1.5, -1]], [1 + 2 * 1.5, 1]],
            # s2
            [[0, 0], [1 + 1.5, 0], [-1 + 1.5, 0], [1.5 + 1.5 ** 5, 0], [-1.5 + 1.5 ** 5, 0],
             [[1.5 + 1.5 ** 5], [0]], [[1.5 + 1.5 ** 5, -1 + 1.5], [0, 0]],
             [1.5, 1], [1.5, -1], [1.5 ** 5, 1.5], [1.5 ** 5, -1.5],
             [[1.5 ** 5], [1.5]], [[1.5 ** 5, 1.5], [1.5, -1]], [1 + 4 * 1.5, 1]],
            # s3
            [[0, 0], [1, 1.5], [-1, 1.5], [1.5, 1.5 ** 3], [-1.5, 1.5 ** 3],
             [[1.5], [1.5 ** 3]], [[1.5, -1], [1.5 ** 3, 1.5]],
             [0, 1 + 1.5], [0, -1 + 1.5], [0, 1.5 + 1.5 ** 3], [0, -1.5 + 1.5 ** 3],
             [[0], [1.5 + 1.5 ** 3]], [[0, 0], [1.5 + 1.5 ** 3, -1 + 1.5]], [1, 1 + 2 * 1.5]],
            # s4
            [[0, 0], [1, 1.5], [-1, 1.5], [1.5, 1.5 ** 5], [-1.5, 1.5 ** 5],
             [[1.5], [1.5 ** 5]], [[1.5, -1], [1.5 ** 5, 1.5]],
             [0, 1 + 1.5], [0, -1 + 1.5], [0, 1.5 + 1.5 ** 5], [0, -1.5 + 1.5 ** 5],
             [[0], [1.5 + 1.5 ** 5]], [[0, 0], [1.5 + 1.5 ** 5, -1 + 1.5]], [1, 1 + 4 * 1.5]]
        ]

        for dist, sols in zip(dist_coefs, solus):

            model = self.Class(**dist) # pyright: ignore[reportArgumentType]

            for inp, solu in zip(inputs, sols):
                with self.subTest(**dist, inp=inp):
                    gnom_dist = model.apply_distortion(np.array(inp))

                    np.testing.assert_array_almost_equal(gnom_dist, solu)

    def test_get_projections(self):

        points = [[0, 0, 1], [-0.1, 0.2, 2.2], [[-0.1], [0.2], [2.2]], [[-0.1, 0], [0.2, 0], [2.2, 1]]]

        model = self.Class(fx=4050.5, fy=3050.25, alpha=1.5, px=1500, py=1500.5,
                           k1=0.5, k2=-0.3, k3=0.15, p1=1e-7, p2=1e-6, a1=1e-1, a2=-1e-6, a3=3e-7,
                           k4=1, k5=-5, k6=11, s1=1e-6, s2=1e2, s3=-3e-3, s4=5e-1,
                           field_of_view=100)

        temps = [0, 1, -1, 10, -10]

        for temp in temps:
            with self.subTest(misalignment=None, temp=temp):

                for point in points:
                    pin, dist, pix = model.get_projections(point, temperature=temp)

                    pin_true = np.array(point[:2]) / point[2]

                    dist_true = model.apply_distortion(pin_true)

                    dist_true *= model.get_temperature_scale(temp)

                    pix_true = (np.matmul(model.intrinsic_matrix[:, :2], dist_true).T + model.intrinsic_matrix[:, 2]).T

                    np.testing.assert_array_equal(pin, pin_true)
                    np.testing.assert_array_equal(dist, dist_true)
                    np.testing.assert_array_equal(pix, pix_true)

        model = self.Class(fx=4050.5, fy=3050.25, alpha=1.5, px=1500, py=1500.5,
                           k1=0.5, k2=-0.3, k3=0.15, p1=1e-7, p2=1e-6,
                           k4=1, k5=-5, k6=11, s1=1e-6, s2=1e2, s3=-3e-3, s4=5e-1,
                           misalignment=[0, 0, np.pi])

        with self.subTest(misalignment=[0, 0, np.pi]):

            for point in points:
                pin, dist, pix = model.get_projections(point)

                pin_true = -np.array(point[:2]) / point[2]

                dist_true = model.apply_distortion(pin_true)

                pix_true = (np.matmul(model.intrinsic_matrix[:, :2], dist_true).T + model.intrinsic_matrix[:, 2]).T

                np.testing.assert_array_almost_equal(pin, pin_true)
                np.testing.assert_array_almost_equal(dist, dist_true)
                np.testing.assert_array_almost_equal(pix, pix_true)

        model = self.Class(fx=4050.5, fy=3050.25, alpha=1.5, px=1500, py=1500.5,
                           k1=0.5, k2=-0.3, k3=0.15, p1=1e-7, p2=1e-6,
                           k4=1, k5=-5, k6=11, s1=1e-6, s2=1e2, s3=-3e-3, s4=5e-1,
                           misalignment=[np.pi, 0, 0])

        with self.subTest(misalignment=[np.pi, 0, 0]):

            for point in points:
                pin, dist, pix = model.get_projections(point)

                pin_true = np.array(point[:2]) / point[2]
                pin_true[0] *= -1

                dist_true = model.apply_distortion(pin_true)

                pix_true = (np.matmul(model.intrinsic_matrix[:, :2], dist_true).T + model.intrinsic_matrix[:, 2]).T

                np.testing.assert_array_almost_equal(pin, pin_true)
                np.testing.assert_array_almost_equal(dist, dist_true)
                np.testing.assert_array_almost_equal(pix, pix_true)

        model = self.Class(fx=4050.5, fy=3050.25, alpha=1.5, px=1500, py=1500.5,
                           k1=0.5, k2=-0.3, k3=0.15, p1=1e-7, p2=1e-6,
                           k4=1, k5=-5, k6=11, s1=1e-6, s2=1e2, s3=-3e-3, s4=5e-1,
                           misalignment=[0, np.pi, 0])

        with self.subTest(misalignment=[0, np.pi, 0]):

            for point in points:
                pin, dist, pix = model.get_projections(point)

                pin_true = np.array(point[:2]) / point[2]
                pin_true[1] *= -1

                dist_true = model.apply_distortion(pin_true)

                pix_true = (np.matmul(model.intrinsic_matrix[:, :2], dist_true).T + model.intrinsic_matrix[:, 2]).T

                np.testing.assert_array_almost_equal(pin, pin_true)
                np.testing.assert_array_almost_equal(dist, dist_true)
                np.testing.assert_array_almost_equal(pix, pix_true)

        model = self.Class(fx=4050.5, fy=3050.25, alpha=1.5, px=1500, py=1500.5,
                           k1=0.5, k2=-0.3, k3=0.15, p1=1e-7, p2=1e-6,
                           k4=1, k5=-5, k6=11, s1=1e-6, s2=1e2, s3=-3e-3, s4=5e-1,
                           misalignment=[1, 0.2, 0.3])

        with self.subTest(misalignment=[1, 0.2, 0.3]):

            rot_mat = at.rotvec_to_rotmat([1, 0.2, 0.3]).squeeze()

            for point in points:
                point_new = rot_mat @ point

                pin, dist, pix = model.get_projections(point)

                pin_true = np.array(point_new[:2]) / np.array(point_new[2])

                dist_true = model.apply_distortion(pin_true)

                pix_true = (np.matmul(model.intrinsic_matrix[:, :2], dist_true).T + model.intrinsic_matrix[:, 2]).T

                np.testing.assert_array_almost_equal(pin, pin_true)
                np.testing.assert_array_almost_equal(dist, dist_true)
                np.testing.assert_array_almost_equal(pix, pix_true)

        model = self.Class(fx=4050.5, fy=3050.25, alpha=1.5, px=1500, py=1500.5,
                           k1=0.5, k2=-0.3, k3=0.15, p1=1e-7, p2=1e-6,
                           k4=1, k5=-5, k6=11, s1=1e-6, s2=1e2, s3=-3e-3, s4=5e-1,
                           misalignment=[[1, 0.2, 0.3], [0, 0, np.pi]])

        model.estimate_multiple_misalignments = True

        with self.subTest(misalignment=[[1, 0.2, 0.3], [0, 0, np.pi]]):

            rot_mat = at.rotvec_to_rotmat([1, 0.2, 0.3]).squeeze()

            for point in points:
                point_new = rot_mat @ point

                pin, dist, pix = model.get_projections(point, image=0)

                pin_true = np.array(point_new[:2]) / np.array(point_new[2])

                dist_true = model.apply_distortion(pin_true)

                pix_true = (np.matmul(model.intrinsic_matrix[:, :2], dist_true).T + model.intrinsic_matrix[:, 2]).T

                np.testing.assert_array_almost_equal(pin, pin_true)
                np.testing.assert_array_almost_equal(dist, dist_true)
                np.testing.assert_array_almost_equal(pix, pix_true)

                pin, dist, pix = model.get_projections(point, image=1)

                pin_true = -np.array(point[:2]) / point[2]

                dist_true = model.apply_distortion(pin_true)

                pix_true = (np.matmul(model.intrinsic_matrix[:, :2], dist_true).T + model.intrinsic_matrix[:, 2]).T

                np.testing.assert_array_almost_equal(pin, pin_true)
                np.testing.assert_array_almost_equal(dist, dist_true)
                np.testing.assert_array_almost_equal(pix, pix_true)

    def test_project_onto_image(self):

        points = [[0, 0, 1], [-0.1, 0.2, 2.2], [[-0.1], [0.2], [2.2]], [[-0.1, 0], [0.2, 0], [2.2, 1]]]

        model = self.Class(fx=4050.5, fy=3050.25, alpha=1.5, px=1500, py=1500.5,
                           k1=0.5, k2=-0.3, k3=0.15, p1=1e-7, p2=1e-6, a1=1, a2=2, a3=-3,
                           k4=1, k5=-5, k6=11, s1=1e-6, s2=1e2, s3=-3e-3, s4=5e-1,
                           field_of_view=100)

        temps = [0, 1, -1, 10, -10]

        for temp in temps:

            with self.subTest(temp=temp, misalignment=None):

                for point in points:
                    _, __, pix = model.get_projections(point, temperature=temp)

                    pix_proj = model.project_onto_image(point, temperature=temp)

                    np.testing.assert_array_equal(pix, pix_proj)

        model = self.Class(fx=4050.5, fy=3050.25, alpha=1.5, px=1500, py=1500.5,
                           k1=0.5, k2=-0.3, k3=0.15, p1=1e-7, p2=1e-6,
                           k4=1, k5=-5, k6=11, s1=1e-6, s2=1e2, s3=-3e-3, s4=5e-1,
                           misalignment=[[1, 0.2, 0.3], [0, 0, np.pi]])

        model.estimate_multiple_misalignments = True

        with self.subTest(misalignment=[[1, 0.2, 0.3], [0, 0, np.pi]]):

            for point in points:
                _, __, pix = model.get_projections(point, image=0)

                pix_proj = model.project_onto_image(point, image=0)

                np.testing.assert_array_equal(pix, pix_proj)

                _, __, pix = model.get_projections(point, image=1)

                pix_proj = model.project_onto_image(point, image=1)

                np.testing.assert_array_equal(pix, pix_proj)

    def test_compute_pixel_jacobian(self):

        def num_deriv(uvec, cmodel, delta=1e-8, image=0, temperature=0) -> np.ndarray:

            uvec = np.array(uvec).reshape(3, -1)

            pix_true = cmodel.project_onto_image(uvec, image=image, temperature=temperature)

            uvec_pert = uvec + [[delta], [0], [0]]

            pix_pert_x_f = cmodel.project_onto_image(uvec_pert, image=image, temperature=temperature)

            uvec_pert = uvec + [[0], [delta], [0]]

            pix_pert_y_f = cmodel.project_onto_image(uvec_pert, image=image, temperature=temperature)

            uvec_pert = uvec + [[0], [0], [delta]]

            pix_pert_z_f = cmodel.project_onto_image(uvec_pert, image=image, temperature=temperature)

            uvec_pert = uvec - [[delta], [0], [0]]

            pix_pert_x_b = cmodel.project_onto_image(uvec_pert, image=image, temperature=temperature)

            uvec_pert = uvec - [[0], [delta], [0]]

            pix_pert_y_b = cmodel.project_onto_image(uvec_pert, image=image, temperature=temperature)

            uvec_pert = uvec - [[0], [0], [delta]]

            pix_pert_z_b = cmodel.project_onto_image(uvec_pert, image=image, temperature=temperature)

            return np.array([(pix_pert_x_f-pix_pert_x_b)/(2*delta),
                             (pix_pert_y_f-pix_pert_y_b)/(2*delta),
                             (pix_pert_z_f-pix_pert_z_b)/(2*delta)]).swapaxes(0, -1)

        inputs = [np.array([[0.5, 0, 1]]).T,
                  np.array([[0, 0.5, 1], [0.5, 0.5, 1], [-0.5, 0, 1]]).T,
                  np.array([[0.1, -0.5, 1], [-0.5, -0.5, 1], [5, 10, 1000.23], [1, 2, 1200.23]]).T]

        temperatures = [0, 1, -1, 10.5, -10.5]

        model = self.Class(fx=4050.5, fy=5050.25, alpha=1.5, px=1500, py=1500.5,
                           a1=0.15e-7, a2=-0.01e-8, a3=1e-9,
                           k1=0.5, k2=-0.3, k3=0.15, p1=1e-7, p2=1e-6,
                           k4=-0.5, k5=0.02, k6=0.01, s1=-0.45, s2=0.0045, s3=-0.8, s4=0.9,
                           misalignment=[[1e-9, -1e-9, 2e-9], [-1e-9, 2e-9, -1e-9]])

        model.estimate_multiple_misalignments = True

        for temp in temperatures:

            for input in inputs:

                for image in range(2):

                    with self.subTest(image=image, temp=temp, input=input):

                        jac_ana = model.compute_pixel_jacobian(input, image=image, temperature=temp)

                        jac_num = num_deriv(input, model, image=image, temperature=temp, delta=1e-2)

                        np.testing.assert_allclose(jac_ana, jac_num, rtol=1e-3, atol=1e-10)

    def test__compute_dcamera_point_dgnomic(self):

        def num_deriv(gnomic_locations, cmodel, delta=1e-6) -> np.ndarray:

            def g2u(g):

                v = np.vstack([g, cmodel.focal_length*np.ones(g.shape[-1])])

                return v/np.linalg.norm(v, axis=0, keepdims=True)

            gnomic_locations = np.asarray(gnomic_locations).reshape(2, -1)

            gnom_pert = gnomic_locations + [[delta], [0]]

            cam_loc_pert_x_f = g2u(gnom_pert)

            gnom_pert = gnomic_locations + [[0], [delta]]

            cam_loc_pert_y_f = g2u(gnom_pert)

            gnom_pert = gnomic_locations - [[delta], [0]]

            cam_loc_pert_x_b = g2u(gnom_pert)

            gnom_pert = gnomic_locations - [[0], [delta]]

            cam_loc_pert_y_b = g2u(gnom_pert)

            return np.array([(cam_loc_pert_x_f -cam_loc_pert_x_b)/(2*delta),
                             (cam_loc_pert_y_f -cam_loc_pert_y_b)/(2*delta)]).swapaxes(0, -1)

        inputs = [np.array([[0, 0]]).T,
                  np.array([[0, 2000], [2000, 0], [2000, 2000]]).T,
                  np.array([[1000, 1000], [1000, 2000], [2000, 1000], [0, 1000], [1000, 0]]).T]

        model = self.Class(fx=4050.5, fy=5050.25, alpha=1.5, px=1500, py=1500.5,
                           a1=0.15e-7, a2=-0.01e-8, a3=1e-9,
                           k1=0.5, k2=-0.3, k3=0.15, p1=1e-7, p2=1e-6,
                           k4=-0.5, k5=0.02, k6=0.01, s1=-0.45, s2=0.0045, s3=-0.8, s4=0.9,
                           misalignment=[[1e-9, -1e-9, 2e-9], [-1e-9, 2e-9, -1e-9]])

        model.estimate_multiple_misalignments = True

        for input in inputs:

            with self.subTest(input=input):
                jac_ana = []
                for gnom in input.T:
                    jac_ana.append(
                        model._compute_dcamera_point_dgnomic(gnom, np.sqrt(np.sum(gnom*gnom) + model.focal_length**2)))

                jac_ana = np.array(jac_ana)

                jac_num = num_deriv(input, model)

                np.testing.assert_almost_equal(jac_ana, jac_num)

    def test__compute_dgnomic_ddist_gnomic(self):

        def num_deriv(dist_gnomic_locations, cmodel, delta=1e-6) -> np.ndarray:

            def dg2g(dg):

                gnomic_guess = dg.copy()

                # perform the fpa
                for _ in np.arange(20):

                    # get the distorted location assuming the current guess is correct
                    gnomic_guess_distorted = cmodel.apply_distortion(gnomic_guess)

                    # subtract off the residual distortion from the gnomic guess
                    gnomic_guess += dg - gnomic_guess_distorted

                    # check for convergence
                    if np.all(np.linalg.norm(gnomic_guess_distorted - dg, axis=0) <= 1e-15):
                        break

                return gnomic_guess

            dist_gnomic_locations = np.asarray(dist_gnomic_locations).reshape(2, -1)

            dist_gnom_pert = dist_gnomic_locations + [[delta], [0]]

            gnom_loc_pert_x_f = dg2g(dist_gnom_pert)

            dist_gnom_pert = dist_gnomic_locations + [[0], [delta]]

            gnom_loc_pert_y_f = dg2g(dist_gnom_pert)

            dist_gnom_pert = dist_gnomic_locations - [[delta], [0]]

            gnom_loc_pert_x_b = dg2g(dist_gnom_pert)

            dist_gnom_pert = dist_gnomic_locations - [[0], [delta]]

            gnom_loc_pert_y_b = dg2g(dist_gnom_pert)

            return np.array([(gnom_loc_pert_x_f - gnom_loc_pert_x_b)/(2*delta),
                             (gnom_loc_pert_y_f - gnom_loc_pert_y_b)/(2*delta)]).swapaxes(0, -1)

        inputs = [np.array([[0, 0]]).T,
                  np.array([[0, 0.1], [0.1, 0], [0.1, 0.1]]).T/100,
                  np.array([[-0.1, 0], [0, -0.1], [-0.1, -0.1], [0.1, -0.1], [-0.1, 0.1]]).T/100]

        model = self.Class(fx=4050.5, fy=5050.25, alpha=1.5, px=1500, py=1500.5,
                           a1=0.15e-7, a2=-0.01e-8, a3=1e-9,
                           k1=0.005, k2=-0.003, k3=0.0015, p1=1e-7, p2=1e-6,
                           k4=-0.005, k5=0.0002, k6=0.0001, s1=-0.0045, s2=0.000045, s3=-0.008, s4=0.009,
                           misalignment=[[1e-9, -1e-9, 2e-9], [-1e-9, 2e-9, -1e-9]])

        model.estimate_multiple_misalignments = True

        for input in inputs:

            with self.subTest(input=input):
                jac_ana = []
                for dist_gnom in input.T:
                    jac_ana.append(model._compute_dgnomic_ddist_gnomic(dist_gnom))

                jac_ana = np.array(jac_ana)

                jac_num = num_deriv(input, model, delta=1e-4)

                np.testing.assert_allclose(jac_ana, jac_num, rtol=1e-1, atol=1e-10)

    def test_compute_unit_vector_jacobian(self):

        def num_deriv(pixels, cmodel, delta=1e-6, image=0, temperature=0) -> np.ndarray:

            pixels = np.array(pixels).reshape(2, -1)

            pix_pert = pixels + [[delta], [0]]

            uvec_pert_x_f = cmodel.pixels_to_unit(pix_pert, image=image, temperature=temperature)

            pix_pert = pixels + [[0], [delta]]

            uvec_pert_y_f = cmodel.pixels_to_unit(pix_pert, image=image, temperature=temperature)

            pix_pert = pixels - [[delta], [0]]

            uvec_pert_x_b = cmodel.pixels_to_unit(pix_pert, image=image, temperature=temperature)

            pix_pert = pixels - [[0], [delta]]

            uvec_pert_y_b = cmodel.pixels_to_unit(pix_pert, image=image, temperature=temperature)

            return np.array([(uvec_pert_x_f-uvec_pert_x_b)/(2*delta),
                             (uvec_pert_y_f-uvec_pert_y_b)/(2*delta)]).swapaxes(0, -1)

        inputs = [np.array([[0, 0]]).T,
                  np.array([[0, 2000], [2000, 0], [2000, 2000]]).T/10,
                  np.array([[1000, 1000], [1000, 2000], [2000, 1000], [0, 1000], [1000, 0]]).T/10]

        temperatures = [0, 1, -1, 10.5, -10.5]

        model = self.Class(fx=4050.5, fy=5050.25, alpha=1.5, px=100, py=100.5,
                           a1=0.15e-7, a2=-0.01e-8, a3=1e-9,
                           k1=0.005, k2=-0.003, k3=0.0015, p1=1e-7, p2=1e-6,
                           k4=-0.005, k5=0.0002, k6=0.0001, s1=-0.0045, s2=0.000045, s3=-0.008, s4=0.009,
                           misalignment=[[1e-9, -1e-9, 2e-9], [-1e-9, 2e-9, -1e-9]])

        model.estimate_multiple_misalignments = True

        for temp in temperatures:

            for input in inputs:

                for image in range(2):

                    with self.subTest(image=image, temp=temp, input=input):

                        jac_ana = model.compute_unit_vector_jacobian(input, image=image, temperature=temp)

                        jac_num = num_deriv(input, model, image=image, temperature=temp, delta=1)

                        np.testing.assert_allclose(jac_ana, jac_num, rtol=1e-3, atol=1e-10)

    def test__compute_ddistorted_gnomic_dgnomic(self):

        def num_deriv(loc, cmodel, delta=1e-8) -> np.ndarray:
            loc_pert = np.array(loc) + [delta, 0]
            dist_pert_x_f = cmodel.apply_distortion(loc_pert)
            loc_pert = np.array(loc) + [0, delta]
            dist_pert_y_f = cmodel.apply_distortion(loc_pert)

            loc_pert = np.array(loc) - [delta, 0]
            dist_pert_x_b = cmodel.apply_distortion(loc_pert)
            loc_pert = np.array(loc) - [0, delta]
            dist_pert_y_b = cmodel.apply_distortion(loc_pert)

            return np.array(
                [(dist_pert_x_f - dist_pert_x_b) / (2 * delta), (dist_pert_y_f - dist_pert_y_b) / (2 * delta)]).T

        dist_coefs = [{"k1": 1.5},
                      {"k2": 1.5},
                      {"k3": 1.5},
                      {"p1": 1.5},
                      {"p2": 1.5},
                      {"k4": 1.5},
                      {"k5": 1.5},
                      {"k6": 1.5},
                      {"s1": 1.5},
                      {"s2": 1.5},
                      {"s3": 1.5},
                      {"s4": 1.5},
                      {"k1": -1.5, "k2": -1.5, "k3": -1.5, "p1": -1.5, "p2": -1.5,
                       "k4": -1.5, "k5": -1.5, "k6": -1.5, "s1": -1.5, "s2": -1.5, "s3": -1.5, "s4": -1.5}]

        inputs = [[0, 0], [1, 0], [-1, 0], [1.5, 0], [-1.5, 0], [0, 1], [0, -1], [0, 1.5], [0, -1.5], [1, 1]]

        for dist_coef in dist_coefs:

            model = self.Class(**dist_coef) # pyright: ignore[reportArgumentType]

            for inp in inputs:
                with self.subTest(**dist_coef, inp=inp):
                    r = np.sqrt(inp[0] ** 2 + inp[1] ** 2)
                    r2 = r ** 2
                    r4 = r ** 4
                    r6 = r ** 6

                    num = num_deriv(inp, model)

                    ana = model._compute_ddistorted_gnomic_dgnomic(np.array(inp), r2, r4, r6)

                    np.testing.assert_allclose(num, ana, atol=1e-14)

    def test__compute_dpixel_ddistorted_gnomic(self):

        def num_deriv(loc, cmodel, delta=1e-8, temperature=0) -> np.ndarray:
            loc_pert = np.array(loc) + [delta, 0]
            loc_pert *= cmodel.get_temperature_scale(temperature)
            pix_pert_x_f = cmodel.intrinsic_matrix[:, :2] @ loc_pert + cmodel.intrinsic_matrix[:, 2]
            loc_pert = np.array(loc) + [0, delta]
            loc_pert *= cmodel.get_temperature_scale(temperature)
            pix_pert_y_f = cmodel.intrinsic_matrix[:, :2] @ loc_pert + cmodel.intrinsic_matrix[:, 2]

            loc_pert = np.array(loc) - [delta, 0]
            loc_pert *= cmodel.get_temperature_scale(temperature)
            pix_pert_x_b = cmodel.intrinsic_matrix[:, :2] @ loc_pert + cmodel.intrinsic_matrix[:, 2]
            loc_pert = np.array(loc) - [0, delta]
            loc_pert *= cmodel.get_temperature_scale(temperature)
            pix_pert_y_b = cmodel.intrinsic_matrix[:, :2] @ loc_pert + cmodel.intrinsic_matrix[:, 2]

            return np.array(
                [(pix_pert_x_f - pix_pert_x_b) / (2 * delta), (pix_pert_y_f - pix_pert_y_b) / (2 * delta)]).T

        intrins_coefs = [{"fx": 1.5, "fy": 0, "alpha": 0, "px": 0, "py": 0},
                         {"fx": 0, "fy": 1.5, "alpha": 0, "px": 0, "py": 0},
                         {"fx": 0, "fy": 0, "alpha": 1.5, "px": 0, "py": 0},
                         {"fx": 0, "fy": 0, "alpha": 0, "px": 1.5, "py": 0},
                         {"fx": 0, "fy": 0, "alpha": 0, "px": 0, "py": 1.5},
                         {"fx": -1.5, "fy": -1.5, "alpha": -1.5, "px": 1.5, "py": 1.5}]

        inputs = [[0, 0], [1, 0], [-1, 0], [1.5, 0], [-1.5, 0], [0, 1], [0, -1], [0, 1.5], [0, -1.5], [1, 1]]

        temps = [0, 1, -1, 10.5, -10.5]

        for temp in temps:

            for intrins_coef in intrins_coefs:

                model = self.Class(**intrins_coef)

                with self.subTest(**intrins_coef, temp=temp):

                    for inp in inputs:
                        num = num_deriv(np.array(inp), model, temperature=temp)

                        ana = model._compute_dpixel_ddistorted_gnomic(temperature=temp)

                        np.testing.assert_allclose(num, ana, atol=1e-14)

    def test__compute_dpixel_dintrinsic(self):
        def num_deriv(loc, cmodel, delta=1e-6) -> np.ndarray:

            model_pert = cmodel.copy()
            model_pert.kx += delta
            pix_pert_kx_f = model_pert.intrinsic_matrix[:, :2] @ loc + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.kxy += delta
            pix_pert_kxy_f = model_pert.intrinsic_matrix[:, :2] @ loc + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.ky += delta
            pix_pert_ky_f = model_pert.intrinsic_matrix[:, :2] @ loc + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.px += delta
            pix_pert_px_f = model_pert.intrinsic_matrix[:, :2] @ loc + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.py += delta
            pix_pert_py_f = model_pert.intrinsic_matrix[:, :2] @ loc + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.kx -= delta
            pix_pert_kx_b = model_pert.intrinsic_matrix[:, :2] @ loc + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.kxy -= delta
            pix_pert_kxy_b = model_pert.intrinsic_matrix[:, :2] @ loc + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.ky -= delta
            pix_pert_ky_b = model_pert.intrinsic_matrix[:, :2] @ loc + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.px -= delta
            pix_pert_px_b = model_pert.intrinsic_matrix[:, :2] @ loc + model_pert.intrinsic_matrix[:, 2]

            model_pert = cmodel.copy()
            model_pert.py -= delta
            pix_pert_py_b = model_pert.intrinsic_matrix[:, :2] @ loc + model_pert.intrinsic_matrix[:, 2]

            return np.array([(pix_pert_kx_f - pix_pert_kx_b) / (2 * delta),
                             (pix_pert_ky_f - pix_pert_ky_b) / (2 * delta),
                             (pix_pert_kxy_f - pix_pert_kxy_b) / (2 * delta),
                             (pix_pert_px_f - pix_pert_px_b) / (2 * delta),
                             (pix_pert_py_f - pix_pert_py_b) / (2 * delta)]).T

        intrins_coefs = [{"fx": 1.5, "fy": 0, "alpha": 0, "px": 0, "py": 0},
                         {"fx": 0, "fy": 1.5, "alpha": 0, "px": 0, "py": 0},
                         {"fx": 0, "fy": 0, "alpha": 1.5, "px": 0, "py": 0},
                         {"fx": 0, "fy": 0, "alpha": 0, "px": 1.5, "py": 0},
                         {"fx": 0, "fy": 0, "alpha": 0, "px": 0, "py": 1.5},
                         {"fx": -1.5, "fy": -1.5, "alpha": -1.5, "px": 1.5, "py": 1.5}]

        inputs = [[1e-6, 1e-6], [1, 0], [-1, 0], [1.5, 0], [-1.5, 0], [0, 1], [0, -1], [0, 1.5], [0, -1.5], [1, 1]]

        for intrins_coef in intrins_coefs:

            model = self.Class(**intrins_coef)

            for inp in inputs:
                with self.subTest(**intrins_coef, inp=inp):
                    num = num_deriv(inp, model, delta=1e-5)

                    ana = model._compute_dpixel_dintrinsic(np.array(inp))

                    np.testing.assert_allclose(num, ana, atol=1e-14, rtol=1e-5)

    def test__compute_ddistorted_gnomic_ddistortion(self):

        def num_deriv(loc, cmodel, delta=1e-8) -> np.ndarray:

            model_pert = cmodel.copy()
            model_pert.k1 += delta
            loc_pert_k1_f = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.k2 += delta
            loc_pert_k2_f = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.k3 += delta
            loc_pert_k3_f = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.k4 += delta
            loc_pert_k4_f = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.k5 += delta
            loc_pert_k5_f = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.k6 += delta
            loc_pert_k6_f = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.p1 += delta
            loc_pert_p1_f = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.p2 += delta
            loc_pert_p2_f = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.s1 += delta
            loc_pert_s1_f = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.s2 += delta
            loc_pert_s2_f = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.s3 += delta
            loc_pert_s3_f = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.s4 += delta
            loc_pert_s4_f = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.k1 -= delta
            loc_pert_k1_b = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.k2 -= delta
            loc_pert_k2_b = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.k3 -= delta
            loc_pert_k3_b = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.k4 -= delta
            loc_pert_k4_b = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.k5 -= delta
            loc_pert_k5_b = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.k6 -= delta
            loc_pert_k6_b = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.p1 -= delta
            loc_pert_p1_b = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.p2 -= delta
            loc_pert_p2_b = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.s1 -= delta
            loc_pert_s1_b = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.s2 -= delta
            loc_pert_s2_b = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.s3 -= delta
            loc_pert_s3_b = model_pert.apply_distortion(loc)

            model_pert = cmodel.copy()
            model_pert.s4 -= delta
            loc_pert_s4_b = model_pert.apply_distortion(loc)

            return np.array([(loc_pert_k1_f - loc_pert_k1_b) / (2 * delta),
                             (loc_pert_k2_f - loc_pert_k2_b) / (2 * delta),
                             (loc_pert_k3_f - loc_pert_k3_b) / (2 * delta),
                             (loc_pert_k4_f - loc_pert_k4_b) / (2 * delta),
                             (loc_pert_k5_f - loc_pert_k5_b) / (2 * delta),
                             (loc_pert_k6_f - loc_pert_k6_b) / (2 * delta),
                             (loc_pert_p1_f - loc_pert_p1_b) / (2 * delta),
                             (loc_pert_p2_f - loc_pert_p2_b) / (2 * delta),
                             (loc_pert_s1_f - loc_pert_s1_b) / (2 * delta),
                             (loc_pert_s2_f - loc_pert_s2_b) / (2 * delta),
                             (loc_pert_s3_f - loc_pert_s3_b) / (2 * delta),
                             (loc_pert_s4_f - loc_pert_s4_b) / (2 * delta)]).T

        dist_coefs = [{"k1": 1.5},
                      {"k2": 1.5},
                      {"k3": 1.5},
                      {"p1": 1.5},
                      {"p2": 1.5},
                      {"k4": 1.5},
                      {"k5": 1.5},
                      {"k6": 1.5},
                      {"s1": 1.5},
                      {"s2": 1.5},
                      {"s3": 1.5},
                      {"s4": 1.5},
                      {"k1": -1.5, "k2": -1.5, "k3": -1.5, "p1": -1.5, "p2": -1.5,
                       "k4": -1.5, "k5": -1.5, "k6": -1.5, "s1": -1.5, "s2": -1.5, "s3": -1.5, "s4": -1.5}]

        inputs = [[0, 0], [1, 0], [-1, 0], [1.5, 0], [-1.5, 0], [0, 1], [0, -1], [0, 1.5], [0, -1.5], [1, 1]]

        for dist_coef in dist_coefs:

            model = self.Class(**dist_coef) # pyright: ignore[reportArgumentType]

            with self.subTest(**dist_coef):

                for inp in inputs:
                    r = np.sqrt(inp[0] ** 2 + inp[1] ** 2)
                    r2 = r ** 2
                    r4 = r ** 4
                    r6 = r ** 6

                    num = num_deriv(np.array(inp), model)

                    ana = model._compute_ddistorted_gnomic_ddistortion(np.array(inp), r2, r4, r6)

                    np.testing.assert_allclose(num, ana, rtol=1e-5, atol=1e-14)

    def test__compute_dgnomic_dcamera_point(self):
        def num_deriv(loc, cmodel, delta=1e-8) -> np.ndarray:
            loc_pert = np.array(loc) + [delta, 0, 0]
            gnom_pert_x_f = cmodel.get_projections(loc_pert)[0]
            loc_pert = np.array(loc) + [0, delta, 0]
            gnom_pert_y_f = cmodel.get_projections(loc_pert)[0]
            loc_pert = np.array(loc) + [0, 0, delta]
            gnom_pert_z_f = cmodel.get_projections(loc_pert)[0]

            loc_pert = np.array(loc) - [delta, 0, 0]
            gnom_pert_x_b = cmodel.get_projections(loc_pert)[0]
            loc_pert = np.array(loc) - [0, delta, 0]
            gnom_pert_y_b = cmodel.get_projections(loc_pert)[0]
            loc_pert = np.array(loc) - [0, 0, delta]
            gnom_pert_z_b = cmodel.get_projections(loc_pert)[0]

            return np.array([(gnom_pert_x_f - gnom_pert_x_b) / (2 * delta),
                             (gnom_pert_y_f - gnom_pert_y_b) / (2 * delta),
                             (gnom_pert_z_f - gnom_pert_z_b) / (2 * delta)]).T

        inputs = [[0, 0, 1], [0.5, 0, 1], [0, 0.5, 1], [0.5, 0.5, 1], [-0.5, 0, 1], [0, -0.5, 1], [-0.5, -0.5, 1],
                  [5, 10, 1000.23], [0.5, 1e-14, 1]]

        model = self.Class()

        for inp in inputs:
            num = num_deriv(inp, model)

            ana = model._compute_dgnomic_dcamera_point(np.array(inp))

            np.testing.assert_allclose(num, ana, atol=1e-9, rtol=1e-5)

    def test_get_jacobian_row(self):

        def num_deriv(loc, temp, cmodel, delta=1e-8, image=0) -> np.ndarray:
            model_pert = cmodel.copy()
            model_pert.fx += delta

            pix_pert_fx_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.fy += delta
            pix_pert_fy_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.alpha += delta
            pix_pert_skew_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.px += delta
            pix_pert_px_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.py += delta
            pix_pert_py_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.a1 += delta
            pix_pert_a1_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.a2 += delta
            pix_pert_a2_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.a3 += delta
            pix_pert_a3_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.fx -= delta
            pix_pert_fx_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.fy -= delta
            pix_pert_fy_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.alpha -= delta
            pix_pert_skew_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.px -= delta
            pix_pert_px_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.py -= delta
            pix_pert_py_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.k1 += delta
            pix_pert_k1_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.k2 += delta
            pix_pert_k2_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.k3 += delta
            pix_pert_k3_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.k4 += delta
            pix_pert_k4_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.k5 += delta
            pix_pert_k5_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.k6 += delta
            pix_pert_k6_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.p1 += delta
            pix_pert_p1_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.p2 += delta
            pix_pert_p2_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.s1 += delta
            pix_pert_s1_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.s2 += delta
            pix_pert_s2_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.s3 += delta
            pix_pert_s3_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.s4 += delta
            pix_pert_s4_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.k1 -= delta
            pix_pert_k1_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.k2 -= delta
            pix_pert_k2_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.k3 -= delta
            pix_pert_k3_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.k4 -= delta
            pix_pert_k4_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.k5 -= delta
            pix_pert_k5_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.k6 -= delta
            pix_pert_k6_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.p1 -= delta
            pix_pert_p1_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.p2 -= delta
            pix_pert_p2_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.s1 -= delta
            pix_pert_s1_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.s2 -= delta
            pix_pert_s2_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.s3 -= delta
            pix_pert_s3_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.s4 -= delta
            pix_pert_s4_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][0] += delta
            pix_pert_mx_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][1] += delta
            pix_pert_my_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][2] += delta
            pix_pert_mz_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][0] -= delta
            pix_pert_mx_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][1] -= delta
            pix_pert_my_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][2] -= delta
            pix_pert_mz_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.a1 -= delta
            pix_pert_a1_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.a2 -= delta
            pix_pert_a2_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.a3 -= delta
            pix_pert_a3_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            return np.vstack([(pix_pert_fx_f - pix_pert_fx_b) / (delta * 2),
                              (pix_pert_fy_f - pix_pert_fy_b) / (delta * 2),
                              (pix_pert_skew_f - pix_pert_skew_b) / (delta * 2),
                              (pix_pert_px_f - pix_pert_px_b) / (delta * 2),
                              (pix_pert_py_f - pix_pert_py_b) / (delta * 2),
                              (pix_pert_k1_f - pix_pert_k1_b) / (delta * 2),
                              (pix_pert_k2_f - pix_pert_k2_b) / (delta * 2),
                              (pix_pert_k3_f - pix_pert_k3_b) / (delta * 2),
                              (pix_pert_k4_f - pix_pert_k4_b) / (delta * 2),
                              (pix_pert_k5_f - pix_pert_k5_b) / (delta * 2),
                              (pix_pert_k6_f - pix_pert_k6_b) / (delta * 2),
                              (pix_pert_p1_f - pix_pert_p1_b) / (delta * 2),
                              (pix_pert_p2_f - pix_pert_p2_b) / (delta * 2),
                              (pix_pert_s1_f - pix_pert_s1_b) / (delta * 2),
                              (pix_pert_s2_f - pix_pert_s2_b) / (delta * 2),
                              (pix_pert_s3_f - pix_pert_s3_b) / (delta * 2),
                              (pix_pert_s4_f - pix_pert_s4_b) / (delta * 2),
                              (pix_pert_a1_f - pix_pert_a1_b) / (delta * 2),
                              (pix_pert_a2_f - pix_pert_a2_b) / (delta * 2),
                              (pix_pert_a3_f - pix_pert_a3_b) / (delta * 2),
                              np.zeros((image * 3, 2)),
                              (pix_pert_mx_f - pix_pert_mx_b) / (delta * 2),
                              (pix_pert_my_f - pix_pert_my_b) / (delta * 2),
                              (pix_pert_mz_f - pix_pert_mz_b) / (delta * 2)]).T

        model = self.Class(fx=4050.5, fy=5050.25, alpha=1.5, px=1500, py=1500.5,
                           a1=0.15e-7, a2=-0.01e-8, a3=1e-9,
                           k1=0.5, k2=-0.3, k3=0.15, p1=1e-7, p2=1e-6,
                           k4=-0.5, k5=0.02, k6=0.01, s1=-0.45, s2=0.0045, s3=-0.8, s4=0.9,
                           misalignment=[[1e-9, -1e-9, 2e-9], [-1e-9, 2e-9, -1e-9]])

        model.estimate_multiple_misalignments = True

        inputs = [[0.5, 0, 1], [0, 0.5, 1], [0.5, 0.5, 1], [-0.5, 0, 1], [0, -0.5, 1], [-0.5, -0.5, 1],
                  [5, 10, 1000.23], [[1], [2], [1200.23]]]

        temps = [0, 1.5, -10]

        # TODO: investigate if this is actually correct
        for temperature in temps:
            for inp in inputs:
                with self.subTest(temperature=temperature, inp=inp):
                    num = num_deriv(inp, temperature, model, delta=1e-2)
                    ana = model._get_jacobian_row(np.array(inp), 0, 1, temperature=temperature)

                    np.testing.assert_allclose(ana, num, rtol=1e-1, atol=1e-10)

                    num = num_deriv(inp, temperature, model, delta=1e-2, image=1)
                    ana = model._get_jacobian_row(np.array(inp), 1, 2, temperature=temperature)

                    np.testing.assert_allclose(ana, num, atol=1e-10, rtol=1e-1)

    def test_compute_jacobian(self):

        def num_deriv(loc, temp, cmodel, delta=1e-8, image=0, nimages=2) -> np.ndarray:

            model_pert = cmodel.copy()
            model_pert.fx += delta
            pix_pert_fx_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.fy += delta
            pix_pert_fy_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.alpha += delta
            pix_pert_skew_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.px += delta
            pix_pert_px_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.py += delta
            pix_pert_py_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.a1 += delta
            pix_pert_a1_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.a2 += delta
            pix_pert_a2_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.a3 += delta
            pix_pert_a3_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.fx -= delta
            pix_pert_fx_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.fy -= delta
            pix_pert_fy_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.alpha -= delta
            pix_pert_skew_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.px -= delta
            pix_pert_px_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.py -= delta
            pix_pert_py_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.k1 += delta
            pix_pert_k1_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.k2 += delta
            pix_pert_k2_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.k3 += delta
            pix_pert_k3_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.k4 += delta
            pix_pert_k4_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.k5 += delta
            pix_pert_k5_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.k6 += delta
            pix_pert_k6_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.p1 += delta
            pix_pert_p1_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.p2 += delta
            pix_pert_p2_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.s1 += delta
            pix_pert_s1_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.s2 += delta
            pix_pert_s2_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.s3 += delta
            pix_pert_s3_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.s4 += delta
            pix_pert_s4_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.k1 -= delta
            pix_pert_k1_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.k2 -= delta
            pix_pert_k2_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.k3 -= delta
            pix_pert_k3_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.k4 -= delta
            pix_pert_k4_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.k5 -= delta
            pix_pert_k5_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.k6 -= delta
            pix_pert_k6_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.p1 -= delta
            pix_pert_p1_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.p2 -= delta
            pix_pert_p2_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.s1 -= delta
            pix_pert_s1_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.s2 -= delta
            pix_pert_s2_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.s3 -= delta
            pix_pert_s3_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.s4 -= delta
            pix_pert_s4_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][0] += delta
            pix_pert_mx_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][1] += delta
            pix_pert_my_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][2] += delta
            pix_pert_mz_f = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][0] -= delta
            pix_pert_mx_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][1] -= delta
            pix_pert_my_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.misalignment[image][2] -= delta
            pix_pert_mz_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.a1 -= delta
            pix_pert_a1_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.a2 -= delta
            pix_pert_a2_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            model_pert = cmodel.copy()
            model_pert.a3 -= delta
            pix_pert_a3_b = model_pert.project_onto_image(loc, image=image, temperature=temp).ravel()

            return np.vstack([(pix_pert_fx_f - pix_pert_fx_b) / (delta * 2),
                              (pix_pert_fy_f - pix_pert_fy_b) / (delta * 2),
                              (pix_pert_skew_f - pix_pert_skew_b) / (delta * 2),
                              (pix_pert_px_f - pix_pert_px_b) / (delta * 2),
                              (pix_pert_py_f - pix_pert_py_b) / (delta * 2),
                              (pix_pert_k1_f - pix_pert_k1_b) / (delta * 2),
                              (pix_pert_k2_f - pix_pert_k2_b) / (delta * 2),
                              (pix_pert_k3_f - pix_pert_k3_b) / (delta * 2),
                              (pix_pert_k4_f - pix_pert_k4_b) / (delta * 2),
                              (pix_pert_k5_f - pix_pert_k5_b) / (delta * 2),
                              (pix_pert_k6_f - pix_pert_k6_b) / (delta * 2),
                              (pix_pert_p1_f - pix_pert_p1_b) / (delta * 2),
                              (pix_pert_p2_f - pix_pert_p2_b) / (delta * 2),
                              (pix_pert_s1_f - pix_pert_s1_b) / (delta * 2),
                              (pix_pert_s2_f - pix_pert_s2_b) / (delta * 2),
                              (pix_pert_s3_f - pix_pert_s3_b) / (delta * 2),
                              (pix_pert_s4_f - pix_pert_s4_b) / (delta * 2),
                              (pix_pert_a1_f - pix_pert_a1_b) / (delta * 2),
                              (pix_pert_a2_f - pix_pert_a2_b) / (delta * 2),
                              (pix_pert_a3_f - pix_pert_a3_b) / (delta * 2),
                              np.zeros((image * 3, 2)),
                              (pix_pert_mx_f - pix_pert_mx_b) / (delta * 2),
                              (pix_pert_my_f - pix_pert_my_b) / (delta * 2),
                              (pix_pert_mz_f - pix_pert_mz_b) / (delta * 2),
                              np.zeros(((nimages - image - 1) * 3, 2))]).T

        model = self.Class(fx=4050.5, fy=5050.25, alpha=1.5, px=1500, py=1500.5,
                           k1=0.5, k2=-0.3, k3=0.15, p1=1e-7, p2=1e-6,
                           k4=-0.5, k5=0.02, k6=0.01, s1=-0.45, s2=0.0045, s3=-0.8, s4=0.9,
                           misalignment=[[1e-9, -1e-9, 2e-9],
                                         [-1e-9, 2e-9, -1e-9],
                                         [1e-10, 2e-11, 3e-12]],
                           a1=0.15e-6, a2=-0.01e-7, a3=0.5e-8,
                           estimation_parameters=['intrinsic', 'temperature dependence', 'multiple misalignments'])

        inputs = [np.array([[0.5, 0, 1]]).T,
                  np.array([[0, 0.5, 1], [0.5, 0.5, 1], [-0.5, 0, 1]]).T,
                  np.array([[0.1, -0.5, 1], [-0.5, -0.5, 1], [5, 10, 1000.23], [1, 2, 1200.23]]).T]

        model.use_a_priori = False

        temps = [0, -20, 20.5]

        jac_ana = model.compute_jacobian(inputs, temperature=temps)

        jac_num = []

        numim = len(inputs)

        for ind, inp in enumerate(inputs):

            temperature = temps[ind]

            for vec in inp.T:
                jac_num.append(num_deriv(vec.T, temperature, model, delta=1e-3, image=ind, nimages=numim))

        np.testing.assert_allclose(jac_ana, np.vstack(jac_num), rtol=1e-1, atol=1e-9)

        model.use_a_priori = True

        jac_ana = model.compute_jacobian(inputs, temperature=temps)

        jac_num = []

        numim = len(inputs)

        for ind, inp in enumerate(inputs):

            temperature = temps[ind]

            for vec in inp.T:
                jac_num.append(num_deriv(vec.T, temperature, model, delta=1e-3, image=ind, nimages=numim))

        jac_num = np.vstack(jac_num)

        jac_num = np.pad(jac_num, [(0, jac_num.shape[1]), (0, 0)], 'constant', constant_values=0)

        jac_num[-jac_num.shape[1]:] = np.eye(jac_num.shape[1])

        np.testing.assert_allclose(jac_ana, jac_num, rtol=1e-1, atol=1e-9)

    def test_apply_update(self):
        model_param = {"fx": 0, "fy": 0, "alpha": 0, "k1": 0,
                       "k2": 0, "k3": 0, "p1": 0, "p2": 0,
                       'k4': 0, 'k5': 0, 'k6': 0, 's1': 0, 's2': 0, 's3': 0, 's4': 0,
                       'a1': 0, 'a2': 0, 'a3': 0, "px": 0, "py": 0,
                       "misalignment": [[0, 0, 0], [0, 0, 0]]}

        model = self.Class(**model_param,
                           estimation_parameters=['intrinsic', 'temperature dependence', 'multiple misalignments'])

        update_vec = np.arange(26)

        model.apply_update(update_vec)

        keys = list(model_param.keys())

        keys.remove('misalignment')

        for key in keys:
            self.assertEqual(getattr(model, key), update_vec[model.element_dict[key][0]])

        for ind, vec in enumerate(update_vec[20:].reshape(-1, 3)):
            np.testing.assert_array_almost_equal(at.Rotation(vec).quaternion, at.Rotation(model.misalignment[ind]).quaternion)

    def test_pixels_to_gnomic(self):

        intrins_param = {"fx": 3000, "fy": 4000, "alpha": 0.5,
                         "px": 4005.23, 'py': 2000.33, 'a1': 1e-5, 'a2': 1e-6, 'a3': -1e-7,
                         "field_of_view": 100}

        dist_coefs = [{"k1": 1.5e-1},
                      {"k2": 1.5e-1},
                      {"k3": 1.5e-1},
                      {"p1": 1.5e-6},
                      {"p2": 1.5e-6},
                      {"k4": 1.5e-1},
                      {"k5": 1.5e-1},
                      {"k6": 1.5e-1},
                      {"s1": 1.5e-1},
                      {"s2": 1.5e-1},
                      {"s3": 1.5e-1},
                      {"s4": 1.5e-1},
                      {"k1": -1.5e-1, "k2": -1.5e-1, "k3": -1.5e-1, "p1": -1.5e-6, "p2": -1.5e-6,
                       "k4": -1.5e-1, "k5": -1.5e-1, "k6": -1.5e-1,
                       "s1": -1.5e-1, "s2": -1.5e-1, "s3": -1.5e-1, "s4": -1.5e-1}]

        pinhole = [[0, 0], [0.1, 0], [-0.1, 0], [0.15, 0], [-0.15, 0], [[0.15], [0]], [[0.15, -0.1], [0, 0]],
                   [0, 0.1], [0, -0.1], [0, 0.15], [0, -0.15], [[0], [0.15]], [[0, 0], [0.15, -0.1]], [0.1, 0.1]]

        temperatures = [0, 1, -1, 10.5, -10.5]

        for temp in temperatures:
            for dist in dist_coefs:

                model = self.Class(**dist, **intrins_param)

                for fp_pinhole in pinhole:
                    with self.subTest(**dist, temp=temp, fp_pinhole=fp_pinhole):
                        fp_dist = model.apply_distortion(np.array(fp_pinhole))

                        fp_dist *= model.get_temperature_scale(temp)

                        pix_dist = ((model.intrinsic_matrix[:, :2] @ fp_dist).T + model.intrinsic_matrix[:, 2]).T

                        fp_undist = model.pixels_to_gnomic(pix_dist, temperature=temp)

                        np.testing.assert_allclose(fp_undist, fp_pinhole, atol=1e-13)

    def test_undistort_pixels(self):

        intrins_param = {"fx": 3000, "fy": 4000, "alpha": 0.5,
                         "px": 4005.23, 'py': 2000.33, 'a1': 1e-5, 'a2': 1e-6, 'a3': -1e-7, "field_of_view": 100}

        dist_coefs = [{"k1": 1.5e-1},
                      {"k2": 1.5e-1},
                      {"k3": 1.5e-1},
                      {"p1": 1.5e-6},
                      {"p2": 1.5e-6},
                      {"k4": 1.5e-1},
                      {"k5": 1.5e-1},
                      {"k6": 1.5e-1},
                      {"s1": 1.5e-1},
                      {"s2": 1.5e-1},
                      {"s3": 1.5e-1},
                      {"s4": 1.5e-1},
                      {"k1": -1.5e-1, "k2": -1.5e-1, "k3": -1.5e-1, "p1": -1.5e-6, "p2": -1.5e-6,
                       "k4": -1.5e-1, "k5": -1.5e-1, "k6": -1.5e-1,
                       "s1": -1.5e-1, "s2": -1.5e-1, "s3": -1.5e-1, "s4": -1.5e-1}]
        pinhole = [[0, 0], [0.1, 0], [-0.1, 0], [0.15, 0], [-0.15, 0], [[0.15], [0]], [[0.15, -0.1], [0, 0]],
                   [0, 0.1], [0, -0.1], [0, 0.15], [0, -0.15], [[0], [0.15]], [[0, 0], [0.15, -0.1]], [0.1, 0.1]]

        temperatures = [0, 1, -1, 10.5, -10.5]

        for temp in temperatures:
            for dist in dist_coefs:

                model = self.Class(**dist, **intrins_param)

                with self.subTest(**dist, temp=temp):

                    for fp_pinhole in pinhole:
                        fp_pinhole = np.array(fp_pinhole).astype(np.float64)
                        fp_dist = model.apply_distortion(fp_pinhole)

                        fp_dist *= model.get_temperature_scale(temp)

                        pix_dist = ((model.intrinsic_matrix[:, :2] @ fp_dist).T + model.intrinsic_matrix[:, 2]).T

                        pix_undist = model.undistort_pixels(pix_dist, temperature=temp)

                        fp_pinhole *= model.get_temperature_scale(temp)

                        pix_pinhole = ((model.intrinsic_matrix[:, :2] @ fp_pinhole).T + model.intrinsic_matrix[:, 2]).T

                        np.testing.assert_allclose(pix_undist, pix_pinhole, atol=1e-13)

    def test_pixels_to_unit(self):
        intrins_param = {"fx": 3000, "fy": 4000, "alpha": 0.5,
                         "px": 4005.23, 'py': 2000.33, 'a1': 1e-6, 'a2': -2e-7, 'a3': 4.5e-8,
                         "field_of_view": 100}

        dist_coefs = [{"k1": 1.5e-1},
                      {"k2": 1.5e-1},
                      {"k3": 1.5e-1},
                      {"p1": 1.5e-6},
                      {"p2": 1.5e-6},
                      {"k4": 1.5e-1},
                      {"k5": 1.5e-1},
                      {"k6": 1.5e-1},
                      {"s1": 1.5e-1},
                      {"s2": 1.5e-1},
                      {"s3": 1.5e-1},
                      {"s4": 1.5e-1},
                      {"k1": -1.5e-1, "k2": -1.5e-1, "k3": -1.5e-1, "p1": -1.5e-6, "p2": -1.5e-6,
                       "k4": -1.5e-1, "k5": -1.5e-1, "k6": -1.5e-1,
                       "s1": -1.5e-1, "s2": -1.5e-1, "s3": -1.5e-1, "s4": -1.5e-1},
                      {"misalignment": np.array([1e-11, 2e-12, -1e-10])},
                      {"misalignment": np.array([[1e-11, 2e-12, -1e-10], [-1e-13, 1e-11, 2e-12]]),
                       "estimation_parameters": "multiple misalignments"}]

        camera_vecs = [[0, 0, 1], [0.1, 0, 1], [-0.1, 0, 1], [0, 0.1, 1], [0, -0.1, 1], [0.1, 0.1, 1],
                       [-0.1, -0.1, 1], [[0.1, -0.1], [-0.1, 0.1], [1, 1]]]

        temperatures = [0, 1, -1, 10.5, -10.5]

        for temp in temperatures:
            for dist in dist_coefs:

                model = self.Class(**dist, **intrins_param)

                with self.subTest(**dist, temp=temp):
                    for vec in camera_vecs:
                        pixel_loc = model.project_onto_image(vec, image=-1, temperature=temp)

                        unit_vec = model.pixels_to_unit(pixel_loc, image=-1, temperature=temp)

                        unit_true = np.array(vec).astype(np.float64)

                        unit_true /= np.linalg.norm(unit_true, axis=0, keepdims=True)

                        np.testing.assert_allclose(unit_vec, unit_true, atol=1e-13)

    def test_overwrite(self):

        model1 = self.Class(field_of_view=10, intrinsic_matrix=np.array([[1, 2, 3], [0, 5, 6]]),
                            distortion_coefficients=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
                            misalignment=[[1, 2, 3], [4, 5, 6]], use_a_priori=False,
                            estimation_parameters=['multiple misalignments'])

        model2 = self.Class(field_of_view=20, intrinsic_matrix=np.array([[11, 12, 13], [0, 15, 16]]),
                            distortion_coefficients=np.array([11, 12, 13, 14, 15, 16, 16, 18, 19, 20, 21, 22]),
                            misalignment=[[11, 12, 13], [14, 15, 16]], use_a_priori=True,
                            estimation_parameters=['single misalignment'])

        modeltest = model1.copy()

        modeltest.overwrite(model2)

        self.assertEqual(model2.field_of_view, modeltest.field_of_view)
        self.assertEqual(model2.use_a_priori, modeltest.use_a_priori)
        self.assertEqual(model2.estimate_multiple_misalignments, modeltest.estimate_multiple_misalignments)
        np.testing.assert_array_equal(model2.intrinsic_matrix, modeltest.intrinsic_matrix)
        np.testing.assert_array_equal(model2.distortion_coefficients, modeltest.distortion_coefficients)
        np.testing.assert_array_equal(model2.misalignment, modeltest.misalignment)
        np.testing.assert_array_equal(model2.estimation_parameters, modeltest.estimation_parameters)

        self.assertEqual(modeltest, model2)

        modeltest = model2.copy()

        modeltest.overwrite(model1)

        self.assertEqual(model1.field_of_view, modeltest.field_of_view)
        self.assertEqual(model1.use_a_priori, modeltest.use_a_priori)
        self.assertEqual(model1.estimate_multiple_misalignments, modeltest.estimate_multiple_misalignments)
        np.testing.assert_array_equal(model1.intrinsic_matrix, modeltest.intrinsic_matrix)
        np.testing.assert_array_equal(model1.distortion_coefficients, modeltest.distortion_coefficients)
        np.testing.assert_array_equal(model1.misalignment, modeltest.misalignment)
        np.testing.assert_array_equal(model1.estimation_parameters, modeltest.estimation_parameters)

        self.assertEqual(modeltest, model1)

    def test_distort_pixels(self):

        model = self.Class(kx=1000, ky=-950.5, px=4500, py=139.32, a1=1e-3, a2=1e-4, a3=1e-5,
                           kxy=0.5, radial2n=1e-5, radial4n=1e-5, radial6n=1e-7,
                           tiptilt_x=1e-6, tiptilt_y=2e-12, k4=0.5e-6, k5=-0.2e-6, k6=0.01e-6,
                           s1=-0.5e-6, s2=0.5e-6, s3=-0.7e-6, s4=0.6e-6)

        pixels = [[0, 1], [1, 0], [-1, 0], [0, -1], [9000., 200.2],
                  [[4500, 100, 10.98], [0, 139.23, 200.3]]]

        temperatures = [0, 1, -1, 10.5, -10.5]

        for pix in pixels:

            for temp in temperatures:
                with self.subTest(pix=pix, temp=temp):
                    undist_pix = model.undistort_pixels(pix, temperature=temp)
                    dist_pix = model.distort_pixels(undist_pix, temperature=temp)

                    np.testing.assert_allclose(dist_pix, pix, atol=1e-10)

    def test_to_from_elem(self):

        element = etree.Element(self.Class.__name__)

        model = self.Class(field_of_view=5, use_a_priori=True,
                           misalignment=[1, 2, 3], kx=2, ky=200, px=50, py=300, kxy=12123,
                           a1=37, a2=1, a3=-1230, k1=5, k2=10, k3=20, p1=-10, p2=35,
                           estimation_parameters=['px', 'multiple misalignments'], n_rows=20, n_cols=30)

        model_copy = model.copy()

        with self.subTest(misalignment=True):
            element = model.to_elem(element, misalignment=True)

            self.assertEqual(model, model_copy)

            model_new = self.Class.from_elem(element)

            self.assertEqual(model, model_new)

        with self.subTest(misalignment=False):
            element = model.to_elem(element, misalignment=False)

            self.assertEqual(model, model_copy)

            model_new = self.Class.from_elem(element)

            model.estimation_parameters[-1] = 'single misalignment'

            model.estimate_multiple_misalignments = False

            model.misalignment = np.zeros(3)

            self.assertEqual(model, model_new)

    def test_distortion_map(self):

        model = self.Class(kx=100, ky=-985.234, px=1000, py=1095, kxy=10,
                           k1=1e-6, k2=1e-12, k3=-4e-10, p1=6e-7, p2=-1e-5,
                           k4=0.1, k5=0.01, k6=0.001, s1=0.1, s2=-0.1, s3=0.5, s4=-0.6,
                           a1=1e-6, a2=-1e-7, a3=4e-12, field_of_view=100)

        rows, cols, dist = model.distortion_map((2000, 250), step=10)

        rl, cl = np.arange(0, 2000, 10), np.arange(0, 250, 10)

        rs, cs = np.meshgrid(rl, cl, indexing='ij')

        np.testing.assert_array_equal(rows, rs)
        np.testing.assert_array_equal(cols, cs)

        distl = model.distort_pixels(np.vstack([cs.ravel(), rs.ravel()]).astype(np.float64))

        np.testing.assert_array_equal(distl - np.vstack([cs.ravel(), rs.ravel()]), dist)


class TestSaveLoad(TestCase):
    def test_save_load(self):
        import os

        file = 'temp'

        models = [PinholeModel(focal_length=10, field_of_view=20, use_a_priori=True,
                               misalignment=[[1, 2, 3], [4, 5, 6]], estimation_parameters=['kx', 'a1'],
                               kx=5, ky=4, px=100, py=-1000., n_rows=50, n_cols=60, a1=10, a2=20, a3=30),
                  OwenModel(focal_length=20, field_of_view=30, use_a_priori=True,
                            misalignment=[[6, 2, 8], [5, 5, 3]],
                            estimation_parameters=['e1', 'a2', 'multiple misalignments'],
                            kx=5.5, ky=4, px=100.5, py=-1000., n_rows=50, n_cols=60, a1=10, a2=20, a3=30,
                            kyx=30, kxy=40, e1=10, e2=30, e3=30, e4=50, e5=20, e6=-200),
                  BrownModel(field_of_view=30, use_a_priori=True,
                             misalignment=[[6, 2, 8], [5, 5, 3]],
                             estimation_parameters=['k2', 'a1', 'multiple misalignments'],
                             kx=5.5, ky=4, px=100.5, py=-1000., n_rows=50, n_cols=60, a1=10, a2=20, a3=30,
                             kxy=40, k1=10, k2=30, k3=30, p1=50, p2=20),
                  OpenCVModel(field_of_view=30, use_a_priori=True,
                              misalignment=[np.array([1, -1, 2.]) for _ in range(100)],
                              estimation_parameters=['a1', 'a2', 'multiple misalignments'],
                              kx=5.5, ky=4, px=100.5, py=-1000., n_rows=50, n_cols=60, a1=10, a2=20, a3=30,
                              kxy=40, k1=10, k2=30, k3=30, p1=50, p2=20, k4=100, k5=-2342.2, k6=23.3,
                              s1=30, s2=-234., s3=235, s4=678)]

        names = ['a', 'b', 'c', 'd']
        groups = [None, 'taco', 'tuesday', 'forever']

        with TemporaryDirectory() as tmp:
            file = Path(tmp) / "save_test.xml"

            for group, name, model in zip(groups, names, models):

                save(file, name, model, group=group, misalignment=True)

            for group, name, orig in zip(groups, names, models):

                with self.subTest(misalignment=True, group=group, name=name):

                    model = load(file, name, group=group)

                    self.assertEqual(model, orig)

            for name, orig in zip(names, models):

                with self.subTest(misalignment=True, group=None, name=name):

                    model = load(file, name, group=None)

                    self.assertEqual(model, orig)

        # os.remove(file)
        with TemporaryDirectory() as tmp:
            file = Path(tmp) / "save_test.xml"

            for group, name, model in zip(groups, names, models):

                save(file, name, model, group=group, misalignment=False)

            for group, name, orig in zip(groups, names, models):

                with self.subTest(misalignment=False, group=group, name=name):

                    model = load(file, name, group=group)

                    self.assertNotEqual(model, orig)
                    norig = orig.copy()

                    norig.misalignment = np.zeros(3)
                    if 'multiple misalignments' in norig.estimation_parameters:
                        norig.estimation_parameters.remove('multiple misalignments')
                        norig.estimation_parameters.append('single misalignment')
                        norig.estimate_multiple_misalignments = False

                    self.assertEqual(norig, model)

            for name, orig in zip(names, models):

                with self.subTest(misalignment=False, group=None, name=name):

                    model = load(file, name, group=None)

                    self.assertNotEqual(model, orig)
                    norig = orig.copy()

                    norig.misalignment = np.zeros(3)
                    if 'multiple misalignments' in norig.estimation_parameters:
                        norig.estimation_parameters.remove('multiple misalignments')
                        norig.estimation_parameters.append('single misalignment')
                        norig.estimate_multiple_misalignments = False

                    self.assertEqual(norig, model)

        # os.remove(file)
