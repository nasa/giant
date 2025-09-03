from copy import deepcopy
from unittest import TestCase

import numpy as np

from scipy.stats import chi2

from giant.point_spread_functions import *


rng = np.random.default_rng(193339)


class TestGaussian(TestCase):

    def setUp(self) -> None:
        self.Class = Gaussian
        self.update_length = 5

        cx, cy = 20.5, -2.1
        self.params = {'sigma_x': 3, 'sigma_y': 1, 'amplitude': 500,
                       'centroid_x': cx, 'centroid_y': cy}

        half_size = 1
        self.x, self.y = np.meshgrid(np.arange(np.round(cx)-half_size, np.round(cx)+half_size+1),
                                     np.arange(np.round(cy)-half_size, np.round(cy)+half_size+1))

        dx, dy = self.x - cx, self.y-cy
        self.expected = self.params['amplitude']*np.exp(-(dx**2/(2*self.params['sigma_x']**2) +
                                                          dy**2/(2*self.params['sigma_y']**2)))

        self.noise = np.abs(np.median(self.expected)/100)*rng.standard_normal(self.expected.shape)

        self.perturbed = self.expected + self.noise
        test = self.perturbed < 0
        self.perturbed[test] *= -1
        self.noise[test] *= -1

        self.state_order = ['centroid_x', 'centroid_y', 'sigma_x', 'sigma_y', 'amplitude']

        self.pert = 1e-6
        # set the sigma expected to be the one that should capture 99.9% of cases.
        self.sigma_expected = np.sqrt(chi2.ppf(0.999, len(self.state_order)))

    def numeric_jacobian(self, psf, x: np.ndarray, y: np.ndarray):
        jac = np.zeros((x.size, self.update_length), dtype=np.float64)

        for u in range(self.update_length):
            pert_psf = deepcopy(psf)

            pert_vec = np.zeros(self.update_length)
            pert_vec[u] += self.pert

            pert_psf.update_state(pert_vec)

            positive = pert_psf.evaluate(x.ravel(), y.ravel())

            pert_psf = deepcopy(psf)

            pert_vec = np.zeros(self.update_length)
            pert_vec[u] -= self.pert

            pert_psf.update_state(pert_vec)
            negative = pert_psf.evaluate(x.ravel(), y.ravel())

            jac[:, u] = (positive - negative) / (2 * self.pert)

        return jac

    def test_evaluate(self) -> None:

        psf = self.Class(**self.params)

        attained = psf.evaluate(self.x, self.y)

        np.testing.assert_allclose(attained, self.expected)

    def test_update_state(self) -> None:
        update_vec = np.arange(1, 1+self.update_length)

        psf = self.Class(**self.params)

        psf.update_state(update_vec)

        for i, s in zip(range(self.update_length), self.state_order):
            with self.subTest(param=s):
                self.assertEqual(self.params[s]+update_vec[i], getattr(psf, s, None))

    def test_compute_jacobian(self) -> None:

        psf = self.Class(**self.params)

        jacobian = psf.compute_jacobian(self.x.ravel(), self.y.ravel(), psf.evaluate(self.x.ravel(), self.y.ravel()))

        num_jac = self.numeric_jacobian(psf, self.x.ravel(), self.y.ravel())

        np.testing.assert_allclose(jacobian, num_jac, atol=self.pert)

    def test_covariance(self) -> None:

        self.Class.save_residuals = True
        fit = self.Class.fit(self.x, self.y, self.perturbed)
        cov = fit.covariance
        jac = fit.compute_jacobian(self.x.ravel(), self.y.ravel(), fit.evaluate(self.x, self.y).ravel())
        expected_cov = fit.residual_std**2*np.linalg.inv(jac.T@jac)
        np.testing.assert_allclose(cov, expected_cov, atol=1e-6, rtol=.05)
        self.Class.save_residuals = False

    def test_fit(self) -> None:

        self.Class.save_residuals = True
        fit = self.Class.fit(self.x, self.y, self.perturbed)
        cov = fit.covariance
        self.Class.save_residuals = False

        for i, s in enumerate(self.state_order):
            expected = self.params[s]
            actual = getattr(fit, s, None)
            with self.subTest(param=s):
                self.assertAlmostEqual(expected, actual,
                                       delta=self.sigma_expected*np.sqrt(cov[i, i]))

    def test_generate_kernel(self) -> None:

        size = 3
        half_size = 1

        obj = self.Class(**self.params, size=size)

        expected = obj.evaluate(*np.meshgrid(np.arange(obj.centroid[0]-half_size, obj.centroid[0]+half_size+1),
                                             np.arange(obj.centroid[1]-half_size, obj.centroid[1]+half_size+1)))

        expected /= expected.sum()

        np.testing.assert_allclose(obj.generate_kernel(), expected)

    # TODO: figure out how to test __call__ and apply_1d


class TestIterativeGaussian(TestGaussian):

    def setUp(self) -> None:
        self.Class = IterativeGaussian
        self.update_length = 5

        cx, cy = 20.5, -2.1
        self.params = {'sigma_x': 3, 'sigma_y': 1, 'amplitude': 1000,
                       'centroid_x': cx, 'centroid_y': cy}

        half_size = 7
        self.x, self.y = np.meshgrid(np.arange(np.round(cx)-half_size, np.round(cx)+half_size+1),
                                     np.arange(np.round(cy)-half_size, np.round(cy)+half_size+1))
        dx, dy = self.x - cx, self.y-cy
        self.expected = self.params['amplitude']*np.exp(-(dx**2/(2*self.params['sigma_x']**2) +
                                                          dy**2/(2*self.params['sigma_y']**2)))

        self.noise = np.abs(np.median(self.expected)/20)*rng.standard_normal(self.expected.shape)

        self.perturbed = self.expected + self.noise

        self.state_order = ['centroid_x', 'centroid_y', 'sigma_x', 'sigma_y', 'amplitude']

        self.pert = 1e-6

        # set the sigma expected to be the one that should capture 99.9% of cases.
        self.sigma_expected = np.sqrt(chi2.ppf(0.999, len(self.state_order)))


class TestIterativeGaussianWBackground(TestGaussian):

    def setUp(self) -> None:
        self.Class = IterativeGaussianWBackground
        self.update_length = 8

        cx, cy = 20.5, -2.1
        self.params = {'sigma_x': 2, 'sigma_y': 1, 'amplitude': 1000,
                       'centroid_x': cx, 'centroid_y': cy, 'bg_b_coef': 10, 'bg_c_coef': -20.6, 'bg_d_coef': 10}

        half_size = 8
        self.x, self.y = np.meshgrid(np.arange(np.round(cx)-half_size, np.round(cx)+half_size+1),
                                     np.arange(np.round(cy)-half_size, np.round(cy)+half_size+1))
        dx, dy = self.x - cx, self.y-cy
        self.expected = (self.params['amplitude']*np.exp(-(dx**2/(2*self.params['sigma_x']**2) +
                                                           dy**2/(2*self.params['sigma_y']**2))) +
                         self.params['bg_b_coef']*self.x + self.params['bg_c_coef']*self.y+self.params['bg_d_coef'])

        self.noise = np.abs(np.median(self.expected)/50)*rng.standard_normal(self.expected.shape)

        self.perturbed = self.expected + self.noise

        self.state_order = ['centroid_x', 'centroid_y', 'sigma_x', 'sigma_y', 'amplitude',
                            'bg_b_coef', 'bg_c_coef', 'bg_d_coef']

        self.pert = 1e-6

        # set the sigma expected to be the one that should capture 99.9% of cases.
        self.sigma_expected = np.sqrt(chi2.ppf(0.999, len(self.state_order)))


class TestGeneralizedGaussian(TestGaussian):

    def setUp(self) -> None:
        self.Class = GeneralizedGaussian
        self.update_length = 6

        cx, cy = 20.5, -2.1
        self.params = {'a_coef': 1/2*2.5**2, 'b_coef': 0.05, 'c_coef': 1/2*1.5**2, 'amplitude': 1000,
                       'centroid_x': cx, 'centroid_y': cy}

        half_size = 1
        self.x, self.y = np.meshgrid(np.arange(np.round(cx)-half_size, np.round(cx)+half_size+1),
                                     np.arange(np.round(cy)-half_size, np.round(cy)+half_size+1))
        dx, dy = self.x - cx, self.y-cy
        self.expected = self.params['amplitude']*np.exp(-(self.params['a_coef']*dx**2 +
                                                          2*self.params['b_coef']*dx*dy +
                                                          self.params['c_coef']*dy**2))

        self.noise = np.abs(np.median(self.expected)/100)*rng.standard_normal(self.expected.shape)

        self.perturbed = self.expected + self.noise
        test = self.perturbed < 0
        self.perturbed[test] *= -1
        self.noise[test] *= -1

        self.state_order = ['centroid_x', 'centroid_y', 'a_coef', 'b_coef', 'c_coef', 'amplitude']

        self.pert = 1e-6

        # set the sigma expected to be the one that should capture 99.9% of cases.
        self.sigma_expected = np.sqrt(chi2.ppf(0.999, len(self.state_order)))


class TestIterativeGeneralizedGaussian(TestGaussian):

    def setUp(self) -> None:
        self.Class = IterativeGeneralizedGaussian
        self.update_length = 6

        cx, cy = 20.5, -2.1
        self.params = {'a_coef': 1/(2*2.5**2), 'b_coef': 0.05, 'c_coef': 1/(2*1.5**2), 'amplitude': 1000,
                       'centroid_x': cx, 'centroid_y': cy}

        half_size = 3
        self.x, self.y = np.meshgrid(np.arange(np.round(cx)-half_size, np.round(cx)+half_size+1),
                                     np.arange(np.round(cy)-half_size, np.round(cy)+half_size+1))
        dx, dy = self.x - cx, self.y-cy
        self.expected = self.params['amplitude']*np.exp(-(self.params['a_coef']*dx**2 +
                                                          2*self.params['b_coef']*dx*dy +
                                                          self.params['c_coef']*dy**2))

        self.noise = np.abs(np.median(self.expected)/100)*rng.standard_normal(self.expected.shape)

        self.perturbed = self.expected + self.noise
        test = self.perturbed < 0
        self.perturbed[test] *= -1
        self.noise[test] *= -1

        self.state_order = ['centroid_x', 'centroid_y', 'a_coef', 'b_coef', 'c_coef', 'amplitude']

        self.pert = 1e-6

        # set the sigma expected to be the one that should capture 99.9% of cases.
        self.sigma_expected = np.sqrt(chi2.ppf(0.999, len(self.state_order)))


class TestIterativeGeneralizedGaussianWBackground(TestGaussian):

    def setUp(self) -> None:
        self.Class = IterativeGeneralizedGaussianWBackground
        self.update_length = 9

        cx, cy = 20.5, -2.1
        self.params = {'a_coef': 1/(2*0.75**2), 'b_coef': 0.05, 'c_coef': 1/(2*0.5**2), 'amplitude': 1000,
                       'centroid_x': cx, 'centroid_y': cy, 'bg_b_coef': 10, 'bg_c_coef': -20.6, 'bg_d_coef': 10}

        half_size = 8
        self.x, self.y = np.meshgrid(np.arange(np.round(cx)-half_size, np.round(cx)+half_size+1),
                                     np.arange(np.round(cy)-half_size, np.round(cy)+half_size+1))

        dx, dy = self.x - cx, self.y-cy
        self.expected = (self.params['amplitude']*np.exp(-(self.params['a_coef']*dx**2 +
                                                           2*self.params['b_coef']*dx*dy +
                                                           self.params['c_coef']*dy**2)) +
                         self.params['bg_b_coef']*self.x + self.params['bg_c_coef']*self.y + self.params['bg_d_coef'])

        self.noise = np.abs(np.median(self.expected)/50)*rng.standard_normal(self.expected.shape)

        self.perturbed = self.expected + self.noise

        self.state_order = ['centroid_x', 'centroid_y', 'a_coef', 'b_coef', 'c_coef', 'amplitude',
                            'bg_b_coef', 'bg_c_coef', 'bg_d_coef']

        self.pert = 1e-6

        # set the sigma expected to be the one that should capture 99.9% of cases.
        self.sigma_expected = np.sqrt(chi2.ppf(0.999, len(self.state_order)))


if __name__ == '__main__':
    import unittest

    unittest.main()