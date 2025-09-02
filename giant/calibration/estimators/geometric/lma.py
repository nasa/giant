from dataclasses import dataclass

import warnings

import numpy as np

from giant.calibration.estimators.geometric.iterative_nonlinear_lstsq import IterativeNonlinearLSTSQ, IterativeNonlinearLstSqOptions
from giant.camera_models import CameraModel


@dataclass
class LMAEstimatorOptions(IterativeNonlinearLstSqOptions):
    
    max_divergence_steps: int = 5
    """
    The maximum number of steps in a row that can diverge before breaking iteration    
    """


class LMAEstimator(IterativeNonlinearLSTSQ, LMAEstimatorOptions):
    """
    This implements a Levenberg-Marquardt Algorithm estimator, which is analogous to a damped iterative non-linear
    least squares.

    This class is nearly exactly the same as the :class:`IterativeNonlinearLSTSQ` except that it adds damping to the
    update step of the iterative non-linear least squares algorithm and allows a few diverging steps in a row where the
    damping parameter is updated before failing.  The number of diverging steps that are allowed is controlled by the
    :attr:`max_divergence_steps` setting.  This represents only difference from the :class:`IterativeNonlinearLSTSQ`
    interface from the user's perspective.

    In general, this algorithm will result in the same answer as the :class:`IterativeNonlinearLSTSQ` algorithm but at a
    slower convergence rate.  In a few cases however, this estimator can be more robust to initial guess errors,
    achieving convergence when the standard iterative nonlinear least squares diverges.  Therefore, it is likely best to
    start with the :class:`IterativeNonlinearLSTSQ` class an only switch to this if you experience convergence issues.

    The implementation of the LMA in this class is inspired by
    https://link.springer.com/article/10.1007/s40295-016-0091-3
    """

    def __init__(self, model: CameraModel, options: LMAEstimatorOptions | None = None):
        r"""
        :param model: The camera model instance to be estimated set with an initial guess of the state.
        :param options: The options dataclass to configure the class with
        """
        if options is None:
            # need to manually do this due to the way the inheritance is structured
            options = LMAEstimatorOptions()
        super().__init__(model=model, options=options)

    def estimate(self) -> None:
        """
        This method estimates the postfit residuals based on the model, weight matrix, lma coefficient, etc.
        Convergence is achieved once the standard deviation of the computed residuals is less than the absolute
        tolerance or the difference between the prefit and postfit residuals is less than the relative tolerance.

        """
        if self.measurements is None:
            raise ValueError("measurements must not be None before a call to estimate")
        if self.camera_frame_directions is None:
            raise ValueError("camera_frame_directions must not be None before a call to estimate")
        if self.weighted_estimation and (self.measurement_covariance is None):
            raise ValueError("measurement_covariance must not be None before a call to estimate "
                             "if weighed_estimation is True")
        if self.model.use_a_priori and (self.a_priori_model_covariance is None):
            raise ValueError("a_priori_model_covariance must not be None before a call to estimate "
                             "if model.use_a_priori is True")

        # get the size of the state vector
        a_priori_state = np.array(self.model.state_vector)
        state_size = len(a_priori_state)

        # get the number of measurements
        num_meas = self.measurements.size

        # get the weight matrix
        weight_matrix = self._compute_weight_matrix(state_size, num_meas)

        # calculate the prefit residuals
        prefit_residuals = self.compute_residuals()
        pre_ss = prefit_residuals.ravel() @ prefit_residuals.ravel()

        # a flag specifying this is the first time through so we need to initialize the lma_coefficient
        first = True
        lma_coefficient = 0
        n_diverge = 0

        # iterate to convergence
        for _ in range(self.max_iter):

            # get the jacobian matrix
            jacobian = self.model.compute_jacobian(self.camera_frame_directions, temperature=self.temperatures)

            if first:
                # initialize the lma_coefficient
                lma_coefficient = 0.001 * np.trace(jacobian.T @ jacobian) / jacobian.shape[1]

            if self.model.use_a_priori:
                residuals_vec = np.concatenate([prefit_residuals.reshape((-1, 1), order='F'),
                                                np.zeros((state_size, 1))], axis=0)

            else:
                residuals_vec = prefit_residuals.reshape((-1, 1), order='F')

            if np.isscalar(weight_matrix):
                lhs: np.ndarray = np.sqrt(weight_matrix) * jacobian.T @ jacobian
                rhs: np.ndarray = np.sqrt(weight_matrix) * jacobian.T @ residuals_vec
            else:
                lhs: np.ndarray = jacobian.T @ weight_matrix @ jacobian
                rhs: np.ndarray = jacobian.T @ weight_matrix @ residuals_vec

            # get the update vector using LMA
            update_vec = np.linalg.solve(lhs + lma_coefficient*np.diag(np.diag(lhs)), rhs).astype(np.float64)

            model_copy = self.model.copy()

            model_copy.apply_update(update_vec)

            postfit_residuals = self.compute_residuals(model=model_copy)
            post_ss = postfit_residuals.ravel() @ postfit_residuals.ravel()
            resid_change = abs(pre_ss - post_ss)

            # check for convergence
            if resid_change <= (self.residual_atol + self.residual_rtol * pre_ss):
                self._successful = True
                self._postfit_residuals = postfit_residuals
                self.model = model_copy
                self._jacobian = self.model.compute_jacobian(self.camera_frame_directions, temperature=self.temperatures)
                return

            elif (np.abs(update_vec) <= (self.state_atol + self.state_rtol * a_priori_state)).all():
                self._successful = True
                self._postfit_residuals = postfit_residuals
                self.model = model_copy
                self._jacobian = self.model.compute_jacobian(self.camera_frame_directions, temperature=self.temperatures)
                return

            elif pre_ss < post_ss:  # check for divergence

                n_diverge += 1

                if n_diverge > self.max_divergence_steps:
                    warnings.warn('Solution is diverging.  Stopping iteration.'
                                  '\n\tpre-update residuals {}'
                                  '\n\tpost-update residuals {}'
                                  '\n\tdiverged for {} iterations'.format(pre_ss, post_ss, n_diverge))
                    self._successful = False
                    self._postfit_residuals = None
                    self._jacobian = None
                    return

                # update the lma coefficient
                lma_coefficient *= 10

            else:  # converging
                # reset the divergence counter
                n_diverge = 0
                # update the lma coefficient
                lma_coefficient /= 10
                # prepare for the next iteration
                self.model = model_copy
                prefit_residuals = postfit_residuals
                pre_ss = post_ss
                a_priori_state = np.array(self.model.state_vector)

        warnings.warn("Solution didn't converge in the requested number of iterations")
        self._successful = False
        self._postfit_residuals = prefit_residuals
        self.model
        self._jacobian = self.model.compute_jacobian(self.camera_frame_directions, temperature=self.temperatures)



