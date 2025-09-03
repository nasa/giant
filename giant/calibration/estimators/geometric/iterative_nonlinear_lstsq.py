from dataclasses import dataclass

from typing import Optional, cast, Sequence

import warnings

import numpy as np
from numpy.typing import NDArray

from giant.calibration.estimators.geometric.geometric_estimator import GeometricEstimatorBC, GeometricEstimatorOptions, ModelT
from giant.camera_models import CameraModel
from giant._typing import DOUBLE_ARRAY, ARRAY_LIKE


@dataclass
class IterativeNonlinearLstSqOptions(GeometricEstimatorOptions):
    max_iter: int = 20
    """
    The maximum number of iteration steps to attempt to reach convergence.  
    
    If convergence has not been reached after attempting ``max_iter`` steps, a warning will be raised that the model has
    not converged and :attr:`successful` will be set to ``False``.
    """
    
    residual_atol: float = 1e-10
    """
    The absolute convergence tolerance criteria for the sum of squares of the residual.
    """
    
    residual_rtol: float = 1e-10
    """
    The relative convergence tolerance criteria for the sum of squares of the residual.
    """
    
    state_atol: float = 1e-10
    """
    The absolute convergence tolerance criteria for the elements of the state vecto.
    """
    
    state_rtol: float = 1e-10
    """
    The relative convergence tolerance criteria for the elements of the state vector.
    """
    

class IterativeNonlinearLSTSQ(GeometricEstimatorBC[ModelT], IterativeNonlinearLstSqOptions):
    r"""
    This concrete estimator implements iterative non-linear least squares for estimating an updated camera model.

    Iterative non-linear least squares estimation is done by estimating updates to the "state" vector (in this case the
    camera model parameters being updated) iteratively.  At each step, the system is linearized about the current
    estimate of the state and the additive update is estimated.  This iteration is repeated until convergence (or
    divergence) based on the pre/post update residuals and the update vector itself.

    The state vector that is being estimated by this class is controlled by the
    :attr:`.CameraModel.estimation_parameters` attribute of the provided camera model.  This class does not actually use
    the :attr:`.CameraModel.estimation_parameters` attribute since it is handled by the
    :meth:`.CameraModel.compute_jacobian` and :meth:`.CameraModel.apply_update` methods of the provided camera model
    internally, but it is mentioned here to show how to control what exactly is being estimated.

    Because this class linearizes about the current estimate of the state, it requires an initial guess for the camera
    model that is "close enough" to the actual model to ensure convergence.  Defining "close enough" in any broad sense
    is impossible, but based on experience, using the manufacturer defined specs for focal length/pixel pitch and
    assuming no distortion is generally "close enough" even for cameras with heavy distortion (star identification may
    require a better initial model than this anyway).

    As this class converges the state estimate, it updates the supplied camera model in place, therefore, if you wish to
    keep a copy of the original camera model, you should manually create a copy of it before calling the
    :meth:`estimate` method on this class.

    In the :meth:`estimate` method, convergence is checked on both the sum of squares of the residuals and the update
    vector for the state.  That is convergence is reached when either of

    .. math::
        :nowrap:

        \begin{gather*}
        \left\|\mathbf{r}_{pre}^T\mathbf{r}_{pre} - \mathbf{r}_{post}^T\mathbf{r}_{post}\right\|
        \le(a_r+r_r\mathbf{r}_{pre}^T\mathbf{r}_{pre}) \\
        \text{all}\left[\left\|\mathbf{u}\right\|\le(a_s+r_s\mathbf{s}_{pre})\right]
        \end{gather*}

    is ``True``. Here :math:`\mathbf{r}_{pre}` is the nx1 vector of residuals before the update is applied,
    :math:`\mathbf{r}_{post}` is the nx1 vector of residuals after the update is applied, :math:`a_r` is the
    :attr:`residual_atol` absolute residual tolerance, :math:`r_r` is the :attr:`residual_rtol` relative residual
    tolerance, :math:`\mathbf{u}` is the update vector, :math:`\text{all}` indicates that the contained expression is
    ``True`` for all elements, :math:`a_s` is the :attr:`state_atol` absolute tolerance for the state vector,
    :math:`r_s` is the :attr:`state_rtol` relative tolerance for the state vector, and :math:`\mathbf{s}_{pre}` is the
    state vector before the update is applied.  Divergence is only checked on the sum of squares of the residuals, that
    is, divergence is occurring when

    .. math::
        \mathbf{r}_{pre}^T\mathbf{r}_{pre} < \mathbf{r}_{post}^T\mathbf{r}_{post}

    where all is as defined as before.  If a case is diverging then a warning will be printed, the iteration will cease,
    and :attr:`successful` will be set to ``False``.

    Typically this class is not used by the user, and instead it is used internally by the :class:`.Calibration` class
    which handles data preparation for you. If you wish to use this externally from the :class:`.Calibration` class you
    must first set

    * :attr:`model`
    * :attr:`measurements`
    * :attr:`camera_frame_directions`
    * :attr:`temperatures`
    * :attr:`weighted_estimation`
    * :attr:`measurement_covariance` *if* :attr:`weighted_estimation` *is* ``True``
    * :attr:`a_priori_state_covariance` *if* :attr:`~.CameraModel.use_a_priori` *is set to* ``True`` for the camera
      model.

    according to their documentation.  Once those have been set, you can perform the estimation using :meth:`estimate`
    which will iterate until convergence (or divergence).  If the fit successfully converges, :attr:`successful` will be
    set to ``True`` and attributes :attr:`postfit_covariance` and :attr:`postfit_residuals` will both return numpy
    arrays instead of ``None``.  If you wish to use the same instance of this class to do another estimation you should
    call :meth:`reset` before setting the new data to ensure that data is not mixed between estimation runs and all
    flags are set correctly.
    """

    def __init__(self, model: ModelT, options: IterativeNonlinearLstSqOptions | None = None):
        r"""
        :param model: The camera model instance to be estimated set with an initial guess of the state.
        :param options: the dataclass containing the options to configure the class with
        """
        
        super().__init__(IterativeNonlinearLstSqOptions, options=options)

        self._model: ModelT = model 
        """
        The instance attribute to store the camera model being estimated.
        """

        self._measurements: Optional[DOUBLE_ARRAY] = None  
        """
        The instance attribute to store the measurement array
        """

        self._base_frame_directions: Optional[list[DOUBLE_ARRAY | list[list]]] = None  
        """
        The instance attribute to store the base frame direction list
        """

        self._temperatures: Optional[list[float]] = None   
        """
        The instance attribute to store the camera temperature values for each image.
        """

        # set the internal success flag
        self._successful: bool = False 
        """
        The instance attribute to store the success flag
        """

        self._measurement_covariance: Optional[float | DOUBLE_ARRAY] = None 
        """
        The instance attribute to store the measurement covariance matrix
        """

        self._jacobian: Optional[DOUBLE_ARRAY] = None 
        """
        A place to store the Jacobian matrix
        """

        self._postfit_covariance: Optional[DOUBLE_ARRAY] = None 
        """
        A place to cache the post-fit covariance matrix
        """
        
        self._postfit_residuals: Optional[DOUBLE_ARRAY] = None 
        """
        A place to cache the post-fit residual vector
        """

    @property
    def model(self) -> ModelT:
        """
        The camera model that is being estimated.

        Typically this should be a subclass of :class:`.CameraModel`.
        """
        return self._model

    @model.setter
    def model(self, val: ModelT):
        if not isinstance(val, CameraModel):
            warnings.warn("You are setting a camera model that is not a subclass of CameraModel.  We'll assume duck "
                          "typing for now, but be sure that you have implemented all required interfaces or you'll end "
                          "up with an error.")

        self._model = val

    @property
    def successful(self) -> bool:
        """
        A boolean flag indicating whether the fit was successful or not.

        If the fit was successful this should return ``True``, and ``False`` if otherwise.  A fit is defined as
        successful if convergence criteria were reached before the maximum number of iterations.  Divergence and
        non-convergence are both considered an unsuccessful fit resulting in this being set to ``False``
        """

        return self._successful

    @property
    def measurement_covariance(self) -> Optional[DOUBLE_ARRAY | float]:
        """
        A square numpy array containing the covariance matrix for the measurements or a scalar containing the variance
        for all of the measurements.

        If :attr:`weighted_estimation` is set to ``True`` then this property will contain the measurement covariance
        matrix as a square, full rank, numpy array or the measurement variance as a scalar float.  If
        :attr:`weighted_estimation` is set to ``False`` then this property may be ``None`` and will be ignored.

        If specified as a scalar, it is treated as the **variance** for each measurement (that is ``cov = v*I(n,n)``
        where ``cov`` is the covariance matrix, ``v`` is the specified scalar variance, and ``I(n,n)`` is a nxn identity
        matrix) in a memory efficient way.

        :raises ValueError: When attempting to set an array that does not have the proper shape for the
                            :attr:`measurements` vector
        """

        return self._measurement_covariance

    @measurement_covariance.setter
    def measurement_covariance(self, val: Optional[float | ARRAY_LIKE]):
        if self._measurements is not None:
            if not np.isscalar(val):
                val = np.asanyarray(val, dtype=np.float64)
                if val.shape[0] != self._measurements.size:
                    raise ValueError('The measurement covariance matrix must be a square matrix of nxn where n is the '
                                    'number of measurements being used in estimation.\n\tmeasurement_covariance shape = {}'
                                    '\n\tnumber of measurements = {}'.format(val.shape, self._measurements.size))
        self._postfit_covariance = None  # drop the cache
        self._measurement_covariance = val  # type: ignore

    @property
    def measurements(self) -> np.ndarray:
        """
        A nx2 numpy array of the observed pixel locations for stars across all images

        Each column of this array corresponds to the same column of the :attr:`camera_frame_directions` concatenated
        down the last axis. (That is ``measurements[:, i] <-> np.concatenate(camera_frame_directions, axis=-1)[:, i]``)

        This must always be set before a call to :meth:`estimate`.
        """
        assert self._measurements is not None, "measurements shouldn't be none at this point"
        return self._measurements

    @measurements.setter
    def measurements(self, val: Optional[ARRAY_LIKE]):
        if val is not None:
            self._measurements = np.asanyarray(val, dtype=np.float64)
        else:
            self._measurements = None
        

    @property
    def camera_frame_directions(self) -> list[DOUBLE_ARRAY | list[list]]:
        """
        A length m list of unit vectors in the camera frame as numpy arrays for m images corresponding to the
        :attr:`measurements` attribute.

        Each element of this list corresponds to a unique image that is being considered for estimation and the
        subsequent element in the :attr:`temperatures` list. Each column of this concatenated array will correspond to
        the same column of the :attr:`measurements` array. (That is
        ``np.concatenate(camera_frame_directions, axis=-1)[:, i] <-> measurements[:, i]``).

        Any images for which no stars were identified (due to any number of reasons) will have a list of empty arrays in
        the corresponding element of this list (that is ``camera_frame_directions[i] == [[], [], []]`` where ``i`` is an
        image with no measurements identified).  These will be automatically dropped by numpy's concatenate, but are
        included to notify the which temperatures/misalignments to use.

        This must always be set before a call to :meth:`estimate`.
        """

        assert self._base_frame_directions is not None, 'base frame directions should not be None at this point'
        return self._base_frame_directions

    @camera_frame_directions.setter
    def camera_frame_directions(self, val: Optional[Sequence[DOUBLE_ARRAY | list[list]]]):

        self._base_frame_directions = list(val) if val is not None else None

    @property
    def temperatures(self) -> list[float]:
        """
        A length m list of temperatures of the camera for each image being considered in estimation.

        Each element of this list corresponds to a unique image that is being considered for estimation and the
        subsequent element in the :attr:`camera_frame_directions` list.

        This must always be set before a call to :meth:`estimate` (although sometimes it may be a list of all zeros if
        temperature data is not available for the camera).
        """

        assert self._temperatures is not None, "Temperatures should not be None at this point"
        return self._temperatures

    @temperatures.setter
    def temperatures(self, val: Optional[list[float]]):
        self._temperatures = val

    @property
    def postfit_covariance(self) -> Optional[DOUBLE_ARRAY]:
        """
        The post-fit state covariance matrix, taking into account the measurement covariance matrix (if applicable).

        This returns the post-fit state covariance matrix after a call to :meth:`estimate`.  The covariance matrix will
        be in the order according to :attr:`~.CameraModel.estimation_parameters` and if :attr:`weighted_estimation` is
        ``True`` will return the state covariance matrix taking into account the measurement covariance matrix.  If
        :attr:`weighted_estimation` is ``False``, then this will return the post-fit state covariance matrix assuming no
        measurement weighting (that is a measurement covariance matrix of the identity matrix).  If :meth:`estimate`
        has not been called yet or the fit was unsuccessful then this will return ``None``
        """
        return self._calc_covariance()

    @property
    def postfit_residuals(self) -> Optional[DOUBLE_ARRAY]:
        """
        The post-fit observed-computed measurement residuals as a 2xn numpy array.

        This returns the post-fit observed minus computed measurement residuals after a call to :meth:`estimate`.  If
        :meth:`estimate` has not been called yet or the fit was unsuccessful then this will return ``None``.
        """
        if self._successful:
            return self._postfit_residuals
        else:
            return None

    def reset(self) -> None:
        """
        This method resets all of the data attributes to their default values to prepare for another estimation.

        Specifically

        * :attr:`successful`
        * :attr:`measurement_covariance`
        * :attr:`a_priori_state_covariance`
        * :attr:`measurements`
        * :attr:`camera_frame_directions`
        * :attr:`temperatures`
        * :attr:`postfit_covariance`
        * :attr:`postfit_residuals`

        are reset to their default values (typically ``None``).  This also clears the caches for some internally used
        attributes.
        """
        self._successful = False
        self._measurement_covariance = None
        self._measurements = None
        self._base_frame_directions = None
        self._temperatures = None
        self._postfit_covariance = None
        self._postfit_residuals = None
        self._jacobian = None

    def _calc_covariance(self):
        r"""
        This method calculates the post fit covariance (if a fit was successful) using the cached value if available.

        The post-fit covariance is defined as

        .. math::
            \mathbf{C}=\left((\mathbf{J}^T\mathbf{J})^{-1}
            \mathbf{J}^T\mathbf{W}\mathbf{J}
            (\mathbf{J}^T\mathbf{J})^{-1}\right)^{-1}

        where :math:`\mathbf{J}` is the Jacobian matrix evaluated at the final state estimate and
        :math:`\mathbf{W}=\mathbf{R}^{-1}` is the weight matrix, which is the inverse of the measurement covariance
        matrix (if applicable).

        If the fit was not successful (or it has not been performed yet) this will return ``None``.
        """

        # if the fit was unsuccessful return None
        if not self.successful:
            return None

        # if the covariance is cached return it
        if self._postfit_covariance is not None:
            return self._postfit_covariance

        # otherwise compute it
        weight_matrix = self._compute_weight_matrix(len(self.model.state_vector), self.measurements.size)
        
        assert self._jacobian is not None, "The jacobian hasn't been computed, something is wrong"

        if not np.isscalar(weight_matrix):
            orthogonal_project_mat = np.linalg.inv(self._jacobian.T @ self._jacobian) @ self._jacobian.T
            self._postfit_covariance = np.linalg.inv(orthogonal_project_mat @
                                                     weight_matrix @
                                                     orthogonal_project_mat.T).astype(np.float64)
        else:
            self._postfit_covariance = np.linalg.inv(self._jacobian.T @
                                                     self._jacobian * weight_matrix).astype(np.float64) # type: ignore

        return self._postfit_covariance

    def compute_residuals(self, model: Optional[CameraModel] = None) -> np.ndarray:
        """
        This method computes the observed minus computed residuals for the current model (or an input model).

        The residuals are returned as a 2xn numpy array where n is the number of stars observed with units of pixels.

        The computed values are determined by calls to ``model.project_onto_image`` for the
        :attr:`camera_frame_directions` for each image.

        :param model: An optional model to compute the residuals using.  If ``None``, then will use :attr:`model`.
        :return: The observed minus computed residuals as a numpy array
        """
        # use the model attribute if necessary
        if model is None:
            model = self.model
        return self.measurements - np.concatenate(
            [model.project_onto_image(vecs, image=ind, temperature=self.temperatures[ind])
             for ind, vecs in enumerate(self.camera_frame_directions)], axis=1
        )

    def _compute_weight_matrix(self, state_vector_size, number_of_measurements) -> float | DOUBLE_ARRAY:
        """
        This method computes the weight matrix based on whether weighted estimation is being performed, and whether
        using the a priori state as a measurement.

        :param state_vector_size: The size of the state vector
        :param number_of_measurements: The number of measurements
        :return: the weight matrix, or 1 if a weight matrix is not needed
        """

        if self.weighted_estimation and self.model.use_a_priori:
            weight_matrix = np.zeros((state_vector_size + number_of_measurements,
                                      state_vector_size + number_of_measurements), dtype=np.float64)

            if self.a_priori_model_covariance is not None:
                weight_matrix[number_of_measurements:,
                              number_of_measurements:] = np.linalg.inv(np.asanyarray(self.a_priori_model_covariance))
            else:
                weight_matrix[number_of_measurements:, number_of_measurements:] = np.eye(state_vector_size)

            if self._measurement_covariance is not None:
                if np.isscalar(self.measurement_covariance):
                    measurement_info = 1.0 / cast(float, self.measurement_covariance)
                    for i in range(number_of_measurements):
                        weight_matrix[i, i] = measurement_info
                else:
                    weight_matrix[:number_of_measurements,
                                  :number_of_measurements] = np.linalg.inv(np.asanyarray(self.measurement_covariance))
            else:
                for i in range(number_of_measurements):
                    weight_matrix[i, i] = 1

        elif self.weighted_estimation:
            if self._measurement_covariance is not None:
                if np.isscalar(self.measurement_covariance):
                    weight_matrix = 1.0 / cast(float, self.measurement_covariance)
                else:
                    weight_matrix = np.linalg.inv(np.asanyarray(self.measurement_covariance))
            else:
                weight_matrix = 1

        elif self.model.use_a_priori:
            weight_matrix = np.eye(state_vector_size + number_of_measurements, dtype=np.float64)
            if self.a_priori_model_covariance is not None:
                weight_matrix[number_of_measurements:,
                              number_of_measurements:] = np.linalg.inv(np.asanyarray(self.a_priori_model_covariance))

        else:
            weight_matrix = 1

        return weight_matrix

    def estimate(self) -> None:
        """
        Estimates an updated camera model that better transforms the camera frame directions into pixel locations to
        minimize the residuals between the observed and the predicted star locations.

        Upon successful completion, the updated camera model is stored in the :attr:`model` attribute, the
        :attr:`successful` will return ``True``, and :attr:`postfit_residuals` and :attr:`postfit_covariance` should
        both be not None.  If estimation is unsuccessful, then :attr:`successful` should be set to ``False``.

        The estimation is done using nonlinear iterative least squares, as discussed in the class documentation
        (:class:`IterativeNonlinearLSTSQ`).

        :raises ValueError: if :attr:`measurements` or :attr:`camera_frame_directions` are ``None``.
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
        pre_ss = prefit_residuals.ravel()@prefit_residuals.ravel()

        for _ in range(self.max_iter):

            jacobian = self.model.compute_jacobian(self.camera_frame_directions, temperature=self.temperatures)

            if self.model.use_a_priori:
                residuals_vec = np.concatenate([prefit_residuals.reshape((-1, 1), order='F'),
                                                np.zeros((state_size, 1))], axis=0)

            else:
                residuals_vec = prefit_residuals.reshape((-1, 1), order='F')

            if np.isscalar(weight_matrix):
                lhs = np.sqrt(weight_matrix)*jacobian.T@jacobian
                rhs = np.sqrt(weight_matrix)*jacobian.T@residuals_vec
            else:
                lhs = jacobian.T@weight_matrix@jacobian
                rhs = jacobian.T@weight_matrix@residuals_vec

            update_vec = np.linalg.solve(lhs, rhs).astype(np.float64)

            model_copy: ModelT = self.model.copy()

            model_copy.apply_update(update_vec)

            postfit_residuals = self.compute_residuals(model=model_copy)
            post_ss = postfit_residuals.ravel()@postfit_residuals.ravel()
            resid_change = abs(pre_ss-post_ss)

            # check for convergence
            if resid_change <= (self.residual_atol+self.residual_rtol*pre_ss):
                self._successful = True
                self._postfit_residuals = postfit_residuals
                self.model = model_copy
                self._jacobian = self.model.compute_jacobian(self.camera_frame_directions, temperature=self.temperatures)
                return

            elif (np.abs(update_vec) <= (self.state_atol+self.state_rtol*a_priori_state)).all():
                self._successful = True
                self._postfit_residuals = postfit_residuals
                self.model = model_copy
                self._jacobian = self.model.compute_jacobian(self.camera_frame_directions, temperature=self.temperatures)
                return

            elif pre_ss < post_ss:  # check for divergence
                warnings.warn('Solution is diverging.  Stopping iteration.'
                              '\n\tpre-update residuals {}'
                              '\n\tpost-update residuals {}'.format(pre_ss, post_ss))
                self._successful = False
                self._postfit_residuals = None
                self._jacobian = None
                return

            else:  # converging
                self.model = model_copy
                prefit_residuals = postfit_residuals
                pre_ss = post_ss
                a_priori_state = np.array(self.model.state_vector)

        warnings.warn("Solution didn't converge in the requested number of iterations")
        self._successful = False
        self._postfit_residuals = prefit_residuals
        self.model
        self._jacobian = self.model.compute_jacobian(self.camera_frame_directions, temperature=self.temperatures)


