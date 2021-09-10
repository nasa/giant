# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


r"""
This module provides the ability to estimate geometric camera models as well as static and temperature dependent
attitude alignment based off of observations of stars in monocular images.

In general, a user will not directly interface with the classes defined in this module, and instead will work with the
:class:`.Calibration` class.

Description of the Problem
--------------------------

There are 3 minimization problems we are solving in this module. The first is to minimize the residuals
between predicted and observed pixel locations for stars in an image.  The predicted pixel locations are determined by
taking a star direction from a catalogue, rotating it into the camera frame (based on an estimate of the rotation
between the camera frame and the star catalogue frame also determined from the observation of the star images), and then
projecting it onto the image using the :meth:`.CameraModel.project_onto_image` method.  We are really estimating a
change to the projection here, as it is assumed that the rotation between the camera frame and the catalogue frame is
already well known.

The second is to minimize the residual angles between sets of unit vectors expressed in 2 different frames.  This is
exactly the same problem solved in the :mod:`.stellar_opnav.estimators` module, but in this case we generally are doing
the solution over many different images, instead of for a single image.  This is done to estimate a constant static
alignment between 2 frames.

The final is to minimize the residuals between Euler angles between one frame an another over many images.  This is done
to estimate a temperature dependent alignment between 2 frames.
"""

from abc import ABCMeta, abstractmethod

from typing import List, Optional, Union, Iterable

import numpy as np

import warnings

from giant.stellar_opnav.estimators import DavenportQMethod
from giant.camera_models import CameraModel
from giant.rotations import Rotation, quaternion_to_euler
from giant._typing import NONEARRAY, Real, SCALAR_OR_ARRAY


_BFRAME_TYPE = Optional[List[Union[np.ndarray, List[List]]]]
"""
An alias for the base frame type so it doesn't need to be written each time.

Essentially this is equivalent to a list of 2d numpy arrays/empty lists of lists.
"""


class CalibrationEstimator(metaclass=ABCMeta):
    """
    This abstract base class serves as the template for implementing a class for doing camera model estimation in
    GIANT.

    Camera model estimation in GIANT is primarily handled by the :class:`.Calibration` class, which does the steps of
    extracting observed stars in an image, pairing the observed stars with a star catalogue, and then passing the
    observed star-catalogue star pairs to a subclass of this meta-class, which estimates an update to the camera model
    in place (the input camera model is modified, not a copy).  In order for this to work, this ABC defines the minimum
    required interfaces that the :class:`.Calibration` class expects for an estimator.

    The required interface that the :class:`.Calibration` class expects consists of a few readable/writeable properties,
    and a couple of standard methods, as defined below.  Beyond that the implementation is left to the user.

    If you are just doing a typical calibration, then you probably need not worry about this ABC and instead can use one
    of the 2 concrete classes defined in this module, which work well in nearly all cases.  If you do have a need to
    implement your own estimator, then you should subclass this ABC, and study the concrete classes from this module for
    an example of what needs to be done.

    .. note:: Because this is an ABC, you cannot create an instance of this class (it will raise a ``TypeError``)
    """

    @property
    @abstractmethod
    def model(self) -> CameraModel:
        """
        The camera model that is being estimated.

        Typically this should be a subclass of :class:`.CameraModel`.

        This should be a read/write property
        """
        pass

    @model.setter
    @abstractmethod
    def model(self, val: CameraModel):  # model must be writeable
        pass

    @property
    @abstractmethod
    def successful(self) -> bool:
        """
        A boolean flag indicating whether the fit was successful or not.

        If the fit was successful this should return ``True``, and ``False`` if otherwise.

        This should be a read-only property.
        """
        pass

    @property
    @abstractmethod
    def weighted_estimation(self) -> bool:
        """
        A boolean flag specifying whether to do weighted estimation.

        If set to ``True``, the estimator should use the provided measurement weights in :attr:`measurement_covariance`
        during the estimation process.  If set to ``False``, then no measurement weights should be considered.

        This should be a read/write property
        """
        pass

    @weighted_estimation.setter
    @abstractmethod
    def weighted_estimation(self, val: bool):  # weighted_estimation must be writeable
        pass

    @property
    @abstractmethod
    def measurement_covariance(self) -> Optional[SCALAR_OR_ARRAY]:
        """
        A square numpy array containing the covariance matrix for the measurements.

        If :attr:`weighted_estimation` is set to ``True`` then this property will contain the measurement covariance
        matrix as a square, full rank, numpy array.  If :attr:`weighted_estimation` is set to ``False`` then this
        property may be ``None`` and should be ignored.

        This should be a read/write property.
        """
        pass

    @measurement_covariance.setter
    @abstractmethod
    def measurement_covariance(self, val: Optional[SCALAR_OR_ARRAY]):  # measurement_covariance must be writeable
        pass

    @property
    @abstractmethod
    def a_priori_state_covariance(self) -> NONEARRAY:
        """
        A square numpy array containing the covariance matrix for the a priori estimate of the state vector.

        This is only considered if :attr:`weighted_estimation` is set to ``True`` and if
        :attr:`.CameraModel.use_a_priori` is set to ``True``, otherwise it is ignored.  If both are set to ``True`` then
        this should be set to a square, full rank, lxl numpy array where ``l=len(model.state_vector)`` containing the
        covariance matrix for the a priori state vector.  The order of the parameters in the state vector can be
        determined from :meth:`.CameraModel.get_state_labels`.

        This should be a read/write property.
        """
        pass

    @a_priori_state_covariance.setter
    @abstractmethod
    def a_priori_state_covariance(self, val: NONEARRAY):  # measurement_covariance must be writeable
        pass

    @property
    @abstractmethod
    def measurements(self) -> NONEARRAY:
        """
        A nx2 numpy array of the observed pixel locations for stars across all images

        Each column of this array will correspond to the same column of the :attr:`camera_frame_directions` concatenated
        down the last axis. (That is ``measurements[:, i] <-> np.concatenate(camera_frame_directions, axis=-1)[:, i]``)

        This will always be set before a call to :meth:`estimate`.

        This should be a read/write property.
        """
        pass

    @measurements.setter
    @abstractmethod
    def measurements(self, val: NONEARRAY):  # measurements must be writeable
        pass

    @property
    @abstractmethod
    def camera_frame_directions(self) -> _BFRAME_TYPE:
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
        included to notify the user which temperatures to use.

        This will always be set before a call to :meth:`estimate`.

        This should be a read/write property.
        """
        pass

    @camera_frame_directions.setter
    @abstractmethod
    def camera_frame_directions(self, val: list):  # camera_frame_directions must be writeable
        pass

    @property
    @abstractmethod
    def temperatures(self) -> Optional[List[Real]]:
        """
        A length m list of temperatures of the camera for each image being considered in estimation.

        Each element of this list corresponds to a unique image that is being considered for estimation and the
        subsequent element in the :attr:`camera_frame_directions` list.

        This will always be set before a call to :meth:`estimate` (although sometimes it may be a list of all zeros if
        temperature data is not available for the camera).

        This should be a read/write property.
        """
        pass

    @temperatures.setter
    @abstractmethod
    def temperatures(self, val: List[Real]):  # temperatures must be writeable
        pass

    @property
    @abstractmethod
    def postfit_covariance(self) -> NONEARRAY:
        """
        The post-fit state covariance matrix, taking into account the measurement covariance matrix (if applicable).

        This returns the post-fit state covariance matrix after a call to :meth:`estimate`.  The covariance matrix will
        be in the order according to :attr:`~.CameraModel.estimation_parameters` and if :attr:`weighted_estimation` is
        ``True`` will return the state covariance matrix taking into account the measurement covariance matrix.  If
        :attr:`weighted_estimation` is ``False``, then this will return the post-fit state covariance matrix assuming no
        measurement weighting (that is a measurement covariance matrix of the identity matrix).  If :meth:`estimate`
        has not been called yet then this will return ``None``

        This is a read only property
        """
        pass

    @property
    @abstractmethod
    def postfit_residuals(self) -> NONEARRAY:
        """
        The post-fit observed-computed measurement residuals as a 2xn numpy array.

        This returns the post-fit observed minus computed measurement residuals after a call to :meth:`estimate`.  If
        :meth:`estimate` has not been called yet then this will return ``None``.

        This is a read only property
        """
        pass

    @abstractmethod
    def estimate(self) -> None:
        """
        Estimates an updated camera model that better transforms the camera frame directions into pixel locations to
        minimize the residuals between the observed and the predicted star locations.

        Typically, upon successful completion, the updated camera model is stored in the :attr:`model` attribute, the
        :attr:`successful` should return ``True``, and :attr:`postfit_residuals` and :attr:`postfit_covariance` should
        both be not None.  If estimation is unsuccessful, then :attr:`successful` should be set to ``False`` and
        everything else will be ignored so you can do whatever you want with it.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        This method resets all of the data attributes to their default values to prepare for another estimation.

        This should reset

        * :attr:`successful`
        * :attr:`measurement_covariance`
        * :attr:`a_priori_state_covariance`
        * :attr:`measurements`
        * :attr:`camera_frame_directions`
        * :attr:`temperatures`
        * :attr:`postfit_covariance`
        * :attr:`postfit_residuals`

        to their default values (typically ``None``) to ensure that data from one estimation doesn't get mixed with data
        from a subsequent estimation.  You may also choose to reset some other attributes depending on the
        implementation of the estimator.
        """
        pass


class IterativeNonlinearLSTSQ(CalibrationEstimator):
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

    def __init__(self, model: Optional[CameraModel] = None, weighted_estimation: bool = False, max_iter: int = 20,
                 residual_atol: float = 1e-10, residual_rtol: float = 1e-10,
                 state_atol: float = 1e-10, state_rtol: float = 1e-10,
                 measurements: NONEARRAY = None, camera_frame_directions: _BFRAME_TYPE = None,
                 measurement_covariance: Optional[SCALAR_OR_ARRAY] = None, a_priori_state_covariance: NONEARRAY = None,
                 temperatures: Optional[List[Real]] = None):
        r"""
        :param model: The camera model instance to be estimated set with an initial guess of the state.
        :param weighted_estimation: A boolean flag specifying whether to do weighted estimation.  ``True`` indicates
                                    that the measurement weights (and a priori state covariance if applicable) should be
                                    used in the estimation.
        :param max_iter: The maximum number of iteration steps to attempt to reach convergence.  If convergence has not
                         been reached after attempting ``max_iter`` steps, a warning will be raised that the model has
                         not converged and :attr:`successful` will be set to ``False``.
        :param residual_atol: The absolute convergence tolerance criteria for the sum of squares of the residuals
        :param residual_rtol: The relative convergence tolerance criteria for the sum of squares of the residuals
        :param state_atol: The absolute convergence tolerance criteria for the elements of the state vector
        :param state_rtol: The relative convergence tolerance criteria for the elements of the state vector
        :param measurements: A 2xn numpy array of measurement pixel locations to be fit to
        :param camera_frame_directions: A length m list of 3xj numpy arrays or empty 3x1 list of empty lists where m is
                                        the number of unique images the data comes from (and is the same length as
                                        :attr:`temperatures`) and j is the number of measurements from each image.  A
                                        list of empty lists indicates that no measurements were identified for the
                                        corresponding image.
        :param measurement_covariance: An optional nxn numpy array containing the covariance matrix for the ravelled
                                       measurement vector (in fortran order such that the ravelled measurement vector is
                                       [x1, y1, x2, y2, ... xk, yk] where k=n//2)
        :param a_priori_state_covariance: An optional lxl numpy array containing the a priori covariance matrix for the
                                          a priori estimate of the state, where l is the number of parameters in the
                                          state vector.  This is used only if :attr:`.CameraModel.use_a_priori` is set
                                          to ``True``.  The length of the state vector can be determined by
                                          ``len(``:attr:`.CameraModel.state_vector`\ ``)``
        :param temperatures: A length m list of floats containing the camera temperature at the time of each
                             corresponding image.  These may be used by the :class:`.CameraModel` to perform temperature
                             dependent estimation of parameters like the focal length, depending on what is set for
                             :attr:`.CameraModel.estimation_parameters`
        """

        self._model = None  # type: Optional[CameraModel]
        """
        The instance attribute to store the camera model being estimated.
        """

        # set the camera model using the property
        self.model = model

        self._weighted_estimation = False  # type: bool
        """
        The instance attribute to store the weighted estimation flag.
        """

        # set the weighted estimation flag based on what the user specified
        self.weighted_estimation = weighted_estimation

        self._measurements = None  # type: NONEARRAY
        """
        The instance attribute to store the measurement array
        """

        self._base_frame_directions = None  # type: _BFRAME_TYPE
        """
        The instance attribute to store the base frame direction list
        """

        self._temperatures = None   # type: Optional[List[Real]]
        """
        The instance attribute to store the camera temperature values for each image.
        """

        # set the measurements, camera_frame_directions, and temperature based on what the user specified
        self.measurements = measurements
        self.camera_frame_directions = camera_frame_directions
        self.temperatures = temperatures

        # set the internal success flag
        self._successful = False  # type: bool
        """
        The instance attribute to store the success flag
        """

        self._a_priori_state_covariance = None  # type: NONEARRAY
        """
        The instance attribute to store the a priori state covariance
        """
        self._measurement_covariance = None  # type: Optional[SCALAR_OR_ARRAY]
        """
        The instance attribute to store the measurement covariance matrix
        """

        # set the measurement and a priori state covariance based on user input
        self.measurement_covariance = measurement_covariance
        self.a_priori_state_covariance = a_priori_state_covariance

        # set the iteration controls
        self.max_iter = max_iter  # type: int
        """
        The maximum number of iteration steps to attempt for convergence
        """

        self.residual_atol = residual_atol  # type: float
        """
        The absolute tolerance for the sum of square of the residuals to indicate convergence
        """

        self.residual_rtol = residual_rtol  # type: float
        """
        The relative tolerance for the sum of square of the residuals to indicate convergence
        """

        self.state_atol = state_atol  # type: float
        """
        The absolute tolerance for the state vector to indicate convergence
        """

        self.state_rtol = state_rtol  # type: float
        """
        The relative tolerance for the state vector to indicate convergence
        """

        self._jacobian = None  # type: NONEARRAY
        """
        A place to store the Jacobian matrix
        """

        self._postfit_covariance = None  # type: NONEARRAY
        """
        A place to cache the post-fit covariance matrix
        """
        self._postfit_residuals = None  # type: NONEARRAY
        """
        A place to cache the post-fit residual vector
        """

    @property
    def model(self) -> Optional[CameraModel]:
        """
        The camera model that is being estimated.

        Typically this should be a subclass of :class:`.CameraModel`.
        """
        return self._model

    @model.setter
    def model(self, val: CameraModel):
        if val is None:
            warnings.warn("You have set the camera model to be None. This must be changed to a functional camera model "
                          "before a call to estimate.")

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
    def weighted_estimation(self) -> bool:
        """
        A boolean flag specifying whether to do weighted estimation.

        If set to ``True``, the estimator will use the provided measurement weights in :attr:`measurement_covariance`
        during the estimation process.  If set to ``False``, then no measurement weights will be considered.
        """

        return self._weighted_estimation

    @weighted_estimation.setter
    def weighted_estimation(self, val: bool):
        if not isinstance(val, bool):
            warnings.warn("you set weighted_estimation to a non bool value. We'll convert it to bool but make sure "
                          "you didn't do this mistakenly.")
        self._weighted_estimation = bool(val)

    @property
    def measurement_covariance(self) -> Optional[SCALAR_OR_ARRAY]:
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
    def measurement_covariance(self, val: Optional[SCALAR_OR_ARRAY]):
        if val is None:
            self._measurement_covariance = None
        elif (self._measurements is not None) and (not np.isscalar(val)):
            if val.shape[0] != self._measurements.size:
                raise ValueError('The measurement covariance matrix must be a square matrix of nxn where n is the '
                                 'number of measurements being used in estimation.\n\tmeasurement_covariance shape = {}'
                                 '\n\tnumber of measurements = {}'.format(val.shape, self._measurements.size))
        self._postfit_covariance = None  # drop the cache
        self._measurement_covariance = val

    @property
    def a_priori_state_covariance(self) -> NONEARRAY:
        """
        A square numpy array containing the covariance matrix for the a priori estimate of the state vector.

        This is only considered if :attr:`weighted_estimation` is set to ``True`` and if
        :attr:`.CameraModel.use_a_priori` is set to ``True``, otherwise it is ignored.  If both are set to ``True`` then
        this should be set to a square, full rank, lxl numpy array where ``l=len(model.state_vector)`` containing the
        covariance matrix for the a priori state vector.  The order of the parameters in the state vector can be
        determined from :meth:`.CameraModel.get_state_labels`.

        :raises ValueError: If the shape of the input matrix is not appropriate for the size of the state vector
        """

        return self._a_priori_state_covariance

    @a_priori_state_covariance.setter
    def a_priori_state_covariance(self, val: NONEARRAY):
        if val is not None:
            state_length = len(self._model.state_vector)

            if state_length != val.shape[0]:
                raise ValueError("The a priori state covariance matrix must be a square matrix of shape lxl where "
                                 "l=len(model.state_vector)."
                                 "\n\tinput covariance shape = {}"
                                 "\n\tstate vector size = {}".format(val.shape, state_length))

        self._postfit_covariance = None  # drop the cache
        self._a_priori_state_covariance = val

    @property
    def measurements(self) -> NONEARRAY:
        """
        A nx2 numpy array of the observed pixel locations for stars across all images

        Each column of this array corresponds to the same column of the :attr:`camera_frame_directions` concatenated
        down the last axis. (That is ``measurements[:, i] <-> np.concatenate(camera_frame_directions, axis=-1)[:, i]``)

        This must always be set before a call to :meth:`estimate`.
        """

        return self._measurements

    @measurements.setter
    def measurements(self, val: NONEARRAY):

        self._measurements = val

    @property
    def camera_frame_directions(self) -> _BFRAME_TYPE:
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

        return self._base_frame_directions

    @camera_frame_directions.setter
    def camera_frame_directions(self, val: _BFRAME_TYPE):

        self._base_frame_directions = val

    @property
    def temperatures(self) -> Optional[List[Real]]:
        """
        A length m list of temperatures of the camera for each image being considered in estimation.

        Each element of this list corresponds to a unique image that is being considered for estimation and the
        subsequent element in the :attr:`camera_frame_directions` list.

        This must always be set before a call to :meth:`estimate` (although sometimes it may be a list of all zeros if
        temperature data is not available for the camera).
        """

        return self._temperatures

    @temperatures.setter
    def temperatures(self, val: Optional[List[Real]]):

        self._temperatures = val

    @property
    def postfit_covariance(self) -> NONEARRAY:
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
    def postfit_residuals(self) -> NONEARRAY:
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
        self._a_priori_state_covariance = None

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

        if not np.isscalar(weight_matrix):
            orthogonal_project_mat = np.linalg.inv(self._jacobian.T @ self._jacobian) @ self._jacobian.T
            self._postfit_covariance = np.linalg.inv(orthogonal_project_mat @
                                                     weight_matrix @
                                                     orthogonal_project_mat.T)
        else:
            self._postfit_covariance = np.linalg.inv(self._jacobian.T @
                                                     self._jacobian * weight_matrix)

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

    def _compute_weight_matrix(self, state_vector_size, number_of_measurements):
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

            if self._a_priori_state_covariance is not None:
                weight_matrix[number_of_measurements:,
                              number_of_measurements:] = np.linalg.inv(self.a_priori_state_covariance)
            else:
                weight_matrix[number_of_measurements:, number_of_measurements:] = np.eye(state_vector_size)

            if self._measurement_covariance is not None:
                if np.isscalar(self.measurement_covariance):
                    measurement_info = 1 / self.measurement_covariance
                    for i in range(number_of_measurements):
                        weight_matrix[i, i] = measurement_info
                else:
                    weight_matrix[:number_of_measurements,
                                  :number_of_measurements] = np.linalg.inv(self.measurement_covariance)
            else:
                for i in range(number_of_measurements):
                    weight_matrix[i, i] = 1

        elif self.weighted_estimation:
            if self._measurement_covariance is not None:
                if np.isscalar(self.measurement_covariance):
                    weight_matrix = 1/self.measurement_covariance
                else:
                    weight_matrix = np.linalg.inv(self.measurement_covariance)
            else:
                weight_matrix = 1

        elif self.model.use_a_priori:
            weight_matrix = np.eye(state_vector_size + number_of_measurements, dtype=np.float64)
            if self._a_priori_state_covariance is not None:
                weight_matrix[number_of_measurements:,
                              number_of_measurements:] = np.linalg.inv(self.a_priori_state_covariance)

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

        :raises ValueError: if :attr:`model`, :attr:`measurements`, or :attr:`camera_frame_directions` are ``None``.
        """

        if self.model is None:
            raise ValueError("Model must not be None before a call to estimate")
        if self.measurements is None:
            raise ValueError("measurements must not be None before a call to estimate")
        if self.camera_frame_directions is None:
            raise ValueError("camera_frame_directions must not be None before a call to estimate")
        if self.weighted_estimation and (self.measurement_covariance is None):
            raise ValueError("measurement_covariance must not be None before a call to estimate "
                             "if weighed_estimation is True")
        if self.model.use_a_priori and (self.a_priori_state_covariance is None):
            raise ValueError("a_priori_state_covariance must not be None before a call to estimate "
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

            update_vec = np.linalg.solve(lhs, rhs)

            model_copy = self.model.copy()

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


class LMAEstimator(IterativeNonlinearLSTSQ):
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

    def __init__(self, model: Optional[CameraModel] = None, weighted_estimation: bool = False, max_iter: int = 20,
                 residual_atol: float = 1e-10, residual_rtol: float = 1e-10, state_atol: float = 1e-10,
                 state_rtol: float = 1e-10, max_divergence_steps: int = 5, measurements: NONEARRAY = None,
                 camera_frame_directions: _BFRAME_TYPE = None, measurement_covariance: Optional[SCALAR_OR_ARRAY] = None,
                 a_priori_state_covariance: NONEARRAY = None, temperatures: Optional[List[Real]] = None):
        r"""
        :param model: The camera model instance to be estimated set with an initial guess of the state.
        :param weighted_estimation: A boolean flag specifying whether to do weighted estimation.  ``True`` indicates
                                    that the measurement weights (and a priori state covariance if applicable) should be
                                    used in the estimation.
        :param max_iter: The maximum number of iteration steps to attempt to reach convergence.  If convergence has not
                         been reached after attempting ``max_iter`` steps, a warning will be raised that the model has
                         not converged and :attr:`successful` will be set to ``False``.
        :param residual_atol: The absolute convergence tolerance criteria for the sum of squares of the residuals
        :param residual_rtol: The relative convergence tolerance criteria for the sum of squares of the residuals
        :param state_atol: The absolute convergence tolerance criteria for the elements of the state vector
        :param state_rtol: The relative convergence tolerance criteria for the elements of the state vector
        :param max_divergence_steps: The maximum number of steps in a row that can diverge before breaking iteration
        :param measurements: A 2xn numpy array of measurement pixel locations to be fit to
        :param camera_frame_directions: A length m list of 3xj numpy arrays or empty 3x1 list of empty lists where m is
                                        the number of unique images the data comes from (and is the same length as
                                        :attr:`temperatures`) and j is the number of measurements from each image.  A
                                        list of empty lists indicates that no measurements were identified for the
                                        corresponding image.
        :param measurement_covariance: An optional nxn numpy array containing the covariance matrix for the ravelled
                                       measurement vector (in fortran order such that the ravelled measurement vector is
                                       [x1, y1, x2, y2, ... xk, yk] where k=n//2)
        :param a_priori_state_covariance: An optional lxl numpy array containing the a priori covariance matrix for the
                                          a priori estimate of the state, where l is the number of parameters in the
                                          state vector.  This is used only if :attr:`.CameraModel.use_a_priori` is set
                                          to ``True``.  The length of the state vector can be determined by
                                          ``len(``:attr:`.CameraModel.state_vector`\ ``)``
        :param temperatures: A length m list of floats containing the camera temperature at the time of each
                             corresponding image.  These may be used by the :class:`.CameraModel` to perform temperature
                             dependent estimation of parameters like the focal length, depending on what is set for
                             :attr:`.CameraModel.estimation_parameters`
        """
        super().__init__(model=model, weighted_estimation=weighted_estimation, max_iter=max_iter,
                         residual_atol=residual_atol, residual_rtol=residual_rtol,
                         state_atol=state_atol, state_rtol=state_rtol,
                         measurements=measurements, camera_frame_directions=camera_frame_directions,
                         measurement_covariance=measurement_covariance,
                         a_priori_state_covariance=a_priori_state_covariance,
                         temperatures=temperatures)

        self.max_divergence_steps = max_divergence_steps  # type: int
        """
        The maximum number of steps in a row that can diverge before breaking iteration
        """

    def estimate(self) -> None:
        """
        This method estimates the postfit residuals based on the model, weight matrix, lma coefficient, etc.
        Convergence is achieved once the standard deviation of the computed residuals is less than the absolute
        tolerance or the difference between the prefit and postfit residuals is less than the relative tolerance.

        """
        if self.model is None:
            raise ValueError("Model must not be None before a call to estimate")
        if self.measurements is None:
            raise ValueError("measurements must not be None before a call to estimate")
        if self.camera_frame_directions is None:
            raise ValueError("camera_frame_directions must not be None before a call to estimate")
        if self.weighted_estimation and (self.measurement_covariance is None):
            raise ValueError("measurement_covariance must not be None before a call to estimate "
                             "if weighed_estimation is True")
        if self.model.use_a_priori and (self.a_priori_state_covariance is None):
            raise ValueError("a_priori_state_covariance must not be None before a call to estimate "
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
                lhs = np.sqrt(weight_matrix) * jacobian.T @ jacobian
                rhs = np.sqrt(weight_matrix) * jacobian.T @ residuals_vec
            else:
                lhs = jacobian.T @ weight_matrix @ jacobian
                rhs = jacobian.T @ weight_matrix @ residuals_vec

            # get the update vector using LMA
            update_vec = np.linalg.solve(lhs + lma_coefficient*np.diag(np.diag(lhs)), rhs)

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


class StaticAlignmentEstimator:
    """
    This class estimates a static attitude alignment between one frame and another.

    The static alignment is estimated using Davenport's Q-Method solution to Wahba's problem, using the
    :class:`.DavenportQMethod` class.  To use, simply specify the unit vectors from the base frame and the unit vectors
    from the target frame, and then call :meth:`estimate`.  The estimated alignment from frame 1 to frame 2 will
    be stored as a :class:`.Rotation` object in :attr:`alignment`.

    In general this class should not be used by the user, and instead you should use the :class:`.Calibration` class and
    its :meth:`~.Calibration.estimate_static_alignment` method which will handle set up and tear down of this class for
    you.

    For more details about the algorithm used see the :class:`.DavenportQMethod` documentation.
    """

    def __init__(self, frame1_unit_vecs: NONEARRAY = None, frame2_unit_vecs: NONEARRAY = None):
        """
        :param frame1_unit_vecs: Unit vectors in the base frame as a 3xn array where each column is a unit vector.
        :param frame2_unit_vecs: Unit vectors in the destination (camera) frame as a 3xn array where each column is a
                                 unit vector
        """

        self.frame1_unit_vecs = frame1_unit_vecs  # type: NONEARRAY
        """
        The base frame unit vectors.
        
        Each column of this 3xn matrix should correspond to the same column in the :attr:`frame2_unit_vecs` attribute.
        
        Typically this data should come from multiple images to ensure a good alignment can be estimated over 
        time.
        """

        self.frame2_unit_vecs = frame2_unit_vecs  # type: NONEARRAY
        """
        The target frame unit vectors.

        Each column of this 3xn matrix should correspond to the same column in the :attr:`frame1_unit_vecs` attribute.

        Typically this data should come from multiple images to ensure a good alignment can be estimated over 
        time.
        """

        self.alignment = None  # type: Optional[Rotation]
        """
        The location where the estimated alignment is stored
        """

    def estimate(self):
        """
        Estimate the static alignment between the frame 1 and frame 2 using Davenport's Q Method Solution.

        The estimated alignment is stored in the :attr:`alignment` attribute.
        """

        solver = DavenportQMethod(target_frame_directions=np.hstack(self.frame2_unit_vecs),
                                  base_frame_directions=np.hstack(self.frame1_unit_vecs))

        solver.estimate()

        self.alignment = solver.rotation


class TemperatureDependentAlignmentEstimator:
    r"""
    This class estimates a temperature dependent attitude alignment between one frame and another.

    The temperature dependent alignment is found by fitting linear temperature dependent euler angles (or
    Tait-Bryan angles) to transform from the first frame to the second.  That is

    .. math::
        \mathbf{T}_B=\mathbf{R}_m(\theta_m(t))\mathbf{R}_n(\theta_n(t))\mathbf{R}_p(\theta_p(t))\mathbf{T}_A

    where :math:`\mathbf{T}_B` is the target frame, :math:`\mathbf{R}_i` is the rotation matrix about the :math:`i^{th}`
    axis, :math:`\mathbf{T}_A` is the base frame, and :math:`\theta_i(t)` are the linear angles.

    This fit is done in a least squares sense by computing the values for :math:`\theta_i(t)` across a range of
    temperatures (by estimating the attitude for multiple single images) and then solving the system

    .. math::
        \left[\begin{array}{cc} 1 & t_1 \\ 1 & t_2 \\ \vdots & \vdots \\ 1 & t_n \end{array}\right]
        \left[\begin{array}{ccc} \theta_{m0} & \theta_{n0} & \theta_{p0} \\
        \theta_{m1} & \theta_{n1} & \theta_{p1}\end{array}\right] =
        \left[\begin{array}{ccc}\vphantom{\theta}^0\theta_m &\vphantom{\theta}^0\theta_n &\vphantom{\theta}^0\theta_p\\
        \vdots & \vdots & \vdots \\
        \vphantom{\theta}^k\theta_m &\vphantom{\theta}^k\theta_n &\vphantom{\theta}^k\theta_p\end{array}\right]

    where :math:`\vphantom{\theta}^k\theta_i` is the measured Euler/Tait-Bryan angle for the :math:`k^{th}` image.

    In general a user should not use this class and instead the
    :meth:`.Calibration.estimate_temperature_dependent_alignment` should be used which handles the proper setup.
    """

    def __init__(self, frame_1_rotations: Optional[Iterable[Rotation]] = None,
                 frame_2_rotations: Optional[Iterable[Rotation]] = None,
                 temperatures: Optional[List[Real]] = None, order: str = 'xyz'):
        """
        :param frame_1_rotations: The rotation objects from the inertial frame to the base frame
        :param frame_2_rotations: The rotation objects from the inertial frame to the target frame
        :param temperatures: The temperature of the camera corresponding to the times the input rotations were
                             estimated.
        :param order: The order of the rotations to perform according to the convention in :func:`.quaternion_to_euler`
        """

        self.frame_1_rotations = frame_1_rotations  # type: Optional[Iterable[Rotation]]
        """
        An iterable containing the rotations from the inertial frame to the base frame for each image under 
        consideration.
        """

        self.frame_2_rotations = frame_2_rotations  # type: Optional[Iterable[Rotation]]
        """
        An iterable containing the rotations from the inertial frame to the target frame for each image under 
        consideration.
        """

        self.temperatures = temperatures  # type: Optional[List[Real]]
        """
        A list containing the temperatures of the camera for each image under consideration
        """

        self.order = order  # type: str
        """
        The order of the Euler angles according to the convention in :func:`.quaternion_to_euler`
        """

        self.angle_m_offset = None  # type: Optional[float]
        """
        The estimated constant angle offset for the m rotation axis in radians.
        
        This will be ``None`` until :meth:`estimate` is called.
        """

        self.angle_m_slope = None  # type: Optional[float]
        """
        The estimated angle temperature slope for the m rotation axis in radians.
        
        This will be ``None`` until :meth:`estimate` is called.
        """

        self.angle_n_offset = None  # type: Optional[float]
        """
        The estimated constant angle offset for the n rotation axis in radians.
        
        This will be ``None`` until :meth:`estimate` is called.
        """

        self.angle_n_slope = None  # type: Optional[float]
        """
        The estimated angle temperature slope for the n rotation axis in radians.
        
        This will be ``None`` until :meth:`estimate` is called.
        """

        self.angle_p_offset = None  # type: Optional[float]
        """
        The estimated constant angle offset for the p rotation axis in radians.
        
        This will be ``None`` until :meth:`estimate` is called.
        """

        self.angle_p_slope = None  # type: Optional[float]
        """
        The estimated angle temperature slope for the p rotation axis in radians.
        
        This will be ``None`` until :meth:`estimate` is called.
        """

    def estimate(self) -> None:
        """
        This method estimates the linear temperature dependent alignment as 3 linear temperature dependent euler
        angles according to :attr:`order`.

        This is done by first converting the relative rotation from the base frame to the target frame into euler angles
        for each image under consideration, and then performing a linear least squares estimate of the temperature
        dependence.  The resulting fit is store in the ``angle_..._...`` attributes in units of radians.

        :raises ValueError: if any of :attr:`temperatures`, :attr:`frame_1_rotations`, :attr:`frame_2_rotations` are
                            still ``None``
        """

        if self.frame_1_rotations is None:
            raise ValueError('frame_1_rotations must be set before a call to estimate')
        if self.frame_2_rotations is None:
            raise ValueError('frame_2_rotations must be set before a call to estimate')
        if self.temperatures is None:
            raise ValueError('temperatures must be set before a call to estimate')

        relative_euler_angles = []

        # get the independent euler angles
        for f1, f2 in zip(self.frame_1_rotations, self.frame_2_rotations):

            relative_euler_angles.append(list(quaternion_to_euler(f2*f1.inv(), order=self.order)))

        # make the coefficient matrix
        coef_mat = np.vstack([np.ones(len(self.temperatures)), self.temperatures]).T

        # solve for the solution
        solution = np.linalg.lstsq(coef_mat, relative_euler_angles)[0]

        # store the solution
        self.angle_m_offset = solution[0, 0]
        self.angle_m_slope = solution[1, 0]
        self.angle_n_offset = solution[0, 1]
        self.angle_n_slope = solution[1, 1]
        self.angle_p_offset = solution[0, 2]
        self.angle_p_slope = solution[1, 2]
