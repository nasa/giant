
from dataclasses import dataclass

from typing import Optional, Protocol, runtime_checkable, Generic, TypeVar

from giant.camera_models import CameraModel
from giant._typing import DOUBLE_ARRAY

from giant.utilities.options import UserOptions
from giant.utilities.mixin_classes import AttributeEqualityComparison, AttributePrinting, UserOptionConfigured   


ModelT = TypeVar("ModelT", bound=CameraModel)
"""
Type variable bound to CameraModel for type safety
"""

@dataclass
class GeometricEstimatorOptions(UserOptions):
    
    weighted_estimation: bool = False
    """
    A boolean flag specifying whether to do weighted estimation.

    If set to ``True``, the estimator should use the provided measurement weights in :attr:`measurement_covariance`
    during the estimation process.  If set to ``False``, then no measurement weights should be considered.
    """
    
    a_priori_model_covariance: Optional[DOUBLE_ARRAY] = None
    """
    A square numpy array containing the covariance matrix for the a priori estimate of the state vector.

    This is only considered if :attr:`weighted_estimation` is set to ``True`` and if
    :attr:`.CameraModel.use_a_priori` is set to ``True``, otherwise it is ignored.  If both are set to ``True`` then
    this should be set to a square, full rank, lxl numpy array where ``l=len(model.state_vector)`` containing the
    covariance matrix for the a priori state vector.  The order of the parameters in the state vector can be
    determined from :meth:`.CameraModel.get_state_labels`.
    """


@runtime_checkable
class GeometricEstimator(Protocol, Generic[ModelT]):
    """
    This protocol class serves as the template for implementing a class for doing geometric camera model estimation in
    GIANT.

    Camera model estimation in GIANT is primarily handled by the :class:`.Calibration` class, which does the steps of
    extracting observed stars in an image, pairing the observed stars with a star catalog, and then passing the
    observed star-catalog star pairs to a subclass of this protocol-class, which estimates an update to the camera model
    in place (the input camera model is modified, not a copy).  In order for this to work, this protocol defines the minimum
    required interfaces that the :class:`.Calibration` class expects for an estimator.

    The required interface that the :class:`.Calibration` class expects consists of a few readable/writeable properties,
    and a couple of standard methods, as defined below.  Beyond that the implementation is left to the user.

    If you are just doing a typical calibration, then you probably need not worry about this protocol and instead can use one
    of the 2 concrete classes defined in this package, which work well in nearly all cases.  If you do have a need to
    implement your own estimator, then you should implement all the requirements in the protocol, and study the concrete 
    classes from this subpackage for an example of what needs to be done.
    """
    
    weighted_estimation: bool
    """
    A boolean flag specifying whether to do weighted estimation.

    If set to ``True``, the estimator should use the provided measurement weights in :attr:`measurement_covariance`
    during the estimation process.  If set to ``False``, then no measurement weights should be considered.
    """
    
    a_priori_model_covariance: Optional[DOUBLE_ARRAY]
    """
    A square numpy array containing the covariance matrix for the a priori estimate of the state vector.

    This is only considered if :attr:`weighted_estimation` is set to ``True`` and if
    :attr:`.CameraModel.use_a_priori` is set to ``True``, otherwise it is ignored.  If both are set to ``True`` then
    this should be set to a square, full rank, lxl numpy array where ``l=len(model.state_vector)`` containing the
    covariance matrix for the a priori state vector.  The order of the parameters in the state vector can be
    determined from :meth:`.CameraModel.get_state_labels`.
    """
    
    def __init__(self, model: ModelT, options: GeometricEstimatorOptions | None) -> None:
        ...

    @property
    def model(self) -> ModelT:
        """
        The camera model that is being estimated.

        Typically this should be a subclass of :class:`.CameraModel`.

        This should be a read/write property
        """
        ...

    @model.setter
    def model(self, val: ModelT):  # model must be writeable
        ...

    @property
    def successful(self) -> bool:
        """
        A boolean flag indicating whether the fit was successful or not.

        If the fit was successful this should return ``True``, and ``False`` if otherwise.

        This should be a read-only property.
        """
        ...

    @property
    def measurement_covariance(self) -> Optional[float | DOUBLE_ARRAY]:
        """
        A square numpy array containing the covariance matrix for the measurements.

        If :attr:`weighted_estimation` is set to ``True`` then this property will contain the measurement covariance
        matrix as a square, full rank, numpy array.  If :attr:`weighted_estimation` is set to ``False`` then this
        property may be ``None`` and should be ignored.

        This should be a read/write property.
        """
        ...

    @measurement_covariance.setter
    def measurement_covariance(self, val: Optional[float | DOUBLE_ARRAY]):  # measurement_covariance must be writeable
        ...

    @property
    def measurements(self) -> Optional[DOUBLE_ARRAY]:
        """
        A 2xn numpy array of the observed pixel locations for stars across all images

        Each column of this array will correspond to the same column of the :attr:`camera_frame_directions` concatenated
        down the last axis. (That is ``measurements[:, i] <-> np.concatenate(camera_frame_directions, axis=-1)[:, i]``)

        This will always be set before a call to :meth:`estimate`.

        This should be a read/write property.
        """
        ...

    @measurements.setter
    def measurements(self, val: DOUBLE_ARRAY):  # measurements must be writeable
        ...

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
        included to notify the user which temperatures to use.

        This will always be set before a call to :meth:`estimate`.

        This should be a read/write property.
        """
        ...

    @camera_frame_directions.setter
    def camera_frame_directions(self, val: list[DOUBLE_ARRAY | list[list]]):  # camera_frame_directions must be writeable
        ...

    @property
    def temperatures(self) -> list[float]:
        """
        A length m list of temperatures of the camera for each image being considered in estimation.

        Each element of this list corresponds to a unique image that is being considered for estimation and the
        subsequent element in the :attr:`camera_frame_directions` list.

        This will always be set before a call to :meth:`estimate` (although sometimes it may be a list of all zeros if
        temperature data is not available for the camera).

        This should be a read/write property.
        """
        ...

    @temperatures.setter
    def temperatures(self, val: list[float]):  # temperatures must be writeable
        ...

    @property
    def postfit_covariance(self) -> Optional[DOUBLE_ARRAY]:
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
        ...

    @property
    def postfit_residuals(self) -> Optional[DOUBLE_ARRAY]:
        """
        The post-fit observed-computed measurement residuals as a 2xn numpy array.

        This returns the post-fit observed minus computed measurement residuals after a call to :meth:`estimate`.  If
        :meth:`estimate` has not been called yet then this will return ``None``.

        This is a read only property
        """
        ...

    def estimate(self) -> None:
        """
        Estimates an updated camera model that better transforms the camera frame directions into pixel locations to
        minimize the residuals between the observed and the predicted star locations.

        Typically, upon successful completion, the updated camera model is stored in the :attr:`model` attribute, the
        :attr:`successful` should return ``True``, and :attr:`postfit_residuals` and :attr:`postfit_covariance` should
        both be not None.  If estimation is unsuccessful, then :attr:`successful` should be set to ``False`` and
        everything else will be ignored so you can do whatever you want with it.
        """
        ...

    def reset(self) -> None:
        """
        This method resets all of the data attributes to their default values to prepare for another estimation.

        This should reset

        * :attr:`successful`
        * :attr:`measurement_covariance`
        * :attr:`measurements`
        * :attr:`camera_frame_directions`
        * :attr:`temperatures`
        * :attr:`postfit_covariance`
        * :attr:`postfit_residuals`

        to their default values (typically ``None``) to ensure that data from one estimation doesn't get mixed with data
        from a subsequent estimation.  You may also choose to reset some other attributes depending on the
        implementation of the estimator.
        """
        ...
        
    def reset_settings(self) -> None:
        """
        This method resets all the setting back to their original values as specified at instance creation.
        
        Generally this is provided for you through the :class:`.UserOptionConfigured` mixin class.
        """
        ...


class GeometricEstimatorBC(UserOptionConfigured[GeometricEstimatorOptions], GeometricEstimatorOptions, AttributeEqualityComparison, AttributePrinting, GeometricEstimator[ModelT]):
    """
    This base class is used to make it easy to make a GeometricEstimator class using other beneficial mixin classes from GIANT.
    
    Generally, therefore, custom classes should inherit from this, and not GeometricEstimator directly, though that's not strictly necessary
    """
    pass
