# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


from abc import ABCMeta, abstractmethod

from datetime import datetime

from uuid import uuid4

from typing import Optional, Hashable

import numpy as np

from giant.ufo.dynamics import Dynamics
from giant.image import OpNavImage
from giant.camera_models import CameraModel

from giant._typing import F_SCALAR_OR_ARRAY


class Measurement(metaclass=ABCMeta):
    """
    This ABC defines the interface for a measurement that is expected by the :class:`.ExtendedKalmanFilter` class for a
    measurement.

    To implement a new measurement type, simply subclass this class, implement the abstract methods and properties (plus
    whatever you need for the methods/properties) and then use it to feed measurements into the EKF.
    """

    @property
    @abstractmethod
    def covariance(self) -> np.ndarray:
        """
        Returns the covariance of the measurement as a numpy array
        """
        pass

    @abstractmethod
    def predict(self, state: Dynamics.State) -> F_SCALAR_OR_ARRAY:
        """
        Predicts what the measurement should be given the current state.

        :param state: The state object defining the current state of the target.
        :return: The predicted measurement as a scalar or numpy array
        """

        pass

    @property
    @abstractmethod
    def observed(self) -> F_SCALAR_OR_ARRAY:
        """
        Returns the observed measurement.
        """
        pass

    @abstractmethod
    def compute_jacobian(self, state: Dynamics.State) -> np.ndarray:
        """
        Computes and returns the change in the predicted measurement given a change in the state.

        This is also known as the observation matrix.  It linearly maps changes in the state to changes in the
        measurement.  As such, it should be a matrix of length nxm where n is the number of elements in the measurement
        (for instance n=2 for a single pixel measurement) and m is the length of the state vector (``len(state)``).

        The state vector is always guaranteed to contain position and velocity as the first 2 components of the state
        vector.

        :param state: The state object defining the current state of the target.
        :return: The Jacobian of the measurement as a numpy array
        """

        pass

    @property
    @abstractmethod
    def observer_location(self) -> Dynamics.State:
        """
        Returns the state of the observer at the time the measurement was captured in the base dynamics frame
        """

        pass

    @property
    @abstractmethod
    def time(self) -> datetime:
        """
        Returns the time of the measurement as a python datetime object
        """
        pass

    @property
    @abstractmethod
    def identity(self) -> Hashable:
        """
        The identity of this measurement.

        This is used primarily by the tracker to link measurement back to observation ids
        """
        pass

    @staticmethod
    @abstractmethod
    def compare_residuals(first: F_SCALAR_OR_ARRAY, second: F_SCALAR_OR_ARRAY) -> bool:
        """
        This compares residuals computed using these measurement models.

        If this returns ``True``, then ``first`` is smaller than or equal to ``second`` (according to the definition of
        the residuals for this measurement).  If ``False, then ``first`` is larger than ``second``.  This is used to
        check for divergence.

        :param first: The first residual
        :param second: The second residual
        :return: ``True`` if the first residual <= second residual otherwise ``False``
        """
        pass


class OpticalBearingMeasurement(Measurement):
    """
    This class implements a concrete measurement model for optical bearing measurements presented as pixel/line pairs.

    This class serves both as a functional measurement model and as an example for how to define your own measurement
    model.  Because of the power of the :class:`.CameraModel` classes in GIANT, this largely just serves as a wrapper
    around the :class:`.CameraModel` that describes the camera the measurements are being generated from.  As such, at
    initialization the camera model must be specified, as well as the image the measurement came from.

    Note that this class does not implement the ability to extract measurements from an image, just the ability to model
    them.  GIANT provides lots of capabilities for extracting measurements from images in the :mod:`.ufo` and
    :mod:`.relative_opnav` modules that you could then feed into this class for modelling in the GIANT filter.
    """

    def __init__(self, observed_measurement: np.ndarray, camera_model: CameraModel, image: OpNavImage,
                 observer_location: Dynamics.State, covariance: Optional[np.ndarray] = None,
                 identity: Optional[Hashable] = None):
        """
        :param observed_measurement: The observed measurement as a length 2 array of x, y in pixels
        :param camera_model: The camera model that represents the camera
        :param image: The image that the measurement was extracted from
        :param observer_location: The location of the camera in the base dynamics frame when the image was captured
        :param covariance: The measurement covariance matrix as a 2x2 numpy array or ``None``.  If this is ``None``
                           then the identity matrix will be assumed.  The units should be pixels squared.
        :param identity: A unique identity
        """

        self.observed_measurement: np.ndarray = observed_measurement
        """
        The observed measurement as a length 2 array of x, y pixels
        """

        self.camera_model: CameraModel = camera_model
        """
        The camera model that represents the camera
        """

        self.image: OpNavImage = np.zeros(1, dtype=np.uint8).view(image.__class__)
        """
        The image that the measurement was extracted from, which gives the temperature of the camera as well as the 
        orientation of the camera at the time the image was captured.
        
        This only includes the header data for the image, not the image data itself for memory management purposes
        """
        self.image.__array_finalize__(image)

        self._observer_location: Dynamics.State = observer_location
        """
        The location of the camera in the base dynamics frame when the measurement was captured.
        """

        self._covariance: np.ndarray = covariance if covariance is not None else np.eye(2, dtype=np.float64)
        """
        The measurement covariance matrix as a 2x2 numpy array or ``None``.  
        
        If this is ``None`` then the identity matrix will be assumed.  The units should be pixels squared.
        """

        self._identity: Hashable = identity if identity is not None else uuid4()
        """
        The identity of this measurement.  
        
        This is used primarily by the tracker to link measurement back to observation ids
        """

    @property
    def identity(self) -> Hashable:
        """
        The identity of this measurement.

        This is used primarily by the tracker to link measurement back to observation ids
        """

        return self._identity

    @property
    def observed(self) -> np.ndarray:
        """
        The observed measurement as a length 2 numpy array
        """

        return self.observed_measurement

    @property
    def covariance(self) -> np.ndarray:
        """
        The measurement covariance matrix as a 2x2 numpy array.

        The units of this should be pixels squared for the main diagonal.
        """

        return self._covariance

    @property
    def observer_location(self) -> Dynamics.State:
        """
        The location of the camera when this measurement was captured in the base Dynamics frame.
        """
        return self._observer_location

    @property
    def time(self) -> datetime:
        """
        The time when this measurement was captured as a datetime object.
        """
        return self.image.observation_date

    def predict(self, state: Dynamics.State) -> F_SCALAR_OR_ARRAY:
        """
        Predicts the pixel location of an observation for the given state of the target and the location of the camera
        at the time the target was observed.

        This is computed by getting the relative state between the ``state``` and the ``observer_location`` in the
        camera frame and then projecting the relative state using :meth:`.CameraModel.project_onto_image`.

        :param state: The estimated state of the target at the time the target was observed
        :return: The length 2 array giving the predicted pixel location of the target in the image
        """

        # compute the relative state
        relative_state = state - self._observer_location

        # put the relative state in the "inertial" frame
        relative_state_inertial = relative_state.orientation.matrix.T @ relative_state.position

        # put it into the camera frame
        relative_state_camera = self.image.rotation_inertial_to_camera.matrix @ relative_state_inertial

        # return the predicted location
        return self.camera_model.project_onto_image(relative_state_camera, temperature=self.image.temperature)

    def compute_jacobian(self, state: Dynamics.State) -> np.ndarray:
        """
        Computes and returns the change in the predicted measurement given a change in the state.

        This is also known as the observation matrix.  It linearly maps changes in the state to changes in the
        measurement.  As such, it should be a matrix of length nxm where n is the number of elements in the measurement
        (for instance n=2 for a single pixel measurement) and m is the length of the state vector (``len(state)``).

        The state vector is always guaranteed to contain position and velocity as the first 2 components of the state
        vector.

        :param state: The state object defining the current state of the target.
        :return: The Jacobian of the measurement as a numpy array
        """

        # compute the relative state
        relative_state = state - self._observer_location

        # put the relative state in the "inertial" frame
        relative_state_inertial = relative_state.orientation.matrix.T @ relative_state.position

        # put it into the camera frame
        relative_state_camera = self.image.rotation_inertial_to_camera.matrix @ relative_state_inertial

        # initialize the observation matrix
        obs_mat = np.zeros((2, len(state)), dtype=np.float64)

        # store the jacobian with respect to the position which is the only thing that matters
        obs_mat[:, :3] = (self.camera_model.compute_pixel_jacobian(relative_state_camera.reshape(3, 1)) @
                          self.image.rotation_inertial_to_camera.matrix)

        return obs_mat

    @staticmethod
    def compare_residuals(first: np.ndarray, second: np.ndarray) -> bool: # pyright: ignore[reportIncompatibleMethodOverride]
        """
        This compares residuals computed using these measurement models.

        If this returns ``True``, then ``first`` is smaller than or equal to ``second`` (according to the definition of
        the residuals for this measurement).  If ``False, then ``first`` is larger than ``second``.  This is used to
        check for divergence.

        Here the residuals are compared by their 2 norm.

        :param first: The first residual
        :param second: The second residual
        :return: ``True`` if the first residual <= second residual otherwise ``False``
        """

        return bool(np.linalg.norm(first) <= np.linalg.norm(second))
