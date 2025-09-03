


"""
This module defines an Extended Kalman Filter (EKF) to be used for tracking of non-cooperative targets as part of the
:mod:`.ufo` package.

Description
-----------

An EKF is a filter which linearizes both measurements and the dynamics about the current best estimate of the state to
estimate updates to the state based on ingested measurements.  It is powerful and fast, but can occasionally be finicky
when you have bad a priori conditions and the measurement/state are very non-linear.  There are many great resources on
EKFs and how they work available therefore we don't go into details here.

Use
---

This module defines a single class, :class:`.ExtendedKalmanFilter`, which implements the EKF.  Generally you won't
interact with this class directly and instead will interact with the :class:`.EKFTracker` class from the
:mod:`.ekf_tracker` module.  If you need more details on using this class directly refer to the following class
documentation.
"""

from typing import Callable, Type, Optional, List, Union, Tuple

from uuid import uuid4

from copy import deepcopy, copy

import logging

import numpy as np

from giant.ufo.dynamics import Dynamics, PN_TYPE
from giant.ufo.measurements import Measurement

from giant._typing import F_SCALAR_OR_ARRAY, DOUBLE_ARRAY


# TODO: provide a Monte filter interface at some point


_LOGGER: logging.Logger = logging.getLogger(__name__)
"""
This is the logging interface for reporting status, results, issues, and other information.
"""


STATE_INITIALIZER_TYPE = Callable[[Measurement, Type[Dynamics.State]], Dynamics.State]
"""
This defines the callable sequent for state initializer functions.

See :attr:`.state_initializer` for more details.
"""


def _negate_pn(process_noise: PN_TYPE) -> PN_TYPE:
    """
    This simple wrapper negates the process noise (as either a numpy array or a callable) for doing backwards smoothing.

    Typically this is not used directly by a user.

    :param process_noise: The process noise function or array to be negated
    :return: Either the negated array or the function wrapped to return a negated array
    """

    if isinstance(process_noise, np.ndarray):
        # just return the negated array
        return -process_noise
    else:
        # need to wrap the function and negate it maintaining the appropriate call sequence
        def negated_pn(state: np.ndarray, time: float) -> np.ndarray:
            """
            Negates a process noise function.

            :param state: The state vector
            :param time: The time
            :return: the negated process noise
            """

            return -process_noise(state, time)

        return negated_pn


class ExtendedKalmanFilter:
    """
    This class implements a simple extended kalman filter for processing measurements and estimating a state for a
    target that generated those measurements.

    The EKF works by linearizing about the current best estimate of the state at each measurement time, computing an
    update using the linearized space, updating the state, and then returning to the non-linear domain to propagate to
    the next measurement time.  This is relatively fast and powerful, but it does require an adequate initial guess for
    the state vector for the linearization to be reasonable.

    Using this EKF is fairly straight forward.  Simply specify the :class:`.Dynamics` model that governs your system
    (and your state vector) and provide a function that takes in a :class:`.Measurement` instance and a
    :class:`.Dynamics.State` class object which initializes the state off of the first measurement of the target.
    Then you can simply call :meth:`initialize` to initialize the state vector and :meth:`process_measurement` to
    process measurements (note that you should not call :meth:`process_measurement` to process the measurement you used
    to initialize the state vector).  Once you have processed all of your measurements, you can optionally call method
    :meth:`smooth` to perform backwards smoothing and get a best fit estimate of all of the residuals.

    Tuning the filter is done through the :attr:`~.SpiceGravityDynamics.process_noise` attribute of the
    :class:`.Dynamics` class (if it has one), :attr:`.Measurement.covariance` attribute, and the
    :attr:`state_initializer` function which can be used to set the initial state covariance.

    This is certainly not the most feature rich EKF and is intended primarily for light-weight work in determining
    tracks of non-cooperative targets observed in images as part of the :mod:`.ufo` package.  That being said, it is
    general enough that you could use it for other things if you wanted to, but doing that is beyond the scope of this
    documentation.
    """

    def __init__(self, dynamics: Dynamics, state_initializer: STATE_INITIALIZER_TYPE,
                 initial_measurement: Optional[Measurement] = None):
        """
        :param dynamics: The :class:`.Dynamics` instance to use to propagate the state and covariance
        :param state_initializer: A callable that initializes the state given an initial measurement.  This should take
                                  in the initial :class:`.Measurement` and the :class:`.Dynamics.State` type (class
                                  object) and return the initialized :class:`.Dynamics.State` instance with at minimum
                                  position, velocity, and covariance filled out.
        :param initial_measurement: Optionally provide the initial measurement.  If this is not ``None`` then the
                                    :meth:`initialize` method will be called.  If it is ``None`` then the
                                    :meth:`initialize` method will need to be called manually.
        """

        self.dynamics: Dynamics = dynamics
        """
        The :class:`.Dynamics` instance to use to propagate the state and covariance 
        """

        self.state_initializer: STATE_INITIALIZER_TYPE = state_initializer
        """
        A callable that initializes the state given an initial measurement.  
        
        This should take in the initial :class:`.Measurement` and the :class:`.Dynamics.State` type (class 
        object) and return the initialized :class:`.Dynamics.State` instance with at minimum 
        position, velocity, and covariance filled out. 
        """

        self.state_history: List[Tuple[Dynamics.State, Dynamics.State]] = []
        """
        This contains the history of all of the best state objects in time order.  

        Initially this will contain the history from forward filtering.  After a call to :meth:`smooth` though, this 
        will instead include the history from the backwards smoothing (in time order, not processing order).
        
        Each element is a tuple with the first element of the tuple being the pre-update state and the second element of 
        the tuple being the post-update state.
        """

        self.long_state_history: List[Tuple[Dynamics.State, Dynamics.State]] = []
        """
        This contains the history of all of the state objects as they are generated.  
        
        This includes the state history for both the forwards filtering and backwards smoothing.  For the best fit 
        history see the :attr:`state_history` parameter.
        
        Each element is a tuple with the first element of the tuple being the pre-update state and the second element of 
        the tuple being the post-update state.
        """

        self.measurement_history: List[Measurement] = []
        """
        This contains the history of the measurement objects as they are processed.
        """

        self.residuals: List[Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]] = []
        """
        This contains the history of the measurement residuals as a list of tuples of floats or numpy arrays 
        (depending on whether the ingested measurements are scalars or arrays).

        Each tuple contains the pre-update residual first followed by the post-update residual.

        These are the best residuals (after smoothing).  For a full history of residuals see :attr:`long_residuals`.
        """

        self.long_residuals: List[Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]] = []
        """
        This contains the history of all of the measurement residuals as a list of tuples of floats or numpy arrays 
        (depending on whether the ingested measurements are scalars or arrays).
        
        Each tuple contains the pre-update residual first followed by the post-update residual.
        
        These are all the residuals (both forwards and backwards smoothing).  For the history of just the best residuals
        (after smoothing) see :attr:`residuals`
        """

        self.identity = uuid4()
        """
        A unique identifier for this EKF
        """

        if initial_measurement is not None:
            self.initialize(initial_measurement)

    def __deepcopy__(self, memodict: Optional[dict]):

        other = ExtendedKalmanFilter(self.dynamics, self.state_initializer)

        other.measurement_history = copy(self.measurement_history)
        other.long_state_history = copy(self.long_state_history)
        other.state_history = copy(self.state_history)
        other.residuals = copy(self.residuals)
        other.long_residuals = copy(self.long_residuals)

        return other

    def initialize(self, initial_measurement: Measurement):
        """
        This method (re)initializes the EKF by setting the initial state using the :attr:`state_initializer` function
        and resetting all of the history lists.

        Note that this method deletes all history from the EKF, so if you want to keep the history, make sure you make a
        copy before calling this method (or create an entirely new EKF).

        :param initial_measurement: The initial measurement to use to initialize the state
        """

        # reset the state history
        initial_state = self.state_initializer(initial_measurement, self.dynamics.State)
        self.state_history = [(initial_state, initial_state)]
        self.long_state_history = [self.state_history[0]]

        # reset the measurement and residual history
        self.measurement_history = [initial_measurement]

        residual = initial_measurement.observed - initial_measurement.predict(self.state_history[0][0])
        self.residuals = [(residual*np.nan, residual)]
        self.long_residuals = [(residual*np.nan, residual)]

    def propagate_and_predict(self, measurement: Measurement) -> Tuple[Optional[Dynamics.State],
                                                                       Optional[F_SCALAR_OR_ARRAY]]:
        """
        This function integrates to the new measurement time and predicts the measurement at that time based on the
        propagated state.

        :param measurement: The measurement instance which defines at minimum the new time and can predict the
                            measurement
        :return: The state at the requested time and the predicted measurement
        """
        try:
            new_state = self.dynamics.propagate(self.long_state_history[-1][1], measurement.time)
        except ValueError as e:
            _LOGGER.error(f'EKF {self.identity} failed to propagate', exc_info=e)
            return None, None

        predicted_measurement = measurement.predict(new_state)

        return new_state, predicted_measurement

    def process_measurement(self, measurement: Measurement,
                            pre_update_state: Optional[Dynamics.State] = None,
                            pre_update_predicted_measurement: Optional[F_SCALAR_OR_ARRAY] = None,
                            backwards: bool = False, backwards_index: int = 0) -> Optional[np.ndarray]:
        """
        This does a update step for a new measurement.

        The predicted state is used along with the measurement to compute the state update, which is applied.
        The residuals are also computed.  Everything is stored in the appropriate history attributes.

        :param measurement: The new measurement to ingest
        :param pre_update_state: The predicted state at the measurement time.  If ``None`` then the state will be
                                 propagated to the measurement time
        :param pre_update_predicted_measurement: The predicted measurement using the predicted state at the measurement
                                                 time.  If ``None`` then the predicted measurement will be generated for
                                                 you
        :param backwards: A boolean flag indicating if we are doing backwards smoothing.  If ``True`` then the way we
                          store the results changes.
        :param backwards_index: An integer specifying the index into the short history lists where we should insert the
                                results.  This is ignored if ``backwards`` is not ``True``
        :return: The applied state update as a numpy array the same length of the state vector (including covariance)
        """

        if not backwards:
            # store the measurement
            self.measurement_history.append(measurement)

        # get the state
        if pre_update_state is None:
            pre_update_state, pre_update_predicted_measurement = self.propagate_and_predict(measurement)
            if pre_update_state is None:
                return None
        else:
            if pre_update_state.time != measurement.time:
                _LOGGER.warning(f"Something fishy is going on.  The state time does not match the measurement time. "
                                f"Did you forget to propagate_and_predict? STATE TIME: {pre_update_state.time} "
                                f"MEASUREMENT TIME: {measurement.time}")

        # compute the residuals
        if pre_update_predicted_measurement is None:
            pre_update_residuals = measurement.observed - measurement.predict(pre_update_state)
        else:
            pre_update_residuals = measurement.observed - pre_update_predicted_measurement

        # compute the observation matrix
        observation_matrix = measurement.compute_jacobian(pre_update_state)

        # compute the Kalman gain
        state_covariance = pre_update_state.covariance
        measurement_covariance = measurement.covariance
        try:
            kalman_gain = ((state_covariance @ observation_matrix.T) @
                           np.linalg.inv(observation_matrix @ state_covariance @ observation_matrix.T +
                                         measurement_covariance))
        except np.linalg.linalg.LinAlgError:
            _LOGGER.debug('Unable to invert kalman gain. '
                            'Falling back to pseudo inverse but something is probably wrong')

            kalman_gain = ((state_covariance @ observation_matrix.T) @
                           np.linalg.pinv(observation_matrix @ state_covariance @ observation_matrix.T +
                                          measurement_covariance))

        # update the state
        state_update = kalman_gain @ pre_update_residuals
        updated_state = deepcopy(pre_update_state)
        updated_state.update_state(state_update)

        identity_minus_ko: DOUBLE_ARRAY = np.eye(len(updated_state), dtype=np.float64) - kalman_gain@observation_matrix

        if updated_state.covariance is not None:
            updated_state.covariance = (identity_minus_ko@updated_state.covariance@identity_minus_ko.T +
                                        kalman_gain @ measurement_covariance @ kalman_gain.T)

        # get the post-fit residuals
        post_update_residuals = measurement.observed - measurement.predict(updated_state)

        if not measurement.compare_residuals(post_update_residuals, pre_update_residuals):
            _LOGGER.debug(f'Filter might be diverging. Post-update residual is not smaller than pre-update. '
                          f'PRE: {pre_update_residuals} '
                          f'POST: {post_update_residuals}')

        # store the results
        if not backwards:
            self.long_residuals.append((pre_update_residuals, post_update_residuals))
            self.residuals.append((pre_update_residuals, post_update_residuals))
            self.state_history.append((pre_update_state, updated_state))
            self.long_state_history.append((pre_update_state, updated_state))
        else:
            self.long_residuals.append((pre_update_residuals, post_update_residuals))
            self.long_state_history.append((pre_update_state, updated_state))
            self.residuals[backwards_index] = (pre_update_residuals, post_update_residuals)
            self.state_history[backwards_index] = (pre_update_state, updated_state)

        return state_update

    def smooth(self, maximum_sigma_update: float = 5) -> bool:
        """
        This method performs backwards smoothing (kind-of) for all measurements processed by this EKF.

        This is done by starting at the end of the "arc" and processing the measurements that were already ingested in
        reverse order.  In order to do this we need to negate the process noise (otherwise it causes the covariance to
        collapse instead of grow).  We then step through each measurement in reverse order and reprocess it using
        :meth:`process_measurement`.

        After calling the method, :attr:`long_residuals` and :attr:`long_state_history` will be twice as long and
        :attr:`residuals` and :attr:`state_history` will have the "smoothed" residuals and state history.

        If we can't complete the smoothing (because of NaN in the covariance) then this function will return ``False``.
        If we did complete the smoothing it will return ``True``

        :param maximum_sigma_update: The maximum state update allowed expressed as a multiple of the pre-update
                                     covariance
        """

        original_pn = None
        if (original_pn := getattr(self.dynamics, 'process_noise')) is not None:
            setattr(self.dynamics, "process_noise", _negate_pn(original_pn))

        skip = False

        for backwards_index in range(len(self.measurement_history)-2, -1, -1):
            state_update = self.process_measurement(self.measurement_history[backwards_index], backwards=True,
                                                    backwards_index=backwards_index)

            if state_update is None:
                _LOGGER.debug('Failed to propagate while smoothing')
                skip = True
                break

            
            if (b1c := self.state_history[backwards_index][1].covariance) is not None and np.isnan(b1c).any():
                _LOGGER.debug("Got a NaN in the covariance while smoothing.  Stopping.")
                skip = True
                break

            state_size = len(self.state_history[backwards_index][0])

            # compute how big of an update we made
            assert (b0c := self.state_history[backwards_index][0].covariance) is not None
            sigma_jump = np.abs(np.linalg.pinv(b0c) @ state_update[:state_size])

            if (sigma_jump >= maximum_sigma_update).any():
                _LOGGER.debug(f"Had too large of an update in the smoothing {sigma_jump}. Stopping")

                skip = True
                break

        if hasattr(self.dynamics, 'process_noise'):
            setattr(self.dynamics, "process_noise", original_pn)

        return not skip

    def compute_residual_statistics(self) -> Tuple[float, float]:
        """
        This method computes the mean and standard deviation of the residual history from the EKF

        :return: The residual mean and residual history of the best (post-smoothed) post-update residuals from the ekf
        """

        resids = np.concatenate([r[1] for r in self.residuals])

        return resids.mean(), resids.std()
