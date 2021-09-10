# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
This module defines dynamics models to be used in an EKF for propagating the state and covariance of an estimated
target.

Description
-----------

The dynamics specify how the target is expected to move in space given initial conditions.  They are usually specified
as an ODE initial value problem and integrated numerically, although for simple cases a few closed for analytic
solutions exist. These models are generally used by the :mod:`extended_kalman_filter` module in order to link
observations together and to propagate the state from one time to the next.

Use
---

This module defines 3 classes for defining Dynamics models.  The first, :class:`.Dynamics` is an abstract base class
that provides a template for creating a new dynamics model in GIANT.ufo.  If you want to create your own custom model,
you should subclass this and implement/update the :attr:`.Dynamics.State` class describing the state vector for your
model, and the abstract methods defined by the abstract class.  If you do this then you will be able to use your
dynamics model in the EKF (presuming your dynamics model works and describes what it going on).

Alternatively, the there are 2 basic dynamics models provided in this module which you can use directly or subclass and
extend with more features.  The first :class:`SpiceGravityDynamics` implements a simple n-body problem dynamics model
assuming point mass gravity, using NAIF spice to query the n-body positions at each integration step.  The second,
:class:`SolRadAndGravityDynamics` adds cannonball model solar radiation pressure to the simple n-body gravity model.
These 2 simple dynamics models are generally sufficient for most things you'll be tracking in UFO and thus can be used
directly.  They also serve as examples for implementing/extending your own dynamics models.
"""

from dataclasses import dataclass, field

from copy import deepcopy

from datetime import datetime

from abc import ABCMeta, abstractmethod

from pathlib import Path

# we minimize the security risk here by using FTPS
import ftplib  # nosec

from typing import Optional, List, Dict, Tuple, Union, Callable, ClassVar

import numpy as np

import spiceypy as spice

from scipy.integrate import solve_ivp
from scipy.optimize import OptimizeResult

from giant.rotations import Rotation

from giant.utilities.spice_interface import datetime_to_et


_I3: np.ndarray = np.eye(3, dtype=np.float64)
"""
Just a constant 3x3 identity matrix.  

This is used to avoid having to allocate matrices all the time for efficiency.  Probably doesn't really do anything but 
its low hanging fruit.
"""


def _numeric_jacobian(dynamics: 'Dynamics', state: np.ndarray, state_size: int, time: float):

    pert = 1e-6

    jacobian = np.zeros((state_size, state_size))

    for i in range(state_size):
        pert_vect = np.zeros(len(state))

        pert_vect[i] = pert

        positive_pert = state + pert_vect

        negative_pert = state - pert_vect

        jacobian[i, :] = (dynamics.compute_dynamics(time, positive_pert)[:state_size] -
                          dynamics.compute_dynamics(time, negative_pert)[:state_size]) / (2*pert)

    return jacobian


def _download_kernels():
    """
    This function downloads the standard spice kernels
    """

    # security threat is not risky due to FTPS
    ftp = ftplib.FTP_TLS('naif.jpl.nasa.gov')  # nosec

    ftp.login(secure=False)
    ftp.cwd('pub/naif/generic_kernels/')

    files = ['spk/planets/de440.bsp',
             'pck/pck00010.tpc',
             'pck/gm_de431.tpc',
             'lsk/latest_leapseconds.tls']

    data = (Path(__file__).parent / "data")

    for file in files:
        local = data.joinpath(file)

        local.parent.mkdir(exist_ok=True, parents=True)

        with local.open('wb') as out_file:
            ftp.retrbinary('RETR {}'.format(file), out_file.write)


def zero3() -> np.ndarray:
    """
    This simple function returns a length 3 numpy array of zeros.

    It is used as the default_factory for State fields sometimes.

    :return: np.zeros(3, dtype=np.float64)
    """

    return np.zeros(3, dtype=np.float64)


class Dynamics(metaclass=ABCMeta):
    """
    Dynamics classes are used to propagate state and covariance through time.

    This is an abstract Dynamics class and it defines the interface that is expected for a Dynamics class.  Namely, a
    dynamics class needs to define a ``class State`` which defines the state vector used in that dynamics class as well
    as a :meth:`compute_dynamics` which computes the time derivative of the state vector.  You may also wish to
    override the :meth:`propagate` method if you would like to do your own propagation instead of the default.
    """

    @dataclass
    class State:
        """
        Each Dynamics subclass should define a State class which defines that the state is for that dynamics case.

        At minimum the state class should provide time, position, velocity, and covariance attributes.  You should also
        define the __sub__ method to compute at least the relative state for position/velocity and define a vector
        property which returns the state as a 1D vector representation or sets the state from a 1D vector
        representation, at least if you want to use the default propagate method provided by the :class:`Dynamics`
        class.

        Dataclasses can make these easy (so you don't have to right your own init method) but you don't need to use them

        For many cases this simple class is probably sufficient for what you need.
        """

        time: datetime
        """
        The time that this state is for
        """

        position: np.ndarray
        """
        The position of this state in km as a numpy array
        """

        velocity: np.ndarray = field(default_factory=zero3)
        """
        The velocity of this state in km/sec as a numpy array
        """

        orientation: Rotation = field(default_factory=Rotation)
        """
        The orientation with respect to the base frame that this state is represented in.  
         
        Typically the base frame is the inertial frame centered at the CB.  It doesn't have to be this, but it must be
        inertial (not rotating).
        """

        covariance: Optional[np.ndarray] = None
        """
        State covariance matrix as a n x n numpy array
        """

        length: ClassVar[int] = 6
        """
        The length of the state vector for this state instance
        """

        def __len__(self) -> int:
            """
            Returns the number of elements in the state vector
            """
            return self.length

        @property
        def vector(self) -> List[np.ndarray]:
            """
            This property returns a list of arrays that can be concatenated together to form the state vector.

            Therefore, you can do ``np.concatenate(state_instance.vector)`` to get a 1D array of the state vector.

            This is done so that subclasses can override the order if they so choose.

            :return: a list of numpy arrays
            """

            return list(map(np.ravel, [self.position, self.velocity, self.covariance]))

        @vector.setter
        def vector(self, val: np.ndarray):

            self.position = val[:3]
            self.velocity = val[3:6]
            self.covariance = val[6:].reshape(6, 6)

        def __sub__(self, other: __qualname__) -> __qualname__:
            """
            This computes the relative state from other to self.

            Mathematically we have:

            .. math::
                a-b = c \\
                c+b = a \\
                a-c = b

            where a is self, b is other, and c is the result of this method (the relative state)

            The relative state that is returned will be expressed in the orientation of a (or self).  It is assumed that
            the states share the same origin (although the returned relative state will not share the same origin)

            Covariance is only rotated in this computation and is assumed to be the same as other

            :param other: The state that we are computing the relative state with respect to
            :return: the relative state from other to self
            """

            if not isinstance(other, self.__class__):
                return NotImplemented

            # make a copy of self
            out = deepcopy(self)

            delta_orientation = self.orientation*other.orientation.inv()

            out.position = self.position - delta_orientation.matrix@other.position
            out.velocity = self.velocity - delta_orientation.matrix@other.velocity
            out.orientation = self.orientation
            if other.covariance is not None:
                out.covariance = delta_orientation.matrix@other.covariance@delta_orientation.matrix.T

            return out

        def __add__(self, other) -> __qualname__:
            """
            This computes adds the relative state other to self.

            Mathematically we have:

            .. math::
                a+b = c \\
                c-b = a \\
                a-c = b

            where a is self, b is other, and c is the result of this method

            The state that is returned will be expressed in the orientation of other.

            Covariance is ignored in this computation (only position and velocity are modified from self).

            :param other: The relative state to be added to this one
            :return: The new state expressed in the orientation of other
            """

            if not isinstance(other, self.__class__):
                return NotImplemented

            # make a copy of self
            out = deepcopy(other)

            delta_orientation = other.orientation*self.orientation.inv()

            out.position = other.position + delta_orientation.matrix@self.position
            out.velocity = other.velocity + delta_orientation.matrix@self.velocity

            out.covariance = delta_orientation.matrix@self.covariance@delta_orientation.matrix.T

            return out

        def update_state(self, state_update: np.ndarray):
            """
            Perform an additive update to ``self``.

            This method updates ``self`` by adding the input update to ``self``.

            This is used inside of the :class:`.ExtendedKalmanFilter` to apply the update to the state vector from a
            processed measurement.  The ``state_update`` input will be a 1d numpy array the same length as the state
            vector (``len(self)``).

            :param state_update: The update vector to add to the state
            """

            self.position += state_update[:3]

            self.velocity += state_update[3:6]


    def propagate(self, state: State, new_time: datetime) -> State:
        """
        This method propagates the input state the the requested time.

        By default an RK45 ODE integrator is used to integrate the state from the first time to the second, though you
        can override this method if you wish.

        :param state: The current state
        :param new_time: the time the state is to be propagated to
        :return: The updated state at the new time (a copy)
        """

        # get the ephemeris seconds since J2000 for the start/end times to avoid leap second non-sense
        start = datetime_to_et(state.time)
        stop = datetime_to_et(new_time)

        # get the state vector at the initial time
        # this will give use a 1D array of [position, velocity, covariance], at least for the default State case
        state_vector = np.concatenate(state.vector)

        # do the integration
        # noinspection PyTypeChecker
        solution: OptimizeResult = solve_ivp(self.compute_dynamics, [start, stop], state_vector, method='RK45')

        out_state = deepcopy(state)

        out_state.vector = solution.y[:, -1]

        out_state.time = new_time

        return out_state

    @abstractmethod
    def compute_dynamics(self, time: float, state: np.ndarray) -> np.ndarray:
        """
        This method should compute the dynamics for the state vector (which normally is the position/velocity/other
        state parameters + the raveled covariance matrix)

        The dynamics should be a 1d array that is the same length as the state vector.  It should give the time
        derivative of the state vector.

        :param time: the time at which the dynamics are to be computed.  Normally this is as ephemeris seconds since
                     J2000
        :param state: The state vector to compute the dynamics for
        :return: The time derivative of the state vector
        """

        pass



PN_TYPE = Union[np.ndarray, Callable[[np.ndarray, float], np.ndarray]]
"""
This describes the type that the process noise can be.

It can either be a numpy array, or it can be a callable that takes in the state vector and current time and returns a 
numpy array
"""


class SpiceGravityDynamics(Dynamics):
    """
    This class implements a simple N-Body gravity dynamics model using Spice as the source of the planet locations.

    To use this class, specify the central (primary body) as well as any bodies/barycenters you want to include as
    n-body sources through the ``center_body`` and ``other_bodies`` arguments respectively.  The class will then
    automatically compute the gravity for you.  Note that you should not include the central body in the list of n-body
    perturbations.

    By default, this class will retrieve the GM values from spice for each body/barycenter being considered.  You can
    override these values by specifying the key word argument ``gravity_parameters`` which should be a dictionary
    mapping the name of the planet to the gravity parameter.

    This class can optionally furnsh the planetary ephemeris and planetary constants files for you if you so desire.
    This is controlled through key word argument ``load_data``.  If you provide this option, the class will attempt to
    locate the files in the ``data`` directory in the directory containing this file.  If it cannot find the files there
    then it will ask you if it can download them from the naif site.  If you have already loaded kernels that provide
    the required data (namely the name of the planets, the planetary GM (unless you are providing your own)), and the
    locations of the planets) then you should leave this option off as it could override the values you have already
    loaded.

    The covariance in this dynamics model is integrated directly instead of using the state transformation matrix
    (because that is how I learned EKFs...).  This means that the process noise is added to the covariance derivative
    directly (which may be different from what many are use to).  Therefore be sure you carefully consider how to set
    the process noise when using this class.
    """

    def __init__(self, center_body: str, process_noise: PN_TYPE,
                 other_bodies: Optional[List[str]] = None,
                 gravity_parameters: Optional[Dict[str, float]] = None, load_data: bool = False,
                 minimum_time_step: Optional[float] = 0.001):
        """
        :param center_body: The center of integration (and the center of the integration frame)
        :param process_noise: The process noise either as a numpy array of shape 7x7 (constant process noise) or as a
                              callable object which takes in the current state and time (np.ndarray, float) and outputs
                              a 7x7 numpy array containing the process noise matrix
        :param other_bodies: Other bodies whose gravity should be included as a list of body/barycenter names.  If
                             ``None`` no other bodies will be considered.
        :param gravity_parameters: A dictionary mapping names to GM values.  If this is ``None`` or it does not provide
                                   the data for one of the bodies being considered we will try to query this from spice.
                                   The values should be in km**3/sec**2
        :param load_data: A boolean flag specifying whether this class should furnsh the required datafiles.  If you
                          have already loaded files then you should leave this as ``False``.
        :param minimum_time_step: The minimum time step to allow the integrator to take in seconds.  If ``None`` (or 0)
                                  then no minimum time step is enforced.
        """

        self.center_body: str = center_body
        """
        The name of the center of integration and center of the integration frame
        """

        self.process_noise: PN_TYPE = process_noise
        """
        The process noise either as a numpy array of shape 6x6 (constant process noise) or as a 
        callable object which takes in the current state and time (np.ndarray, float) and outputs
        a 6x6 numpy array containing the process noise matrix
        """

        self.other_bodies: List[str] = other_bodies if other_bodies is not None else []
        """
        A list of other bodies to consider the gravity effects for.
        """

        self.gravity_parameters: Dict[str, float] = gravity_parameters if gravity_parameters is not None else dict()
        """
        A dictionary mapping planet/barycenter name to GM in km**3/sec**2
        """

        if load_data:

            data = Path(__file__).parent / 'data'

            files = list(map(data.joinpath, ['spk/planets/de440.bsp',
                                             'pck/pck00010.tpc',
                                             'pck/gm_de431.tpc',
                                             'lsk/latest_leapseconds.tls']))

            for file in files:
                if not file.exists():
                    download = input("Missing spice data.  Would you like to download it (y/n)?")

                    if download.lower() == 'y':
                        _download_kernels()
                    else:
                        raise ValueError('Requested to load data but data is missing')

                spice.furnsh(str(file))

        self.gm_cb: float = self.get_gm(self.center_body)
        """
        The gm of the central body
        """

        self.gm_other_bodies: List[float] = [self.get_gm(body) for body in self.other_bodies]
        """
        A list of the GMs of other bodies
        """

        self.previous_time: Optional[float] = None
        """
        The previous time step for tracking the minimum step size
        """

        self.minimum_time_step: Optional[float] = minimum_time_step
        """
        The minimum time step to allow the integrator to take in seconds. 
        
        If ``None`` or ``0`` no minimum time step is enforced.
        
        If the minimum time step is encountered then a Value error is raised
        """

    def get_gm(self, body: str) -> float:

        gm = self.gravity_parameters.get(body)

        if gm is None:
            gm = spice.bodvrd(body, 'GM', 1)[1]

        return gm

    def compute_state_dynamics(self, state: np.ndarray, et_time: float,
                               return_intermediaries: bool = False) -> Union[List[np.ndarray],
                                                                             Tuple[List[np.ndarray],
                                                                                   Tuple[float,
                                                                                         List[np.ndarray],
                                                                                         List[np.ndarray],
                                                                                         List[float],
                                                                                         List[float]]]]:
        """
        This computes the dynamics for just the "state" part of the state vector (not the covariance)

        Optionally this can also return the distance from the CB to the spacecraft, the vectors from the central body to
        the other bodies, the vectors from the spacecraft to the other bodies, the distances from the central body to
        the other bodies, and the distances from the other bodies to the spacecraft if ``return_intermediaries`` is
        ``True``.

        The first component of the return is always a list of the dynamics for the state vector in order of position,
        velocity.

        :param state: The state vector at the current time
        :param et_time: The ephemeris time
        :param return_intermediaries: A flag specifying whether to return the intermediate distances/vectors for use
                                      elsewhere
        :return: A list containing [dposition/dt, dvelocity/dt].  Optionally return a second tuple containing
                 (radial_distance_cb, position_cp_to_bodies, position_sc_to_bodies, radial_distance_cb_to_bodies,
                 radial_distance_sc_to_bodies)

        """

        position: np.ndarray = state[:3]  # position is from CB to S/C
        velocity: np.ndarray = state[3:6]

        # distance to central body
        radial_distance_cb: float = np.linalg.norm(position)

        # compute the gravitational acceleration due to the central body
        acceleration_gravity: np.ndarray = -self.gm_cb*position/radial_distance_cb**3

        # gravity due to other bodies
        position_sc_to_bodies: List[np.ndarray] = []
        position_cb_to_bodies: List[np.ndarray] = []
        radial_distance_sc_to_bodies: List[float] = []
        radial_distance_cb_to_bodies: List[float] = []
        for body, gm in zip(self.other_bodies, self.gm_other_bodies):
            position_cb_to_bodies.append(spice.spkpos(body, et_time, 'J2000', 'LT+S', self.center_body)[0])
            position_sc_to_bodies.append(position_cb_to_bodies[-1] - position)
            radial_distance_sc_to_bodies.append(np.linalg.norm(position_sc_to_bodies[-1]))
            radial_distance_cb_to_bodies.append(np.linalg.norm(position_cb_to_bodies[-1]))

            acceleration_gravity += gm*(position_sc_to_bodies[-1]/radial_distance_sc_to_bodies[-1]**3 +
                                        position_cb_to_bodies[-1]/radial_distance_cb_to_bodies[-1]**3)

        if return_intermediaries:
            return [velocity, acceleration_gravity], (radial_distance_cb,
                                                      position_cb_to_bodies, position_sc_to_bodies,
                                                      radial_distance_cb_to_bodies, radial_distance_sc_to_bodies)
        else:
            return [velocity, acceleration_gravity]

    def compute_covariance_dynamics(self, state: np.ndarray, et_time: float, radial_distance_cb: float,
                                    position_sc_to_bodies: List[np.ndarray],
                                    radial_distance_sc_to_bodies: List[float]) -> np.ndarray:
        r"""
        This method computes the dynamics for the covariance matrix.

        The dynamics for the covariance matrix is the product of the Jacobian matrix of the dynamics for the state with
        respect to the state and the current covariance matrix, plus the process noise matrix

        .. math::
            \mathbf{\partial \mathbf{P}}{\partial t} = \mathbf{J}\mathbf{P} + \mathbf{P}\mathbf{J}^T + \mathbf{Q}

        where :math:`\mathbf{P}` is the covariance matrix, :math:`t` is time,

        .. math::
            \mathbf{J}=\frac{\partial \mathbf{f}(\mathbf{x})}{\partial\mathbf{x}}

        with :math:`\mathbf{J}` being the Jacobian matrix, :math:`\mathbf{f}(\mathbf{x})` is the state dynamics
        function, and :math:`\mathbf{x}` is the state vector.

        :param state: The state vector at the current time
        :param et_time: The ephemeris time
        :param radial_distance_cb: The distance from the central body to the body we are estimating
        :param position_sc_to_bodies: The position from the spacecraft to the other bodies considered for gravity
        :param radial_distance_sc_to_bodies: The distance from the spacecraft to the other bodies
        :return: The covariance time derivative.
        """

        jacobian = np.zeros((6, 6), dtype=np.float64)

        jacobian[:3, 3:] = _I3
        jacobian[3:, :3] = self._compute_d_acceleration_d_position(state, radial_distance_cb,
                                                                   position_sc_to_bodies,
                                                                   radial_distance_sc_to_bodies)

        covariance = state[6:].reshape(6, 6)

        if isinstance(self.process_noise, np.ndarray):
            pn = self.process_noise
        else:
            pn = self.process_noise(state, et_time)

        return jacobian@covariance + covariance@jacobian.T + pn

    def _compute_d_acceleration_d_position(self, state: np.ndarray, radial_distance_cb: float,
                                           position_sc_to_bodies: List[np.ndarray],
                                           radial_distance_sc_to_bodies: List[float]) -> np.ndarray:
        r"""
        The returns the jacobian of the acceleration with respect to the position vector as a 3x3 numpy array.

        .. math::
            \frac{\partial\mathbf{a}_G}{\partial\mathbf{x}_{pos}}=\mu_{cb}
            \left(\frac{3}{d_{cb2sc}**5}\mathbf{x}_{pos}\mathbf{x}_{pos}^T -
            \frac{\mathbf{I}_{3\times 3}}{d_{cb2sc}**3}\right) -
            \sum_{bod}\left(\mu_{bod}\left(\frac{3}{d_{sc2bod}**5}\mathbf{x_{pos,sc2bod}\mathbf{x_{pos,sc2bod}^T-
            \frac{\mathbf{I}_{3\times 3}}{d_{sc2bod})


        :param state: The state vector at the current time
        :param radial_distance_cb: The distance from the central body to the body we are estimating
        :param position_sc_to_bodies: The position from the spacecraft to the other bodies considered for gravity
        :param radial_distance_sc_to_bodies: The distance from the spacecraft to the other bodies
        :return: The jacobian of the acceleration with respect to the state
        """

        jac = self.gm_cb*(3*np.outer(state[:3], state[:3])/radial_distance_cb**5 -
                          _I3/radial_distance_cb**3)

        for gm, position_sc_to_bod, radial_distance_sc_to_bod in zip(self.gm_other_bodies,
                                                                     position_sc_to_bodies,
                                                                     radial_distance_sc_to_bodies):

            jac += gm*(3*np.outer(position_sc_to_bod, position_sc_to_bod)/(radial_distance_sc_to_bod**5) -
                       _I3/radial_distance_sc_to_bod**3)

        return jac

    def compute_dynamics(self, time: float, state: np.ndarray) -> np.ndarray:
        """
        This method computes the dynamics for the state vector

        The dynamics are returned as a 1d array of length 42.  It gives the time derivative of the state vector.

        The first 6 elements of the dynamics array are the position and velocity components of the state vector
        respectively.  The last 36 elements are the dynamics of the covariance matrix raveled in c-order.

        :param time: the time at which the dynamics are to be computed in ephemeris seconds since J2000
        :param state: The state vector to compute the dynamics for
        :return: The time derivative of the state vector
        """

        if self.minimum_time_step is not None:
            if self.previous_time is None:
                self.previous_time = time
            elif abs(time - self.previous_time) < self.minimum_time_step:
                raise ValueError('The time step is too small')

        dynamics,  (radial_distance_cb,
                    position_cb_to_bodies,
                    position_sc_to_bodies,
                    radial_distance_cb_to_bodies,
                    radial_distance_sc_to_bodies) = self.compute_state_dynamics(state, time, return_intermediaries=True)

        out_dynamics = np.concatenate(dynamics + [
            self.compute_covariance_dynamics(state, time, radial_distance_cb,
                                             position_sc_to_bodies, radial_distance_sc_to_bodies).ravel()
        ])

        if np.isnan(out_dynamics).any():
            raise ValueError('NaN in Dynamics')

        return out_dynamics


class SolRadAndGravityDynamics(SpiceGravityDynamics):
    r"""
    This class adds spherical solar radiation pressure dynamics to the :class:`.SpiceGravityDynamics` class.

    Everything is the same except the solar radiation pressure is added to the :attr:`State` vector and the
    dynamics for the solar radiation pressure are added to the appropriate methods.

    The solar radiation pressure is modelled as a cannonball model

    .. math::
        \mathbf{a}_{sr}=\frac{C_rA\Phi}{cmd_{sun}**2}\hat{\mathbf{s}}

    where :math:`C_r` is the radiation pressure coefficient, :math:`A` is the cross sectional area in meters
    squared, :math:`\Phi` the the solar constant at 1 AU in kW/m**2, :math:`c` is the speed of light in m/s**2,
    :math:`m` is the mass of the spacecraft in kg, :math:`d_{sun}` is the distance from the sun in AU, and
    :math:`\hat{\mathbf{s}}` is the unit vector from the sun to the spacecraft.

    Because this is not intended to be a high fidelity model we combine :math:`\frac{C_rA}{m} into a single parameter
    ``cram`` that is estimated in the filter.  If you want to back out one of the parameters from this estimated value,
    you must hold 2 of them fixed and then perform the arithmetic to get your answer.

    For more details on using this class, see the :class:`.SpiceGravityDynamics` documentation
    """

    @dataclass
    class State(SpiceGravityDynamics.State):
        """
        This extends the default State class to also contain the cram parameter for solar radiation pressure.

        See the default :class:`.Dynamics.State` class documentation for details.
        """

        cram: float = 1.0
        r"""
        The estimated portion of the solar radiation pressure model.
        
        This is equivalent to
        
        .. math::
            cram=\frac{C_rA}{m}
        """

        length: ClassVar[int] = 7
        """
        The length of the state vector for this representation
        """

        @property
        def vector(self) -> List[np.ndarray]:
            """
            This property returns a list of arrays that can be concatenated together to form the state vector.

            Therefore, you can do ``np.concatenate(state_instance.vector)`` to get a 1D array of the state vector.

            This is done so that subclasses can override the order if they so choose.

            :return: a list of numpy arrays
            """

            out = super().vector

            # add in the cram before the covariance
            out.insert(2, np.array([self.cram], dtype=np.float64))

            return out

        @vector.setter
        def vector(self, val: np.ndarray):

            self.position = val[:3]
            self.velocity = val[3:6]
            self.cram = val[6]
            self.covariance = val[7:].reshape(7, 7)

        def update_state(self, state_update: np.ndarray):
            """
            Perform an additive update to ``self``.

            This method updates ``self`` by adding the input update to ``self``.

            This is used inside of the :class:`.ExtendedKalmanFilter` to apply the update to the state vector from a
            processed measurement.  The ``state_update`` input will be a 1d numpy array the same length as the state
            vector (``len(self)``).

            :param state_update: The update vector to add to the state
            """

            self.position += state_update[:3]

            self.velocity += state_update[3:6]

            self.cram += state_update[6]


    def __init__(self, center_body: str, process_noise: PN_TYPE,
                 other_bodies: Optional[List[str]] = None,
                 gravity_parameters: Optional[Dict[str, float]] = None, load_data: bool = False,
                 minimum_time_step: Optional[float] = 0.001):
        """
        :param center_body: The center of integration (and the center of the integration frame)
        :param process_noise: The process noise either as a numpy array of shape 6x6 (constant process noise) or as a
                              callable object which takes in the current state and time (np.ndarray, float) and outputs
                              a 6x6 numpy array containing the process noise matrix
        :param other_bodies: Other bodies whose gravity should be included as a list of body/barycenter names.  If
                             ``None`` no other bodies will be considered.
        :param gravity_parameters: A dictionary mapping names to GM values.  If this is ``None`` or it does not provide
                                   the data for one of the bodies being considered we will try to query this from spice.
                                   The values should be in km**3/sec**2
        :param load_data: A boolean flag specifying whether this class should furnsh the required datafiles.  If you
                          have already loaded files then you should leave this as ``False``.
        :param minimum_time_step: The minimum time step to allow the integrator to take in seconds.  If ``None`` (or 0)
                                  then no minimum time step is enforced.
        """

        super().__init__(center_body, process_noise, other_bodies=other_bodies, gravity_parameters=gravity_parameters,
                         load_data=load_data, minimum_time_step=minimum_time_step)

        self.speed_of_light: float = 2.99792458e8  # m/s
        """
        The speed of light in meters per second
        """

        self.solar_constant: float = 1360.8  # kW/m**2
        """
        The solar constant in kW/m**2
        """

        self.km_to_au = 1/149597870.7  # 1AU/km
        """
        The conversion from kilometers to AU
        """

    def compute_state_dynamics(self, state: np.ndarray, et_time: float,
                               return_intermediaries: bool = False) -> Union[List[np.ndarray],
                                                                             Tuple[List[np.ndarray],
                                                                                   Tuple[float,
                                                                                         List[np.ndarray],
                                                                                         List[np.ndarray],
                                                                                         List[float],
                                                                                         List[float],
                                                                                         np.ndarray,
                                                                                         float]]]:
        """
        This computes the dynamics for just the "state" part of the state vector (not the covariance)

        Optionally this can also return the distance from the CB to the spacecraft, the vectors from the central body to
        the other bodies, the vectors from the spacecraft to the other bodies, the distances from the central body to
        the other bodies, and the distances from the other bodies to the spacecraft if ``return_intermediaries`` is
        ``True``.

        The first component of the return is always a list of the dynamics for the state vector in order of position,
        velocity, cram.

        :param state: The state vector at the current time
        :param et_time: The ephemeris time
        :param return_intermediaries: A flag specifying whether to return the intermediate distances/vectors for use
                                      elsewhere
        :return: A list containing [dposition/dt, dvelocity/dt, dcram/dt].  Optionally return a second tuple containing
                 (radial_distance_cb, position_cp_to_bodies, position_sc_to_bodies, radial_distance_cb_to_bodies,
                 radial_distance_sc_to_bodies, sun_to_sc_position, sun_to_sc_distance)

        """
        dynamics, intermediaries = super().compute_state_dynamics(state, et_time,
                                                                  return_intermediaries=return_intermediaries)

        others = [o.lower() for o in self.other_bodies]

        if 'sun' in others:
            location = others.index('sun')

            sun_direction = intermediaries[2][location].copy()
            sun_distance = intermediaries[3][location]
        else:
            sun_direction = state[:3] - spice.spkpos('sun', et_time, 'J2000', 'LT+S', self.center_body)[0]
            sun_distance = np.linalg.norm(sun_direction)

        sun_direction /= sun_distance
        sun_distance *= self.km_to_au

        dynamics[1] += self.compute_solar_radiation_acceleration(state, sun_direction, sun_distance)

        dynamics.append(np.array([0.0]))

        return dynamics, intermediaries + (sun_direction, sun_distance)

    def compute_solar_radiation_acceleration(self, state: np.ndarray,
                                             direction_sun_to_sc: np.ndarray,
                                             distance_sun_to_sc: float):
        """
        This computes the acceleration due to the solar radiation pressure on the spacecraft assuming a cannonball model
        in km/s**2.

        :param state:  The state vector
        :param direction_sun_to_sc: The position vector from the sun to the spacecraft
        :param distance_sun_to_sc: The distance from the sun to the spacecraft in AU
        :return: The solar radiation acceleration in km/s**2 as a numpy array
        """

        return (state[6]*self.solar_constant/(distance_sun_to_sc**2)*direction_sun_to_sc/self.speed_of_light)/1000

    # noinspection PyMethodOverriding
    def _compute_d_acceleration_d_position(self, state: np.ndarray, radial_distance_cb: float,
                                           position_sc_to_bodies: List[np.ndarray],
                                           radial_distance_sc_to_bodies: List[float],
                                           direction_sun_to_sc: np.ndarray,
                                           distance_sun_to_sc: float) -> np.ndarray:
        r"""
        The returns the Jacobian of the acceleration with respect to the position vector as a 3x3 numpy array.

        .. math::
            \frac{\partial\mathbf{a}_G}{\partial\mathbf{x}_{pos}}=\mu_{cb}
            \left(\frac{3}{d_{cb2sc}**5}\mathbf{x}_{pos}\mathbf{x}_{pos}^T -
            \frac{\mathbf{I}_{3\times 3}}{d_{cb2sc}**3}\right) -
            \sum_{bod}\left(\mu_{bod}\left(\frac{3}{d_{sc2bod}**5}\mathbf{x_{pos,sc2bod}\mathbf{x_{pos,sc2bod}^T-
            \frac{\mathbf{I}_{3\times 3}}{d_{sc2bod})


        :param state: The state vector at the current time
        :param radial_distance_cb: The distance from the central body to the body we are estimating
        :param position_sc_to_bodies: The position from the spacecraft to the other bodies considered for gravity
        :param radial_distance_sc_to_bodies: The distance from the spacecraft to the other bodies
        :param direction_sun_to_sc: The unit direction vector from the sun to the spacecraft
        :param distance_sun_to_sc: The distance between the sun and the sc at the current time in units of AU
        :return: The jacobian of the acceleration with respect to the state
        """

        d_distance_sun_to_sc_2_d_r_cb_to_sc = 2*self.km_to_au*direction_sun_to_sc*distance_sun_to_sc

        d_direction_sun_to_sc_d_r_cp_to_sc = (_I3/distance_sun_to_sc -
                                              np.outer(direction_sun_to_sc, direction_sun_to_sc*distance_sun_to_sc) /
                                              distance_sun_to_sc)

        dasr_dpos = state[6]*self.solar_constant/(self.speed_of_light*1000)*(
                d_direction_sun_to_sc_d_r_cp_to_sc/distance_sun_to_sc**2 -
                np.outer(direction_sun_to_sc, d_distance_sun_to_sc_2_d_r_cb_to_sc)/distance_sun_to_sc**4
        )

        return super()._compute_d_acceleration_d_position(state, radial_distance_cb, position_sc_to_bodies,
                                                          radial_distance_sc_to_bodies) + dasr_dpos

    # noinspection PyMethodOverriding
    def compute_covariance_dynamics(self, state: np.ndarray, et_time: float, radial_distance_cb: float,
                                    position_sc_to_bodies: List[np.ndarray],
                                    radial_distance_sc_to_bodies: List[float],
                                    direction_sun_to_sc: np.ndarray,
                                    distance_sun_to_sc: float) -> np.ndarray:
        r"""
        This method computes the dynamics for the covariance matrix.

        The dynamics for the covariance matrix is the product of the Jacobian matrix of the dynamics for the state with
        respect to the state and the current covariance matrix, plus the process noise matrix

        .. math::
            \mathbf{\partial \mathbf{P}}{\partial t} = \mathbf{J}\mathbf{P} + \mathbf{P}\mathbf{J}^T + \mathbf{Q}

        where :math:`\mathbf{P}` is the covariance matrix, :math:`t` is time,

        .. math::
            \mathbf{J}=\frac{\partial \mathbf{f}(\mathbf{x})}{\partial\mathbf{x}}

        with :math:`\mathbf{J}` being the Jacobian matrix, :math:`\mathbf{f}(\mathbf{x})` is the state dynamics
        function, and :math:`\mathbf{x}` is the state vector.

        :param state: The state vector at the current time
        :param et_time: The ephemeris time
        :param radial_distance_cb: The distance from the central body to the body we are estimating
        :param position_sc_to_bodies: The position from the spacecraft to the other bodies considered for gravity
        :param radial_distance_sc_to_bodies: The distance from the spacecraft to the other bodies
        :param direction_sun_to_sc: The unit direction vector from the sun to the spacecraft
        :param distance_sun_to_sc: The distance between the sun and the sc at the current time in units of AU
        :return: The covariance time derivative.
        """

        jacobian = np.zeros((7, 7), dtype=np.float64)

        jacobian[:3, 3:6] = _I3
        jacobian[3:6, :3] = self._compute_d_acceleration_d_position(state, radial_distance_cb,
                                                                    position_sc_to_bodies,
                                                                    radial_distance_sc_to_bodies,
                                                                    direction_sun_to_sc, distance_sun_to_sc)

        jacobian[3:6, -1] = self.solar_constant*direction_sun_to_sc/distance_sun_to_sc**2/self.speed_of_light/10000

        covariance = state[7:].reshape(7, 7)

        if isinstance(self.process_noise, np.ndarray):
            pn = self.process_noise
        else:
            pn = self.process_noise(state, et_time)

        return jacobian @ covariance + covariance @ jacobian.T + pn

    def compute_dynamics(self, time: float, state: np.ndarray) -> np.ndarray:
        """
        This method computes the dynamics for the state vector

        The dynamics are returned as a 1d array of length 56.  It gives the time derivative of the state vector.

        The first 6 elements of the dynamics array are the position and velocity components of the state vector
        respectively.  The next element is cram. The last 49 elements are the dynamics of the covariance matrix
        raveled in c-order.

        :param time: the time at which the dynamics are to be computed in ephemeris seconds since J2000
        :param state: The state vector to compute the dynamics for
        :return: The time derivative of the state vector
        """

        if self.minimum_time_step is not None:
            if self.previous_time is None:
                self.previous_time = time
            elif 0 < abs(time - self.previous_time) < self.minimum_time_step:
                raise ValueError('The time step is too small')

        dynamics,  (radial_distance_cb,
                    position_cb_to_bodies,
                    position_sc_to_bodies,
                    radial_distance_cb_to_bodies,
                    radial_distance_sc_to_bodies,
                    sun_direction, sun_distance) = self.compute_state_dynamics(state, time, return_intermediaries=True)

        out_dynamics = np.concatenate(dynamics + [
            self.compute_covariance_dynamics(state, time, radial_distance_cb,
                                             position_sc_to_bodies, radial_distance_sc_to_bodies,
                                             sun_direction, sun_distance).ravel()])

        if np.isnan(out_dynamics).any():
            raise ValueError('NaN in Dynamics')

        return out_dynamics


# TODO: Provide a monte dynamics interface at some point...
