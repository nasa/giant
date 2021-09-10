# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
This module provides a working example state initializer callable for use with the :class:`.ExtendedKalmanFilter`.

Description
-----------

A state initializer is a Callable object which takes in the initial :class:`.Measurement` instance and the
:class:`.Dynamics.State` class object and returns an initialized :class:`.Dynamics.State` with position, velocity,
covariance and other state parameters filled in.  This callable can either be a function or a class which implements the
``__call__`` method, although it is more typically implemented as a class to enable easy changing of parameters.

Use
---

The process of initializing the state from a measurement will likely be different in every scenario, so the provided
initializer really just serves as an example (taken from the OSIRIS-REx particle tracking code) for building your own.
If your case is sufficiently similar to the OSIRIS-REx case then you can use this initializer directly (it is fully
functional) by supplying an instance of it to the :class:`.ExtendedKalmanFilter` class.
"""


from dataclasses import dataclass

from typing import Optional, Type

import numpy as np

from giant.ray_tracer.scene import Scene
from giant.ufo.dynamics import Dynamics
from giant.ufo.measurements import OpticalBearingMeasurement, Measurement

from giant._typing import Real, SCALAR_OR_ARRAY



@dataclass
class ORExInitializer:

    scene: Scene
    """
    The :class:`.Scene` instance that has the central body of the integrator as the first "target".
     
    The first target need not have any geometry associated with it (in fact it is better if it doesn't for speed/memory 
    reasons)
    """

    initial_range: Optional[Real] = None
    """
    The initial range to the target as a number or ``None``.
    
    If this is ``None``, then the initial range to the target will be the same as the range from the camera to the 
    central body assumed to be the first target in the :attr:`scene` attribute.
    """

    initial_cram: Real = 1e-3
    """
    The initial CrAm value for the filter in m**2/kg.  
    
    See the :class:`.SolRadAndGravityDynamics` class for details.
    """

    range_variance: Real = 49.0
    """
    The initial range variance in km**2 for computing the initial state covariance.  
    
    This is used along with the measurement covariance to compute the initial position covariance matrix.
    """

    measurement_covariance_multiplier: Real = 9.0
    """
    The multiplier to use on the measurement covariance when computing the state covariance.  
    
    Because this is applied to the covariance it is essentially a sigma**2 multiplier (so a value of 9 is saying use the 
    3 sigma measurement uncertainty).
    """

    initial_velocity_variance: SCALAR_OR_ARRAY = 1e-6
    """
    The initial velocity variance in units of (km/s)**2
    
    This can be specified as a scalar, in which case it is assumed that all 3 axes are the specified value, a 1d array 
    of length 3 in which case the covariance matrix is formed with this as the main diagonal, or a 2d array of shape 3x3 
    in which case the covariance matrix is directly used.  Note that if you specify the full covariance matrix it must 
    be symmetric.
    """

    initial_cram_variance: Real = 1.0
    """
    The initial CrAm variance in units of (m**2/kg)**2.
    """

    minimum_initial_radius: Real = 0.250
    """
    The minimum initial radius to the central body in km.  
    
    If the initial determined particle location is less than this the initial range will be decreased until 
    this condition is met.
    """

    def __call__(self, measurement: Measurement, state_type: Type[Dynamics.State]) -> Dynamics.State:
        """
        Implement the initializer function.

        This is only valid for :class:`.OpticalBearingMeasurement` measurements.  It may work with others through duck
        typing but not likely.

        :param measurement: The bearing measurement we are using to initialize the state
        :param state_type: The type of the state we are initializing
        :return: The initialized state
        """

        if not isinstance(measurement, OpticalBearingMeasurement):
            raise ValueError('Currently only OpticalBearingMeasurement measurements are accepted')

        # retrieve the image form the measurement
        image = measurement.image

        # update the scene to be at this image time
        self.scene.update(image)

        # determine what the initial range should be
        if self.initial_range is None:
            # if it wasn't specified then use the distance from the spacecraft to the central body as an initial guess
            initial_range = np.linalg.norm(self.scene.target_objs[0].position)
        else:
            # otherwise use the user specified value
            initial_range = self.initial_range

        # get the initial position of the target in the camera fixed/centered frame
        los_camera = measurement.camera_model.pixels_to_unit(measurement.observed, temperature=image.temperature)

        position_camera = initial_range * los_camera

        # rotate and translate into the inertial frame centered ont he central body
        position_inertial = image.rotation_inertial_to_camera.matrix.T @ (position_camera -
                                                                          self.scene.target_objs[0].position)

        cb_distance = np.linalg.norm(position_inertial)

        while cb_distance < self.minimum_initial_radius:

            # shrink the initial range
            initial_range *= 0.9

            position_camera = initial_range*los_camera

            # rotate and translate into the inertial frame centered ont he central body
            position_inertial = image.rotation_inertial_to_camera.matrix.T@(position_camera -
                                                                            self.scene.target_objs[0].position)

            cb_distance = np.linalg.norm(position_inertial)

        # initialize the covariance matrix
        covariance = np.zeros((state_type.length, state_type.length), dtype=np.float64)

        # determine the covariance for the line-of-sight vector in the camera frame
        jacobian_los_pix = measurement.camera_model.compute_unit_vector_jacobian(
            measurement.observed.reshape(2, 1), temperature=image.temperature
        ).squeeze()

        covariance_los_camera = (jacobian_los_pix @
                                 measurement.covariance * self.measurement_covariance_multiplier @
                                 jacobian_los_pix.T)

        # determine the covariance of the position vector in the camera frame
        covariance_position_camera = (self.range_variance*np.outer(los_camera, los_camera) +
                                      initial_range**2 * covariance_los_camera)

        # get the position covariance in the inertial frame and store it into the covariance matrix
        covariance[:3, :3] = (image.rotation_inertial_to_camera.matrix.T @
                              covariance_position_camera @
                              image.rotation_inertial_to_camera.matrix)

        # determine the initial velocity covariance
        if np.isscalar(self.initial_velocity_variance):
            covariance[3:6, 3:6] = np.diag([self.initial_velocity_variance]*3)
        elif self.initial_velocity_variance.size == 3:
            covariance[3:6, 3:6] = np.diag(self.initial_velocity_variance)
        else:
            covariance[3:6, 3:6] = self.initial_velocity_variance

        # determine if this state uses solar radiation and treat it differently if it doesn't
        if hasattr(state_type, 'cram'):

            covariance[6, 6] = self.initial_cram_variance

            # noinspection PyArgumentList
            state = state_type(image.observation_date, position_inertial, velocity=np.zeros(3, dtype=np.float64),
                               covariance=covariance, cram=self.initial_cram)

        else:

            # noinspection PyArgumentList
            state = state_type(image.observation_date, position_inertial, velocity=np.zeros(3, dtype=np.float64),
                               covariance=covariance)

        return state
