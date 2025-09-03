


r"""
This module provides a class for doing observation planning of a target given details about a camera, as well as a
helper class for specifying discretized trajectories.

Description
-----------
By observation planning, we refer to 3 interrelated processes with respect to determining when an object will be visible
in a monocular camera.

1. Determining first acquisition (the distance/time at which a specified snr can be reached for a given phase angle or
   trajectory and exposure length)
2. Determining exposure length (the exposure length required to achieve a specified snr for a given distance/phase angle
   or trajectory)
3. Determining SNR (the SNR value expected to be achieved for a given exposure length and distance/phase angle or
   trajectory)

These 3 capabilities are crucial for planning the capture of OpNav images.  They do however require more information
about the electronics of the camera as well as the properties of the target being observed beyond what is required for
typical OpNav scenarios. 

Use
---
To do observation planning in GIANT, create an instance of the :class:`.ObservationPlanner` with the appropriate data.
Once initialized, you can use methods :meth:`.predict_acquisition`, :meth:`.predict_exposure`, and :meth:`.predict_snr`
to cover most of your needs.  For all 3 methods, you can either specify the geometry of the observations, or you
can specify a discretized trajectory to do the calculations over, using the :class:`.DiscretizedPhotometryTrajectory` class.
"""

import numpy as np
import warnings
from copy import copy
from datetime import datetime, timedelta
from scipy.optimize import least_squares, OptimizeResult
from typing import Callable, Tuple, Union, Optional, cast, Iterator

# GIANT IMPORTS
from giant.image import OpNavImage
from giant.ray_tracer.scene import Scene
from giant.rotations import Rotation
from giant.rotations.frames import two_vector_frame
# PHOTOMETRY IMPORTS
from giant.photometry.magnitude import PhaseMagnitudeModel
from giant.photometry.photometry import Photometry


class DiscretizedTrajectory:
    """
    This class is used to setup a span in which to query the photometry model over. When iterating over 
    this class, the photometry model is placed in the correct geometry within the camera frame and points
    to the defined target for each time step. 
    
    This class should only be used to step through a photometry model by time.
    
    If a target_index value is not provided by the user, then the camera will point to the first target 
    defined in the photometry model's scene. 
    """

    def __init__(self, scene: Scene,
                 camera_position_function: Callable,
                 start_bound: datetime, stop_bound: datetime, step_size: timedelta,
                 target_index: Optional[int] = None):
        """
        :param scene: A Scene in which to find trajectory parameters
        :param camera_position_function: a function that returns the position of the camera 
            for an input datetime
        :param start_bound: A datetime object defining the beginning of the window over which the calculations are to be
                            done (inclusive)
        :param stop_bound: A datetime object defining the end of the window over which the calculations are to be
                            done (inclusive)
        :param step_size: A timedelta object specifying how to discretize the trajectory.
        :param target_index: The index representing the body to point to during the trajectory in the scene objects
                            defined in the photometry model
        """
        self.scene: Scene = scene
        """
        A Photometry object containing the scene in which to calculate photometry
        """
        self.camera_position_function = camera_position_function
        """
        a function that returns the position of the camera for an input datetime
        """

        self.target_index: int = 0
        self.target_index = target_index if target_index is not None else 0
        """
        The index representing the body to point to during the trajectory in the scene objects
        defined in the photometry model
        """

        self.start_bound: datetime = start_bound
        """
        A datetime object defining the beginning of the window over which the calculations are to be done (inclusive)
        """

        self.stop_bound: datetime = stop_bound
        """
        A datetime object defining the end of the window over which the calculations are to be done (inclusive)
        """

        self.step_size: timedelta = step_size
        """
        A timedelta object defining the size of the steps to take when discretizing the trajectory
        """

    def _point_camera_at_body(self, date: datetime) -> Rotation:
        """
        Return the rotation neccesary to point the camera directly at the target
        """

        camera_to_body = self.scene.target_objs[self.target_index].position - self.camera_position_function(date)
        z = camera_to_body.ravel() / np.linalg.norm(camera_to_body)  # unit vector camera to body
        x_const = np.array([1, 0, 0])  # xdir of intertial frame
        return cast(Rotation, two_vector_frame(z, x_const, 'z', 'x', True))

    def _place_in_camera_frame(self, date: datetime) -> None:
        """
        Use the Scene.update() function to place the scene in the camera frame based on a blank image
        
        :param date: the date at which to calculate things
        """

        # create a blank image to place the scene in the camera frame
        dummy_image = OpNavImage(
            data=[],
            observation_date=date,
            position=self.camera_position_function(date),
            rotation_inertial_to_camera=self._point_camera_at_body(date),
        )
        self.scene.update(dummy_image)

    def place_and_point_to_body(self, date: datetime) -> None:
        r"""
        This method computes places the scene at the input date, changes the origin of the scene to the camera, 
        and points the camera directly at the target at ``target_index``. 

        This method requires that the scene contains at least one target object and the position functions defining each
        object in the scene has the same observer.  
        
        :param date: the datetime in which to place the scene
        """
        self._place_in_camera_frame(date)

    def __iter__(self) -> Iterator[datetime]:
        """
        Yields the time and places the Photomerty model at the particular time pointed to the target
        """

        current = copy(self.start_bound)
        # determine if the current time is before th end time and place the photometry model then
        while current < self.stop_bound:
            self.place_and_point_to_body(current)
            yield current
            # set the next time at the next timestep
            current += self.step_size

        # now yield for the stop bound and place the photometric model there
        self.place_and_point_to_body(self.stop_bound)
        yield self.stop_bound


class ObservationPlanner:
    """
    This class implements observation planning functionality for GIANT.

    This class acts as a container for details about the target/camera that are not normally used for navigation
    purposes.  It then exposes the ability to use this extra information for OpNav planning purposes.

    This class contains and Photometry model and DiscetizedPhotometricTrajectory in order to execute
    planning operations on a target body over the trajectory.
    
    The magnitude model and luminocity function are used to determine the size and brightness of the target 
    as seen by the camera.
    
    * :meth:`predict_acquisition` - determine the minimum time over a DiscretizedTrajectory in which an SNR value is 
                          acquired.
    * :meth:`predict_exposure` - Determine the required exposure time needed to achieve an SNR value over each step of
                          a DiscretizedTrajectory.
    * :meth:`predict_snr` - Determine the SNR of an image taken at each step in a DiscretizedTrajectory.

    The following shows an example use of the ObservationPlanner type.  First, let's assume you already have a SceneObject set
    up for the target body (target_obj) and the sun (sun_obj) as well as a GIANT CameraModel (camera_model). After importing the 
    planning and photometry packages, we'll want to define our objects within a Scene.
    :
        >>> from giant.photometry.planning import ObservationPlanner, DiscretizedTrajectory
        >>> from giant.photometry.modelling import PhotometricCameraModel,scatteredLight, au, Photometry
        >>> from giant.photometry.magnitude import LinearPhaseMagnitudeModel
        >>> import numpy
        >>> from datetime import timedelta, datetime
        >>> target_scene = Scene(target_objs=[target_obj], light_obj=sun_obj)
        
    Next, lets set up the photometric camera model with the proper inputs. This example uses a LLORRI camera. 
    
        >>> LORRI = PhotometricCameraModel(gain = 21.1, #electrons/dn, transfer_time = 0.01178, 
                                standard_mag = 18.97, bin_mode = 1, resolved_rows_threshold = 300,
                                name='lorri',dn_readnoise = 0.88, dn_rate_standard = 15816137,
                                dark_current=0.0003,psf_factor=0.1, camera_model = camera_model)
                                
    Next, we will set up a Photometry object with the Scene and PhotomertricCameraModel. 
    
        >>> photometry = Photometry(scene=targetdinkinesh_scene, 
                        photometric_camera_model=LORRI))
                        
    Then, we will set up the trajectory as a DiscretizedTrajectory. This will place the scene in the camera frame for 
    each time step specified. This will require a camera position function in order to place the scene in the correct frame.
    Note that it is assumed that the camera_position_function will return a position relative to the same central body used in
    the position_function for the target_obj. For demo purposes, our function will return a constant, but this is typically defined
    by spice routines.
    
        >>> def camera_position_function(date):
        ... return numpy.array([-5, 6, 7])
        >>> trajectory = DiscretizedTrajectory(scene=target_scene, camera_position_function, 
        ... start_bound = datetime(2025, 2, 16), stop_bound=datetime(2026, 2, 16), step_size=timedelta(hours=24))
        
    Now, we need to set up a magnitude and luminocity function used to determine the magnitude and I over F of the target based
    on the phase angle. Again, for simplicity of the demo, these will be simple functions and models. We will use a different magnitude 
    models for when the target is resolved and unresolved.
    
        *** Note: the magnitude models must be set up as a PhaseMagnitudeModel object, a LinearPhaseMagnitudeModel and 
        HGPhaseMagnitudeModel objects are already set up in giant. However, the user can create their own. 
    
        >>> def target_IoverF(target_index, photometry):
        ... phase_angle = photometry.scene.target_objs[target_index]
        ... return 1*np.cos(phase_angle)
        >>> resolved_mag_model = LinearPhaseMagnitudeModel(phase_slope=0.4)
        >>> unresolved_mag_model = LinearPhaseMagmitudeModel(phase_slope=0.8)
        
    Finally, we can place all these objects into an ObservationPlanner. 

        >>> planner = ObservationPlanner(photometry_model = photometry, trajectory = trajectory, luminosity_function = LumPhaseTable,
            magnitude_model = {'unresolved': unresolved_mag_model,
                                'resolved': resolved_mag_model
                                },
        )
        
    Now that the planner is set up, you can use this to predict acquisition times, exposure times, and SNR. 
    
        >>> aqu_time, aqu_dist = planner.predict_acquisition(goal_snr=10, exposure_time=5)
        >>> exp, snr = planner.predict_exposure(goal_snr=10, exposure_time_guess=5)
        >>> snr = planner.predict_snr(exposure_time=5)
        
    """

    def __init__(self, photometry_model: Photometry, trajectory: DiscretizedTrajectory,
                 magnitude_model: Union[dict, list, PhaseMagnitudeModel],
                 luminosity_function: Callable):
        """
        :param photometry_model: giant.photometry.Photometry object containing the scene to plan with
        :param trajectory: DiscretizedPhotometricTrajectory object containing the span in which to 
                            search the trajectory
        :param magnitude_model: A PhaseMagnitudeModel object that calculates the target magnitude based on the 
                            phase angle of the current trajectory
        :param luminosity_function: A function with positional arguments (target_index : int, photometry : Photometry)
                            to return the I over F of the target based on geometry and camera model
        """
        self.photometry_model = photometry_model
        """
        giant.photometry.Photometry object containing the scene to plan with
        search the trajectory
        """

        self.trajectory = trajectory
        """
        DiscretizedPhotometricTrajectory object containing the span in which to search the trajectory
        """

        self.luminosity_function = luminosity_function
        """
        A function with positional arguments (target_index : int, photometry : Photometry)
        to return the I over F of the target based on geometry and camera model
        """

        self.magnitude_model = magnitude_model
        """
        A PhaseMagnitudeModel object that calculates the target magnitude based on the 
        phase angle of the current trajectory
        """

    @property
    def magnitude_model(self):
        return self._magnitude_model

    @magnitude_model.setter
    def magnitude_model(self, val):
        '''
        magnitude_model can be set as as the following types:
            list [unresolved, resolved],
            dictionary {'unresolved','resolved'}
            PhaseMagnitudeModel (same model used for resolved and unresolved targets).
            
        It is not recommended to use the same magnitude model for resolved and unresolved targets.
        '''
        self._magnitude_model = val
        if isinstance(self._magnitude_model, PhaseMagnitudeModel):
            self._resolved_magnitude_model = self._magnitude_model
            self._unresolved_magnitude_model = self._magnitude_model
            warnings.warn(
                'Only one Magnitude Model provided, this will be used for both resolved and unresolved targets')
        elif isinstance(self._magnitude_model, dict):
            self._resolved_magnitude_model = self._magnitude_model['resolved']
            self._unresolved_magnitude_model = self._magnitude_model['unresolved']
        elif isinstance(self._magnitude_model, list):
            self._resolved_magnitude_model = self._magnitude_model[1]
            self._unresolved_magnitude_model = self._magnitude_model[0]
        else:
            warnings.warn('Attempting to set invalid Magnitude Model.')
            self._resolved_magnitude_model = None
            self._unresolved_magnitude_model = None

    def _target_mag(self) -> None:
        '''
        Calculate the apparent magnitude of the Target based on the magnitude model
        
        This parameter is saved within the photometry object
        '''
        if self.photometry_model.resolved(self.trajectory.target_index):
            assert self._resolved_magnitude_model is not None
            self.photometry_model.apparent_mag(self.trajectory.target_index, self._resolved_magnitude_model)
        else:
            assert self._unresolved_magnitude_model is not None
            self.photometry_model.apparent_mag(self.trajectory.target_index, self._unresolved_magnitude_model)

    def _target_lum(self) -> None:
        '''
        Calculate the luminosity of the Target based on the luminosity model
        
        This parameter is saved within the photometry object
        '''
        self.photometry_model.iof(self.trajectory.target_index, self.luminosity_function)

    def predict_acquisition(self, goal_snr: float, exposure_time: float) -> Union[
        Tuple[Optional[datetime], Optional[float]], float]:
        """
        Predicts the acquisition distance (and approximate time) for the desired SNR.

        :param desired_snr: The signal to noise ratio that you want to see as a float
        :param exposure_time: The length of the exposure in seconds
        :return: A tuple containing the time as a datetime and distance in km to the target when the SNR is met
        """

        # preallocate SNR to be improved
        best_snr = 0

        for idx, time in enumerate(self.trajectory):
            # determine the magnitude of the target at the current time
            self._target_mag()
            self._target_lum()
            snr = self.photometry_model.snr(self.trajectory.target_index, exposure_time)

            best_snr = max(best_snr, snr)

            # if we meet the constraints, return the time and the distance from the camera to the target
            if snr >= goal_snr:
                if idx == 0:
                    warnings.warn(
                        f'ObservationPlanner.predict_acquisition found that the target is acquired at the first step of the trajectory.'
                        f'This may not represent the true acquisition time of the target')
                return time, self.photometry_model.scene.target_objs[self.trajectory.target_index].distance

        # if we get here then the snr was never met.  Throw a warning and return None
        warnings.warn(f'Unable to meet the requested SNR constraint for the requested trajectory.'
                      f'Best SNR found was {best_snr}')

        return None, None

    def predict_exposure(self, goal_snr: float, exposure_time_guess: float = 1.0) -> Tuple[list[float | None], list[float | None]]:
        """
        Optimize the exposure time based on the required/goal signal-to-noise ratio

        param: goal_snr: The desired SNR to achieve
        param: exposure_time_guess: An initial guess for the exposure time in seconds for the optimizer.
        :returns: tuple containing an array of the optimized exposure times over the trajectory steps and an array of
                  the SNR that is calculated using that exposure time
        """

        def optimized_snr(x, goal):
            """
            This function is used to minimize the difference between the SNR from the model and the 
            requested SNR
            """
            return abs(self.photometry_model.snr(self.trajectory.target_index, x) - goal)

        # setup exp and snr as blank arrays to append to with each trajectory step
        exp: list[float | None] = []
        snr: list[float | None] = []

            
        assert self.photometry_model.photometric_camera_model is not None
        # step through the trajectory and get the exposure time needed for the requested SNR based
        # on the trajectory geometry
        for _ in self.trajectory:

            # calculate target magnitude as seen by the camera
            self._target_mag()
            self._target_lum()

            # set up a least-squares optimizer to minimize the result from optimize_snr
            optimize: OptimizeResult = least_squares(optimized_snr, exposure_time_guess, bounds=(0, self.photometry_model.photometric_camera_model.max_exposure), verbose=0, args=(goal_snr,))

            if optimize.success:

                # save the exposure time and snr if the optimizer successfully found a solution
                exp.append(optimize.x[-1])
                snr.append(self.photometry_model.snr(self.trajectory.target_index, optimize.x[-1]))

            else:

                # print the error message and do not save the exposure time and snr at this step if the
                # optimizer failed to find a solution
                print(optimize.message)
                exp.append(None)
                snr.append(None)

        return exp, snr

    def predict_snr(self, exposure_time: float) -> np.ndarray:
        """
        Calculate the SNR for each step along the trajectory with a provided exposure time
        
        :param exposure_time: the exposure time in seconds to use
        """

        # setup snr as blank array to append to with each trajectory step
        snr = []

        # calculate the snr at each step in the trajectory for a static exposure time
        for _ in self.trajectory:
            self._target_mag()
            self._target_lum()
            snr.append(self.photometry_model.snr(self.trajectory.target_index, exposure_time))
        return np.array(snr)
