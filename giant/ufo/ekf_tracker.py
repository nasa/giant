# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
This module provides a class for tracking UFO detections across monocular images.

Description
-----------

UFO tracking is done using Extended Kalman Filters to predict the locations of UFOs in subsequent images and then match
UFO detections in those images to the predict locations.  Multiple paths are followed for each possible particle
resulting in many possible tracks across all of the images.  Each of these possible tracks is then filtered to remove
extraneous tracks (and tracks that are subsets of other tracks) leaving only the tracks we are relatively confident in
the quality of.  Further description of the process can be found in the paper at
https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2019EA000843

Use
---

To track UFOs from image to image, initialize the :class:`.Tracker` class with the required information and then call
method :meth:`.Tracker.track`.  This will go through the whole process of initial tracking and then filtering for all
the possible particles in the loaded data, storing the results in the :attr:`.confirmed_tracks` and
:attr:`.confirmed_standard_deviations` lists.  You can also save the results to a csv file using method
:meth:`~.Tracker.save_results`.

You may also be interested in using the :class:`.UFO` class which combines both detection and tracking instead of using
this class directly.
"""

import logging

from copy import deepcopy, copy

from multiprocessing import Process, cpu_count, Array

from tempfile import TemporaryFile

from time import time, sleep

from pathlib import Path

from queue import Empty, Full

from datetime import timedelta

# no real risk hear because the pickle files are created by this script itself
import pickle  # nosec

from typing import List, Optional, Callable, Union, Tuple, Hashable, Iterable

from scipy.spatial.ckdtree import cKDTree
import numpy as np

from giant.image import OpNavImage
from giant.ray_tracer.scene import Scene
from giant.camera import Camera
from giant.ufo.extended_kalman_filter import ExtendedKalmanFilter, STATE_INITIALIZER_TYPE
from giant.ufo.measurements import OpticalBearingMeasurement
from giant.ufo.dynamics import Dynamics
from giant.ufo.clearable_queue import ClearableQueue

from giant._typing import Real, PATH


_LOGGER: logging.Logger = logging.getLogger(__name__)
"""
The logger to use to report status/errors/warning/results/etc
"""


_ZERO_TIMEDELTA: timedelta = timedelta()
"""
The 0 timedelta constant
"""


def _pickle_generator(file: TemporaryFile) -> Iterable[ExtendedKalmanFilter]:
    """
    Simple generator to work through the pickled objects in a file
    :param file: the file object to work on
    """

    while True:
        try:
            # we are using pickle for swap essentially therefore there is no real risk
            yield pickle.load(file)  # nosec
        except EOFError:
            break


class Tracker:
    """
    This class provides an interface for autonomously tracking UFOs through subsequent images in time.

    There are 2 main components to this tracker.  The first is the use of EKFs to follow most of the possible paths
    forward from a single starting particle in an image.  This can result in hundreds of thousands (if not millions) of
    possible tracks for a set of images with fairly dense UFO detections.  The second component is then the filtering of
    these EKFs, which is done based on length (the number of measurements included in the EKF), post-fit residual
    statistics, and uniqueness (each starting particle only gets the best track assigned to it, and once a particle has
    been assigned to a track it can't be assigned to others).  This filtering process normally brings the number of
    tracks down to a much more manageable 10s to 100s.

    Explicit details on how this tracker works are provided in the paper at
    https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2019EA000843 and are not repeated here.

    To use this class, provide the (numerous) initialization values (the defaults will be fine for many cases) and then
    call method :meth:`track`.  The results will then be stored in the :attr:`confirmed_filters`,
    :attr:`confirmed_particles`, and :attr:`confirmed_standard_deviations` attributes.  In general, you may not directly
    interact with this class however, and instead will use the :class:`UFO` interface class to both detect and track
    particles.
    """

    def __init__(self, camera: Camera, scene: Scene,
                 dynamics: Dynamics,
                 state_initializer: STATE_INITIALIZER_TYPE,
                 search_distance_function: Callable[[ExtendedKalmanFilter], Real],
                 observation_trees: Optional[List[Optional[cKDTree]]] = None,
                 observation_ids: Optional[List[Optional[List[int]]]] = None,
                 initial_euclidean_threshold: Real = 300,
                 measurement_covariance: Optional[np.ndarray] = None,
                 maximum_image_timedelta: timedelta = timedelta(hours=1),
                 maximum_paths_per_image: int = 10,
                 maximum_paths_total: int = 15,
                 maximum_forward_images: int = 2,
                 maximum_track_length: int = 15,
                 maximum_mahalanobis_distance_squared: float = 25,
                 expected_convergence_number: int = 4,
                 reduced_paths_forward_per_image: int = 3,
                 minimum_number_of_measurements: int = 4,
                 maximum_residual_standard_deviation: Real = 20,
                 maximum_time_outs: int = 50,
                 maximum_tracking_time_per_image: Real = 3600):
        """
        :param camera: The :class:`.Camera` containing the images to process and the camera model
        :param scene: The :class:`.Scene` describing the location of the central body with respect to the camera.
        :param dynamics: The dynamics model to use in the EKF for propagating the state from one time to another
        :param state_initializer: A callable which takes in a :class:`.Measurement` instance and
                                  :class:`.Dynamics.State` class object and returns an initialized state.
        :param search_distance_function: A callable which takes in an :class:`.ExtendedKalmanFilter` and returns what
                                         the Euclidean search distance should be for that EKF in pixels.  This is only
                                         applied after the first pair has been made
        :param observation_trees: A list of scipy.spatial cKDTree objects that are built on the pixel locations of the
                                  detections for each image
        :param observation_ids: A list of the ids for each observation contained in the trees in order.
        :param initial_euclidean_threshold: The threshold in pixels for points to be paired form the first image to a
                                            subsequent image.  This is not applied after the first pair has been made.
        :param measurement_covariance: The covariance matrix of the measurements as a 2x2 array.  This should have units
                                       of pixels squared.
        :param maximum_image_timedelta: The maximum time separation between images to attempt either initial or
                                        subsequent pairings as a ``timedelta`` object
        :param maximum_paths_per_image: The maximum number of forward paths for a single particle to the next image to
                                        consider
        :param maximum_paths_total: The total maximum number of forward paths for a single image across all images to
                                    consider
        :param maximum_forward_images: The maximum number of images to retrieve potential pairs from (images will be
                                       processed in time order)
        :param maximum_track_length: The maximum length of a track before it is artificially terminated.  The length of
                                     a track is defined as the number of measurements that it has ingested
        :param maximum_mahalanobis_distance_squared: The maximum squared Mahalanobis distance for subsequent (not
                                                     initial) observations to be paired to a track.
        :param expected_convergence_number: The number of ingested measurements at which point tracks are expected to be
                                            fairly converged.  At this point, more stringent filtering is used in
                                            determining forward paths.
        :param reduced_paths_forward_per_image: The number of paths forward per image that are considered once a track
                                                is considered converged.
        :param minimum_number_of_measurements: The minimum number of measurements in a track for it to be considered a
                                               potential good track.  Typically 4 should be the absolute minimum, but in
                                               some cases it may need to be even higher than that.
        :param maximum_residual_standard_deviation: The maximum standard deviation of the post-fit residuals in an EKF
                                                    for it to be considered a potentially good track.
        :param maximum_time_outs: The maximum number of time outs that can occur when attempting to retrieve tracking
                                  results from the children processes before we assume something has gone wrong and
                                  terminate all of the processes.
        :param maximum_tracking_time_per_image: The maximum amount of time to attempt tracking in the image before we
                                                assume something has gone wrong and terminate all of the processes.
        """

        self.camera: Camera = camera
        """
        The :class:`.Camera` containing the images to process and the camera model
        """

        self.scene: Scene = scene
        """
        The :class:`.Scene` describing the location of the central body with respect to the camera.
        
        The central body should be the first "target" in the scene.  It doesn't need to have a shape associated with it
        (it can be a ``Point`` object) and in fact is encouraged not to (to avoid large memory overhead)
        """

        self.observation_trees: List[Optional[cKDTree]] = observation_trees
        """
        A list of KDTrees built on the observed UFO locations.
        
        If any elements are ``None`` then it is assumed that no detections exist for that image.
        """

        self.observation_ids: List[Optional[List[Hashable]]] = observation_ids
        """
        A list of the ids for each observation contained in the trees.
        
        These should be unique ``Hashable`` objects like ints or strings, not mutable like lists/arrays/dictionaries.
        They should uniquely identify an observation.
        """

        self.processes: List[Process] = []
        """
        A List of Processes that are working
        """

        self.confirmed_particles = set()
        """
        A set of particles that have been assigned to a track already
        """

        self.dynamics: Dynamics = dynamics
        """
        The dynamics model to use to propagate the states from one time to another.
        """

        self.measurement_covariance: Optional[np.ndarray] = measurement_covariance
        """
        The 2x2 measurement covariance matrix (or ``None`` to use the identify matrix) with units of pixels squared.
        """

        self.state_initializer: STATE_INITIALIZER_TYPE = state_initializer
        """
        The state initializer callable to use to initialize the state vector given the initial measurement and the type
        of the state vector to be initialized
        """

        self.maximum_image_timedelta: timedelta = maximum_image_timedelta
        """
        The maximum separation in time between images for them to be paired subsequently 
        """

        self.initial_euclidean_threshold: float = float(initial_euclidean_threshold)
        """
        The threshold in pixels for points to be paired from the first image to a subsequent image
        """

        self.maximum_paths_per_image: int = maximum_paths_per_image
        """
        The maximum number of pairs from one image to the next for a single possible particle
        """

        self.maximum_paths_total: int = maximum_paths_total
        """
        The maximum number of pairs from one image to all images for a single particle
        """

        self.maximum_forward_images: int = maximum_forward_images
        """
        The maximum number of images to pair with
        """

        self.maximum_track_length: int = maximum_track_length
        """
        The maximum length of a track to consider before terminating.  
        
        the length of a track is defined as the number of measurements that it has ingested.
        
        This is necessary for memory management purposes as long tracks frequently have many variants that are being 
        tracked and can take up a lot of memory.
        """

        self.search_distance_function: Callable[[ExtendedKalmanFilter], Real] = search_distance_function
        """
        A function which returns what the search distance should be in pixels when trying to match the current EKF to 
        new images.
        """

        self.maximum_mahalanobis_distance_squared: float = maximum_mahalanobis_distance_squared
        """
        This specifies the maximum Mahalanobis distance squared for a potential detection to be paired to a track
        
        The Mahalanobis distance is roughly equivalent to the sigma normalized error between the predicted and observed 
        location.  Therefore a Mahalanobis distance squares of 25 roughly corresponds to only accepting pairings that 
        are within 5 sigma of the predicted location.
        """

        self.expected_convergence_number: int = expected_convergence_number
        """
        This specifies the number of measurements at which it is expected the filter will have mostly converged and we 
        should become more selective with which paths forward we follow.
        """

        self.reduced_paths_forward_per_image: int = reduced_paths_forward_per_image
        """
        This specifies the reduced number of paths forward we should follow once the filter should be converged.
        """

        self.minimum_number_of_measurements: int = minimum_number_of_measurements
        """
        The minimum number of measurements for an EKF to be considered a track.
        """

        self.maximum_residual_standard_deviation: float = float(maximum_residual_standard_deviation)
        """
        The maximum post-fit standard deviation of the residuals in an EKF for it to be considered a valid fit
        """

        self.maximum_time_outs: int = maximum_time_outs
        """
        The maximum number of time outs in a row before we stop trying to process an image
        """

        self.maximum_tracking_time_per_image: float = float(maximum_tracking_time_per_image)
        """
        The maximum amount of time to attempt tracking in an image in seconds.
        """

        self._ekfs_to_process: Optional[ClearableQueue] = None
        """
        A Queue used to specify what ekfs need to be processed
        """

        self._working: Optional[Array] = None
        """
        A shared array of boolean values specifying whether each processor is actively working
        """

        self._results: Optional[ClearableQueue] = None
        """
        A Queue used to communicate the results for the processes.
        """

        self._smoothing_input: Optional[ClearableQueue] = None

        self._smoothing_output: Optional[ClearableQueue] = None

        self._smoothing_working: Optional[Array] = None

        self.smoothing_processes: Optional[List[Process]] = None
        """
        A List of Processes that are working on smoothing
        """

        self.confirmed_filters: List[ExtendedKalmanFilter] = []
        """
        This list stores the confirmed EKFs
        """

        self.confirmed_standard_deviations: List[float] = []
        """
        This list stores the standard deviation of the post-fit residuals of the confirmed EKFs
        """

        self._n_respawns: int = 0
        """
        A counter for the number of times we have respawned our processes
        """

        self._n_smoothing_respawns: int = 0
        """
        A counter for the number of times we have respawned our smoothing processes
        """

    def find_initial_pairs(self, image_ind: int, image: OpNavImage):
        """
        This method finds the initial pairs for the input image.

        This is done by identifying all observations in subsequent images within a specified number of pixels of the
        first identification (corrected for the movement of the observer in the time period)

        :param image_ind: The index of the image we are identifying the initial pairs for
        :param image: The image that we are identifying the initial pairs for
        """

        if self._ekfs_to_process is None:
            _LOGGER.warning("The work queue is not initialized.  Initializing it")
            self._initialize_tracking_workers()

        _LOGGER.info(f'Processing image {image_ind}, {image.observation_date}')

        # get the detections and ids
        detections = self.observation_trees[image_ind].data if self.observation_trees[image_ind] is not None else None
        detection_ids = self.observation_ids[image_ind]

        if detections is None:
            _LOGGER.warning(f'No detections for {image_ind}, {image.observation_date}')

            return

        # initialize a list to store the ekfs in
        filters = []

        # update the scene to the current time
        self.scene.update(image)

        # get the camera position in the central body centered inertial frame
        camera_position = -image.rotation_inertial_to_camera.matrix.T @ self.scene.target_objs[0].position.ravel()

        # loop through and create the filter for each possible point in the current image
        for possible_id, possible_detection in zip(detection_ids, detections):
            if possible_id in self.confirmed_particles:
                continue

            camera_location = self.dynamics.State(image.observation_date, camera_position)
            initial_measurement = OpticalBearingMeasurement(possible_detection, self.camera.model, image,
                                                            camera_location, self.measurement_covariance,
                                                            identity=possible_id)
            filters.append(ExtendedKalmanFilter(copy(self.dynamics), self.state_initializer,
                                                initial_measurement=initial_measurement))

        number_of_linked_images = 0

        for next_image_index, next_image in self.camera:
            if next_image_index <= image_ind:
                # don't want to go backwards
                continue

            if (next_image.observation_date - image.observation_date) >= self.maximum_image_timedelta:
                # if this image too much later then break because we're done processing
                break

            _LOGGER.info(f'linking image {image_ind}, {image.observation_date} with '
                         f'{next_image_index}, {next_image.observation_date}')

            # update the scene to the time for the next image
            self.scene.update(next_image)

            # get the camera position for this image wrt the central body
            next_camera_position = (-next_image.rotation_inertial_to_camera.matrix.T @
                                    self.scene.target_objs[0].position.ravel())

            next_camera_state = self.dynamics.State(next_image.observation_date, next_camera_position)

            # make a temporary measurement for telling the filter how to predict
            temporary_next_measurement = OpticalBearingMeasurement(np.zeros(2), self.camera.model,
                                                                   next_image, next_camera_state,
                                                                   self.measurement_covariance)

            # get the predicted location of each point from the first image in this image
            predicted_pixels = []
            predicted_states = []
            removes = []
            for ind, ekf in enumerate(filters):
                predicted_state, predicted_observation = ekf.propagate_and_predict(temporary_next_measurement)
                if predicted_state is None:
                    removes.append(ind)
                    continue
                predicted_states.append(predicted_state)
                predicted_pixels.append(predicted_observation)

            for rm in removes[::-1]:
                filters.pop(rm)

            if not predicted_pixels:
                _LOGGER.debug(f'No predicted locations for image {image_ind} to {next_image_index}')
                continue

            # build the kdtree for the predicted locations in the next image so we can do a ball query
            predicted_kdtree = cKDTree(np.vstack(predicted_pixels))

            # do a ball query with the initial euclidean tolerance to figure out the initial pairs
            full_pairs = predicted_kdtree.query_ball_tree(self.observation_trees[next_image_index],
                                                          self.initial_euclidean_threshold)

            # loop through each pair and create an EKF to follow the path
            valid_pairs = False
            for predicted_index, (ekf, pairs) in enumerate(zip(filters, full_pairs)):

                number_of_pairs = len(pairs)

                if number_of_pairs == 0:
                    continue

                if number_of_pairs > self.maximum_paths_per_image:
                    _LOGGER.debug(f'Too many forward paths {number_of_pairs}.  '
                                  f'Only taking {self.maximum_paths_per_image} closest')
                    # pairs = self.observation_trees[next_image_index].query_ball_point(
                    #     predicted_kdtree.data[predicted_index], self.initial_euclidean_threshold / 2
                    # )

                    sorted_pairs = sorted(zip(np.linalg.norm(self.observation_trees[next_image_index].data[pairs] -
                                                             predicted_kdtree.data[predicted_index], axis=-1), pairs))

                    pairs = [x[1] for x in list(sorted_pairs)[:self.maximum_paths_per_image]]

                _LOGGER.debug(f'{len(pairs)} paths forward for track {ekf.identity}')

                valid_pairs = True

                for pair in pairs:
                    if self.observation_ids[next_image_index][pair] in self.confirmed_particles:
                        # skip particles we have already tracked
                        continue

                    # clone the ekf to follow the path
                    new_ekf = deepcopy(ekf)

                    # get rid of the state initializer because it isn't needed anymore and makes problems with pickling
                    new_ekf.state_initializer = None

                    # do the measurement update
                    new_measurement = OpticalBearingMeasurement(self.observation_trees[next_image_index].data[pair],
                                                                self.camera.model, next_image, next_camera_state,
                                                                self.measurement_covariance,
                                                                identity=self.observation_ids[next_image_index][pair])

                    state_update = new_ekf.process_measurement(
                        new_measurement, pre_update_state=predicted_states[predicted_index],
                        pre_update_predicted_measurement=predicted_pixels[predicted_index]
                    )

                    if state_update is None:
                        _LOGGER.debug(f'Failed to propagate EKF {new_ekf.identity}.')
                    else:

                        self._ekfs_to_process.put_retry(new_ekf)

            if valid_pairs:
                # update the image counter
                number_of_linked_images += 1

            if number_of_linked_images >= self.maximum_forward_images:
                break

    def _initialize_tracking_workers(self) -> None:
        """
        This method sets up the processes and queues for working on data in parallel for initial tracking.
        """

        self._results = ClearableQueue()

        self._ekfs_to_process = ClearableQueue()

        self._working = Array('i', cpu_count())

        self.processes = [Process(target=self._follow, args=(pid,), name=f'EKF Follower {pid}')
                          for pid in range(cpu_count())]

    def _reset_tracking_workers(self) -> None:
        """
        This method resets the Pool of workers, respawning any that have died and clears the queues/the work array
        for tracking workers
        """

        self._n_respawns += 1

        self._results.clear()

        self._ekfs_to_process.clear()

        # fix any dead processes
        for ind, process in enumerate(self.processes):
            self._working[ind] = False
            try:
                process.join()
                process.close()
            except ValueError:
                pass
            self.processes[ind] = Process(target=self._follow, args=(ind,),
                                          name=f'EKF Follower {ind}.{self._n_respawns}')

    def _tear_down_tracking_workers(self):
        """
        This method terminates the pool of workers and the queues used to communicate with them
        """

        for process in self.processes:
            try:
                process.join(timeout=0.1)
            except TimeoutError:
                process.terminate()
            except ValueError:
                continue
            process.close()

        self._results.close()
        self._ekfs_to_process.close()

    def _initialize_smoothing_workers(self) -> None:
        """
        This method sets up the processes and queues for working on data in parallel for initial tracking.
        """

        self._smoothing_input = ClearableQueue()

        self._smoothing_output = ClearableQueue()

        self._smoothing_working = Array('i', cpu_count())

        self.smoothing_processes = [Process(target=self._smooth, args=(pid,), name=f'Smoother {pid}')
                                    for pid in range(cpu_count())]

    def _reset_smoothing_workers(self) -> None:
        """
        This method resets the Pool of workers, respawning any that have died and clears the queues/the work array
        for smoothing workers
        """

        self._n_smoothing_respawns += 1

        self._smoothing_input.clear()

        self._smoothing_output.clear()

        # fix any dead processes
        for ind, process in enumerate(self.smoothing_processes):
            self._smoothing_working[ind] = False
            process.join()
            process.close()
            self.smoothing_processes[ind] = Process(target=self._smooth, args=(ind,),
                                                    name=f'Smoother {ind}.{self._n_smoothing_respawns}')

    def _tear_down_smoothing_workers(self):
        """
        This method terminates the pool of workers and the queues used to communicate with them
        """

        for process in self.processes:
            try:
                process.join(timeout=0.1)
            except TimeoutError:
                process.terminate()
            except ValueError:
                continue
            process.close()

        self._results.close()
        self._ekfs_to_process.close()

    def _follow(self, process_number: int):
        """
        This method retrieves an EKF from the queue and follows it to completion

        This is used as the target for the Processes that are created.

        :param process_number: The number of the process.
        """

        if self._ekfs_to_process is None:
            raise ValueError("The work queue hasn't been initialized")
        if self._results is None:
            raise ValueError("The results queue hasn't been initialized")
        if self._working is None:
            raise ValueError("The working array hasn't been initialized")
        while True:
            try:
                # retrieve an ekf from the queue to process
                ekf: ExtendedKalmanFilter = self._ekfs_to_process.get(timeout=2)
                self._working[process_number] = True
            except Empty:
                self._working[process_number] = False
                if True in self._working:
                    _LOGGER.debug(f'QUEUE was empty for process {process_number} but another process is still working.'
                                  f'Process status: {str(list(self._working))}')
                    continue

                else:
                    _LOGGER.info('No more EKFs to follow')
                    self._results.put_retry('DONE')
                    break

            if len(ekf.measurement_history) > self.maximum_track_length:
                _LOGGER.warning(f'Too long of a track for ekf {ekf.identity}, length {len(ekf.measurement_history)}, '
                                f'stopping')
                continue

            # create a list to store the paths forward for the current ekf
            local_splits = []

            _LOGGER.info(f'Following EKF: {ekf.identity}.  Current length {len(ekf.measurement_history)}')

            # get the time of the last image included in the ekf
            last_image_time = ekf.measurement_history[-1].time

            # determine the accuracy this filter should have
            search_distance = float(self.search_distance_function(ekf))

            # start a counter for the number of images we've linked with
            number_of_images_considered = 0

            for next_image_index, next_image in self.camera:

                time_delta = next_image.observation_date - last_image_time

                if time_delta <= _ZERO_TIMEDELTA:
                    # skip if we are still before the current image.
                    continue

                if time_delta >= self.maximum_image_timedelta:
                    # stop if the next image is too far ahead in time
                    break

                if self.observation_trees[next_image_index] is None:
                    # skip if there are no observations for this image
                    continue

                # update the scene to the time of the current image
                self.scene.update(next_image)

                # get the current camera position with respect to the central body in the inertial frame
                next_camera_position = (-next_image.rotation_inertial_to_camera.matrix.T @
                                        self.scene.target_objs[0].position)

                next_camera_state = self.dynamics.State(next_image.observation_date, next_camera_position)

                temporary_next_measurement = OpticalBearingMeasurement(np.zeros(2), self.camera.model,
                                                                       next_image, next_camera_state,
                                                                       self.measurement_covariance)

                # predict the measurement at this time
                predicted_state, predicted_pixels = ekf.propagate_and_predict(temporary_next_measurement)
                # the state failed to propagate
                if predicted_state is None:
                    continue

                # get the innovation matrix
                last_measurement = ekf.measurement_history[-1]
                measurement_jacobian = last_measurement.compute_jacobian(predicted_state)
                innovation_covariance = (last_measurement.covariance +
                                         measurement_jacobian@predicted_state.covariance@measurement_jacobian.T)

                # get the innovation information matrix by inverting the covariance matrix
                try:
                    information_matrix = np.linalg.inv(innovation_covariance)
                except np.linalg.linalg.LinAlgError:
                    information_matrix = np.linalg.pinv(innovation_covariance)

                # query the tree to get the potential next points using the search distance
                pairs = self.observation_trees[next_image_index].query_ball_point(predicted_pixels.ravel(),
                                                                                  search_distance)

                # loop through and filter out paths that are already taken and don't meet the mahalanobis distance

                # create these lists to store pairs that require more consideration
                keep_pairs = []
                keep_mahalanobis_distances = []

                for pair in pairs:
                    if self.observation_ids[next_image_index][pair] in self.confirmed_particles:
                        # skip if we've already used this particle
                        continue

                    # compute the mahalanobis distance
                    pixel_separation = (self.observation_trees[next_image_index].data[pair].ravel() -
                                        predicted_pixels.ravel())

                    mahalanobis_distance_squared = pixel_separation @ information_matrix @ pixel_separation

                    # check the squared mahalanobis distance
                    if mahalanobis_distance_squared > self.maximum_mahalanobis_distance_squared:
                        _LOGGER.debug(f'Rejected potential pair for {ekf.identity} due to mahalanobis distance '
                                      f'of {mahalanobis_distance_squared}')
                        continue

                    # store this as a valid path forward
                    keep_pairs.append(pair)
                    keep_mahalanobis_distances.append(mahalanobis_distance_squared)

                number_of_paths = len(keep_pairs)
                if ((len(ekf.measurement_history) > self.expected_convergence_number) and
                        (number_of_paths > self.reduced_paths_forward_per_image)):
                    _LOGGER.warning(f"The filter should be converged but there are still {number_of_paths} forward."
                                    f"Only keeping the {self.reduced_paths_forward_per_image} best ones")

                    keep_pairs = [x[1] for x in sorted(zip(keep_mahalanobis_distances,
                                                           keep_pairs))[:self.reduced_paths_forward_per_image]]

                elif number_of_paths > self.maximum_paths_per_image:
                    _LOGGER.warning(f"There are {number_of_paths} paths forward but only "
                                    f"{self.maximum_paths_per_image} are allowed."
                                    f"Only keeping the {self.maximum_paths_per_image} best ones")
                    keep_pairs = [x[1] for x in sorted(zip(keep_mahalanobis_distances,
                                                           keep_pairs))[:self.maximum_paths_per_image]]

                # loop through all of the paths we are going to follow
                for pair in keep_pairs:
                    # clone the ekf to follow this path
                    new_ekf = deepcopy(ekf)

                    # make a measurement for this pair
                    new_measurement = OpticalBearingMeasurement(self.observation_trees[next_image_index].data[pair],
                                                                self.camera.model, next_image, next_camera_state,
                                                                self.measurement_covariance,
                                                                identity=self.observation_ids[next_image_index][pair])

                    # perform a measurement update using this path
                    new_ekf.process_measurement(new_measurement, predicted_state, predicted_pixels)

                    # add the new ekf to the queue to be processed
                    _LOGGER.debug(f'Adding {new_ekf.identity} to the queue. {self._ekfs_to_process.size.value}')
                    self._ekfs_to_process.put_retry(new_ekf)

                    # store how many times we've spit this ekf
                    local_splits.append(new_ekf)

                if len(local_splits) > self.maximum_paths_total:
                    _LOGGER.warning(f'Too many forward splits for EKF {ekf.identity}. {len(local_splits)} already. '
                                    f'Skipping more splits')

                    break

                number_of_images_considered += 1

                if number_of_images_considered > self.maximum_forward_images:
                    break

            # put the split ekfs as potential ending ekfs that need to be smoothed and analyzed further
            self._results.put_retry([x for x in local_splits
                                     if len(x.measurement_history) > self.minimum_number_of_measurements])

            self._working[process_number] = False

    def _smooth(self, process_number: int):
        """
        This method retrieves an EKF from the queue and smooths it.

        The smoothed EKF is then placed into the output queue

        This is used as the target for the Processes that are created for smoothing.

        :param process_number: The number of the process.
        """
        if self._smoothing_input is None:
            raise ValueError("The work queue hasn't been initialized")
        if self._smoothing_output is None:
            raise ValueError("The results queue hasn't been initialized")
        if self._smoothing_working is None:
            raise ValueError("The working array hasn't been initialized")

        num_time_out = 0
        while True:
            try:
                incoming: Union[Tuple[int, ExtendedKalmanFilter], str] = self._smoothing_input.get(timeout=2)

                if isinstance(incoming, str) and incoming == "END":
                    self._smoothing_working[process_number] = False
                    self._smoothing_output.put_retry('DONE')
                    break

                ind, ekf = incoming

                num_time_out = 0

                # retrieve an ekf form the queue to process
                self._smoothing_working[process_number] = True

            except Empty:
                self._smoothing_working[process_number] = False
                num_time_out += 1
                if True in self._smoothing_working:
                    _LOGGER.debug(f'QUEUE was empty for process {process_number} but another process is still working.'
                                  f'Process status: {str(list(self._smoothing_working))}')
                    continue

                elif num_time_out >= self.maximum_time_outs:
                    _LOGGER.info('No more EKFs to smooth')
                    self._smoothing_output.put_retry('DONE')
                    break
                else:
                    _LOGGER.debug(f'{process_number} timed out {num_time_out} times')
                    continue

            smooth_successful = ekf.smooth()

            sent = False
            number_failed = 0

            while not sent:
                try:
                    self._smoothing_output.put_retry((smooth_successful, ind, ekf))
                    sent = True
                except Full:
                    number_failed += 1

                    if number_failed > 4:
                        _LOGGER.warning('Unable to put the smoothed results on the queue')
                        break

    def filter_ekfs(self, ekfs_to_filter: Iterable[ExtendedKalmanFilter], number_of_ekfs: int):
        """
        This method does backwards smoothing on each EKF and figures out which are actually valid

        :param ekfs_to_filter: The list of EKFs to filter
        :param number_of_ekfs: The number of ekfs there are to filter
        """

        removes = []
        residual_means = []
        residual_stds = []

        smoothed_ekfs: List[Optional[ExtendedKalmanFilter]] = [None] * number_of_ekfs

        for p in self.smoothing_processes:
            try:
                p.start()
            except AssertionError:
                pass

        for ind, ekf in enumerate(ekfs_to_filter):

            if len(ekf.measurement_history) < self.minimum_number_of_measurements:
                removes.append(ind)
                continue

            self._smoothing_input.put_retry((ind, ekf))

        for _ in range(cpu_count()):
            self._smoothing_input.put_retry('END')

        number_timed_out = 0
        number_not_done = len(self.processes)

        while number_not_done:
            try:
                result = self._smoothing_output.get(timeout=1)

                # reset the counter
                number_timed_out = 0

                if isinstance(result, str) and (result == 'DONE'):
                    number_not_done -= 1
                    _LOGGER.info(f'{number_not_done} processes still smoothing.  Approximately '
                                 f'{self._smoothing_input.qsize()} things still to do')

                elif result[0]:
                    # compute and store the standard deviation and mean
                    resid_mean, resid_std = result[2].compute_residual_statistics()

                    residual_means.append(resid_mean)
                    residual_stds.append(resid_std)

                    # store the smoothed ekf
                    smoothed_ekfs[result[1]] = result[2]

                else:
                    # we didn't smooth so remove this
                    removes.append(result[1])

            except Empty:
                number_timed_out += 1
                _LOGGER.warning(f'Timed out trying to retrieve smoothed results {number_timed_out} times')

        # get rid of things that were too short or didn't smooth
        for rm in sorted(removes, reverse=True):
            smoothed_ekfs.pop(rm)

        # kill the processes
        for p in self.smoothing_processes:
            try:
                p.join(timeout=0.2)
            except TimeoutError:
                p.terminate()

        _LOGGER.info(f'Identifying valid EKFs from {len(smoothed_ekfs)} total')

        id_sets = []

        # filter out where the residuals are greater than the maximum standard deviation
        # also prepare the sets of particle ids
        removes = []
        for ind, (std, ekf) in enumerate(zip(residual_stds, smoothed_ekfs)):
            if std > self.maximum_residual_standard_deviation:
                removes.append(ind)
                continue

            # store the set of the measurement ids
            id_sets.append({meas.identity for meas in ekf.measurement_history})

        for rm in removes[::-1]:
            smoothed_ekfs.pop(rm)
            residual_stds.pop(rm)
            residual_means.pop(rm)

        removes = []

        # loop through and figure out which ekfs are subsets of the others
        for first_ind, first_meas_id_set in enumerate(id_sets):
            for second_ind, second_meas_id_set in enumerate(id_sets):

                if first_ind == second_ind:
                    # if this is the same set then skip it
                    continue

                # get the points which are in the first ekf but not the second ekf
                first_unique = first_meas_id_set - second_meas_id_set

                if len(first_unique) == 0:
                    # if the first ekf is a full subset of the later ekf then remove it
                    removes.append(first_ind)
                    break  # we no longer need to consider this ekf

                # if the first ekf only has 1 unique point and the other has multiple unique points then then other
                # is a longer track and should be kept over this one
                if (len(first_unique) == 1) and len(second_meas_id_set - first_meas_id_set) >= 2:
                    removes.append(first_ind)
                    break  # we no longer need to consider this ekf


        # remove things that should be removed (we might have duplicates so ignore them and unique sorts for us)
        for rm in np.unique(removes)[::-1]:
            smoothed_ekfs.pop(rm)
            residual_stds.pop(rm)
            residual_means.pop(rm)
            id_sets.pop(rm)

        # now figure out which track is best for each starting particle
        starting_dictionary = {}

        for ind, ekf in enumerate(smoothed_ekfs):

            # get the initial particle in this track
            start_id = ekf.measurement_history[0].identity

            # see if we've considered other tracks that contain this particle already
            current_best = starting_dictionary.get(start_id, None)

            # if this is the first track starting on this particle keep it for now
            if current_best is None:
                current_best = (ind, residual_stds[ind])
                starting_dictionary[start_id] = current_best
                continue

            # compare the number of measurements of the current best track starting with this particle and the track
            # under consideration
            length_current_best = len(id_sets[current_best[0]])
            length_under_consideration = len(id_sets[ind])

            if length_under_consideration > (length_current_best + 1):
                # if the EKF under consideration has 2 or more measurements than the current best keep it
                starting_dictionary[start_id] = (ind, residual_stds[ind])

            elif length_current_best > (length_under_consideration + 1):
                # if the current best has 2 or more measurements than the EKF under consideration keep it
                continue

            # if the EKF under consideration and the current best are within 1 measurement of each other keep the
            # one with a lower post-fit std
            elif current_best[-1] < residual_stds[ind]:
                continue

            else:
                starting_dictionary[start_id] = (ind, residual_stds[ind])

        # store the confirmed EKFs and ingested particles
        for ind, std in starting_dictionary.values():

            self.confirmed_filters.append(smoothed_ekfs[ind])
            self.confirmed_standard_deviations.append(std)
            self.confirmed_particles.update(id_sets[ind])

    def track(self) -> None:
        """
        This method tracks particles from image to image using the EKF.

        This works by finding initial pairs for each image and then following those tracks to termination. It also
        handles setup and teardown of the working pool.  It then filters all of the tracks, saving only the good ones.
        """

        # initialize the worker pool
        self._initialize_tracking_workers()

        self._initialize_smoothing_workers()

        # loop through each image and identify the tracks that begin in that image
        first = True

        for ind, image in self.camera:

            start = time()

            # do the initial pairing
            self.find_initial_pairs(ind, image)


            # reset the tracking workers for the next image
            if not first:
                self._reset_tracking_workers()

            # start the process of tracking along these initial pairs
            for process in self.processes:
                try:
                    process.start()
                except AssertionError:
                    pass

            # self._follow(-1)

            # loop through, retrieving the results
            ekfs = 0

            number_to_complete = len(self.processes)
            number_timed_out = 0

            with TemporaryFile('wb+') as ekfs_to_smooth:

                while number_to_complete:
                    try:
                        results = self._results.get(timeout=1)

                        # if the process said it is done processing then decrease the number to complete and move along
                        if results == 'DONE':
                            number_to_complete -= 1
                            if number_to_complete == 0:
                                _LOGGER.info(f'All processes completed.  {ekfs} generated')
                                break
                            else:
                                _LOGGER.info(f'Tracking process has completed, {number_to_complete} still working')
                        else:

                            # otherwise we got new results and should store them
                            # ekfs.extend(results)
                            for ekf in results:
                                pickle.dump(ekf, ekfs_to_smooth)
                                ekfs += 1

                            # reset the number timed out
                            number_timed_out = 0

                    except Empty:
                        number_timed_out += 1
                        if number_timed_out > self.maximum_time_outs:
                            _LOGGER.warning('Exceeded the maximum number of time outs.  Stopping')
                            for process in self.processes:
                                process.terminate()
                                sleep(0.1)
                                try:
                                    process.close()
                                except ValueError:
                                    pass
                            break

                    # check if we've reached our overall time lime
                    if (time() - start) > self.maximum_tracking_time_per_image:
                        _LOGGER.warning('Exceeded maximum execution time. Stopping')
                        for process in self.processes:
                            process.terminate()
                            sleep(0.5)
                            try:
                                process.close()
                            except ValueError:
                                pass
                        break

                _LOGGER.info(f'Tracking complete for image {ind} of {sum(self.camera.image_mask)} in '
                             f'{(time()-start):.3f} seconds')

                # filter the ekfs and save only the good ones
                if not first:
                    self._reset_smoothing_workers()
                else:
                    first = False

                ekfs_to_smooth.seek(0)
                self.filter_ekfs(_pickle_generator(ekfs_to_smooth), ekfs)

        # tear down the workers
        self._tear_down_smoothing_workers()
        self._tear_down_tracking_workers()

    def save_results(self, out: PATH):
        """
        This method saves the final ekfs to a csv file.

        The files in the csv file are

        ================== =============================================================================================
        Column             Description
        ================== =============================================================================================
        id                 The ID for the EKF.  This is usually a UUID hash string
        length             The length of the EKF (number of measurements it ingested)
        residual std       The standard deviation of the post-fit residuals of the EKF
        initial time       The UTC time of the initial state for the EKF
        initial position x The x position of the initial state for the EKF
        initial position y The y position of the initial state for the EKF
        initial position z The z position of the initial state for the EKF
        initial velocity x The x velocity of the initial state for the EKF
        initial velocity y The y velocity of the initial state for the EKF
        initial velocity z The z velocity of the initial state for the EKF
        detection ids      A list of detection ids (dependent on the IDs of the UFO detections) separated by '|'
        ================== =============================================================================================

        :param out: The file to save the csv to
        """

        with Path(out).open('w') as out_file:
            out_format = '{},{},{},{},{},{},{},{},{},{},{}\n'
            out_file.write(out_format.format('id',
                                             'length',
                                             'residual std',
                                             'initial time',
                                             'initial position x',
                                             'initial position y',
                                             'initial position z',
                                             'initial velocity x',
                                             'initial velocity y',
                                             'initial velocity z',
                                             'detection ids'))

            for ekf, std in zip(self.confirmed_filters, self.confirmed_standard_deviations):
                initial_state: Dynamics.State = ekf.state_history[0][1]
                out_file.write(out_format.format(ekf.identity,
                                                 len(ekf.measurement_history),
                                                 std,
                                                 initial_state.time.isoformat(),
                                                 *initial_state.position,
                                                 *initial_state.velocity,
                                                 '|'.join(str(meas.identity) for meas in ekf.measurement_history)))
