# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
This module provides a user interface class for detecting and tracking unidentified objects in monocular images.

Description
-----------

The process of identifying and tracking unidentified objects in monocular images is complex.  It involves extracting
possible detections from images based only on the images themselves and then attempting to link those detections from
image to image to create tracks.  The end results is an autonomous system that is capable of capturing many observed
particles with limited to no human interaction.  For more details refer to the :mod:`.detector` and :mod:`.ekf_tracker`
packages or to the paper at
https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2019EA000843

Use
---

The :class:`.UFO` class is the main interface for autonomous detection and tracking of objects in images in GIANT.  It
provides everything that a user will need to interact with for extracting detections and tracks, reviewing them
visually, and saving the results to the disk.  This includes access to the :class:`.Detector` and :class:`.Tracker`
classes and their attributes.  In addition methods are provides which package everything together and feed the results
of the :class:`.Detector` into the :class:`.Tracker` for you so you don't need to worry about how to transfer the data
around.  Finally, this also provides direct access to the :mod:`.ufo.visualizers` module along with its
classes/functions, making it easy to visually inspect and modify the results.

The typical workflow will be initialize the :class:`.UFO` class, call :meth:`.UFO.detect`, call :meth:`.UFO.track`, call
:meth:`.UFO.save_results`.

For a description of the tuning parameters that can be messed with refer to the :class:`.Detector`/:class:`.Tracker`
documentation or the :mod:`.ufo` documentation.
"""

import logging

from typing import Optional, Dict, Any, Callable

import pandas as pd

from scipy.spatial import KDTree

from giant.camera import Camera
from giant.ray_tracer.scene import Scene

from giant.stellar_opnav.stellar_class import StellarOpNav, StellarOpNavOptions
from giant.ufo.detector import Detector
from giant.ufo.ekf_tracker import Tracker, Dynamics, STATE_INITIALIZER_TYPE, ExtendedKalmanFilter
from giant.point_spread_functions.gaussians import IterativeGeneralizedGaussianWBackground

from giant._typing import PATH


_LOGGER: logging.Logger = logging.getLogger(__name__)
"""
This is the logging interface for reporting status, results, issues, and other information.
"""


class UFO:
    """
    This class provides a simple user interface for doing the combined steps of detecting and tracking UFOs in images.

    The primary benefit to using this class is that it packages the data from the :class:`.Detector` class to the
    :class:`.Tracker` class without you having to interact with it directly.  Beyond that it is primarily just calls to
    the methods of the :class:`.Detector` and :class:`.Tracker` classes, letting them do all of the hard work behind the
    scenes.

    The typical way to use this class it to initialize it with all of the settings needed for the :class:`.Detector` and
    :class:`.Tracker` and then to call the following methods in order:

    #. :meth:`detect`
    #. :meth:`track`
    #. :meth:`save_results`
    #. :meth:`visualize_detection_results`

    If your settings were right, this should result in 2 csv files (1 with possible detections and one with possible
    tracks) and a visualization of the detections for each image displayed either interactively or saved to files.

    For more details about the processes working in this class or for successfully tuning things for identifying tracks,
    see the :mod:.detector`, :mod:`.tracker`, and :mod:`.ufo` documentation.
    """

    def __init__(self, camera: Camera, scene: Scene, dynamics: Dynamics, state_initializer: STATE_INITIALIZER_TYPE,
                 search_distance_function: Callable[[ExtendedKalmanFilter], float],
                 detector_kwargs: Optional[Dict[str, Any]] = None,
                 tracker_kwargs: Optional[Dict[str, Any]] = None,
                 initial_sopnav_options: Optional[StellarOpNavOptions] = None,
                 tracking_quality_code_minimum: int = 3,
                 identify_hot_pixels_and_unmatched_stars: bool = True,
                 clear_detector_before_tracking: bool = True,
                 visual_inspection_quality_code_minimum: int = 4):
        """
        :param camera: The :class:`.Camera` containing the images to be processed
        :param scene: The :class:`.Scene` specifying the central body as the first target and any extra extended
                      bodies that should be ignored as other targets.
        :param dynamics: The dynamics model to use in the EKF for propagating the state from one time to another
        :param state_initializer: A callable which takes in a :class:`.Measurement` instance and
                                  :class:`.Dynamics.State` class object and returns an initialized state.
        :param search_distance_function: A callable which takes in an :class:`.ExtendedKalmanFilter` and returns what
                                         the Euclidean search distance should be for that EKF in pixels.  This is only
                                         applied after the first pair has been made
        :param detector_kwargs: The key word arguments to pass to the :class:`.Detector` class
        :param tracker_kwargs: The key word arguments to pass to the :class:`.Tracker` class
        :param initial_sopnav_options: The options struct to use to initialize the :class:`.StellarOpNav` instance
                                                instance before it is passed to the :class:`.Detector`
        :param tracking_quality_code_minimum: The minimum quality code to pass a possible detection to the tracking
                                              algorithm
        :param identify_hot_pixels_and_unmatched_stars: A flag specifying whether to attempt to filter out hot pixels
                                                        and unmatched stars from the UFO detections
        :param clear_detector_before_tracking: A flag specifying whether to clear out the intermediate lists created in
                                               the :class:`.Detector` before moving on to the :class:`.Tracker`.  This
                                               can be necessary for memory management purposes
        :param visual_inspection_quality_code_minimum: The minimum quality code to pass a possible detection to the
                                                       detection visualizations
        """

        sopnav = StellarOpNav(camera, options=initial_sopnav_options)

        if detector_kwargs is None:
            detector_kwargs = {}

        self.detector = Detector(sopnav, scene=scene, **detector_kwargs)
        """
        The :class:`.Detector` instance to use to identify possible UFOs in the images
        """

        if tracker_kwargs is None:
            tracker_kwargs = {}
        self.tracker: Tracker = Tracker(camera, scene, dynamics, state_initializer, search_distance_function,
                                        **tracker_kwargs)
        """
        The :class:`.Tracker` instance to use to track UFOs from one frame to the next
        """

        self.tracking_quality_code_minimum: int = tracking_quality_code_minimum
        """
        This specifies the minimum quality code for a detection to be passed to the tracker.
        """

        self.identify_hot_pixels_and_unmatched_stars: bool = identify_hot_pixels_and_unmatched_stars
        """
        This boolean indicates that we should attempt to identify hot pixels and stars from the ufo detections from the
        current set of images.
        
        If this is ``True`` the :meth:`.Detector.identify_hot_pixels_and_unmatched_stars` is called as part of
        :meth:`detect`.
        """

        self.visual_inspection_quality_code_minimum: int = visual_inspection_quality_code_minimum
        """
        This specifies the minimum quality code for a detection to be shown in the visualization.
        """

        self.clear_detector_before_tracking: bool = clear_detector_before_tracking
        """
        This clears data from the detector/sopnav classes (besides what is in the :attr:`.detection_data_frame`) before 
        attempting to track UFOs from image to image.  
        
        This can be important for memory management issues, especially if your machine doesn't have a lot of memory. As 
        such it is recommended to always leave this ``True``.  Note that if this is ``True`` however, and you call
        :meth:`track` then you will no longer be able to retrieve data about the detection from the attributes in the 
        :class:`.Detector` class.  Instead you must access all data from the :attr:`.Detector.detection_data_frame` 
        attribute.
        """

    def detect(self):
        """
        This method detects potential UFOs in all images that have been added to the camera and have not been processed
        yet.

        This essentially boils down to a series of calls to :class:`.Detector` methods.

        #. :meth:`.Detector.update_attitude`
        #. :meth:`.Detector.find_ufos`
        #. :meth:`.Detector.package_results`
        #. :meth:`.Detector.remove_duplicate`
        #. Optionally :meth:`.Detector.identify_hot_pixels_and_unmatched_stars`

        where the last is only called if :attr:`identify_hot_pixels_and_unmatched_stars` is set to ``True``

        Once complete, the results of the detections can be found in :attr:`.Detector.detection_data_frame`.
        """

        self.detector.update_attitude()
        self.detector.find_ufos()
        self.detector.package_results()
        self.detector.remove_duplicates()

        if self.identify_hot_pixels_and_unmatched_stars:
            self.detector.identify_hot_pixels_and_unmatched_stars()

    def track(self):
        """
        This method packages the possible UFO detections for the :class:`.Tracker`, passes them to the tracker, and then
        attempts to track the UFOs from image to image.

        The packaging of the detections essentially boils down to building a scipy.spatial ``cKDTree`` on the
        ``(x_raw, y_raw)`` pixel locations for each detection, and retrieving the id for each UFO detection (the index
        of the :attr:`.Detector.detection_data_frame`).  This data is then passed to the :attr:`tracker`, and the
        :meth:`.Tracker.track` method is called.  The results are then stored in the :attr:`.Tracker.confirmed_filters`
        and :attr:`.Tracker.confirmed_standard_deviations` attributes.

        You can filter which possible detections are fed to the tracker using the :attr:`tracking_quality_code_minimum`
        attribute.

        Note that this method can take a while to run, will use multi-processing, and will likely take up a lot of
        memory (depending on the number of images/number of detections per image).
        """

        # clear out the information from the detector before continuing to save memory
        if self.clear_detector_before_tracking:
            _LOGGER.info('Clearing out unnecessary detector lists')
            self.detector.clear_results()

        # get the dataframe of detections
        if self.detector.detection_data_frame is None:
            raise ValueError('must call detect before track')
        ufos: pd.DataFrame = self.detector.detection_data_frame

        _LOGGER.info(f'Filtering {ufos.shape[0]} detections by quality code > {self.tracking_quality_code_minimum}')
        # filter based on the quality code
        ufos = ufos.loc[ufos.loc[:, "quality_code"] >= self.tracking_quality_code_minimum]
        _LOGGER.info(f'{ufos.shape[0]} detections retained')

        for image, grp in ufos.groupby('image_file'):
            _LOGGER.debug(f'{grp.shape[0]} for {image}')

        # build the kdtree and retrieve the identities for each detection
        kd_trees = []
        ids = []

        _LOGGER.info('Building detection KDTrees')
        for image_file, group in ufos.groupby("image_file"):

            kd_trees.append(KDTree(group.loc[:, ['x_raw', 'y_raw']].to_numpy()))
            ids.append(ufos.index.to_numpy())

        # feed this data to the tracker
        self.tracker.observation_trees = kd_trees
        self.tracker.observation_ids = ids

        _LOGGER.info('Tracking')
        # let it do its thing
        self.tracker.track()

    def save_results(self, detection_results: PATH = "./detections.csv", tracking_results: PATH = "./tracks.csv"):
        """
        This method saves both the detection and tracker results to a file (as long as they have been run).

        The detection results are saved to the ``detection_results`` path specified by the user.  The tracking results
        are saved to the ``tracking_results`` path specified by the user.  If the detector or the tracker have not been
        run a warning is printed and nothing happens.

        :param detection_results: The file to save the detections csv file to
        :param tracking_results:  The file to save the tracking csv file to
        """

        if self.detector.detection_data_frame is None:
            _LOGGER.warning('The detector has not been run yet')
        else:
            _LOGGER.info(f'Saving detection results to {detection_results}')
            self.detector.save_results(str(detection_results))

        if not self.tracker.confirmed_filters:
            _LOGGER.warning('The tracker has not been run yet')
        else:
            _LOGGER.info(f'Saving tracking results to {tracking_results}')
            self.tracker.save_results(tracking_results)

    def visualize_detection_results(self, interactive: bool = True, save_frames: bool = False,
                                    frame_output: str = './{}.png'):
        """
        This method visualizes detection results, overlaying them on the images themselves.

        The detections are filtered based off of quality code using :attr:`visual_inspection_quality_code_minimum`.

        They are then plotted on the images.  The plots can either be saved to file (if ``save_frames`` is set to
        ``True``), displayed interactively (if ``interactive`` is set to ``True``) or displayed all at once (if
        ``save_frames`` and ``frame_output`` are both set to ``False``).  Note that this last option can make a lot of
        figures, therefore we encourage you to use the ``interactive`` option instead.

        For more details refer to the :func:`.show_detections` function.

        :param interactive: Generate an interactive GUI for flipping through the frames
        :param save_frames: Save all frames to a file and don't display them on the screen
        :param frame_output: The location to save all of the frames.  Should include a format specifier {} which will be
                             replaced with the name of the image.
        """
        from giant.ufo.visualizer import show_detections
        if self.detector.detection_data_frame is None:
            raise ValueError('Must call detect before visualize_detection_results')
        _LOGGER.info(f'Filtering {self.detector.detection_data_frame.shape[0]} detections by quality code > '
                     f'{self.visual_inspection_quality_code_minimum}')
        quality_test = self.detector.detection_data_frame.quality_code >= self.visual_inspection_quality_code_minimum
        ufos = self.detector.detection_data_frame.loc[quality_test]
        _LOGGER.info(f'{ufos.shape[0]} detections will be displayed')

        show_detections(ufos, self.detector.sopnav.camera, interactive=interactive, save_frames=save_frames,
                        frame_output=frame_output)
