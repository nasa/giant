


"""
This module provides a class for identifying possible unresolved UFOs in a monocular image.

Description
-----------

UFO identification is done through the usual :mod:`.stellar_opnav` process, but instead of being concerned with the
identified stars, we are concerned with the points that were not matched to stars in the image.  As such, the
:class:`.Detector` class provided in this module simply serves as a wrapper around the :class:`.StellarOpNav` class
to combine some steps together and to collect all of the unidentified results and package them into a manageable format.
For a more detailed description of what happens you can refer to the paper at
https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2019EA000843

Use
---

To identify potential UFOs, simply initialize the :class:`.Detector` class with the required inputs, then call
:meth:`.update_attitude` to estimate the updated attitude of the images (which can be important when trying to identify
dim detections), :meth:`.find_ufos` to identify possible ufos in the images, :meth:`.package_results` to package all of
the detections into a dataframe and to compute some extra information about each detection, and
:meth:`remove_duplicates` which attempts to identify points were we accidentally identified the same point twice (which
can happen when 2 detection are very close together).  Once you have called these methods, you can call
:meth:`.Detector.save_results` to dump the results to (a) csv file(s).

For discussion on Tuning for successful UFO identification, refer to the :mod:`.ufo` package documentation.

You may also be interested in using the :class:`.UFO` class which combines both detection and tracking into a single
interface rather than using this class directly.
"""

import time

import os

import logging

from copy import deepcopy

from typing import Optional, Callable, List, Tuple, Dict, Any, Iterator, Union, Iterable

import numpy as np
from numpy.typing import NDArray

from scipy.spatial import KDTree

import pandas as pd

import cv2

from giant.stellar_opnav.stellar_class import StellarOpNav
from giant.point_spread_functions.gaussians import IterativeGeneralizedGaussianWBackground

from giant.catalogs.utilities import RAD2DEG, DEG2RAD
from giant.utilities.spherical_coordinates import unit_to_radec, radec_to_unit, radec_distance
from giant.utilities.spice_interface import datetime_to_et
from giant.ray_tracer.rays import Rays
from giant.ray_tracer.scene import Scene
from giant.stellar_opnav.stellar_class import StellarOpNavOptions

from giant.image import OpNavImage

from giant._typing import F_SCALAR_OR_ARRAY, PATH 
from giant.utilities.boolean_filter_list import boolean_filter_list


_LOGGER: logging.Logger = logging.getLogger(__name__)
"""
This is the logging interface for reporting status, results, issues, and other information.
"""


def unit_to_radec_jacobian(unit: np.ndarray) -> np.ndarray:
    r"""
    This function computes the Jacobian matrix for going from a unit vector to a right ascension/declination in degrees.

    Mathematically this is given by:

    .. math::
        \frac{\partial (\alpha, \delta)}{\partial \hat{\mathbf{x}}} = \frac{180}{\pi}\left[\begin{array}{ccc}
        -\frac{y}{x**2+y**2} & \frac{x}{x**2+y**2} & 0 \\
        0 & \frac{1}{\sqrt{1-z**2) & 0 \end{array}\right]

    This function is vectorized so that multiple jacobians can be computed at once.  In this case the unit vectors in
    the input should be a shape of 3xn where n is the number of unit vectors and the output will be nx2x3.

    :param unit: The unit vectors to compute the jacobian for as a numpy array
    :return: the nx2x3 jacobian matrices.
    """
    # make sure we have the right shape
    unit = unit.reshape(3, -1)
    # initialize the jacobian matrix
    out = np.zeros([unit.shape[-1], 2, 3], dtype=np.float64)
    # compute the jacobian elements
    out[:, 0, 0] = (-unit[1] / (unit[0] ** 2 + unit[1] ** 2) * 180 / np.pi).ravel()
    out[:, 0, 1] = (unit[0] / (unit[0] ** 2 + unit[1] ** 2) * 180 / np.pi)
    out[:, 1, 2] = (1 / np.sqrt(1 - unit[2] ** 2) * 180 / np.pi)

    return out


_IMAGE_INFORMATION_SIGNATURE = Callable[[OpNavImage], Tuple[float, float, float, F_SCALAR_OR_ARRAY]]
"""
This specifies the call signature that the image information function is expected to have.
"""


_MAGNITUDE_FUNCTION_SIGNATURE = Callable[[List[float], OpNavImage], Union[np.ndarray, List[float]]]
"""
This specifies the call signature that the magnitude function is expected to have..
"""


class Detector:
    """
    This class is used to identify possible non-cooperative unresolved targets in optical images.

    This is done by first extracting bright points from the image, then fitting point spread functions to the bright
    spots in an image, and finally removing bright points that correspond to stars.  All of this is primarily handled by
    the :class:.StellarOpNav` class, and this class serves as a wrapper for collecting data from the
    :class:`.StellarOpNav` class and for packaging each possible detection (with additional information) into a pandas
    Dataframe for easy export/processing.

    To use this class simply provide the required initialization inputs, call :meth:`update_attitude`, call
    :meth:`find_ufos`, call :meth:`package_results`, call :meth:`remove_duplicates`, and then optionally call
    :meth:`save_results` to save the results to a csv file.

    For more details on tuning for detection and the full UFO process, including tacking, see the :mod:`.ufo`
    package documentation.
    """

    def __init__(self, sopnav: StellarOpNav, scene: Optional[Scene] = None, dn_offset: float = 0,
                 image_information_function: Optional[_IMAGE_INFORMATION_SIGNATURE] = None,
                 magnitude_function: Optional[_MAGNITUDE_FUNCTION_SIGNATURE] = None,
                 update_attitude_settings: Optional[StellarOpNavOptions] = None,
                 find_ufos_settings: Optional[StellarOpNavOptions] = None,
                 unmatched_star_threshold: int = 3,
                 hot_pixel_threshold: int = 5,
                 create_hashed_index: bool = True):
        """
        :param sopnav: The stellar opnav instance that will be used for star and UFO identification/attitude estimation
        :param scene: The optional scene instance defining the location of the light source and any extended bodies
        :param dn_offset: The dn offset of the detector, used for computing magnitude/statistics about the observed UFO
        :param image_information_function: A function that gives the quantization noise, read noise, and electron to DN
                                           conversion factor for the detector and predicts the dark current for a
                                           input :class:`.OpNavImage` object (or None).  If None then the SNR values
                                           will be less meaningful.
        :param magnitude_function: A function that computes the apparent magnitude for detections based off of the input
                                   5x5 summed DN for the detections and the image the detection came from.  If ``None``
                                   then the magnitude will not be computed and will be stored as 0 for all detections.
        :param update_attitude_settings: an instance of :class:`.StellarOpNavOptions` used to configure the 
                                         :class:`.StellarOpNav` instance for solving for attitude from star fields
        :param find_ufos_settings: An instance of :class:`.StellarOpNavOptions` used to configure the 
                                   :class:`.StellarOpNav` instance for identifying unmatched points (UFOs) in the images
        :param hot_pixel_threshold: The minimum number of images a (x_raw, y_raw) pair must appear in for the detections
                                    to be labeled a possible hot pixel
        :param unmatched_star_threshold: The minimum number of images a (ra, dec) pair must appear in for the detections
                                         to be labeled a possible unmatched star
        :param create_hashed_index: This boolean flag indicates that when packaging the results
                                    (:meth:`.package_results`) the index of the resulting dataframe should be build from
                                    a hash of ``'image_file_{x_raw}_{y_raw}'`` instead of just using an index.  This
                                    makes it easier to identify the detections uniquely and is recommended to be left
                                    ``True``
        """

        self.sopnav: StellarOpNav = sopnav
        """
        The StellarOpNav instance that will be used for star and UFO identification/attitude estimation.
        """

        self.scene: Optional[Scene] = scene
        """
        The Scene instance that defines the location of any known extended bodies in the images to use for determining
        whether the UFOs are part of the target or not.
        """

        self.dn_offset: float = dn_offset
        """
        The dn offset of the detector (typically a fixed value that pixels are always guaranteed to be above).
        
        This is used to determine the noise level for assigning signal to noise values for each detection 
        """

        self.image_information_function: Optional[_IMAGE_INFORMATION_SIGNATURE] = image_information_function
        """
        A function that gives the quantization noise, read noise, and electron to DN conversion factor for the detector 
        and predicts the dark current for an input :class:`.OpNavImage` object.  
        
        If None then the SNR values computed herein will be less meaningful.
        
        The order of the output should be electrons_to_dn, quantization noise (in elections), read noise (in electrons), 
        dark current (in electrons)
        
        The dark current can either be returned as a scalar or as an array the same size as the image (if it is location
        dependent).
        
        The only input to this function will be the :class:`.OpNavImage` that the detector information is to be returned 
        for, which should give the temperature and exposure length (plus possibly the file the image was retrieved from)
        """

        self.magnitude_function: Optional[_MAGNITUDE_FUNCTION_SIGNATURE] = magnitude_function
        """
        A function that computes the apparent magnitude of a detection given the 5x5 summed DN around the detection and
        the image the detection came from.
        
        If this is ``None`` then the magnitude is not calculated for the detections.  If it is not None, then the 
        results of this function are directly stored in the :attr:`magnitude` and :attr:`Star_observed_magnitude`
        attributes.  Note that this function should expect to process all of the observations at once.
        """

        self.update_attitude_settings: Optional[StellarOpNavOptions] = update_attitude_settings
        """
        A dictionary of str -> dictionary where the keys are "star_id_kwargs", "image_processing_kwargs", or 
        "attitude_estimator_kwargs" and the values are dictionaries specifying the key word argument -> value pairs for 
        the appropriate class.  
        
        This is used to update the settings before attempting to solve for the attitude in each image. 
        (:meth:`update_attitude`)
        """

        self.find_ufos_settings: Optional[StellarOpNavOptions] = find_ufos_settings
        """
        A dictionary of str -> dictionary where the keys are "star_id_kwargs", "image_processing_kwargs", or 
        "attitude_estimator_kwargs" and the values are dictionaries specifying the key word argument -> value pairs for 
        the appropriate class.  

        This is used to update the settings before attempting to identify the UFOs in each image. 
        (:meth:`find_ufos`)
        """

        self.hot_pixel_threshold: int = hot_pixel_threshold
        """
        The minimum number of images the same (x_raw, y_raw) pairs must be identified in before detections are labeled 
        as possible hot pixels.
        """

        self.unmatched_star_threshold: int = unmatched_star_threshold
        """
        The minimum number of images the same (ra, dec) pairs must be identified in before detections are labeled 
        as possible unmatched stars.
        """

        self.create_hashed_index: bool = create_hashed_index
        """
        This boolean flag indicates that when packaging the results (:meth:`.package_results`) the index of the 
        resulting dataframe should be build from a hash of ``'image_file_{x_raw}_{y_raw}'`` instead of just using an 
        index. 
        
        This makes it easier to identify the detections uniquely and is recommended to be left ``True`` 
        """

        self.detection_data_frame: Optional[pd.DataFrame] = None
        """
        This is where the big dataframe of the detections will be stored.
        """

        self.invalid_images: List[bool] = [False] * len(self.sopnav.camera.images)
        """
        This is a list of images that are identified as invalid because they have no bright spots identified in them
        """

        self.summed_dn: List[Optional[List[float]]] = [None] * len(self.sopnav.camera.images)
        """
        This list is used to store the sum of the 5x5 grid of pixel DN values minus the background around each 
        UFO.
        
        Until :meth:`package_results` is called this will be filled with ``None``.
        """

        self.magnitude: List[Optional[Union[List[float], np.ndarray]]] = [None] * len(self.sopnav.camera.images)
        """
        This list is used to store the computed magnitude of each UFO .

        Until :meth:`package_results` is called this will be filled with ``None``.
        """

        self.fit_chi2_value: List[Optional[List[float]]] = [None] * len(self.sopnav.camera.images)
        """
        This list is used to store the Chi2 value from the post-fit residuals for each ufo.

        Until :meth:`package_results` is called this will be filled with ``None``.
        """

        self.integrated_psf_uncertainty: List[Optional[List[float]]] = [None] * len(self.sopnav.camera.images)
        """
        This list is used to store the formal uncertainty of the integrated PSF value for each UFO.

        Until :meth:`package_results` is called this will be filled with ``None``.
        """

        self.summed_dn_uncertainty: List[Optional[List[float]]] = [None] * len(self.sopnav.camera.images)
        """
        This list is used to store the formal uncertainty of the integrated PSF value for each UFO.

        Until :meth:`package_results` is called this will be filled with ``None``.
        """

        self.saturated: List[Optional[List[bool]]] = [None] * len(self.sopnav.camera.images)
        """
        This list is used to store a flag specifying whether each UFO has saturated pixels or not.
        
        If a flag is ``True`` then the UFO did contain at least 1 pixel that was saturated.

        Until :meth:`package_results` is called this will be filled with ``None``.
        """

        self.summed_dn_count: List[Optional[List[int]]] = [None] * len(self.sopnav.camera.images)
        """
        This list is used to store the number of pixels summed for computing the summed dn for each UFO.

        This is nearly always 25 (for a 5x5 grid) but occasionally for points near the edge it may be less.

        Until :meth:`package_results` is called this will be filled with ``None``.
        """

        self.max_dn: List[Optional[List[float]]] = [None] * len(self.sopnav.camera.images)
        """
        This list is used to store the maximum DN value for each UFO.

        Until :meth:`package_results` is called this will be filled with ``None``.
        """

        self.bearing: List[Optional[np.ndarray]] = [None] * len(self.sopnav.camera.images)
        """
        This list is used to store inertial bearing of each UFO.

        Until :meth:`package_results` is called this will be filled with ``None``.
        """

        self.integrated_psf: List[Optional[List[float]]] = [None] * len(self.sopnav.camera.images)
        """
        This list is used to store the integrated PSF values for each UFO

        Until :meth:`package_results` is called this will be filled with ``None``.
        """

        self.ra_sigma: List[Optional[List[float]]] = [None] * len(self.sopnav.camera.images)
        """
        This list is used to store the formal uncertainty on the right-ascension component of the bearing in degrees for 
        each UFO.

        Until :meth:`package_results` is called this will be filled with ``None``.
        """

        self.declination_sigma: List[Optional[List[float]]] = [None] * len(self.sopnav.camera.images)
        """
        This list is used to store the formal uncertainty on the declination component of the bearing in degrees for 
        each UFO.

        Until :meth:`package_results` is called this will be filled with ``None``.
        """

        self.x_raw_sigma: List[Optional[List[float]]] = [None] * len(self.sopnav.camera.images)
        """
        This list is used to store the formal uncertainty on the x pixel location for each UFO.

        Until :meth:`package_results` is called this will be filled with ``None``.
        """

        self.y_raw_sigma: List[Optional[List[float]]] = [None] * len(self.sopnav.camera.images)
        """
        This list is used to store the formal uncertainty on the y pixel location for each UFO.

        Until :meth:`package_results` is called this will be filled with ``None``.
        """

        self.occulting: List[Optional[np.ndarray]] = [None] * len(self.sopnav.camera.images)
        """
        This list is used to store the locations where UFOs are in front of any dark portions of any extended bodies.

        Until :meth:`package_results` is called this will be filled with ``None``.
        """

        self.saturation_distance: List[Optional[np.ndarray]] = [None] * len(self.sopnav.camera.images)
        """
        This list is used to store the minimum distance between the centroid of each UFO and the nearest saturated pixel

        Until :meth:`package_results` is called this will be filled with ``None``.
        """

        self.trail_length: List[Optional[np.ndarray]] = [None] * len(self.sopnav.camera.images)
        """
        This list is used to store the approximate length of the trail of each UFO  in pixels (if the detection is 
        trailed)

        Until :meth:`package_results` is called this will be filled with ``None``.
        """

        self.trail_principal_angle: List[Optional[np.ndarray]] = [None] * len(self.sopnav.camera.images)
        """
        This list is used to store the approximate principal angle for each UFO that is trailed in degrees.
        
        The principal angle is the angle between the +x axis and the principal axis of the skewed PSF.

        Until :meth:`package_results` is called this will be filled with ``None``.
        """

        self.star_summed_dn: List[Optional[List[float]]] = [None] * len(self.sopnav.camera.images)
        """
        This list is used to store the sum of the 5x5 grid of pixel DN values minus the background around each 
        matched star.

        Until :meth:`package_results` is called this will be filled with ``None``.
        """

        self.star_observed_magnitude: List[Optional[Union[List[float], np.ndarray]]] = \
            [None] * len(self.sopnav.camera.images)
        """
        This list is used to store the computed magnitude of each matched star.

        Until :meth:`package_results` is called this will be filled with ``None``.
        """

        self.star_summed_dn_count: List[Optional[List[int]]] = [None] * len(self.sopnav.camera.images)
        """
        This list is used to store the number of pixels summed for computing the summed dn for each matched star.

        This is nearly always 25 (for a 5x5 grid) but occasionally for points near the edge it may be less.

        Until :meth:`package_results` is called this will be filled with ``None``.
        """

        self.star_max_dn: List[Optional[List[float]]] = [None] * len(self.sopnav.camera.images)
        """
        This list is used to store the maximum DN value for each matched star.

        Until :meth:`package_results` is called this will be filled with ``None``.
        """

        self.star_bearing: List[Optional[np.ndarray]] = [None] * len(self.sopnav.camera.images)
        """
        This list is used to store inertial bearing of each matched star.

        Until :meth:`package_results` is called this will be filled with ``None``.
        """

        self.star_integrated_psf: List[Optional[List[float]]] = [None] * len(self.sopnav.camera.images)
        """
        This list is used to store the integrated PSF values for each matched star.

        Until :meth:`package_results` is called this will be filled with ``None``.
        """

        self.star_ra_sigma: List[Optional[List[float]]] = [None] * len(self.sopnav.camera.images)
        """
        This list is used to store the formal uncertainty on the right-ascension component of the bearing in degrees for 
        each matched star.

        Until :meth:`package_results` is called this will be filled with ``None``.
        """

        self.star_declination_sigma: List[Optional[List[float]]] = [None] * len(self.sopnav.camera.images)
        """
        This list is used to store the formal uncertainty on the declination component of the bearing in degrees for 
        each matched star.

        Until :meth:`package_results` is called this will be filled with ``None``.
        """

        self.star_x_raw_sigma: List[Optional[List[float]]] = [None] * len(self.sopnav.camera.images)
        """
        This list is used to store the formal uncertainty on the x pixel location for each matched star.

        Until :meth:`package_results` is called this will be filled with ``None``.
        """

        self.star_y_raw_sigma: List[Optional[List[float]]] = [None] * len(self.sopnav.camera.images)
        """
        This list is used to store the formal uncertainty on the y pixel location for each matched star.

        Until :meth:`package_results` is called this will be filled with ``None``.
        """

        self.star_occulting: List[Optional[np.ndarray]] = [None] * len(self.sopnav.camera.images)
        """
        This list is used to store the locations where matched stars are in front of any dark portions of any extended 
        bodies.
        
        This obviously shouldn't be true so if any are then they may be actual detections

        Until :meth:`package_results` is called this will be filled with ``None``.
        """

        self.star_saturation_distance: List[Optional[np.ndarray]] = [None] * len(self.sopnav.camera.images)
        """
        This list is used to store the minimum distance between the centroid of each matched star and the nearest 
        saturated pixel

        Until :meth:`package_results` is called this will be filled with ``None``.
        """

        self.star_integrated_psf_uncertainty: List[Optional[List[float]]] = [None] * len(self.sopnav.camera.images)
        """
        This list is used to store the formal uncertainty of the integrated PSF value for each matched star.

        Until :meth:`package_results` is called this will be filled with ``None``.
        """

        self.star_summed_dn_uncertainty: List[Optional[List[float]]] = [None] * len(self.sopnav.camera.images)
        """
        This list is used to store the formal uncertainty of the integrated PSF value for each matched star.

        Until :meth:`package_results` is called this will be filled with ``None``.
        """

        self.star_saturated: List[Optional[List[bool]]] = [None] * len(self.sopnav.camera.images)
        """
        This list is used to store a flag specifying whether each matched star has saturated pixels or not.

        If a flag is ``True`` then the UFO did contain at least 1 pixel that was saturated.

        Until :meth:`package_results` is called this will be filled with ``None``.
        """

        self.star_fit_chi2_value: List[Optional[List[float]]] = [None] * len(self.sopnav.camera.images)
        """
        This list is used to store the Chi2 value from the post-fit residuals for each star.

        Until :meth:`package_results` is called this will be filled with ``None``.
        """

        self._delta_row, self._delta_col = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3), indexing='ij')
        """
        These specify the pixels around the center of each detection that we consider in summing by default
        """

        self._needs_processed: List[bool] = [True] * len(self.sopnav.camera.images)
        """
        A list of flags specifying whether the images need processed or not
        """

    def update_attitude(self):
        """
        This method estimates the attitude for each turned on image in the camera.

        This is done through the usual process.  First the settings for the :attr:`.StellarOpNav.star_id``, `
        :attr:`.StellarOpNav.attitude_estimator`, and :attr:`.StellarOpNav.image_processing` attributes are updated
        according to the settings saved in :attr:`update_attitude_settings`. Then the stars are identified using
        :meth:`.StellarOpNav.id_stars`.  Finally, the attitude is estimated using
        :meth:`.StellarOpNav.estimate_attitude`.  All of the results are stored into the :attr:`sopnav` attribute as
        usual.
        """

        _LOGGER.info('Updating settings for star identification')
        if self.update_attitude_settings is not None:
            self.sopnav.update_attitude_estimator(self.update_attitude_settings.attitude_estimator_options)
            self.sopnav.update_point_of_interest_finder(self.update_attitude_settings.point_of_interest_finder_options)
            self.sopnav.update_star_id(self.update_attitude_settings.star_id_options)

        # Identify stars
        _LOGGER.info('Identifying Stars...')
        # only process images that we need to
        self.sopnav.process_stars = self._needs_processed
        self.sopnav.id_stars()
        _LOGGER.info('DONE')
        _LOGGER.info('\nUpdating Attitude...')
        self.sopnav.estimate_attitude()
        _LOGGER.info('DONE')

    def find_ufos(self):
        """
        This method finds unidentified bright spots (not matching a star or an extended body) in the images turned on in
        the  camera.

        This is done by updating the :attr:`.StellarOpNav.star_id` and :attr:`.StellarOpNav.image_processing` attributes
        using the settings saved in :attr:`find_ufos_settings`.  Then :meth:`.StellarOpNav.id_stars` is called to identify
        the UFOs.  The results are stored in the ``unmatched_*`` attributes of the :attr:`sopnav` attribute.

        To get a summary of UFOs/stars with more information about them call :meth:`package_results` after calling this
        method.  Typically you should call :meth:`update_attitude` before calling this method.
        """

        _LOGGER.info('Updating settings for ufo identification')
        if self.find_ufos_settings is not None:
            self.sopnav.update_star_id(self.find_ufos_settings.star_id_options)
            self.sopnav.update_point_of_interest_finder(self.find_ufos_settings.point_of_interest_finder_options)

        # only process images that we need to
        self.sopnav.process_stars = self._needs_processed
        _LOGGER.info('Identifying UFOs...')
        self.sopnav.id_stars()
        self._needs_processed = [False]*len(self.sopnav.camera.images)

    def package_results(self):
        """
        This method packages information about UFOs (and matched stars) into attributes in this class and as a Pandas
        DataFrame that can be stored to any number of table formats.

        You should have called :meth:`find_ufos` before calling this method.

        Specifically, the results for UFOs are stored into

        - :attr:`bearing`
        - :attr:`ra_sigma`
        - :attr:`declination_sigma`
        - :attr:`fit_chi2_value`
        - :attr:`integrated_psf`
        - :attr:`integrated_psf_uncertainty`
        - :attr:`invalid_images`
        - :attr:`magnitude`
        - :attr:`max_dn`
        - :attr:`occulting`
        - :attr:`saturated`
        - :attr:`saturation_distance`
        - :attr:`summed_dn`
        - :attr:`summed_dn_count`
        - :attr:`summed_dn_uncertainty`
        - :attr:`trail_length`
        - :attr:`trail_principal_angle`
        - :attr:`x_raw_sigma`
        - :attr:`y_raw_sigma`
        - :attr: `.StellarOpNav.unmatched_extracted_image_points`
        - :attr: `.StellarOpNav.unmatched_stats`
        - :attr: `.StellarOpNav.unmatched_psfs`
        - :attr: `.StellarOpNav.unmatched_snrs`
        - :attr: `.StellarOpNav.unmatched_snrs`

        In addition, the results for matched stars are stored into
        - :attr:`star_bearing`
        - :attr:`star_ra_sigma`
        - :attr:`star_declination_sigma`
        - :attr:`star_fit_chi2_value`
        - :attr:`star_integrated_psf`
        - :attr:`star_integrated_psf_uncertainty`
        - :attr:`invalid_images`
        - :attr:`star_observed_magnitude`
        - :attr:`star_max_dn`
        - :attr:`star_occulting`
        - :attr:`star_saturated`
        - :attr:`star_saturation_distance`
        - :attr:`star_summed_dn`
        - :attr:`star_summed_dn_count`
        - :attr:`star_summed_dn_uncertainty`
        - :attr:`star_x_raw_sigma`
        - :attr:`star_y_raw_sigma`
        - :attr: `.StellarOpNav.matched_extracted_image_points`
        - :attr: `.StellarOpNav.matched_stats`
        - :attr: `.StellarOpNav.matched_psfs`
        - :attr: `.StellarOpNav.matched_snrs`
        - :attr: `.StellarOpNav.matched_snrs`

        Both star and UFO results are stored into a dataframe at :attr:`detection_data_frame` with Columns of

        ===================== ==========================================================================================
        Column                Description
        ===================== ==========================================================================================
        image_file            The file name of the image the detection came from as a string
        mid_exposure_utc      The mid exposure time in UTC as a datetime
        mid_exposure_et       The mid exposure time in ET seconds since J2000 as a float
        x_raw                 The x-sub-pixel (column) location of the centroid of the detection as a float
        y_raw                 The y-sub-pixel (column) location of the centroid of the detection as a float
        x_raw_sigma           The x-sub-pixel location of the centroid of the detection formal uncertainty in pixels as
                              a float
        y_raw_sigma           The y-sub-pixel location of the centroid of the detection formal uncertainty in pixels as
                              a float
        ra                    The right ascension of the detection in the inertial frame in degrees as a float.  Note
                              that this is the direction from the camera to the detection in inertial space, **not** the
                              location of the detection in the celestial sphere.
        dec                   The declination of the detection in the inertial frame in degrees as a float.  Note
                              that this is the direction from the camera to the detection in inertial space, **not** the
                              location of the detection in the celestial sphere.
        ra_sigma              The formal uncertainty on the RA of the detection in units of degrees as a float
        dec_sigma             The formal uncertainty on the declination of the detection in units of degrees as a float
        area                  The area of the detection (number of pixels above the specified threshold) as an integer
        peak_dn               The maxim DN value of the detection (minus the background term) as a float
        summed_dn             The sum of the (normally) 5x5 grid of pixels surrounding the centroid of the detection in
                              DN as a float
        summed_dn_sigma       The formal uncertainty of the summed dn value in DN as a float.  This is based off of the
                              expected noise of the pixels where the detection was found at.
        n_pix_summed          The number of pixels summed as an integer. This will nearly always be 25, unless the
                              detection was very close to the edge of the image
        integrated_psf        The value of the integrated fit PSF for the detection in DN as a float.
        integrated_psf_sigma  The formal uncertainty of the integrated fit PSF for the detection in DN as a float
        magnitude             The computed apparent magnitude of the detector or 0, if no :attr:`magnitude_function` was
                              given
        snr                   The signal to noise ratio of the detection
        psf                   The fit point spread function for the detection as a string
        psf_fit_quality       The quality of the PSF fit as a chi**2 parameter as a float
        occulting             A boolean flag specifying whether this detection is between the camera and the dark region
                              of an extended body
        saturation_distance   The distance between the centroid of this detection and the nearest blob of pixels with at
                              least 3 pixels saturated
        is_saturated          A boolean flag specifying if any of the pixels in the detection are saturated
        trail_length          The length of the trail of the detection if it is a trailed detection (or 0) in units of
                              pixels as a float
        trail_principal_angle The angle between the trail semi-major axis and the direction of increasing right
                              ascension in the image in units of degrees as a float
        quality_code          The quality code of the detection.  Attempts to give the detection a quality label of 1-5
                              with 5 being a detection in which there is strong confidence it is not just a noise spike.
                              quality codes of 0 indicate detections paired to known stars.  Quality codes of -1
                              indicate detections that may be hot pixels or un-matched stars.  Quality codes of -1 will
                              only be present after a call to :meth:`identify_hot_pixels_and_unmatched_stars`.
        x_inert2cam           The x component of the quaternion that rotates from the inertial frame to the camera frame
                              at the time of the detection as a float
        y_inert2cam           The y component of the quaternion that rotates from the inertial frame to the camera frame
                              at the time of the detection as a float
        z_inert2cam           The z component of the quaternion that rotates from the inertial frame to the camera frame
                              at the time of the detection as a float
        s_inert2cam           The scalar component of the quaternion that rotates from the inertial frame to the camera
                              frame at the time of the detection as a float
        star_id               A string given the catalog ID of the star this detection was matched to (if it was
                              matched to a star).
        ===================== ==========================================================================================
        """

        # create a counter of how many images we've processed
        processed_images = 1

        # loop through each image
        for ind, image in self.sopnav.camera:

            # get a timer
            start = time.time()
            _LOGGER.info(f'Analyzing results for image {ind+1} of {sum(self.sopnav.camera.image_mask)}')

            if self._needs_processed[ind]:
                _LOGGER.warning(f'Image {ind}, {image.observation_date.isoformat()} needs to be processed still')
                continue

            # update the scene to this time if it is available
            if self.scene is not None:
                self.scene.update(image)

            # get the information we need about the detector (if it is available)
            if self.image_information_function is not None:
                electrons_to_dn, quantization_noise, read_noise, dark_current = self.image_information_function(image)
            else:
                electrons_to_dn = 1.0
                quantization_noise = 0.0
                read_noise = 0.0
                dark_current = 0.0

            # determine the bright spots in the image as points > 98% saturated
            bright = image > (image.saturation * 0.98)

            # clump bright spots into individual blobs with connected components
            _, __, stats, ___ = cv2.connectedComponentsWithStats(bright.astype(np.uint8))

            for stat in stats:
                # filter out areas where the number of saturated pixels is less than 3
                if stat[cv2.CC_STAT_AREA] < 3:
                    bright[stat[cv2.CC_STAT_TOP]:(stat[cv2.CC_STAT_HEIGHT] + stat[cv2.CC_STAT_TOP]),
                           stat[cv2.CC_STAT_LEFT]:(stat[cv2.CC_STAT_LEFT] + stat[cv2.CC_STAT_WIDTH])] = False

            # if there aren't any bright spots in the image then this image is invalid and likely corrupted, skip it
            if not bright.any():
                _LOGGER.warning('invalid image, skipping')
                self.invalid_images[ind] = True
                continue

            # get the coordinates of the bright spots in the image (where bright is true)
            bright_coords: np.ndarray = np.transpose(bright.nonzero()[::-1]).astype(np.float64)

            # build the catalog of bright spots
            # noinspection PyArgumentList
            bright_tree = KDTree(bright_coords)
            ufo_points = self.sopnav.unmatched_extracted_image_points[ind]
            if ufo_points is None:
                continue

            if self.scene is not None:
                # determine which detections are actually due to the surface of any extended targets in the scene
                # compute the rays to trace through the scene, one for each unmatched point
                starts = np.zeros(3, dtype=np.float64)
                

                directions = self.sopnav.camera.model.pixels_to_unit(ufo_points, temperature=image.temperature)

                rays = Rays(starts, directions)

                # do a single bounce ray trace from the camera to the body and then to the sun
                illums_inp, inter = self.scene.get_illumination_inputs(rays, return_intersects=True)

                # identify things that are in line with the illuminated portion of the extended targets and throw them
                # out
                # anything that has visible set to true means that the object is on the illuminated portion of the
                # target (at least, where we think the illuminated portion of the targets are)
                test: np.ndarray = illums_inp['visible']

                # Now check to see where we intersected the body, but didn't make it to the sun.  These are points on
                # the dark side of the targets (occulting).  We are going to throw away the points on the bright side so
                # we can use ~test here to store only the ones we'll keep
                self.occulting[ind] = inter[~test]['check']

                # throw out the points that are due to the illuminated targets
                self.sopnav.unmatched_extracted_image_points[ind] = ufo_points[:, ~test]
                ufo_points = ufo_points[:, ~test]
                self.sopnav.unmatched_psfs[ind] = boolean_filter_list(self.sopnav.unmatched_psfs[ind], ~test) # type: ignore
                self.sopnav.unmatched_snrs[ind] = self.sopnav.unmatched_snrs[ind][~test] # type: ignore
                self.sopnav.unmatched_stats[ind] = self.sopnav.unmatched_stats[ind][~test] # type: ignore
            else:
                self.occulting[ind] = np.zeros(ufo_points.shape[1], dtype=bool)

            # determine the distance from each remaining point to the nearest saturated pixel in the image
            self.saturation_distance[ind], _ = bright_tree.query(ufo_points.T)

            # get the inertial unit vector from the camera through the detection
            inertial_vecs = image.rotation_inertial_to_camera.matrix.T @ self.sopnav.camera.model.pixels_to_unit(
                ufo_points, temperature=image.temperature
            )

            # compute the right ascension and declination of the unit vectors in degrees
            current_bearing = np.array(unit_to_radec(inertial_vecs)) * RAD2DEG
            self.bearing[ind] = current_bearing
            
            ufo_psfs = self.sopnav.unmatched_psfs[ind]
            assert ufo_psfs is not None
            ufo_snrs = self.sopnav.unmatched_snrs[ind]
            assert ufo_snrs is not None
            ufo_stats = self.sopnav.unmatched_stats[ind]
            assert ufo_stats is not None

            # prepare some storage lists
            current_summed_dn = []
            self.summed_dn[ind] = current_summed_dn
            current_summed_dn_count = []
            self.summed_dn_count[ind] = current_summed_dn_count
            current_max_dn = []
            self.max_dn[ind] = current_max_dn
            current_fit_chi2_value = [] 
            self.fit_chi2_value[ind] = current_fit_chi2_value
            current_integrated_psf_uncertainty = []
            self.integrated_psf_uncertainty[ind] = current_integrated_psf_uncertainty
            current_summed_dn_uncertainty = []
            self.summed_dn_uncertainty[ind] = current_summed_dn_uncertainty
            current_saturated = []
            self.saturated[ind] = current_saturated
            current_ra_sigma = []
            current_declination_sigma = []
            self.ra_sigma[ind], self.declination_sigma[ind] = current_ra_sigma, current_declination_sigma
            current_x_raw_sigma = []
            current_y_raw_sigma = []
            self.x_raw_sigma[ind], self.y_raw_sigma[ind] = current_x_raw_sigma, current_y_raw_sigma
            current_trail_length = np.zeros(len(ufo_psfs), dtype=np.float64)
            self.trail_length[ind] = current_trail_length
            current_trail_principal_angle = np.zeros(len(ufo_psfs), dtype=np.float64)
            self.trail_principal_angle[ind] = current_trail_principal_angle

            # compute the integrated psf DN for each detection
            self.integrated_psf[ind] = [psf.volume() for psf in ufo_psfs]

            # compute the jacobian matrix of the unit vector in the camera frame with respect to a change in the
            # pixel location
            jacobian_pixels_to_unit = self.sopnav.camera.model.compute_unit_vector_jacobian(
                ufo_points, temperature=image.temperature
            )

            # compute the jacobian matrix of the right ascension and declination with respect to a change in the
            # unit vector
            jacobian_unit_to_bearing = unit_to_radec_jacobian(inertial_vecs)

            # compute photometry for each detection
            # loop through the unmatched points and their psfs
            iterator: Iterator[Tuple[np.ndarray, IterativeGeneralizedGaussianWBackground]] = zip(
                ufo_points,
                ufo_psfs
            ) # type: ignore
            for lind, (poi, psf) in enumerate(iterator):

                # get the indices into the image around the detection, checking that we're not too close to an edge
                rows: np.ndarray = np.round(poi[1] + self._delta_row).astype(int)
                cols: np.ndarray = np.round(poi[0] + self._delta_col).astype(int)

                check = ((rows >= 0) & (cols >= 0) &
                         (rows < self.sopnav.camera.model.n_rows) & (cols < self.sopnav.camera.model.n_cols))

                rows = rows[check]
                cols = cols[check]

                # check to see if the pixels are saturated so we can set the flag
                dns = image[rows, cols].astype(np.float64)
                current_saturated.append(dns.max() >= 0.98 * image.saturation)

                # subtract off the estimated background from the DN values
                bg = psf.evaluate_bg(cols, rows)
                dns -= bg

                # sum the dn, determine the number of pixels included in the sum, and get the max dn in the sub-image
                current_summed_dn.append(dns.sum())
                current_summed_dn_count.append(check.sum())
                current_max_dn.append(dns.max())

                # compute the average stray light to be the background minus the detector dn_offset at the center
                # of the detection
                avg_stray_light = psf.evaluate_bg(*psf.centroid) - self.dn_offset

                # compute the noise level for the summed DN
                # compute the square of the shot noise in electrons
                sigma_shot2 = dns / electrons_to_dn

                # compute the sum of the squares of the quantization noise, the read noise,
                # the noise due to the stray light, and the dark current in electrons
                extra_noise2 = (quantization_noise ** 2 + read_noise ** 2 + avg_stray_light / electrons_to_dn +
                                dark_current)

                # compute the sum of the squares of the noise terms in electrons
                noise2 = sigma_shot2 + extra_noise2

                # get the total noise in the sub-image in units of DN
                noise = np.sqrt(noise2.sum()) * electrons_to_dn

                # compute and store the signal to noise ratio for the detection
                ufo_snrs[lind] = current_summed_dn[ind][lind] / noise

                # compute and store the noise level for the summed DN term in units of DN
                current_summed_dn_uncertainty.append(np.sqrt(noise ** 2))

                # compute the average noise per each pixel squared in units of DN
                pix_noise_avg2 = np.mean(noise2) * electrons_to_dn ** 2

                # make the weighted covariance matrix for the estimated point spread function by multiplying by the
                # average noise per pixel squared
                psf_cov = psf.covariance / psf.residual_std**2 * pix_noise_avg2

                # store the 1 sigma uncertainty in the estimated subpixel center
                current_x_raw_sigma.append(np.sqrt(psf_cov[0, 0]))
                current_y_raw_sigma.append(np.sqrt(psf_cov[1, 1]))

                # compute the jacobian of the integrated point spread function with respect to a change in the
                # estimated point spread function.  Do this with finite differencing
                jacobian_integ_wrt_psf = np.zeros((1, psf_cov.shape[0]), dtype=np.float64)
                for perturbation_axis in range(psf_cov.shape[0]):
                    pert_vec = np.zeros(psf_cov.shape[0])
                    psf_pert = deepcopy(psf)
                    pert_vec[perturbation_axis] = 1e-6
                    psf_pert.update_state(pert_vec)
                    positive_integ = psf_pert.volume()
                    pert_vec = np.zeros(psf_cov.shape[0])
                    psf_pert = deepcopy(psf)
                    pert_vec[perturbation_axis] = -1e-6
                    psf_pert.update_state(pert_vec)
                    negative_integ = psf_pert.volume()

                    jacobian_integ_wrt_psf[0, perturbation_axis] = (positive_integ - negative_integ)/(2*1e-6)

                # compute the variance on the integrated point spread function value in dn
                integ_cov = jacobian_integ_wrt_psf @ psf_cov @ jacobian_integ_wrt_psf.T

                # compute and store the uncertainty on the integrated point spread function in units of DN
                current_integrated_psf_uncertainty.append(np.sqrt(integ_cov))

                # finalize the chi^2 value for the point spread function fit quality by dividing by the DN uncertainty
                # in each pixel ignoring shot noise
                current_fit_chi2_value.append(psf.residual_rss/np.sqrt(extra_noise2))

                # compute the trail length and pa if extended psf
                # check if the semi-major axis is 3 times bigger than the semi-minor axis.
                # If so this is probably a streaked detection
                if psf.sigma_x / psf.sigma_y > 3:
                    # the trail length is two times the semi-major axis (roughly)
                    current_trail_length[lind] = 2 * psf.sigma_x

                    # determine the direction of the trail in ra/dec space
                    # determine the direction of increasing right ascension in the image
                    ra_dir: np.ndarray = self.sopnav.camera.model.project_onto_image(
                        image.rotation_inertial_to_camera.matrix @
                        radec_to_unit(*((current_bearing[lind] + [0.02, 0]) / RAD2DEG)),
                        temperature=image.temperature
                    ) - ufo_points[:, lind]

                    ra_dir /= np.linalg.norm(ra_dir)
                    # determine the direction of increasing declination in the image
                    dec_dir: np.ndarray = self.sopnav.camera.model.project_onto_image(
                        image.rotation_inertial_to_camera.matrix @
                        radec_to_unit(*((current_bearing[lind] + [0, 0.02]) / RAD2DEG)),
                        temperature=image.temperature
                    ) - ufo_points[:, lind]

                    dec_dir /= np.linalg.norm(dec_dir)

                    # determine the principal axis line for the psf
                    pa_line = np.array([np.cos(psf.theta), np.sin(psf.theta)])

                    # angle between pa_line and dec_dir is the trail orientation
                    pa_theta = np.arccos(dec_dir @ pa_line) * RAD2DEG

                    if pa_line @ ra_dir < 0:
                        pa_theta += 180

                    # store the trail orientation
                    current_trail_principal_angle[lind] = pa_theta

                # transform the covariance of the estimated subpixel center into the unit vector covariance
                cov_unit = (image.rotation_inertial_to_camera.matrix.T @
                            jacobian_pixels_to_unit[lind] @
                            psf_cov[:2, :2] @
                            jacobian_pixels_to_unit[lind].T @
                            image.rotation_inertial_to_camera.matrix)

                # transform the covariance into the bearing covariance
                cov_rad = jacobian_unit_to_bearing[lind] @ cov_unit @ jacobian_unit_to_bearing[lind].T

                # extract the sigma values for the right ascension and declination from the covariance matrix
                current_ra_sigma.append(np.sqrt(cov_rad[0, 0]))
                current_declination_sigma.append(np.sqrt(cov_rad[1, 1]))

            # compute the rough magnitude of the detection using the summed DN if we were provided a magnitude function
            if self.magnitude_function is not None:
                self.magnitude[ind] = self.magnitude_function(current_summed_dn, image)
            else:
                self.magnitude[ind] = np.zeros(len(current_summed_dn), dtype=np.float64)

            # now do all this again for the stars...
            star_points = self.sopnav.matched_catalog_image_points[ind]
            if star_points is not None:
                star_psfs = self.sopnav.matched_psfs[ind]
                star_snrs = self.sopnav.matched_snrs[ind]
                star_stats = self.sopnav.matched_stats[ind]
                star_records = self.sopnav.matched_catalog_star_records[ind]
                assert star_psfs is not None and star_snrs is not None and star_stats is not None and star_records is not None
                if self.scene is not None:
                    # determine which detections are actually due to the surface of any extended targets in the scene
                    # compute the rays to trace through the scene, one for each unmatched point
                    starts = np.zeros(3, dtype=np.float64)

                    directions = self.sopnav.camera.model.pixels_to_unit(
                        ufo_points, temperature=image.temperature
                    )

                    rays = Rays(starts, directions)

                    # do a single bounce ray trace from the camera to the body and then to the sun
                    illums_inp, inter = self.scene.get_illumination_inputs(rays, return_intersects=True)

                    # identify things that are in line with the illuminated portion of the extended targets and throw them
                    # out
                    # anything that has visible set to true means that the object is on the illuminated portion of the
                    # target (at least, where we think the illuminated portion of the targets are)
                    test = illums_inp['visible']

                    # Now check to see where we intersected the body, but didn't make it to the sun.  These are points on
                    # the dark side of the targets (occulting).  We are going to throw away the points on the bright side so
                    # we can use ~test here to store only the ones we'll keep
                    self.star_occulting[ind] = inter[~test]['check']

                    # throw out the points that are due to teh illuminated targets
                    star_points = star_points[:, ~test]
                    self.sopnav.matched_extracted_image_points[ind] = star_points
                    star_psfs = boolean_filter_list(star_psfs, ~test)
                    self.sopnav.matched_psfs[ind] = star_psfs
                    star_snrs = star_snrs[~test]
                    self.sopnav.matched_snrs[ind] = star_snrs
                    star_stats = star_stats[~test]
                    self.sopnav.matched_stats[ind] = star_stats
                    star_records = star_records.loc[~test]
                    self.sopnav.matched_catalog_star_records[ind] = star_records
                else:
                    self.star_occulting[ind] = np.zeros(star_points.shape[1], dtype=bool)

                # determine the distance from each remaining point to the nearest saturated pixel in the image
                self.star_saturation_distance[ind], _ = bright_tree.query(star_points.T)

                # self.sopnav.matched_rss[ind][:, 0] /= noise

                # get the inertial unit vector from the self.sopnav.camera through the detection
                inertial_vecs = image.rotation_inertial_to_camera.matrix.T @ self.sopnav.camera.model.pixels_to_unit(
                    star_points, temperature=image.temperature
                )

                # compute the right ascension and declination of the unit vectors in degrees
                self.star_bearing[ind] = np.array(unit_to_radec(inertial_vecs)) * RAD2DEG

                # prepare some storage lists
                current_star_summed_dn = []
                self.star_summed_dn[ind] = current_star_summed_dn
                current_star_summed_dn_count = []
                self.star_summed_dn_count[ind] = current_star_summed_dn_count
                current_star_max_dn = []
                self.star_max_dn[ind] = current_star_max_dn
                current_star_fit_chi2_value = []
                self.star_fit_chi2_value[ind] = current_star_fit_chi2_value
                current_star_integrated_psf_uncertainty = []
                self.star_integrated_psf_uncertainty[ind] = current_star_integrated_psf_uncertainty
                current_star_summed_dn_uncertainty = []
                self.star_summed_dn_uncertainty[ind] = current_star_summed_dn_uncertainty
                current_star_saturated = []
                self.star_saturated[ind] = current_star_saturated
                current_star_x_raw_sigma, current_star_y_raw_sigma = [], []
                self.star_x_raw_sigma[ind], self.star_y_raw_sigma[ind] = current_star_x_raw_sigma, current_star_y_raw_sigma
                current_star_ra_sigma, current_star_declination_sigma = [], []
                self.star_ra_sigma[ind], self.star_declination_sigma[ind] = current_star_ra_sigma, current_star_declination_sigma

                # compute the integrated psf DN for each detection
                self.star_integrated_psf[ind] = [psf.volume() for psf in star_psfs]

                # compute the jacobian matrix of the unit vector in the camera frame with respect to a change in the
                # pixel location
                jacobian_pixels_to_unit = self.sopnav.camera.model.compute_unit_vector_jacobian(
                    star_points, temperature=image.temperature
                )

                # compute the jacobian matrix of the right ascension and declination with respect to a change in the
                # unit vector
                jacobian_unit_to_bearing = unit_to_radec_jacobian(inertial_vecs)

                # compute photometry for each detection
                # loop through the matched points and their psfs
                iterator: Iterator[Tuple[np.ndarray, IterativeGeneralizedGaussianWBackground]] = zip(
                    star_points,
                    star_psfs
                ) # type: ignore
                for lind, (poi, psf) in enumerate(iterator):

                    # get the indices into the image around the detection, checking that we're not too close to an edge
                    rows = np.round(poi[1] + self._delta_row).astype(int)
                    cols = np.round(poi[0] + self._delta_col).astype(int)

                    check = ((rows >= 0) & (cols >= 0) &
                            (rows < self.sopnav.camera.model.n_rows) & (cols < self.sopnav.camera.model.n_cols))

                    rows = rows[check]
                    cols = cols[check]

                    # check to see if the pixels are saturated so we can set the flag
                    dns = image[rows, cols].astype(np.float64)
                    current_star_saturated[ind].append(dns.max() >= 0.98 * image.saturation)

                    # subtract off the estimated background from the DN values
                    bg = psf.evaluate_bg(cols, rows)
                    dns -= bg

                    # sum the dn, determine the number of pixels included in the sum, and get the max dn in the sub-window
                    current_star_summed_dn.append(dns.sum())
                    current_star_summed_dn_count.append(check.sum())
                    current_star_max_dn.append(dns.max())

                    # compute the average stray light to be the background minus the detector dn_offset at the center
                    # of the detection
                    avg_stray_light = psf.evaluate_bg(*psf.centroid) - self.dn_offset

                    # compute the noise level for the summed DN
                    # compute the square of the shot noise in electrons
                    sigma_shot2 = (dns / electrons_to_dn)

                    # compute the sum of the squares of the quantization noise, the read noise,
                    # the noise due to the stray light, and the dark current in electrons
                    extra_noise2 = (quantization_noise ** 2 + read_noise ** 2 + avg_stray_light / electrons_to_dn +
                                    dark_current)

                    # compute the sum of the squares of the noise terms in electrons
                    noise2 = sigma_shot2 + extra_noise2

                    # get the total noise in the sub-window in units of DN
                    noise = np.sqrt(noise2.sum()) * electrons_to_dn

                    # compute and store the signal to noise ratio for the detection
                    star_snrs[lind] = current_star_summed_dn[lind] / noise

                    # compute and store the noise level for the summed DN term in units of DN
                    current_star_summed_dn_uncertainty.append(np.sqrt(noise ** 2))

                    # compute the average noise per each pixel squared in units of DN
                    pix_noise_avg2 = np.mean(noise2) * electrons_to_dn ** 2

                    # make the weighted covariance matrix for the estimated point spread function by multiplying by the
                    # average noise per pixel squared
                    psf_cov = psf.covariance / psf.residual_std**2 * pix_noise_avg2

                    # store the 1 sigma uncertainty in the estimated subpixel center
                    current_star_x_raw_sigma.append(np.sqrt(psf_cov[0, 0]))
                    current_star_y_raw_sigma.append(np.sqrt(psf_cov[1, 1]))

                    # compute the jacobian of the integrated point spread function with respect to a change in the
                    # estimated point spread function.  Do this with finite differencing
                    jacobian_integ_wrt_psf = np.zeros((1, psf_cov.shape[0]), dtype=np.float64)
                    for perturbation_axis in range(psf_cov.shape[0]):
                        pert_vec = np.zeros(psf_cov.shape[0])
                        psf_pert = deepcopy(psf)
                        pert_vec[perturbation_axis] = 1e-6
                        psf_pert.update_state(pert_vec)
                        positive_integ = psf_pert.volume()
                        pert_vec = np.zeros(psf_cov.shape[0])
                        psf_pert = deepcopy(psf)
                        pert_vec[perturbation_axis] = -1e-6
                        psf_pert.update_state(pert_vec)
                        negative_integ = psf_pert.volume()

                        jacobian_integ_wrt_psf[0, perturbation_axis] = (positive_integ - negative_integ)/(2*1e-6)

                    # compute the variance on the integrated point spread function value in dn
                    integ_cov = jacobian_integ_wrt_psf @ psf_cov @ jacobian_integ_wrt_psf.T

                    # compute and store the uncertainty on the integrated point spread function in units of DN
                    current_star_integrated_psf_uncertainty.append(np.sqrt(integ_cov))

                    # finalize the chi^2 value for the point spread function fit quality by dividing by the DN uncertainty
                    # in each pixel ignoring shot noise
                    current_star_fit_chi2_value.append(psf.residual_rss/np.sqrt(extra_noise2))

                    # transform the covariance of the estimated subpixel center into the unit vector covariance
                    cov_unit = (image.rotation_inertial_to_camera.matrix.T @
                                jacobian_pixels_to_unit[lind] @
                                psf_cov[:2, :2] @
                                jacobian_pixels_to_unit[lind].T @
                                image.rotation_inertial_to_camera.matrix)

                    # transform the covariance into the bearing covariance
                    cov_rad = jacobian_unit_to_bearing[lind] @ cov_unit @ jacobian_unit_to_bearing[lind].T

                    # extract the sigma values
                    current_star_ra_sigma.append(np.sqrt(cov_rad[0, 0]))
                    current_star_declination_sigma.append(np.sqrt(cov_rad[1, 1]))

                # compute the rough magnitude of the detection using the summed DN if we were provided a magnitude function
                if self.magnitude_function is not None:
                    self.star_observed_magnitude[ind] = self.magnitude_function(current_star_summed_dn, image)
                else:
                    self.star_observed_magnitude[ind] = np.zeros(len(current_star_summed_dn), dtype=np.float64)

            _LOGGER.info(
                'image {} of {} analyzed in {:.3f} seconds'.format(processed_images,
                                                                   sum(self.sopnav.camera.image_mask),
                                                                   time.time() - start)
            )

            processed_images += 1

        # print out the filtered results
        self.sopnav.sid_summary()

        # list to store all the tuples
        data = []
        for ind, image in self.sopnav.camera:
            # if this is an invalid image skip it
            if self.invalid_images[ind]:
                continue

            # extract the filename from the image data
            assert isinstance(image.file, PATH)
            filename = os.path.splitext(os.path.basename(image.file))[0]

            # get the image utc time
            date_utc = image.observation_date.isoformat()

            # get the image et
            date_et = datetime_to_et(image.observation_date)

            # get the rotation quaternion from inertial to the camera
            rotation_quat = image.rotation_inertial_to_camera.quaternion

            if self.create_hashed_index:
                max_length = int(np.log10(max(self.sopnav.camera.model.n_rows, self.sopnav.camera.model.n_cols)))
                max_length += 5
                hash_format = f'{{}}_{{:0{max_length}.2f}}_{{:0{max_length}.2f}}'

            else:
                hash_format = ''
            
            if self.sopnav.unmatched_extracted_image_points[ind] is not None:

                # make a list of all the different things we need to loop through to make it easier
                zlist = [self.sopnav.unmatched_extracted_image_points[ind].T,  # poi # type: ignore
                        self.sopnav.unmatched_stats[ind],  # stats
                        self.max_dn[ind],  # mdn
                        self.summed_dn[ind],  # sdn
                        self.summed_dn_count[ind],  # nsum
                        self.bearing[ind].T,  # rd # type: ignore
                        self.sopnav.unmatched_psfs[ind],  # psf
                        list(zip(self.x_raw_sigma[ind], self.y_raw_sigma[ind])),  # sigs # type: ignore
                        self.fit_chi2_value[ind],  # rss
                        self.sopnav.unmatched_snrs[ind],  # snr
                        self.ra_sigma[ind],  # rsig
                        self.declination_sigma[ind],  # dsig
                        self.occulting[ind],  # occ
                        self.saturation_distance[ind],  # dist
                        self.magnitude[ind],  # mg
                        self.integrated_psf[ind],  # ipsf
                        self.trail_length[ind],  # tl
                        self.trail_principal_angle[ind],  # tp
                        self.integrated_psf_uncertainty[ind],  # ipsfsig
                        self.summed_dn_uncertainty[ind],  # sdnsig
                        self.saturated[ind]]  # sat

                if not isinstance(self.sopnav.camera.psf, IterativeGeneralizedGaussianWBackground):
                    raise ValueError('Must be IterativeGeneralizedGaussianWBackground to use package results currently')

                # zip together the stuff we need to loop through and loop through it
                # noinspection SpellCheckingInspection
                for (poi, stats, mdn, sdn, nsum, rd, psf, sigs, rss, snr,
                    rsig, dsig, occ, dist, mg, ipsf, tl, tp, ipsfsig, sdnsig, sat) in zip(*zlist):

                    qcode = np.clip(np.round((np.clip(stats[cv2.CC_STAT_AREA], 1, 5) +
                                            self.sopnav.camera.psf.compare(psf)*5 +
                                            np.clip(snr, 3, 15)/3)/3), 1, 5)

                    if np.isnan(qcode):
                        qcode = 0

                    # append a tuple with the requisite information
                    if self.create_hashed_index:
                        data.append((filename, date_utc, date_et,  # image_file, mid_exposure_utc, mid_exposure_et
                                    poi[0], poi[1],  # x_raw, y_raw
                                    sigs[0], sigs[1],  # x_raw_sigma, y_raw_sigma
                                    rd[0], rd[1],  # ra, dec
                                    rsig, dsig,  # ra_sigma, dec_sigma
                                    stats[cv2.CC_STAT_AREA], mdn,  # area, peak_dn
                                    sdn, sdnsig, nsum,  # summed_dn, summed_dn_sigma, n_pix_summed
                                    ipsf, ipsfsig,  # integrated_psf, integrated_psf_sigma
                                    mg, snr,  # magnitude, snr
                                    str(psf), rss,  # psf, psf_fit_quality
                                    occ, dist, sat,  # occulting, saturation_distance, is_saturated
                                    tl, tp,  # trail_length, trail_principal_angle
                                    qcode,  # quality_code
                                    rotation_quat[0], rotation_quat[1], rotation_quat[2], rotation_quat[3],  # rotation
                                    None,  # star_id (None because these are unmatched))
                                    hash_format.format(filename, *poi)))  # hash id
                    else:
                        data.append((filename, date_utc, date_et,  # image_file, mid_exposure_utc, mid_exposure_et
                                    poi[0], poi[1],  # x_raw, y_raw
                                    sigs[0], sigs[1],  # x_raw_sigma, y_raw_sigma
                                    rd[0], rd[1],  # ra, dec
                                    rsig, dsig,  # ra_sigma, dec_sigma
                                    stats[cv2.CC_STAT_AREA], mdn,  # area, peak_dn
                                    sdn, sdnsig, nsum,  # summed_dn, summed_dn_sigma, n_pix_summed
                                    ipsf, ipsfsig,  # integrated_psf, integrated_psf_sigma
                                    mg, snr,  # magnitude, snr
                                    str(psf), rss,  # psf, psf_fit_quality
                                    occ, dist, sat,  # occulting, saturation_distance, is_saturated
                                    tl, tp,  # trail_length, trail_principal_angle
                                    qcode,  # quality_code
                                    rotation_quat[0], rotation_quat[1], rotation_quat[2], rotation_quat[3],  # rotation
                                    None))  # star_id (None because these are unmatched))

            if self.sopnav.matched_catalog_star_records[ind] is not None:
                # make a list of all the different things we need to loop through to make it easier
                cat_id = self.sopnav.matched_catalog_star_records[ind].index, # type: ignore

                # noinspection SpellCheckingInspection
                zlist = [self.sopnav.matched_extracted_image_points[ind].T,  # poi # type: ignore
                        self.sopnav.matched_stats[ind],  # stats
                        self.star_max_dn[ind],  # mdn
                        self.star_summed_dn[ind],  # sdn
                        self.star_summed_dn_count[ind],  # nsum
                        self.star_bearing[ind].T,  # rd # type: ignore
                        self.sopnav.matched_psfs[ind],  # psf
                        list(zip(self.star_x_raw_sigma[ind], self.star_y_raw_sigma[ind])),  # sigs # type: ignore
                        self.star_fit_chi2_value[ind],  # rss
                        self.sopnav.matched_snrs[ind],  # snr
                        [' '.join([str(x) for x in y]) for y in cat_id],  # label, this is the star id value
                        self.star_ra_sigma[ind],  # rsig
                        self.star_declination_sigma[ind],  # dsig
                        self.star_occulting[ind],  # occ
                        self.star_saturation_distance[ind],  # dist
                        self.star_observed_magnitude[ind],  # mg
                        self.star_integrated_psf[ind],  # ipsf
                        self.star_integrated_psf_uncertainty[ind],  # ipsfsig
                        self.star_summed_dn_uncertainty[ind],  # sdnsig
                        self.star_saturated[ind]]  # sat

                # zip together the stuff we need to loop through and loop through it
                # noinspection SpellCheckingInspection
                for (poi, stats, mdn, sdn, nsum, rd, psf, sigs, rss, snr,
                    label, rsig, dsig, occ, dist, mg, ipsf, ipsfsig, sdnsig, sat) in zip(*zlist):

                    if self.create_hashed_index:
                        data.append((filename, date_utc, date_et,  # image_file, mid_exposure_utc, mid_exposure_et
                                    poi[0], poi[1],  # x_raw, y_raw
                                    sigs[0], sigs[1],  # x_raw_sigma, y_raw_sigma
                                    rd[0], rd[1],  # ra, dec
                                    rsig, dsig,  # ra_sigma, dec_sigma
                                    stats[cv2.CC_STAT_AREA], mdn,  # area peak_dn
                                    sdn, sdnsig, nsum,  # summed_dn, summed_dn_sigma, n_pix_summed
                                    ipsf, ipsfsig,  # integrated_psf, integrated_psf_sigma
                                    mg, snr,  # magnitude, snr
                                    str(psf), rss,  # psf, psf_fit_quality
                                    occ, dist, sat,  # occulting, saturation_distance, is_saturated
                                    0, 0, 0,  # trail_length, trail_principal_angle, quality_code
                                    rotation_quat[0], rotation_quat[1], rotation_quat[2], rotation_quat[3],  # rotation
                                    label,  # star_id
                                    hash_format.format(filename, *poi)))  # hash id
                    else:
                        data.append((filename, date_utc, date_et,  # image_file, mid_exposure_utc, mid_exposure_et
                                    poi[0], poi[1],  # x_raw, y_raw
                                    sigs[0], sigs[1],  # x_raw_sigma, y_raw_sigma
                                    rd[0], rd[1],  # ra, dec
                                    rsig, dsig,  # ra_sigma, dec_sigma
                                    stats[cv2.CC_STAT_AREA], mdn,  # area peak_dn
                                    sdn, sdnsig, nsum,  # summed_dn, summed_dn_sigma, n_pix_summed
                                    ipsf, ipsfsig,  # integrated_psf, integrated_psf_sigma
                                    mg, snr,  # magnitude, snr
                                    str(psf), rss,  # psf, psf_fit_quality
                                    occ, dist, sat,  # occulting, saturation_distance, is_saturated
                                    0, 0, 0,  # trail_length, trail_principal_angle, quality_code
                                    rotation_quat[0], rotation_quat[1], rotation_quat[2], rotation_quat[3],  # rotation
                                    label))  # star_id

        # combine everything into a structured array
        if self.create_hashed_index:
            data = np.array(data, dtype=np.dtype(list(zip(['image_file', 'mid_exposure_utc', 'mid_exposure_et',
                                                           'x_raw', 'y_raw',
                                                           'x_raw_sigma', 'y_raw_sigma',
                                                           'ra', 'dec',
                                                           'ra_sigma', 'dec_sigma',
                                                           'area', 'peak_dn',
                                                           'summed_dn', 'summed_dn_sigma', 'n_pix_summed',
                                                           'integrated_psf', 'integrated_psf_sigma',
                                                           'magnitude', 'snr',
                                                           'psf', 'psf_fit_quality',
                                                           'occulting', 'saturation_distance', 'is_saturated',
                                                           'trail_length', 'trail_principal_angle',
                                                           'quality_code',
                                                           'x_inert2cam', 'y_inert2cam', 'z_inert2cam', 's_inert2cam',
                                                           'star_id', 'hash_id'],
                                                          (object, object, np.float64,
                                                           np.float64, np.float64,
                                                           np.float64, np.float64,
                                                           np.float64, np.float64,
                                                           np.float64, np.float64,
                                                           int, np.float64,
                                                           np.float64, np.float64, int,
                                                           np.float64, np.float64,
                                                           np.float64, np.float64,
                                                           object, np.float64,
                                                           bool, np.float64, bool,
                                                           np.float64, np.float64,
                                                           int,
                                                           np.float64, np.float64, np.float64, np.float64,
                                                           object, object)))))

            # make the dataframe and store it
            self.detection_data_frame = pd.DataFrame(data)

            self.detection_data_frame.set_index('hash_id', inplace=True)

        else:
            data = np.array(data, dtype=np.dtype(list(zip(['image_file', 'mid_exposure_utc', 'mid_exposure_et',
                                                           'x_raw', 'y_raw',
                                                           'x_raw_sigma', 'y_raw_sigma',
                                                           'ra', 'dec',
                                                           'ra_sigma', 'dec_sigma',
                                                           'area', 'peak_dn',
                                                           'summed_dn', 'summed_dn_sigma', 'n_pix_summed',
                                                           'integrated_psf', 'integrated_psf_sigma',
                                                           'magnitude', 'snr',
                                                           'psf', 'psf_fit_quality',
                                                           'occulting', 'saturation_distance', 'is_saturated',
                                                           'trail_length', 'trail_principal_angle',
                                                           'quality_code',
                                                           'x_inert2cam', 'y_inert2cam', 'z_inert2cam', 's_inert2cam',
                                                           'star_id'],
                                                          (object, object, np.float64,
                                                           np.float64, np.float64,
                                                           np.float64, np.float64,
                                                           np.float64, np.float64,
                                                           np.float64, np.float64,
                                                           int, np.float64,
                                                           np.float64, np.float64, int,
                                                           np.float64, np.float64,
                                                           np.float64, np.float64,
                                                           object, np.float64,
                                                           bool, np.float64, bool,
                                                           np.float64, np.float64,
                                                           int,
                                                           np.float64, np.float64, np.float64, np.float64,
                                                           object)))))

            # make the dataframe and store it
            self.detection_data_frame = pd.DataFrame(data)

    def identify_hot_pixels_and_unmatched_stars(self):
        """
        This method is used to attempt to autonomously identify detections due to consistent hot-pixels (where the same
        pixel is very bring in many images) or do to an unmatched star (where the same inertial direction is observed in
        multiple images).

        This method should only be used after a call to :meth:`package_results` as it works on the
        :attr:`detection_data_frame`.  Detections labeled as possible stars or hot pixels will be given a quality_code
        of -1.

        Hot pixels are identified by searching for detections within 2 pixels of each other in the images that occur in
        multiple images (so the same x_raw, y_raw value in multiple images).  If a x_raw, y_raw pair occurs in at least
        :attr:`hot_pixel_threshold` times then it will be labeled as a possible hot pixel.

        Unmatched stars are labeled by searching for detections with the same right ascension/declination (within
        2*IFOV of the detector) in multiple images.  If a ra, dec pair appears in more than
        :attr:`unmatched_star_threshold` times then it will be labeled as a possible star.

        Because both of these require things appearing in multiple images, these techniques are best used when there are
        a number of images that were processed together.  If you are only processing a few images then you will likely
        not have much success with this method.
        """

        # extract to a shorter name
        ufos = self.detection_data_frame

        # make sure package results has been called
        if ufos is None:
            raise ValueError('Must call package_results before this method')

        # ignore known stars
        quality_code_check = ufos.quality_code >= 1

        # make groups based on the image that the detections were found in
        image_groups = ufos.loc[quality_code_check].groupby('image_file')

        # extract to a shorter name
        model = self.sopnav.camera.model

        # compute the IFOV of the detector in radians
        ifov = model.instantaneous_field_of_view()

        # make a list of kd trees to use to compare across images
        kd_trees = []

        for _, group in image_groups:
            pixel_locations: np.ndarray = group.loc[:, ["x_raw", "y_raw"]].to_numpy()
            # noinspection PyArgumentList
            kd_trees.append(KDTree(pixel_locations))

        # loop through the group again
        for first_ind, (first_file, first_group) in enumerate(image_groups):

            # initialize an array of counts for this image
            hot_pixel_counts = np.zeros(first_group.shape[0], dtype=int)
            direction_counts = np.zeros(first_group.shape[0], dtype=int)

            # loop through the other images
            for second_ind, (second_file, second_group) in enumerate(image_groups):

                if second_ind == first_ind:
                    continue

                # identify x_raw, y_raw pairs that are within 2 pixels of each other between the images
                # noinspection PyUnresolvedReferences
                pairs = kd_trees[first_ind].query_ball_tree(kd_trees[second_ind], 2)

                # add to the hot pixel count array where we found a pair within 2 pixels
                for matched_index, pair in enumerate(pairs):
                    hot_pixel_counts[matched_index] += len(pair)  # points where no matches were found are length 0

                # compute the number of shared inertial directions between the images.  Need to compute the distance in
                # ra/dec space so we can't use trees and need to brute force it unfortunately
                direction_counts += (radec_distance(first_group.ra.to_numpy().reshape(-1, 1)*DEG2RAD,
                                                    first_group.dec.to_numpy().reshape(-1, 1)*DEG2RAD,
                                                    second_group.ra.to_numpy().reshape(1, -1)*DEG2RAD,
                                                    second_group.dec.to_numpy().reshape(1, -1)*DEG2RAD) <
                                     2*ifov).sum(axis=-1)

            # make a boolean for this image/detections that were considered
            image_check = (ufos.image_file == first_file) & quality_code_check

            # update anywhere that is a possible hot pixel or unmatched star with a quality_code of -1
            image_check[image_check] = ((hot_pixel_counts >= self.hot_pixel_threshold) |
                                        (direction_counts >= self.unmatched_star_threshold))
            ufos.loc[image_check, "quality_code"] = -1

    def remove_duplicates(self):
        """
        Removes duplicates from the dataframe.

        Occasionally if many points are close to each other we might end up with duplicate detections.  This method
        gets rid of them by looking for pairs of detections that are within 2 pixels of each other and only keeping the
        one with the better quality_code.
        """

        # sometimes we might get duplicate detections from the same image.
        # This function gets rid of them
        assert self.detection_data_frame is not None
        remove = self.detection_data_frame.occulting.copy()
        remove[:] = False

        for image_file, grp in self.detection_data_frame.groupby('image_file'):

            # make a kd tree for points in this image
            # noinspection PyArgumentList
            kd = KDTree(grp.loc[:, "x_raw":"y_raw"].to_numpy())

            # find all pairs separated by less than 2 pixels
            # noinspection PyUnresolvedReferences
            pairs = kd.query_pairs(2)

            # loop through each pair and set it to be removed
            for pair in pairs:

                if grp.iloc[pair[0]].quality_code <= grp.iloc[pair[1]].quality_code:
                    remove[grp.iloc[pair[0]].name] = True
                else:
                    remove[grp.iloc[pair[1]].name] = True

        self.detection_data_frame = self.detection_data_frame.loc[~remove]

    def save_results(self, out: str, split: bool = True):
        """
        This method saves the results into csv files.

        It can optionally split the results by image file, creating a single file for each image.  In this case, the out
        string should be a format string expecting 1 input, the name of the image file.

        :param out: the name of the csv file to save the results to.  If ``split`` is ``True`` then this should be a
                    format string expecting the name of the image file
        :param split: A flag specifying whether to split the output into a file for each image processed instead of 1
                      big file
        """
        assert self.detection_data_frame is not None
        # if we are splitting into multiple files
        if split:
            # split according to image
            for image_file, grp in self.detection_data_frame.groupby('image_file'):
                # get the name of the file to save the results to
                out_file = out.format(image_file.strip('.fits').strip('.FITS')) # type: ignore

                # save the results to the file
                grp.to_csv(out_file, index=False)

        else:
            # just write out everything
            self.detection_data_frame.to_csv(out, index=False)

    def clear_results(self):
        """
        This clears all extracted UFOs/Stars from the instance with the exception of the :attr:`detection_data_frame`
        for memory purposes.

        Note that after calling this method, the attributes containing the results will all be blank again and you will
        only be able to access information from the :attr:`detection_data_frame` attribute.j
        """
        number_images = len(self.sopnav.camera.images)

        self.invalid_images = [False] * number_images
        self._needs_processed = [False] * number_images
        self.summed_dn = [None] * number_images
        self.magnitude = [None] * number_images
        self.fit_chi2_value = [None] * number_images
        self.summed_dn_uncertainty = [None] * number_images
        self.saturated = [None] * number_images
        self.summed_dn_count = [None] * number_images
        self.max_dn = [None] * number_images
        self.bearing = [None] * number_images
        self.integrated_psf = [None] * number_images
        self.ra_sigma = [None] * number_images
        self.declination_sigma = [None] * number_images
        self.x_raw_sigma = [None] * number_images
        self.y_raw_sigma = [None] * number_images
        self.occulting = [None] * number_images
        self.saturation_distance = [None] * number_images
        self.trail_length = [None] * number_images
        self.trail_principal_angle = [None] * number_images

        self.star_summed_dn = [None] * number_images
        self.star_observed_magnitude = [None] * number_images
        self.star_fit_chi2_value = [None] * number_images
        self.star_summed_dn_uncertainty = [None] * number_images
        self.star_saturated = [None] * number_images
        self.star_summed_dn_count = [None] * number_images
        self.star_max_dn = [None] * number_images
        self.star_bearing = [None] * number_images
        self.star_integrated_psf = [None] * number_images
        self.star_ra_sigma = [None] * number_images
        self.star_declination_sigma = [None] * number_images
        self.star_x_raw_sigma = [None] * number_images
        self.star_y_raw_sigma = [None] * number_images
        self.star_occulting = [None] * number_images
        self.star_saturation_distance = [None] * number_images

        self.sopnav.clear_results()

    def add_images(self, data: Union[Iterable[Union[PATH, NDArray]], PATH, NDArray],
                   parse_data: bool = True, preprocessor: bool = True):
        """
        This is essentially an alias to the :meth:`.StellarOpNav.add_images` method, but it also expands various lists
        to account for the new number of images.

        When you have already initialized a :class:`Detector` class you should *always* use this method to add
        images for consideration.

        The lists that are extended by this method are:

        * :attr:`invalid_images`
        * :attr:`summed_dn`
        * :attr:`magnitude`
        * :attr:`fit_chi2_value`
        * :attr:`summed_dn_uncertainty`
        * :attr:`saturated`
        * :attr:`summed_dn_count`
        * :attr:`max_dn`
        * :attr:`bearing`
        * :attr:`integrated_psf`
        * :attr:`ra_sigma`
        * :attr:`declination_sigma`
        * :attr:`x_raw_sigma`
        * :attr:`y_raw_sigma`
        * :attr:`occulting`
        * :attr:`saturation_distance`
        * :attr:`trail_length`
        * :attr:`trail_principal_angle`
        * :attr:`star_summed_dn`
        * :attr:`star_observed_magnitude`
        * :attr:`star_fit_chi2_value`
        * :attr:`star_summed_dn_uncertainty`
        * :attr:`star_saturated`
        * :attr:`star_summed_dn_count`
        * :attr:`star_max_dn`
        * :attr:`star_bearing`
        * :attr:`star_integrated_psf`
        * :attr:`star_ra_sigma`
        * :attr:`star_declination_sigma`
        * :attr:`star_x_raw_sigma`
        * :attr:`star_y_raw_sigma`
        * :attr:`star_occulting`
        * :attr:`star_saturation_distance`

        See the :meth:`.StellarOpNav.add_images` for a description of the valid input for `data`

        :param data:  The image data to be stored in the :attr:`.images` list
        :param parse_data:  A flag to specify whether to attempt to parse the metadata automatically for the images
        :param preprocessor: A flag to specify whether to run the preprocessor after loading an image.
        """

        self.sopnav.add_images(data, parse_data=parse_data, preprocessor=preprocessor)

        if not isinstance(data, (list, tuple)):
            data = [data] # pyright: ignore[reportAssignmentType]

        for _ in data: # type: ignore
            self.invalid_images.append(False)
            self.summed_dn.append(None)
            self.magnitude.append(None)
            self.fit_chi2_value.append(None)
            self.summed_dn_uncertainty.append(None)
            self.saturated.append(None)
            self.summed_dn_count.append(None)
            self.max_dn.append(None)
            self.bearing.append(None)
            self.integrated_psf.append(None)
            self.ra_sigma.append(None)
            self.declination_sigma.append(None)
            self.x_raw_sigma.append(None)
            self.y_raw_sigma.append(None)
            self.occulting.append(None)
            self.saturation_distance.append(None)
            self.trail_length.append(None)
            self.trail_principal_angle.append(None)
            self.star_summed_dn.append(None)
            self.star_observed_magnitude.append(None)
            self.star_fit_chi2_value.append(None)
            self.star_summed_dn_uncertainty.append(None)
            self.star_saturated.append(None)
            self.star_summed_dn_count.append(None)
            self.star_max_dn.append(None)
            self.star_bearing.append(None)
            self.star_integrated_psf.append(None)
            self.star_ra_sigma.append(None)
            self.star_declination_sigma.append(None)
            self.star_x_raw_sigma.append(None)
            self.star_y_raw_sigma.append(None)
            self.star_occulting.append(None)
            self.star_saturation_distance.append(None)
            self._needs_processed.append(True)

