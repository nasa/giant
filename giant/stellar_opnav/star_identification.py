# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
This module provides the star identification routines for GIANT through the :class:`StarID` class.

Algorithm Description
_____________________

Star Identification refers to the process of matching observed stars in an image with a corresponding set of known star
locations from a star catalogue. Making this identification is the first step in performing a number of OpNav tasks,
including attitude estimation, geometric camera calibration, and camera alignment, as well as a number of photometry
tasks like linearity checks and point spread function modelling.

In GIANT, star identification is handled using a random sampling and consensus (RANSAC) approach using the following
steps:

#. The *a priori* attitude information for each image is used to query the star catalogue for the expected stars in the
   field of view of each image.
#. The retrieved catalogue stars are transformed into the camera frame and projected onto the image using the *a priori*
   image attitude and camera model.
#. The projected catalogue locations are paired with points in the image that were identified in the image by the image
   processing algorithm as potential stars using a nearest neighbor approach.
#. The initial pairs are thresholded based on the distance between the points, as well as for stars that are matched
   with 2 image points and image points that are close to 2 stars.
#. The remaining pairs are randomly sampled for 4 star pairs
#. The sample is used to estimate a new attitude for the image using the :class:`.DavenportQMethod` routines.
#. The new solved for attitude is used to re-rotate and project the catalogue stars onto the image.
#. The new projections are compared with their matched image points and the number of inlier pairs (pairs whose distance
   is less than some ransac threshold) are counted.
#. The number of inliers is compared to the maximum number of inliers found by any sample to this point (set to 0 if
   this is the first sample) and:

   * if there are more inliers

     * the maximum number of inliers is set to the number of inliers generated for this sample
     * the inliers for this sample are stored as correctly identified stars
     * the sum of the squares of the distances between the inlier pairs for this sample is stored

   * if there are an equivalent number of inliers to the previous maximum number of inliers then the sum of the squares
     of the distance between the pairs of inliers is compared to the sum of the squares of the previous inliers and if
     the new sum of squares is less than the old sum of squares

     * the maximum number of inliers is set to the number of inliers generated for this sample
     * the inliers are stored as correctly identified stars
     * the sum of the squares of the distances between the inlier pairs is stored

#. Steps 5-9 are repeated for a number of iterations, and the final set of stars stored as correctly identified stars
   become the identified stars for the image.

It is also possible to skip the RANSAC algorithm, stopping at step 4 above and marking any pairs that remain after the
check as correctly identified stars.

.. note::
    For the above algorithm an *a priori* attitude is needed for each image in which stars are being identified.  While 
    most OpNav images will have an *a priori* attitude, in some cases they may not due to anomalies on the spacecraft.  
    This is known as the *lost-in-space* problem.  Currently GIANT does not have the ability to handle the lost-in-space
    problem and the user will first need to use other software to determine an *a priori* attitude for the images (such 
    as `astrometry.net <http://astrometry.net>`_).  We are currently developing the algorithms required to perform lost
    in space star identification using hash code based pattern matching (similar to the techniques used by 
    *astrometry.net*) in GIANT, but they are unfortunately not complete yet.

Unfortunately, the star identification routines do require some human input to be successful.  This involves tuning
various parameters to get a good initial match.  Luckily, once these parameters are tuned for a few images for a
certain camera set under certain conditions, they largely should apply well to all similar images from that camera.
Below we discuss the different tuning parameters that are available in the :class:`StarID` class, and also some
techniques for getting successful identifications.

Tuning the StarID routines
__________________________

There are a few different parameters that can be tuned in the :class:`StarID` class when attempting to get a successful
star identification for a set of images.  Each of these parameters and what they control are described in the following
table.

.. _tuning-parameters-table:

===================================== ==================================================================================
Parameter                             Description
===================================== ==================================================================================
:attr:`~.StarID.max_magnitude`        The maximum magnitude to query the star catalogue to.  This is useful for
                                      limiting the number of catalogue stars that are being matched against.
                                      Remember that stellar magnitude is on an inverse logarithmic scale, therefore
                                      the higher you set this number the dimmer stars that will be returned.
:attr:`~.StarID.min_magnitude`        The minimum magnitude to query the star catalogue to.  This is useful for
                                      limiting the number of catalogue stars that are being matched against.
                                      Remember that stellar magnitude is on an inverse logarithmic scale, therefore
                                      the lower you set this number the brighter stars that will be returned.
                                      Typically this should be left alone.
:attr:`~.StarID.max_combos`           The maximum number of samples to try in the RANSAC algorithm.  The RANSAC
                                      algorithm will try at most :attr:`~StarID.max_combos` combinations when
                                      attempting to identify stars. The only way it will try less than
                                      :attr:`~.StarID.max_combos` is if there are less unique sample combinations
                                      possible, in which case the RANSAC algorithm will try every possible sample
                                      (and becomes just a simple Sampling and Consensus algorithm).  This parameter
                                      is also used to turn off the RANSAC algorithm by setting it to 0.  This stops
                                      the star identification process at step 4 from above.
:attr:`~.StarID.tolerance`            The maximum initial distance that a catalogue-image poi pair can have for it to be
                                      considered a potential match in units of pixels. This is the tolerance that is
                                      applied before the RANSAC to filter out nearest neighbor pairs that are too far
                                      apart to be potential matches.
:attr:`~.StarID.ransac_tolerance`     The maximum post-fit distance that a catalogue-image poi pair can have for it to
                                      be considered an inlier in the RANSAC algorithm in units of pixels.  This is
                                      the tolerance used inside of the RANSAC algorithm to determine the number of
                                      inliers for a given attitude solution from a sample.  This should always be
                                      less than the :attr:`~.StarID.tolerance` parameter.
:attr:`~.StarID.second_closest_check` A flag specifying whether to check if the second closest catalogue star to an
                                      image poi is also within the :attr:`~.StarID.tolerance` distance.  This is
                                      useful for throwing out potential pairs that may be ambiguous.  In general you
                                      should set this flag to ``False`` when your initial attitude/camera model error is
                                      larger, and ``True`` after removing those large errors.
:attr:`~.StarID.unique_check`         A flag specifying whether to allow a single catalogue star to be potentially
                                      paired with multiple image points of interest.  In general you
                                      should set this flag to ``False`` when your initial attitude/camera model error is
                                      larger, and ``True`` after removing those large errors.
===================================== ==================================================================================

By tuning these parameters, you should be able to identify stars in nearly any image with an *a priori* attitude that is
remotely close.  There are a few suggestions that may help you to find the proper tuning faster:

* Getting the initial identification is generally the most difficult; therefore, you should generally have 2 tunings
  for an image set.
* The first tuning should be fairly conservative in order to get a good refined attitude estimate for the image.  
  (Remember that we really only need 4 or 5 correctly identified stars to get a good attitude estimate.) 

  * a large initial :attr:`~.StarID.tolerance`--greater than 10 pixels.  Note that this initial tolerance should include
    the errors in the star projections due to both the *a priori* attitude uncertainty and the camera model
  * a smaller but still relatively large :attr:`~.StarID.ransac_tolerance`--on the order of about 1-5 pixels. This
    tolerance should mostly reflect a very conservative estimate on the errors caused by the camera model as the
    attitude errors should largely be removed
  * a small :attr:`~.StarID.max_magnitude`--only allowing bright stars.  Bright stars generally have more accurate
    catalogue positions and are more likely to be picked up by the :class:`.ImageProcessing` algorithms
  * the :attr:`~.StarID.max_combos` set fairly large--on the order of 500-1000
  
* After getting the initial pairing and updating
  the attitude for the images (note that this is done external to the :class:`StarID` class), you can then attempt a 
  larger identification with dimmer stars

  * decreasing the :attr:`~.StarID.tolerance` to be about the same as your previous :attr:`~.StarID.ransac_tolerance`
  * turning the RANSAC algorithm off by setting the :attr:`~.StarID.max_combos` to 0
  * increasing the :attr:`~.StarID.max_magnitude`.
  
* If you are having problems getting the identification to work it can be useful to visually examine the results for a
  couple of images using the :func:`.show_id_results` function.

.. warning::

    This script loads the lost in space catalogue from python pickle files.  Pickle files can be used to execute
    arbitrary code, so you should never open one from an untrusted source.  While this code should only be reading
    pickle files generated by GIANT itself that are safe, you should verify that the :attr:`LIS_FILE` and the file it
    points to have not been tampered with to be absolutely sure.
"""


# import random
import itertools as it
from datetime import datetime
import warnings
from typing import Optional, Union, Tuple
from multiprocessing import Pool
from pathlib import Path

# added warning to documentation
import pickle  # nosec

from copy import copy

import numpy as np

from scipy import spatial as spat
try:
    from scipy.special import comb

except ImportError:
    from scipy.misc import comb

from pandas import DataFrame

from giant.stellar_opnav.estimators import DavenportQMethod
from giant import catalogues as cat
from giant.ray_tracer.scene import correct_stellar_aberration
from giant.camera_models import CameraModel
from giant.rotations import Rotation
from giant.catalogues.meta_catalogue import Catalogue
from giant.catalogues.utilities import radec_to_unit
from giant._typing import NONEARRAY, Real, PATH
from giant.catalogues.utilities import RAD2DEG, unit_to_radec
from giant.utilities.random_combination import RandomCombinations


LIS_FILE = Path(__file__).parent.parent / "catalogues" / "data" / 'lis.pickle'  # type: Path


# def random_combination(n: int, r: int) -> tuple:
#     """
#     This returns a random sample of r indices from n objects
#
#     :param n: The number of objects to choose from
#     :param r: The number of objects to choose for each sample
#     :return: The indices for a random sample as a tuple
#     """
#     return random.sample(range(n), r)


class StarID:
    """
    The StarID class operates on the result of image processing algorithms to attempt to match image points of interest
    with catalogue star records.

    This is a necessary step in all forms of stellar OpNav and is a critical component of
    GIANT.

    In general, the user will not directly interface with the :class:`StarID` class and instead will use the
    :class:`.StellarOpNav` class.  Below we give a brief description of how to use this class directly for users who
    are just curious or need more direct control over the class.

    There are a couple things that the :class:`StarID` class needs to operate.  The first is a camera model, which
    should be a subclass of :class:`.CameraModel`.  The camera model is used to both project catalogue star locations
    onto the image, as well as generate unit vectors through the image points of interest in the camera frame.  The
    next thing the :class:`StarID` class needs is a star catalogue to query.  This should come from the
    :mod:`.catalogues` package and provides all of the necessary information for retrieving and projecting the expected
    stars in an image. Both the star catalogue and camera model are generally set at the construction of the class
    and apply to every image being considered, so they are rarely updated.  The camera model is stored in the
    :attr:`model` attribute and is also specified as the first positional argument for the class constructor.  The
    catalogue is stored in the :attr:`catalogue` attribute and can also be specified in the class constructor as a
    keyword argument of the same name.

    The :class:`StarID` class also needs some information about the current image being considered.  This information
    includes points of interest for the image that need to be matched to stars, the *a priori* attitude of the image,
    and the position/velocity of the camera at the time the image was captured.  The points of interest are generally
    returned from the :class:`.ImageProcessing` routines, although they don't need to be.  The camera attitude,
    position, and velocity are generally passed from the :class:`.OpNavImage` metadata.  The image attitude is used for
    querying the catalogue and rotating the catalogue stars into the image frame.  The camera positions and velocity
    are used for correcting the star locations for parallax and stellar aberration. The camera position and velocity are
    not required but are generally recommended as they will give a more accurate representation.  All of these
    attributes need to be updated for each new image being considered (the :class:`StarID` class does not directly
    operate on the :class:`.OpNavImage` objects).  The image points of interest are stored and updated in the
    :attr:`extracted_image_points` attribute, the camera attitude is stored in the :attr:`a_priori_rotation_cat2camera`
    attribute, and the camera position and velocity are stored in the :attr:`camera_position` and
    :attr:`camera_velocity` attributes respectively.  They can also be specified in the class constructor as keyword
    arguments of the same name.

    Finally, there are a number of tuning parameters that need set.  These parameters are discussed in depth in the
    :ref:`Tuning Parameters Table <tuning-parameters-table>`.

    When everything is correctly set in an instance of :class:`StarID`, then generally all that needs to be called
    is the :meth:`id_stars` method, which accepts the observation date of the image being considered as an
    optional ``epoch`` keyword argument.  This method will go through the whole processed detailed above, storing the
    results in a number of attributes that are detailed below.

    .. warning::

        This class will load data for the lost in space catalogue.  The lost is space catalogue is a pickle file. Pickle
        files can be used to execute arbitrary code, so you should never open one from an untrusted source.  While this
        code should only be reading pickle files generated by GIANT itself that are safe, you should verify that the
        :attr:`lost_in_space_catalogue_file` and the file it points to have not been tampered with to be absolutely
        sure.

    """

    def __init__(self, model: CameraModel, extracted_image_points: NONEARRAY = None,
                 catalogue: Optional[Catalogue] = None, max_magnitude: Real = 7, min_magnitude: Real = -10,
                 max_combos: int = 100, tolerance: Real = 20, a_priori_rotation_cat2camera: Optional[Rotation] = None,
                 ransac_tolerance: Real = 5, second_closest_check: bool = True, camera_velocity: NONEARRAY = None,
                 camera_position: NONEARRAY = None, unique_check: bool = True, use_mp: bool = False,
                 lost_in_space_catalogue_file: Optional[PATH] = None):
        """
        :param model: The camera model to use to relate vectors in the camera frame with points on the image
        :param extracted_image_points: A 2xn array of the image points of interest to be identified.  The first row
                                       should correspond to the y locations (rows) and the second row should correspond
                                       to the x locations (columns).
        :param catalogue: The catalogue object to use to query for potential stars in an image.
        :param max_magnitude:  the maximum magnitude to return when querying the star catalogue
        :param min_magnitude:  the minimum magnitude to return when querying the star catalogue
        :param max_combos: The maximum number of random samples to try in the RANSAC routine
        :param tolerance: The maximum distance between a catalogue star and a image point of interest for a potential
                          pair to be formed before the RANSAC algorithm
        :param a_priori_rotation_cat2camera: The rotation matrix to go from the inertial frame to the camera frame
        :param ransac_tolerance: The maximum distance between a catalogue star and an image point of interest after
                                 correcting the attitude for a pair to be considered an inlier in the RANSAC algorithm.
        :param second_closest_check: A flag specifying whether to reject pairs where 2 catalogue stars are close to an
                                     image point of interest
        :param camera_velocity: The velocity of the camera in km/s with respect to the solar system barycenter in the
                                inertial frame at the time the image was taken
        :param camera_position: The position of the camera in km with respect to the solar system barycenter in the
                                inertial frame at the time the image was taken
        :param unique_check: A flag specifying whether to allow a single catalogue star to be potentially paired with
                             multiple image points of interest
        :param use_mp: A flag specifying whether to use the multi-processing library to accelerate the RANSAC algorithm
        :param lost_in_space_catalogue_file: The file containing the lost in space catalogue
        """

        # initialize temporary attributes to make multiprocessing easier
        self._temp_image_locs = None
        self._temp_image_dirs = None
        self._temp_catalogue_dirs = None
        self._temp_temperature = 0
        self._temp_image_number = 0
        self._temp_combinations = None
        self._temp_att_est = DavenportQMethod()

        self.model = model  # type: CameraModel
        """
        The camera model which relates points in the camera frame to points in the image and vice-versa.
        """

        self.camera_position = np.zeros(3, dtype=np.float64)  # type: np.ndarray
        """
        The position of the camera with respect to the solar system barycenter in the inertial frame at the time the 
        image was captured as a length 3 numpy array of floats.

        Typically this is stored in the :attr:`.OpNavImage.position` attribute
        """

        if camera_position is not None:
            self.camera_position = camera_position

        self.camera_velocity = np.zeros(3, dtype=np.float64)  # type: np.ndarray
        """
        The velocity of the camera with respect to the solar system barycenter in the inertial frame at the time the 
        image was captured as a length 3 numpy array of floats.
        
        Typically this is stored in the :attr:`.OpNavImage.velocity` attribute
        """

        if camera_velocity is not None:
            self.camera_velocity = camera_velocity

        self.extracted_image_points = extracted_image_points  # type: np.ndarray
        """
        a 2xn array of the image points of interest to be paired with catalogue stars.  
        
        the first row should correspond to the x locations (columns) and the second row should correspond
        to the y locations (rows).
        
        typically this is retrieved from a call to :meth:`.ImageProcessing.locate_subpixel_poi_in_roi`.
        """

        self.catalogue = catalogue  # type: Catalogue
        """
        The star catalogue to use when pairing image points with star locations.
        
        This typically should be a subclass of the :class:`.Catalogue` class.  It defaults to the 
        :class:`.GIANTCatalogue`.
        """

        if self.catalogue is None:
            self.catalogue = cat.GIANTCatalogue()

        # store the a priori attitude of the camera
        self.a_priori_rotation_cat2camera = a_priori_rotation_cat2camera  # type: Rotation
        """
        This contains the a priori rotation knowledge from the catalogue frame (typically the inertial frame) to the
        camera frame at the time of the image.
        
        This typically is stored as the :attr:`.OpNavImage.rotation_inertial_to_camera` attribute.
        """

        self.max_magnitude = max_magnitude  # type: Real
        """
        The maximum star magnitude to query from the star catalogue.

        This specifies how dim stars are expected to be in the :attr:`extracted_image_points` data set.  This is 
        typically dependent on both the detector and the exposure length of the image under consideration.
        """

        self.min_magnitude = min_magnitude  # type: Real
        """
        The minimum star magnitude to query from the star catalogue.
        
        This specifies how dim stars are expected to be in the :attr:`extracted_image_points` data set.  This is 
        typically dependent on both the detector and the exposure length of the image under consideration.
        
        Generally this should be left alone unless you are worried about over exposed stars (in which case 
        :attr:`.ImageProcessing.reject_saturation` may be more useful) or you are doing some special analysis.
        """

        self.tolerance = tolerance  # type: Real
        """
        The maximum distance in units of pixels between a projected catalogue location and an extracted image point
        for a possible pairing to be made for consideration in the RANSAC algorithm.
        """

        # store the maximum number of ransac samples to try
        self.max_combos = max_combos  # type: int
        """
        The maximum number of random combinations to try in the RANSAC algorithm.  
        
        If the total possible number of combinations is less than this attribute then an exhaustive search will be 
        performed instead
        """

        self.ransac_tolerance = ransac_tolerance  # type: Real
        """
        The tolerance that is required after correcting for attitude errors for a pair to be considered an inlier
        in the RANSAC algorithm in units of pixels.
        
        This should always be less than the :attr:`tolerance` attribute.
        """

        # store the second closest check and uniqueness flags
        self.second_closest_check = second_closest_check  # type: bool
        """
        A boolean specifying whether to ignore extracted image points where multiple catalogue points are within the
        specified tolerance.
        """

        self.unique_check = unique_check  # type: bool
        """
        A boolean specifying whether to ignore possible catalogue to image point pairs where multiple image points are 
        within the specified tolerance of a single catalogue point.
        """

        self.use_mp = use_mp  # type: bool
        """
        A boolean flag specifying whether to use multi-processing to speed up the RANSAC process.
        
        If this is set to True then all available CPU cores will be utilized to parallelize the RANSAC algorithm 
        computations.  For small combinations, the overhead associated with this can swamp any benefit that may be 
        realized.
        """

        # initialize the attributes for storing the star identification results
        self.queried_catalogue_image_points = None  # type: NONEARRAY
        """
        A 2xn numpy array of points containing the projected image points for all catalogue stars that were queried from 
        the star catalogue  with x (columns) in the first row and y (rows) in the second row.  
        
        Each column corresponds to the same row in :attr:`queried_catalogue_star_records`.
        
        Until :meth:`project_stars` is called this will be ``None``.
        """

        self.queried_catalogue_star_records = None  # type: Optional[DataFrame]
        """
        A pandas DataFrame of all the catalogue star records that were queried.  

        See the :class:`.Catalogue` class for a description of the columns of the dataframe.
        
        Until :meth:`project_stars` is called this will be ``None``.
        """

        self.queried_catalogue_unit_vectors = None  # type: NONEARRAY
        """
        A 3xn numpy array of unit vectors in the inertial frame for all catalogue stars that were queried from 
        the star catalogue.  
        
        Each column corresponds to the same row in :attr:`queried_catalogue_star_records`.

        Until :meth:`project_stars` is called this will be ``None``.
        """

        self.queried_weights_inertial = None  # type: NONEARRAY
        """
        This contains the formal total uncertainty for each unit vector from the queried catalogue stars.  
        
        Each element in this array corresponds to the same row in the :attr:`queried_catalogue_star_records`.
        
        Until :meth:`id_stars` is called this will be ``None``.
        """

        self.queried_weights_picture = None  # type: NONEARRAY
        """
        This contains the formal total uncertainty for each projected pixel location from the queried catalogue stars in
        units of pixels..  
        
        Each element in this array corresponds to the same row in the :attr:`queried_catalogue_star_records`.

        Until :meth:`id_stars` is called this will be ``None``.
        """

        self.unmatched_catalogue_image_points = None
        """
        A 2xn numpy array of points containing the projected image points for all catalogue stars that not matched
        with an extracted image point, with x (columns) in the first row and y (rows) in the second row.  

        Each column corresponds to the same row in :attr:`unmatched_catalogue_star_records`.

        Until :meth:`id_stars` is called this will be ``None``.
        """

        self.unmatched_catalogue_star_records = None  # type: NONEARRAY
        """
        A pandas DataFrame of all the catalogue star records that were not matched to an extracted image point in the 
        star identification routine.  

        See the :class:`.Catalogue` class for a description of the columns of the dataframe.

        Until :meth:`id_stars` is called this will be ``None``.
        """

        self.unmatched_catalogue_unit_vectors = None  # type: NONEARRAY
        """
        A 3xn numpy array of unit vectors in the inertial frame for all catalogue stars that were not matched to an
        extracted image point in the star identification routine.  

        Each column corresponds to the same row in :attr:`matched_catalogue_star_records`.

        Until :meth:`id_stars` is called this will be ``None``.
        """

        self.unmatched_extracted_image_points = None  # type: NONEARRAY
        """
        A 2xn array of the image points of interest that were not paired with a catalogue star in the star 
        identification routine.

        The first row corresponds to the x locations (columns) and the second row corresponds to the y locations (rows).
        
        Until :meth:`id_stars` is called this will be ``None``.
        """

        self.unmatched_weights_inertial = None  # type: NONEARRAY
        """
        This contains the formal total uncertainty for each unit vector from the queried catalogue stars
        that were not matched with an extracted image point.  
        
        Each element in this array corresponds to the same row in the :attr:`unmatched_catalogue_star_records`.

        Until method :meth:`id_stars` is called this will be ``None``.
        """

        self.unmatched_weights_picture = None  # type: NONEARRAY
        """
        This contains the formal total uncertainty for each projected pixel location from the queried catalogue stars 
        that were not matched with an extracted image point in units of pixels. 
        
        Each element in this array corresponds to the same row in the :attr:`unmatched_catalogue_star_records`.

        Until method :meth:`id_stars` is called this will be ``None``.
        """

        self.matched_catalogue_image_points = None  # type: NONEARRAY
        """
        A 2xn numpy array of points containing the projected image points for all catalogue stars that were matched
        with an extracted image point, with x (columns) in the first row and y (rows) in the second row.  

        Each column corresponds to the same row in :attr:`matched_catalogue_star_records`.

        Until :meth:`id_stars` is called this will be ``None``.
        """

        self.matched_catalogue_star_records = None  # type: NONEARRAY
        """
        A pandas DataFrame of all the catalogue star records that were matched to an extracted image point in the 
        star identification routine.  

        See the :class:`.Catalogue` class for a description of the columns of the dataframe.
        
        Each row of the dataframe corresponds to the same column index in the :attr:`matched_extracted_image_points`.

        Until :meth:`id_stars` is called this will be ``None``.
        """

        self.matched_catalogue_unit_vectors = None  # type: NONEARRAY
        """
        A 3xn numpy array of unit vectors in the inertial frame for all catalogue stars that were matched to an
        extracted image point in the star identification routine.  

        Each column corresponds to the same row in :attr:`matched_catalogue_star_records`.

        Until :meth:`id_stars` is called this will be ``None``.
        """

        self.matched_extracted_image_points = None  # type: NONEARRAY
        """
        A 2xn array of the image points of interest that were not paired with a catalogue star in the star 
        identification routine.

        The first row contains to the x locations (columns) and the second row contains to the y locations (rows).
        
        Each column corresponds to the same row in the :attr:`matched_catalogue_star_records` for its pairing.

        Until :meth:`id_stars` is called this will be ``None``.
        """

        self.matched_weights_inertial = None  # type: NONEARRAY
        """
        This contains the formal total uncertainty for each unit vector from the queried catalogue stars
        that were matched with an extracted image point.  

        Each element in this array corresponds to the same row in the :attr:`matched_catalogue_star_records`.

        Until methods :meth:`id_stars` is called this will be ``None``.
        """

        self.matched_weights_picture = None   # type: NONEARRAY
        """
        This contains the formal total uncertainty for each projected pixel location from the queried catalogue stars 
        that were matched with an extracted image point in units of pixels. 

        Each element in this array corresponds to the same row in the :attr:`matched_catalogue_star_records`.

        Until method :meth:`id_stars` is called this will be ``None``.
        """

        if lost_in_space_catalogue_file is None:
            lis_file = LIS_FILE
        else:
            lis_file = Path(lost_in_space_catalogue_file)

        self.lis_catalogue = (None, None)  # type: Tuple[Optional[spat.cKDTree], NONEARRAY]
        """
        The lost in space catalogue.
        
        Contains a scipy cKDTree containing hash codes as the first element and a numpy array containing star ids
        for each hash element for the second element.
        
        .. warning::

            The lost is space catalogue is a pickle file. Pickle files can be used to execute
            arbitrary code, so you should never open one from an untrusted source.  
        """

        if lis_file.exists():
            with lis_file.open('rb') as in_file:
                # added warning to documentation
                tree = pickle.load(in_file)  # nosec
                inds = pickle.load(in_file)  # nosec
                self.lis_catalogue = (tree, inds)

    def query_catalogue(self, epoch: Union[datetime, Real] = datetime(2000, 1, 1)):
        """
        This method queries stars from the catalogue within the field of view.

        The stars are queried such that any stars within 1.3*the :attr:`.CameraModel.field_of_view` value radial
        distance of the camera frame z axis converted to right ascension and declination are returned between
        :attr:`min_magnitude` and :attr:`max_magnitude`.  The queried stars are updated to the ``epoch`` value
        using proper motion.  They are stored in the :attr:`queried_catalogue_star_records` attribute. The stars are
        stored as a pandas DataFrame.  For more information about this format see the :class:`.Catalogue` class
        documentation.

        The epoch input should either be a python datetime object representation of the UTC time or a float value of the
        MJD years.

        In general, this method does not need to be directly called by the user as it is automatically called in the
        :meth:`project_stars` method.

        :param epoch: The new epoch to move the stars to using proper motion
        """

        # get the ra and dec of the camera frame z axis
        ra_dec_cat = np.array(self.compute_pointing())

        # query the catalogue and store the results
        self.queried_catalogue_star_records = self.catalogue.query_catalogue(
            search_center=ra_dec_cat,
            search_radius=1.3 * self.model.field_of_view,
            min_mag=self.min_magnitude,
            max_mag=self.max_magnitude,
            new_epoch=epoch
        )

    def compute_pointing(self) -> Tuple[float, float]:
        r"""
        This method computes the right ascension and declination of an axis of the camera frame in units of degrees.

        The pointing is computed by extracting the camera frame z axis expressed in the inertial frame from the
        :attr:`a_priori_rotation_cat2camera` and then converting that axis to a right ascension and declination.
        The conversion to right ascension and declination is given as

        .. math::
            ra=\text{atan2}(\mathbf{c}_{yI}, \mathbf{c}_{xI})\\
            dec=\text{asin}(\mathbf{c}_{zI})

        where atan2 is the quadrant aware arc tangent function, asin is the arc sin and :math:`\mathbf{c}_{jI}` is the
        :math:`j^{th}` component of the camera frame axis expressed in the Inertial frame.

        In general this method is not used by the user as it is automatically called in the :meth:`query_catalogue`
        method.

        :return:  The right ascension and declination of the specified axis in the inertial frame as a tuple (ra, dec)
                  in units of degrees.
        """

        boresight_cat = self.a_priori_rotation_cat2camera.matrix[2]

        ra, dec = unit_to_radec(boresight_cat)

        return RAD2DEG * ra, RAD2DEG * dec

    def project_stars(self, epoch: Union[datetime, Real] = datetime(2000, 1, 1), compute_weights: bool = False,
                      temperature: Real = 0, image_number: int = 0):
        """
        This method queries the star catalogue for predicted stars within the field of view and projects those stars
        onto the image using the camera model.

        The star catalogue is queried using the :meth:`query_catalogue` method and the stars are updated to the epoch
        specified by ``epoch`` using the proper motion from the catalogue.  The ``epoch`` should be specified as either
        a datetime object representing the UTC time the stars should be transformed to, or a float value representing
        the MJD year.  The queried Pandas Dataframe containing the star catalogue records is stored in the
        :attr:`queried_catalogue_star_records` attribute.

        After the stars are queried from the catalogue, they are converted to inertial unit vectors and corrected for
        stellar aberration and parallax using the :attr:`camera_position` and :attr:`camera_velocity` values.  The
        corrected inertial vectors are stored in the :attr:`queried_catalogue_unit_vectors`.

        Finally, the unit vectors are rotated into the camera frame using the :attr:`a_priori_rotation_cat2camera`
        attribute, and then projected onto the image using the :attr:`model` attribute.  The projected points are stored
        in the :attr:`queried_catalogue_image_points` attribute.

        If requested, the formal uncertainties for the catalogue unit vectors and pixel locations are computed and
        stored in the :attr:`queried_weights_inertial` and :attr:`queried_weights_picture`.  These are computed by
        transforming the formal uncertainty on the right ascension, declination, and proper motion specified in the
        star catalogue into the proper frame.

        In general this method is not called directly by the user and instead is called in the :meth:`id_stars` method.

        :param epoch: The epoch to get the star locations for
        :param compute_weights: A boolean specifying whether to compute the formal uncertainties for the unit vectors
                                and the pixel locations of the catalogue stars.
        :param temperature: The temperature of the camera at the time of the image being processed
        :param image_number: The number of the image being processed
        """

        # query the star catalogue for predicted stars in the field of view
        self.query_catalogue(epoch=epoch)

        # convert the star locations into unit vectors in the inertial frame
        ra_rad = self.queried_catalogue_star_records['ra'].values / RAD2DEG
        dec_rad = self.queried_catalogue_star_records['dec'].values / RAD2DEG
        catalogue_unit_vectors = radec_to_unit(ra_rad, dec_rad)

        # correct the unit vectors for parallax using the distance attribute of the star records and the camera inertial
        # location
        catalogue_points = catalogue_unit_vectors * self.queried_catalogue_star_records['distance'].values

        camera2stars_inertial = catalogue_points - self.camera_position.reshape(3, 1)

        # correct the stellar aberration
        camera2stars_inertial = correct_stellar_aberration(camera2stars_inertial, self.camera_velocity)

        # form the corrected unit vectors
        camera2stars_inertial /= np.linalg.norm(camera2stars_inertial, axis=0, keepdims=True)

        # rotate the unit vectors into the camera frame
        rot2camera = self.a_priori_rotation_cat2camera.matrix
        catalogue_unit_vectors_camera = np.matmul(rot2camera, camera2stars_inertial)

        # store the inertial corrected unit vectors and the projected image locations
        self.queried_catalogue_unit_vectors = camera2stars_inertial
        self.queried_catalogue_image_points = self.model.project_onto_image(catalogue_unit_vectors_camera,
                                                                            temperature=temperature, image=image_number)

        if compute_weights:
            # compute the covariance of the inertial catalogue unit vectors
            cos_d = np.cos(dec_rad)
            cos_a = np.cos(ra_rad)
            sin_d = np.sin(dec_rad)
            sin_a = np.sin(ra_rad)
            zero = np.zeros(cos_d.shape)

            dv_da = np.array([-cos_d * sin_a, cos_d * sin_a, zero])
            dv_dd = np.array([-sin_d * cos_a, -sin_d * sin_a, cos_d])

            cov_v = (np.einsum('ij,jk->jik', dv_da, dv_da.T) *
                     (self.queried_catalogue_star_records['ra_sigma'].values / RAD2DEG / cos_d).reshape(-1, 1, 1) ** 2
                     + np.einsum('ij,jk->jik', dv_dd, dv_dd.T) *
                     (self.queried_catalogue_star_records['dec_sigma'].values.reshape(-1, 1, 1) / RAD2DEG) ** 2)

            # compute the covariance of the projected catalogue points
            cov_xc = rot2camera @ cov_v @ rot2camera.T
            pj = self.model.compute_pixel_jacobian(catalogue_unit_vectors_camera, temperature=temperature,
                                                   image=image_number)
            cov_xp = pj @ cov_xc @ pj.swapaxes(-1, -2)

            self.queried_weights_inertial = np.trace(cov_v, axis1=-2, axis2=-1)
            self.queried_weights_picture = np.diagonal(cov_xp, axis1=-2, axis2=-1)

    def solve_lis(self, epoch: Union[datetime, Real] = datetime(2000, 1, 1),
                  temperature: Real = 0, image_number: int = 0,) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Solves the lost in space problem (no a priori knowledge) for the orientation between the catalogue and camera
        frames.

        The lost in space problem is solved by first generating hash codes of observed possible star quads in an image
        using  :meth:`_generate_hash`.  Given the hash codes, they are compared with a precomputed database of hash
        codes (see :mod:`.build_lost_in_space_catalogue`) to identify the closest matches.  The closest matches are then
        used to make a guess at the rotation from the catalogue frame to the camera frame, and the usual star ID
        routines (:meth:`.id_stars`) are called using the guess as the a priori attitude knowledge.  The number of
        identified stars found using the usual methods is then compared with the best number of stars found so far,
        and if more stars are found the rotation is kept as the best available.  This is done using the settings already
        provided to the class, so you need to ensure that you have a good setup even when solving the lost in space
        problem.  This continues until all possible hash code pairs have been considered, or until a pair produces an
        a priori attitude that successfully identifies half of the queried stars from the catalogue in the FOV of the
        camera and one quarter of the possible stars.

        The result is saved to the :attr:`.a_priori_rotation_cat2camera` attribute and then the usual star ID routines
        are run again to finish off the identification.

        :param epoch: the epoch of the image
        :param temperature: the temperature of the camera at the time the image was captured
        :param image_number: The number of the image being processed
        :return: The boolean index into the image points that met the original pairing criterion, and a second boolean
                 index into the the result from the previous boolean index that extracts the image points that were
                 successfully matched in the RANSAC algorithms
        """

        if self.lis_catalogue[0] is None:
            raise ValueError('The lost in space catalogue has not been loaded.  Cannot solve lost in space problem.'
                             'See build_lost_in_space_catalogue interface for details.')

        ip_hash, ip_inds = self._generate_hash(self.extracted_image_points)

        distances, pair_indices = self.lis_catalogue[0].query(ip_hash, 100, distance_upper_bound=0.02)

        sorted_dist = np.argsort(distances, axis=0)
        distances = distances[sorted_dist]
        pair_indices = pair_indices[sorted_dist]

        lis_sid = copy(self)

        best_inliers = 0
        best_rotation = None
        keep_out = None
        inliers_out = None

        for ip_ind, dist, pairs in zip(ip_inds, distances, pair_indices):
            valid = np.isfinite(dist)
            pairs = pairs[valid]

            self._temp_att_est.target_frame_directions = self.model.pixels_to_unit(
                self.extracted_image_points[:, ip_ind], temperature=temperature, image=image_number)

            done = False
            for pair in pairs:

                cat_stars = self.catalogue.query_catalogue(ids=self.lis_catalogue[1][pair])

                ra_rad, dec_rad = (cat_stars.loc[:, ["ra", "dec"]]/RAD2DEG).values.T

                unit_inertial = radec_to_unit(ra_rad, dec_rad)

                cam2stars_inertial = correct_stellar_aberration(unit_inertial * cat_stars['distance'].values -
                                                                self.camera_position.reshape(3, 1),
                                                                self.camera_velocity)

                cam2stars_inertial /= np.linalg.norm(cam2stars_inertial, axis=0, keepdims=True)

                self._temp_att_est.base_frame_directions = cam2stars_inertial

                self._temp_att_est.estimate()

                lis_sid.a_priori_rotation_cat2camera = self._temp_att_est.rotation

                keeps, inliers = lis_sid.id_stars(epoch, temperature=temperature)

                if lis_sid.matched_catalogue_image_points is not None:
                    if lis_sid.matched_catalogue_image_points.shape[-1] > best_inliers:
                        best_rotation = self._temp_att_est.rotation
                        best_inliers = lis_sid.matched_catalogue_image_points.shape[-1]

                        in_fov = ((lis_sid.queried_catalogue_image_points > [[0], [0]]) &
                                  (lis_sid.queried_catalogue_image_points <
                                   [[self.model.n_cols], [self.model.n_rows]])).all(axis=0)

                        keep_out = keeps
                        inliers_out = inliers

                        self.__dict__.update(lis_sid.__dict__)

                        if best_inliers/in_fov.sum() > 0.5:
                            if best_inliers / lis_sid.extracted_image_points.shape[-1] > 0.25:
                                done = True
                                break
            if done:
                break

        if best_rotation is not None:
            self.a_priori_rotation_cat2camera = best_rotation

        else:
            warnings.warn('Unable to solve lost in space problem')

        return keep_out, inliers_out

    def id_stars(self, epoch: Union[datetime, Real] = datetime(2000, 1, 1), compute_weights: bool = False,
                 temperature: Real = 0, image_number: int = 0,
                 lost_in_space: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        This method attempts to match the image points of interest with catalogue stars.

        The :meth:`id_stars` method is the primary interface of the :class:`StarID` class.  It performs all the tasks of
        querying the star catalogue, performing the initial pairing using a nearest neighbor search, refining the
        initial pairings with the :attr:`second_closest_check` and :attr:`unique_check`, and passing the refined
        pairings to the RANSAC routines.  The matched and unmatched catalogue stars and image points of interest are
        stored in the appropriate attributes.

        This method also returns a boolean index in the image points of interest vector, which extracts the image points
        that met the initial match criterion, and another boolean index into the image points of interest which
        extracts the image points of interest that were matched by the RANSAC algorithms.  This can be used to select
        the appropriate meta data about catalogue stars or stars found in an image that isn't explicitly considered by
        this class (as is done in the :class:`.StellarOpNav` class), but if you do not have extra information you need
        to keep in sync, then you can ignore the output.

        If requested, the formal uncertainties for the catalogue unit vectors and pixel locations are computed and
        stored in the :attr:`queried_weights_inertial` and :attr:`queried_weights_picture`.  These are computed by
        transforming the formal uncertainty on the right ascension, declination, and proper motion specified in the
        star catalogue into the proper frame.

        :param epoch: The new epoch to move the stars to using proper motion
        :param compute_weights: a flag specifying whether to compute weights for the attitude estimation and
                                calibration estimation.
        :param temperature: The temperature of the camera at the time of the image being processed
        :param image_number: The number of the image being processed
        :param lost_in_space: A flag specifying whether the lost in space algorithm needs to be used
        :return: The boolean index into the image points that met the original pairing criterion, and a second boolean
                 index into the the result from the previous boolean index that extracts the image points that were
                 successfully matched in the RANSAC algorithms
        """

        if lost_in_space or self.a_priori_rotation_cat2camera is None:
            return self.solve_lis(epoch, temperature)

        if self.a_priori_rotation_cat2camera is None:
            warnings.warn('Unable to proceed with star id.  No a priori point knowledge available.')

        # first get the unit vectors and image locations for the stars in the field of view
        self.project_stars(epoch=epoch, compute_weights=compute_weights, temperature=temperature,
                           image_number=image_number)

        # create a kdtree of the catalogue image locations for faster searching
        # noinspection PyArgumentList
        catalogue_image_locations_kdtree = spat.cKDTree(self.queried_catalogue_image_points.T)

        # query the kdtree to get the 2 closest catalogue image locations to each image point of interest
        if not self.extracted_image_points.any():
            return None, None
        distance, inds = catalogue_image_locations_kdtree.query(self.extracted_image_points.T, k=2)

        # check to see which pairs are less than the user specified matching tolerance
        dist_check = distance[:, 0] <= self.tolerance

        # throw out pairs where multiple catalogue locations are < the tolerance to the image points of interest
        if self.second_closest_check:
            dist_check &= distance[:, 1] > self.tolerance

        keep_stars = dist_check.copy()

        # throw out points that are matched twice
        if self.unique_check:
            keep_unique = np.ones(inds.shape[0], dtype=bool)

            for kin, ind in enumerate(inds[:, 0]):
                keep_unique[kin] = (ind == inds[dist_check, 0]).sum() == 1

            keep_stars &= keep_unique
        if not keep_stars.any():
            return None, None
        # either return our current matches or further filter using ransac if desired
        if self.max_combos:
            self.matched_extracted_image_points, self.matched_catalogue_unit_vectors, keep_inliers = self.ransac(
                self.extracted_image_points[:, keep_stars], self.queried_catalogue_unit_vectors[:, inds[keep_stars, 0]],
                temperature=temperature, image_number=image_number
            )

            if keep_inliers is None:  # if none of the stars met the ransac criteria then throw everything out
                warnings.warn("no stars found for epoch {0}".format(epoch))
                self.matched_catalogue_star_records = None
                self.matched_catalogue_image_points = None
                if compute_weights:
                    self.matched_weights_inertial = None
                    self.matched_weights_picture = None

            else:
                # update the matched catalogue star records and image points
                self.matched_catalogue_star_records = self.queried_catalogue_star_records.iloc[
                    inds[keep_stars, 0]][keep_inliers]
                self.matched_catalogue_image_points = self.queried_catalogue_image_points[
                                                      :, inds[keep_stars, 0]][:, keep_inliers]

                if compute_weights:
                    # noinspection PyTypeChecker
                    self.matched_weights_inertial = self.queried_weights_inertial[inds[keep_stars, 0]][keep_inliers]
                    # noinspection PyTypeChecker
                    self.matched_weights_picture = self.queried_weights_picture[inds[keep_stars, 0]][keep_inliers]

        else:
            # set the matches in the proper places
            self.matched_extracted_image_points = self.extracted_image_points[:, keep_stars]
            self.matched_catalogue_image_points = self.queried_catalogue_image_points[:, inds[keep_stars, 0]]
            self.matched_catalogue_unit_vectors = self.queried_catalogue_unit_vectors[:, inds[keep_stars, 0]]
            self.matched_catalogue_star_records = self.queried_catalogue_star_records.iloc[inds[keep_stars, 0]]
            if compute_weights:
                self.matched_weights_inertial = self.queried_weights_inertial[inds[keep_stars, 0]]
                self.matched_weights_picture = self.queried_weights_picture[inds[keep_stars, 0]]

            keep_inliers = np.ones(self.matched_extracted_image_points.shape[1], dtype=bool)

        # use python set notation to determine the list of stars and image points that were never matched
        if keep_inliers is not None:

            # get the stars that weren't matched
            unmatched_inds = list({*np.arange(self.queried_catalogue_image_points.shape[1])} -
                                  {*(inds[keep_stars, 0][keep_inliers])})

            self.unmatched_catalogue_image_points = self.queried_catalogue_image_points[:, unmatched_inds].copy()
            self.unmatched_catalogue_star_records = self.queried_catalogue_star_records.iloc[unmatched_inds].copy()
            self.unmatched_catalogue_unit_vectors = self.queried_catalogue_unit_vectors[:, unmatched_inds].copy()
            if compute_weights:
                self.unmatched_weights_inertial = self.queried_weights_inertial[unmatched_inds].copy()
                self.unmatched_weights_picture = self.queried_weights_picture[unmatched_inds].copy()

            # get the points of interest that weren't matched
            camera_inds = np.arange(self.extracted_image_points.shape[1])
            unmatched_centroid_inds = list({*camera_inds} - {*camera_inds[keep_stars][keep_inliers]})
            self.unmatched_extracted_image_points = self.extracted_image_points[:, unmatched_centroid_inds].copy()

        else:  # nothing was matched
            self.unmatched_extracted_image_points = self.extracted_image_points.copy()
            self.unmatched_catalogue_star_records = self.queried_catalogue_star_records.copy()
            self.unmatched_catalogue_unit_vectors = self.queried_catalogue_unit_vectors.copy()
            self.unmatched_catalogue_image_points = self.queried_catalogue_image_points.copy()
            if compute_weights:
                self.unmatched_weights_inertial = self.queried_weights_inertial.copy()
                self.unmatched_weights_picture = self.queried_weights_picture.copy()

        return keep_stars, keep_inliers

    def ransac(self, image_locs: np.ndarray, catalogue_dirs: np.ndarray,
               temperature: Real, image_number: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This method performs RANSAC on the image poi-catalogue location pairs.

        The RANSAC algorithm is described below

        #. The pairs are randomly sampled for 4 star pairs
        #. The sample is used to estimate a new attitude for the image using the :class:`.DavenportQMethod` routines.
        #. The new solved for attitude is used to re-rotate and project the catalogue stars onto the image.
        #. The new projections are compared with their matched image points and the number of inlier pairs (pairs whose
           distance is less than some ransac threshold) are counted.
        #. The number of inliers is compared to the maximum number of inliers found by any sample to this point (set to
           0 if this is the first sample) and:

           * if there are more inliers

             * the maximum number of inliers is set to the number of inliers generated for this sample
             * the inliers for this sample are stored as correctly identified stars
             * the sum of the squares of the distances between the inlier pairs for this sample is stored

           * if there are an equivalent number of inliers to the previous maximum number of inliers then the sum of the
             squares of the distance between the pairs of inliers is compared to the sum of the squares of the previous
             inliers and if the new sum of squares is less than the old sum of squares

             * the maximum number of inliers is set to the number of inliers generated for this sample
             * the inliers are stored as correctly identified stars
             * the sum of the squares of the distances between the inlier pairs is stored

        #. Steps 1-5 are repeated for a number of iterations, and the final set of stars stored as correctly identified
           stars become the identified stars for the image.

        In order to use this method, the ``image_locs`` input and the ``catalogue_dirs`` input should represent the
        initial pairings between the image points found using image processing and the predicted catalogue star unit
        vectors in the inertial frame. The columns in these 2 arrays should represent the matched pairs (that is column
        10 of ``image_locs`` should correspond to column 10 in ``catalogue_dirs``).

        This method returns the paired image locations and catalogue directions from the best RANSAC iteration
        and the boolean index into the input arrays that extract these values.

        In general this method is not used directly by the user and instead is called as part of the :meth:`id_stars`
        method.

        :param image_locs:  The image points of interest that met the initial matching criteria as a 2xn array
        :param catalogue_dirs:  The catalogue inertial unit vectors that met the initial matching criteria in the same
                                order as the ``image_locs`` input as a 3xn array.
        :param temperature: The temperature of the camera at the time of the image being processed
        :param image_number: The number of the image being processed
        :return: The matched image points of interest, the matched catalogue unit vectors, and the boolean index that
                 represents these arrays
        """

        # initialize the maximum number of inliers and minimum sum of squares variables.
        max_inliers = 0
        max_rs = 2 * self.ransac_tolerance ** 2 * image_locs.shape[1]

        # get the maximum number of combinations that are available to sample
        n_comb = int(comb(image_locs.shape[1], min(image_locs.shape[1] - 1, 4)))

        # convert the image points of interest to unit vectors in the camera frame
        image_dirs = self.model.pixels_to_unit(image_locs, temperature=temperature, image=image_number)

        self._temp_image_locs = image_locs
        self._temp_image_dirs = image_dirs
        self._temp_catalogue_dirs = catalogue_dirs
        self._temp_temperature = temperature
        self._temp_image_number = image_number
        self._temp_combinations = list(RandomCombinations(image_locs.shape[-1],
                                                          min(image_locs.shape[1] - 1, 4),
                                                          min(n_comb, self.max_combos)))

        # perform the ransac
        iters = range(min(n_comb, self.max_combos))
        if self.use_mp:
            with Pool() as pool:
                res = pool.map(self.ransac_iter_test, iters)
        else:
            res = [self.ransac_iter_test(i) for i in iters]

        # initialize the return values in case the RANSAC fails
        keep_image_locs = None
        keep_catalogue_dirs = None
        keep_inliers = None

        # find the best iteration and keep it
        for num_inliers, filtered_image_locs, filtered_catalogue_dirs, inliers, rs in res:

            # check to see if this is the best ransac iteration yet
            if num_inliers > max_inliers:

                max_inliers = num_inliers

                keep_image_locs = filtered_image_locs

                keep_catalogue_dirs = filtered_catalogue_dirs

                keep_inliers = inliers

                max_rs = rs

            elif num_inliers == max_inliers:

                if rs < max_rs:
                    max_inliers = num_inliers

                    keep_image_locs = filtered_image_locs

                    keep_catalogue_dirs = filtered_catalogue_dirs

                    keep_inliers = inliers

                    max_rs = rs

        # clear out the temp data
        self._temp_image_locs = None
        self._temp_image_dirs = None
        self._temp_catalogue_dirs = None
        self._temp_temperature = 0
        self._temp_image_number = 0

        # return the matched results and the boolean index
        return keep_image_locs, keep_catalogue_dirs, keep_inliers

    def ransac_iter_test(self, iter_num: int) -> Tuple[int, NONEARRAY, NONEARRAY, NONEARRAY, NONEARRAY]:
        """
        This performs a single ransac iteration.

        See the :meth:`ransac` method for more details.

        :param iter_num: the iteration number for retrieving the combination to try
        :return: the number of inliers for this iteration, the image location inliers for this iteration, the
                 catalogue direction inliers for this iteration, the boolean index for the inliers for this iteration,
                 and the sum of the squares of the residuals for this iteration
        """
        image_locs = self._temp_image_locs
        image_dirs = self._temp_image_dirs
        catalogue_dirs = self._temp_catalogue_dirs

        # get a random combination of indices into the image_locs and catalogue_dirs arrays
        # inds = random_combination(image_locs.shape[1], min(image_locs.shape[1] - 1, 4))
        inds = self._temp_combinations[iter_num]

        # extract the image directions to use for this ransac iteration
        image_dirs_use = image_dirs[:, inds]

        # extract the catalogue directions ot use for this ransac iteration
        catalogue_dirs_use = catalogue_dirs[:, inds]

        # store the unit vectors into the attitude estimator
        self._temp_att_est.target_frame_directions = image_dirs_use

        self._temp_att_est.base_frame_directions = catalogue_dirs_use

        # estimate an updated attitude
        self._temp_att_est.estimate()

        # get the updated attitude rotation matrix
        new_rot = self._temp_att_est.rotation.matrix

        # rotate the catalogue directions into the camera frame and project them onto the image using the new
        # attitude
        catalogue_dirs_cam = np.matmul(new_rot, catalogue_dirs)

        catalogue_locs = self.model.project_onto_image(catalogue_dirs_cam, temperature=self._temp_temperature,
                                                       image=self._temp_image_number)

        # compute the residual distance in all of the pairs
        resids = np.linalg.norm(catalogue_locs - image_locs, axis=0)

        # check to see which pairs meet the ransac tolerance
        inliers = resids < self.ransac_tolerance

        # get the sum of the squares of the residuals
        rs = np.sum(resids[inliers] * resids[inliers])

        if inliers.any():
            return inliers.sum(), image_locs[:, inliers], catalogue_dirs[:, inliers], inliers, rs

        else:
            return -1, None, None, None, None

    @staticmethod
    def _generate_hash(points, max_pairs=None):
        """
        This function generates a 4d hash code given an array of points.  The points can be in any units so long as both
        the first and second component have the same units, and they can be expressed in any conformal coordinate system
        (most typically expressed as right ascension, declination angular coordinates, or row, column pixel coordinates)

        This method is still under development and should not be used

        :param points: an array like input containing the first component of each point in the first row and the second
                       component in the second row
        :return hash_code: a numpy array containing the hash codes for all permutations of the input points.  Each
                           column corresponds to an element of the hash_code and each row is a hash code for a specific
                           permutation
        :return hash_inds: a numpy array containing the corresponding indices to stars A, B, C, D used to generate each
                           hash code.  Each row corresponds to the same row in hash_code
        """

        # # start by forming all possible combinations of 4 of the input points using the itertools combinations tool
        # if (max_pairs is None) or (max_pairs > comb(points.shape[1], 4)):
        #     # combos_array = np.array([combo for combo in it.combinations(points.T, 4)]).transpose((0, 2, 1))
        #     combos_inds = np.array([combo for combo in it.combinations(np.arange(points.shape[1]), 4)])
        # else:
        #     # combos_inds = np.random.choice(points.shape[-1], (max_pairs, 4), replace=False)
        #     combos_inds_s = set()
        #     while len(combos_inds_s) < max_pairs:
        #         combos_inds_s.add(tuple(sorted(random_combination(points.shape[-1], 4))))
        #
        #     combos_inds = np.array(list(combos_inds_s), dtype=int)
        #     # combos_inds = np.array([random_combination(points.shape[-1], 4) for i in range(max_pairs)])
        combos_inds = np.array(list(RandomCombinations(points.shape[-1], 4, max_pairs)))

        if combos_inds.size == 0:
            return None, None
        combos_array = points.T[combos_inds, :].transpose(0, 2, 1)

        # set up the data type for the numpy arrays that will contain information about the combinations
        # start by determining which pair of points form the a,b pair, and which form the c,d pair
        internal_dtype_ab = [('first', int), ('second', int)]
        internal_dtype_cd = [('third', int), ('fourth', int)]

        values = np.arange(0, 4)

        # create numpy arrays containing all possible combinations that could be ab pairs, do the same for cd pairs
        ab_sets = np.fromiter(it.combinations(values, 2), dtype=internal_dtype_ab)
        cd_sets = np.fromiter((tuple(set(values) - set(k)) for k in ab_sets), dtype=internal_dtype_cd)

        # check the distances between all the points in each set of 4
        distances = np.array([np.sqrt(np.sum(np.power(combos_array[:, :, pair[0]] - combos_array[:, :, pair[1]], 2),
                                             axis=1)) for pair in ab_sets])

        # determine which pair of points is spaced the furthest apart
        pairing_inds = np.argmax(distances, axis=0)

        # store which pair of points is spaced furthest apart
        ab_pairings = ab_sets[pairing_inds]
        cd_pairings = cd_sets[pairing_inds]

        # generate a numpy array containing subscripts for each combination
        hash_subscripts = np.arange(combos_array.shape[0])

        # determine the xy points if the 'first' point in each pair is point A
        xy01 = (combos_array[hash_subscripts, :, ab_pairings['second']] -
                combos_array[hash_subscripts, :, ab_pairings['first']]).T

        # and the same for if the 'second' point in point A
        xy02 = -xy01

        # calculate the rotation angle needed to enter into the hash space
        theta01 = np.pi / 4. - np.arctan2(xy01[1, :], xy01[0, :]).reshape((-1, 1, 1))
        theta02 = np.pi / 4. - np.arctan2(xy02[1, :], xy02[0, :]).reshape((-1, 1, 1))

        # form the rotation matrices to enter into hash space
        r1t = np.dstack((np.cos(theta01), -np.sin(theta01)))
        r1b = np.dstack((np.sin(theta01), np.cos(theta01)))
        rotation1 = np.dstack((r1t, r1b)).reshape((-1, 2, 2))

        r2t = np.dstack((np.cos(theta02), -np.sin(theta02)))
        r2b = np.dstack((np.sin(theta02), np.cos(theta02)))
        rotation2 = np.dstack((r2t, r2b)).reshape((-1, 2, 2))

        # rotate all of the points into hash space
        coords1 = np.matmul(rotation1, combos_array -
                            combos_array[hash_subscripts, :, ab_pairings['first']].reshape((-1, 2, 1)))
        coords2 = np.matmul(rotation2, combos_array -
                            combos_array[hash_subscripts, :, ab_pairings['second']].reshape((-1, 2, 1)))

        # scale all of the points into hash space
        coords1 /= coords1[hash_subscripts, :, ab_pairings['second']].reshape((-1, 2, 1))
        coords2 /= coords2[hash_subscripts, :, ab_pairings['first']].reshape((-1, 2, 1))

        # initialize a numpy array to store the hash codes
        hash_code = np.zeros((coords1.shape[0], 4), dtype=np.float64)
        hash_inds = np.zeros((coords1.shape[0], 4), dtype=int)

        # check to see points where the sum of the x coordinates for the 'third' and 'fourth' points is less than 1
        # this indicates that the first hash space is correct for these point quads
        x_coord_check = (coords1[hash_subscripts, 0, cd_pairings['third']] +
                         coords1[hash_subscripts, 0, cd_pairings['fourth']]) <= 1  # type: np.ndarray

        # determine the points with the minimum x coordinate in the first hash code space
        min_subs = np.argmin([coords1[hash_subscripts, 0, cd_pairings['third']],
                              coords1[hash_subscripts, 0, cd_pairings['fourth']]], axis=0)

        # form a "dictionary" to translate the minimum subscripts into structured array fields
        cd_dict = np.array(['third', 'fourth'])

        if x_coord_check.any():
            # form the hash codes for all points in which the first hash code space is correct
            temp_hash = np.array([[coords1[ind, 0, cd_pairings[ind][cd_dict[min_subs[ind]]]],
                                   coords1[ind, 1, cd_pairings[ind][cd_dict[min_subs[ind]]]],
                                   coords1[ind, 0, cd_pairings[ind][cd_dict[1 - min_subs[ind]]]],
                                   coords1[ind, 1, cd_pairings[ind][cd_dict[1 - min_subs[ind]]]]]
                                  for ind, check in enumerate(x_coord_check) if check])

            temp_inds = np.array([[combos_inds[ind, ab_pairings[ind]['first']],
                                   combos_inds[ind, ab_pairings[ind]['second']],
                                   combos_inds[ind, cd_pairings[ind][cd_dict[min_subs[ind]]]],
                                   combos_inds[ind, cd_pairings[ind][cd_dict[1 - min_subs[ind]]]]]
                                  for ind, check in enumerate(x_coord_check) if check])

            # store those hash codes
            hash_code[x_coord_check, :] = np.atleast_2d(temp_hash)
            hash_inds[x_coord_check, :] = np.atleast_2d(temp_inds)

            # now find the minimum x coordinates in the second hash code space
            min_subs = np.argmin([coords2[hash_subscripts, 0, cd_pairings['third']],
                                  coords2[hash_subscripts, 0, cd_pairings['fourth']]], axis=0)

        if (~x_coord_check).any():
            # generate hash codes for all quads for which the second has code space is correct
            temp_hash = np.array([[coords2[ind, 0, cd_pairings[ind][cd_dict[min_subs[ind]]]],
                                   coords2[ind, 1, cd_pairings[ind][cd_dict[min_subs[ind]]]],
                                   coords2[ind, 0, cd_pairings[ind][cd_dict[1 - min_subs[ind]]]],
                                   coords2[ind, 1, cd_pairings[ind][cd_dict[1 - min_subs[ind]]]]]
                                  for ind, check in enumerate(~x_coord_check) if check])

            temp_inds = np.array([[combos_inds[ind, ab_pairings[ind]['second']],
                                   combos_inds[ind, ab_pairings[ind]['first']],
                                   combos_inds[ind, cd_pairings[ind][cd_dict[min_subs[ind]]]],
                                   combos_inds[ind, cd_pairings[ind][cd_dict[1 - min_subs[ind]]]]]
                                  for ind, check in enumerate(~x_coord_check) if check])

            # store those hash codes and return the full result.
            hash_code[~x_coord_check, :] = temp_hash
            hash_inds[~x_coord_check, :] = temp_inds

        return hash_code, hash_inds
