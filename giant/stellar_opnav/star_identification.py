# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
This module provides the star identification routines for GIANT through the :class:`StarID` class.

Algorithm Description
_____________________

Star Identification refers to the process of matching observed stars in an image with a corresponding set of known star
locations from a star catalog. Making this identification is the first step in performing a number of OpNav tasks,
including attitude estimation, geometric camera calibration, and camera alignment, as well as a number of photometry
tasks like linearity checks and point spread function modelling.

In GIANT, star identification is handled using a random sampling and consensus (RANSAC) approach using the following
steps:

#. The *a priori* attitude information for each image is used to query the star catalog for the expected stars in the
   field of view of each image.
#. The retrieved catalog stars are transformed into the camera frame and projected onto the image using the *a priori*
   image attitude and camera model.
#. The projected catalog locations are paired with points in the image that were identified in the image by the image
   processing algorithm as potential stars using a nearest neighbor approach.
#. The initial pairs are thresholded based on the distance between the points, as well as for stars that are matched
   with 2 image points and image points that are close to 2 stars.
#. The remaining pairs are randomly sampled for 4 star pairs
#. The sample is used to estimate a new attitude for the image using the :class:`.ESOQ2` routines.
#. The new solved for attitude is used to re-rotate and project the catalog stars onto the image.
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
    as `astrometry.net <https://astrometry.net>`_ or `COTS Star Tracker <https://github.com/nasa/COTS-Star-Tracker>)  

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
:attr:`~.StarID.max_magnitude`        The maximum magnitude to query the star catalog to.  This is useful for
                                      limiting the number of catalog stars that are being matched against.
                                      Remember that stellar magnitude is on an inverse logarithmic scale, therefore
                                      the higher you set this number the dimmer stars that will be returned.
:attr:`~.StarID.min_magnitude`        The minimum magnitude to query the star catalog to.  This is useful for
                                      limiting the number of catalog stars that are being matched against.
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
:attr:`~.StarID.tolerance`            The maximum initial distance that a catalog-image poi pair can have for it to be
                                      considered a potential match in units of pixels. This is the tolerance that is
                                      applied before the RANSAC to filter out nearest neighbor pairs that are too far
                                      apart to be potential matches.
:attr:`~.StarID.ransac_tolerance`     The maximum post-fit distance that a catalog-image poi pair can have for it to
                                      be considered an inlier in the RANSAC algorithm in units of pixels.  This is
                                      the tolerance used inside of the RANSAC algorithm to determine the number of
                                      inliers for a given attitude solution from a sample.  This should always be
                                      less than the :attr:`~.StarID.tolerance` parameter.
:attr:`~.StarID.second_closest_check` A flag specifying whether to check if the second closest catalog star to an
                                      image poi is also within the :attr:`~.StarID.tolerance` distance.  This is
                                      useful for throwing out potential pairs that may be ambiguous.  In general you
                                      should set this flag to ``False`` when your initial attitude/camera model error is
                                      larger, and ``True`` after removing those large errors.
:attr:`~.StarID.unique_check`         A flag specifying whether to allow a single catalog star to be potentially
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
    catalog positions and are more likely to be picked up by the :class:`.ImageProcessing` algorithms
  * the :attr:`~.StarID.max_combos` set fairly large--on the order of 500-1000
  
* After getting the initial pairing and updating
  the attitude for the images (note that this is done external to the :class:`StarID` class), you can then attempt a 
  larger identification with dimmer stars

  * decreasing the :attr:`~.StarID.tolerance` to be about the same as your previous :attr:`~.StarID.ransac_tolerance`
  * turning the RANSAC algorithm off by setting the :attr:`~.StarID.max_combos` to 0
  * increasing the :attr:`~.StarID.max_magnitude`.
  
* If you are having problems getting the identification to work it can be useful to visually examine the results for a
  couple of images using the :func:`.show_id_results` function.

"""


# import random
import warnings
from typing import Optional, Tuple, cast
from multiprocessing import Pool
from dataclasses import dataclass, field

from copy import copy

import numpy as np
from numpy.typing import NDArray

from scipy import spatial as spat
from scipy.special import comb

from pandas import DataFrame

from giant.stellar_opnav.estimators import ESOQ2
from giant import catalogs as cat
from giant.camera_models import CameraModel
from giant.catalogs.meta_catalog import Catalog
from giant._typing import NONEARRAY, PATH
from giant.catalogs.utilities import RAD2DEG
from giant.utilities.spherical_coordinates import unit_to_radec
from giant.utilities.random_combination import RandomCombinations
from giant.image import OpNavImage
from giant.utilities.options import UserOptions
from giant.utilities.mixin_classes.attribute_equality_comparison import AttributeEqualityComparison
from giant.utilities.mixin_classes.attribute_printing import AttributePrinting
from giant.utilities.mixin_classes.user_option_configured import UserOptionConfigured
from giant._typing import DOUBLE_ARRAY


@dataclass
class StarIDOptions(UserOptions):
    """
    :param catalog: The catalog object to use to query for potential stars in an image.
    :param max_magnitude:  the maximum magnitude to return when querying the star catalog
    :param min_magnitude:  the minimum magnitude to return when querying the star catalog
    :param max_combos: The maximum number of random samples to try in the RANSAC routine
    :param tolerance: The maximum distance between a catalog star and a image point of interest for a potential
                        pair to be formed before the RANSAC algorithm
    :param ransac_tolerance: The maximum distance between a catalog star and an image point of interest after
                                correcting the attitude for a pair to be considered an inlier in the RANSAC algorithm.
    :param second_closest_check: A flag specifying whether to reject pairs where 2 catalog stars are close to an
                                    image point of interest
    :param unique_check: A flag specifying whether to allow a single catalog star to be potentially paired with
                            multiple image points of interest
    :param use_mp: A flag specifying whether to use the multi-processing library to accelerate the RANSAC algorithm
    """
    catalog: Catalog = field(default_factory=cat.Gaia)
    """
    The star catalog to use when pairing image points with star locations.
    
    This typically should be a subclass of the :class:`.Catalog` class.  It defaults to the 
    :class:`.Gaia`.
    """

    max_magnitude: float = 7
    """
    The maximum star magnitude to query from the star catalog.

    This specifies how dim stars are expected to be in the :attr:`extracted_image_points` data set.  This is 
    typically dependent on both the detector and the exposure length of the image under consideration.
    """

    min_magnitude: float = -10
    """
    The minimum star magnitude to query from the star catalog.
    
    This specifies how dim stars are expected to be in the :attr:`extracted_image_points` data set.  This is 
    typically dependent on both the detector and the exposure length of the image under consideration.
    
    Generally this should be left alone unless you are worried about over exposed stars (in which case 
    :attr:`.ImageProcessing.reject_saturation` may be more useful) or you are doing some special analysis.
    """
    
    max_combos: int = 100
    """
    The maximum number of random combinations to try in the RANSAC algorithm.  
    
    If the total possible number of combinations is less than this attribute then an exhaustive search will be 
    performed instead
        """
        
    tolerance: float = 20
    """
    The maximum distance in units of pixels between a projected catalog location and an extracted image point
    for a possible pairing to be made for consideration in the RANSAC algorithm.
    """
    
    ransac_tolerance: float = 5
    """
    The tolerance that is required after correcting for attitude errors for a pair to be considered an inlier
    in the RANSAC algorithm in units of pixels.
    
    This should always be less than the :attr:`tolerance` attribute.
    """
    
    # store the second closest check and uniqueness flags
    second_closest_check: bool = True 
    """
    A boolean specifying whether to ignore extracted image points where multiple catalog points are within the
    specified tolerance.
    """

    unique_check: bool = True
    """
    A boolean specifying whether to ignore possible catalog to image point pairs where multiple image points are 
    within the specified tolerance of a single catalog point.
    """
    
    use_mp: bool = False
    """
    A boolean flag specifying whether to use multi-processing to speed up the RANSAC process.
    
    If this is set to True then all available CPU cores will be utilized to parallelize the RANSAC algorithm 
    computations.  For small combinations, the overhead associated with this can swamp any benefit that may be 
    realized.
    """
    
    compute_weights: bool = False
    """
    A boolean specifying whether to compute the formal uncertainties for the unit vectors
                                and the pixel locations of the catalog stars.
    """

class StarID(UserOptionConfigured[StarIDOptions], StarIDOptions, AttributePrinting, AttributeEqualityComparison):
    """
    The StarID class operates on the result of image processing algorithms to attempt to match image points of interest
    with catalog star records.

    This is a necessary step in all forms of stellar OpNav and is a critical component of
    GIANT.

    In general, the user will not directly interface with the :class:`StarID` class and instead will use the
    :class:`.StellarOpNav` class.  Below we give a brief description of how to use this class directly for users who
    are just curious or need more direct control over the class.

    There are a couple things that the :class:`StarID` class needs to operate.  The first is a camera model, which
    should be a subclass of :class:`.CameraModel`.  The camera model is used to both project catalog star locations
    onto the image, as well as generate unit vectors through the image points of interest in the camera frame.  The
    next thing the :class:`StarID` class needs is a star catalog to query.  This should come from the
    :mod:`.catalogs` package and provides all of the necessary information for retrieving and projecting the expected
    stars in an image. Both the star catalog and camera model are generally set at the construction of the class
    and apply to every image being considered, so they are rarely updated.  The camera model is stored in the
    :attr:`model` attribute and is also specified as the first positional argument for the class constructor.  The
    catalog is stored in the :attr:`catalog` attribute and can also be specified in the class constructor as a
    keyword argument of the same name.

    The :class:`StarID` class also needs some information about the current image being considered.  This information
    includes points of interest for the image that need to be matched to stars, the *a priori* attitude of the image,
    and the position/velocity of the camera at the time the image was captured.  The points of interest are generally
    returned from the :class:`.PointOfInterestFinder` routines, although they don't need to be.  The camera attitude,
    position, velocity, and a priori orientation are generally passed from the :class:`.OpNavImage` metadata.  The 
    a priori image attitude is used for querying the catalog and rotating the catalog stars into the image frame.  
    The camera positions and velocity are used for correcting the star locations for parallax and stellar aberration. 
    The camera position and velocity are not required but are generally recommended as they will give a more accurate 
    representation.  
     
    Finally, there are a number of tuning parameters that need set.  These parameters are discussed in depth in the
    :ref:`Tuning Parameters Table <tuning-parameters-table>`.

    When everything is correctly set in an instance of :class:`StarID`, then generally all that needs to be called
    is the :meth:`id_stars` method, which accepts the observation date of the image being considered as an
    optional ``epoch`` keyword argument.  This method will go through the whole processed detailed above, storing the
    results in a number of attributes that are detailed below.
    """

    def __init__(self, model: CameraModel, options: Optional[StarIDOptions] = None):
        """
        :param model: The camera model to use to relate vectors in the camera frame with points on the image
        :param options: StarIDOptions
        """
        
        super().__init__(StarIDOptions, options=options)

        # initialize temporary attributes to make multiprocessing easier
        self._temp_image_locs = None
        self._temp_image_dirs = None
        self._temp_catalog_dirs = None
        self._temp_temperature = 0
        self._temp_image_number = 0
        self._temp_combinations = None
        self._temp_att_est = ESOQ2()
        self._temp_att_est.weighted_estimation = False
        self._temp_att_est.n_iter = 0

        self.model: CameraModel = model 
        """
        The camera model which relates points in the camera frame to points in the image and vice-versa.
        """

        # initialize the attributes for storing the star identification results
        self.queried_catalog_image_points: DOUBLE_ARRAY | None = None 
        """
        A 2xn numpy array of points containing the projected image points for all catalog stars that were queried from 
        the star catalog  with x (columns) in the first row and y (rows) in the second row.  
        
        Each column corresponds to the same row in :attr:`queried_catalog_star_records`.
        
        Until :meth:`project_stars` is called this will be ``None``.
        """

        self.queried_catalog_star_records: Optional[DataFrame] = None 
        """
        A pandas DataFrame of all the catalog star records that were queried.  

        See the :class:`.Catalog` class for a description of the columns of the dataframe.
        
        Until :meth:`project_stars` is called this will be ``None``.
        """

        self.queried_catalog_unit_vectors: DOUBLE_ARRAY | None = None 
        """
        A 3xn numpy array of unit vectors in the inertial frame for all catalog stars that were queried from 
        the star catalog.  
        
        Each column corresponds to the same row in :attr:`queried_catalog_star_records`.

        Until :meth:`project_stars` is called this will be ``None``.
        """

        self.queried_weights_inertial: DOUBLE_ARRAY | None = None 
        """
        This contains the formal total uncertainty for each unit vector from the queried catalog stars.  
        
        Each element in this array corresponds to the same row in the :attr:`queried_catalog_star_records`.
        
        Until :meth:`id_stars` is called this will be ``None``.
        """

        self.queried_weights_picture: DOUBLE_ARRAY | None = None 
        """
        This contains the formal total uncertainty for each projected pixel location from the queried catalog stars in
        units of pixels..  
        
        Each element in this array corresponds to the same row in the :attr:`queried_catalog_star_records`.

        Until :meth:`id_stars` is called this will be ``None``.
        """

        self.unmatched_catalog_image_points: DOUBLE_ARRAY | None = None
        """
        A 2xn numpy array of points containing the projected image points for all catalog stars that not matched
        with an extracted image point, with x (columns) in the first row and y (rows) in the second row.  

        Each column corresponds to the same row in :attr:`unmatched_catalog_star_records`.

        Until :meth:`id_stars` is called this will be ``None``.
        """

        self.unmatched_catalog_star_records: DataFrame | None = None 
        """
        A pandas DataFrame of all the catalog star records that were not matched to an extracted image point in the 
        star identification routine.  

        See the :class:`.Catalog` class for a description of the columns of the dataframe.

        Until :meth:`id_stars` is called this will be ``None``.
        """

        self.unmatched_catalog_unit_vectors: DOUBLE_ARRAY | None = None 
        """
        A 3xn numpy array of unit vectors in the inertial frame for all catalog stars that were not matched to an
        extracted image point in the star identification routine.  

        Each column corresponds to the same row in :attr:`matched_catalog_star_records`.

        Until :meth:`id_stars` is called this will be ``None``.
        """

        self.unmatched_extracted_image_points: DOUBLE_ARRAY | None = None 
        """
        A 2xn array of the image points of interest that were not paired with a catalog star in the star 
        identification routine.

        The first row corresponds to the x locations (columns) and the second row corresponds to the y locations (rows).
        
        Until :meth:`id_stars` is called this will be ``None``.
        """

        self.unmatched_weights_inertial: DOUBLE_ARRAY | None = None 
        """
        This contains the formal total uncertainty for each unit vector from the queried catalog stars
        that were not matched with an extracted image point.  
        
        Each element in this array corresponds to the same row in the :attr:`unmatched_catalog_star_records`.

        Until method :meth:`id_stars` is called this will be ``None``.
        """

        self.unmatched_weights_picture: DOUBLE_ARRAY | None = None 
        """
        This contains the formal total uncertainty for each projected pixel location from the queried catalog stars 
        that were not matched with an extracted image point in units of pixels. 
        
        Each element in this array corresponds to the same row in the :attr:`unmatched_catalog_star_records`.

        Until method :meth:`id_stars` is called this will be ``None``.
        """

        self.matched_catalog_image_points: DOUBLE_ARRAY | None = None 
        """
        A 2xn numpy array of points containing the projected image points for all catalog stars that were matched
        with an extracted image point, with x (columns) in the first row and y (rows) in the second row.  

        Each column corresponds to the same row in :attr:`matched_catalog_star_records`.

        Until :meth:`id_stars` is called this will be ``None``.
        """

        self.matched_catalog_star_records: DataFrame | None = None 
        """
        A pandas DataFrame of all the catalog star records that were matched to an extracted image point in the 
        star identification routine.  

        See the :class:`.Catalog` class for a description of the columns of the dataframe.
        
        Each row of the dataframe corresponds to the same column index in the :attr:`matched_extracted_image_points`.

        Until :meth:`id_stars` is called this will be ``None``.
        """

        self.matched_catalog_unit_vectors: DOUBLE_ARRAY | None = None 
        """
        A 3xn numpy array of unit vectors in the inertial frame for all catalog stars that were matched to an
        extracted image point in the star identification routine.  

        Each column corresponds to the same row in :attr:`matched_catalog_star_records`.

        Until :meth:`id_stars` is called this will be ``None``.
        """

        self.matched_extracted_image_points: DOUBLE_ARRAY | None = None 
        """
        A 2xn array of the image points of interest that were not paired with a catalog star in the star 
        identification routine.

        The first row contains to the x locations (columns) and the second row contains to the y locations (rows).
        
        Each column corresponds to the same row in the :attr:`matched_catalog_star_records` for its pairing.

        Until :meth:`id_stars` is called this will be ``None``.
        """

        self.matched_weights_inertial: DOUBLE_ARRAY | None = None 
        """
        This contains the formal total uncertainty for each unit vector from the queried catalog stars
        that were matched with an extracted image point.  

        Each element in this array corresponds to the same row in the :attr:`matched_catalog_star_records`.

        Until methods :meth:`id_stars` is called this will be ``None``.
        """

        self.matched_weights_picture: DOUBLE_ARRAY | None = None  
        """
        This contains the formal total uncertainty for each projected pixel location from the queried catalog stars 
        that were matched with an extracted image point in units of pixels. 

        Each element in this array corresponds to the same row in the :attr:`matched_catalog_star_records`.

        Until method :meth:`id_stars` is called this will be ``None``.
        """

    def query_catalog(self, image: OpNavImage):
        """
        This method queries stars from the catalog within the field of view.

        The stars are queried such that any stars within 1.3*the :attr:`.CameraModel.field_of_view` value radial
        distance of the camera frame z axis converted to right ascension and declination are returned between
        :attr:`min_magnitude` and :attr:`max_magnitude`.  The queried stars are updated to the 
        ``image.observation_date`` value using proper motion.  They are stored in the 
        :attr:`queried_catalog_star_records` attribute. The stars are stored as a pandas DataFrame.  For 
        more information about this format see the :class:`.Catalog` class documentation.

        In general, this method does not need to be directly called by the user as it is automatically called in the
        :meth:`project_stars` method.

        :param image: The image to querry the catalog for (specifying the time and the a priori pointing)
        """

        # get the ra and dec of the camera frame z axis
        ra_dec_cat = np.array(self.compute_pointing(image))

        # query the catalog and store the results
        self.queried_catalog_star_records = self.catalog.query_catalog(
            search_center=ra_dec_cat,
            search_radius=1.3 * self.model.field_of_view,
            min_mag=self.min_magnitude,
            max_mag=self.max_magnitude,
            new_epoch=image.observation_date
        )

    @staticmethod
    def compute_pointing(image: OpNavImage) -> Tuple[float, float]:
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

        In general this method is not used by the user as it is automatically called in the :meth:`query_catalog`
        method.

        :return:  The right ascension and declination of the specified axis in the inertial frame as a tuple (ra, dec)
                  in units of degrees.
        """

        boresight_cat = image.rotation_inertial_to_camera.matrix[2]

        ra, dec = cast(tuple[float, float], unit_to_radec(boresight_cat))

        return RAD2DEG * ra, RAD2DEG * dec

    def project_stars(self, image: OpNavImage, image_number: int = 0):
        """
        This method queries the star catalog for predicted stars within the field of view and projects those stars
        onto the image using the camera model.

        The star catalog is queried using the :meth:`query_catalog` method and the stars are updated to the epoch
        specified by ``image.observation_date`` using the proper motion from the catalog.  The queried Pandas 
        Dataframe containing the star catalog records is stored in the :attr:`queried_catalog_star_records` attribute.

        After the stars are queried from the catalog, they are converted to inertial unit vectors and corrected for
        stellar aberration and parallax using the :attr:`.position` and :attr:`camera_velocity` values.  The
        corrected inertial vectors are stored in the :attr:`queried_catalog_unit_vectors`.

        Finally, the unit vectors are rotated into the camera frame using the :attr:`a_priori_rotation_cat2camera`
        attribute, and then projected onto the image using the :attr:`model` attribute.  The projected points are stored
        in the :attr:`queried_catalog_image_points` attribute.

        If requested, the formal uncertainties for the catalog unit vectors and pixel locations are computed and
        stored in the :attr:`queried_weights_inertial` and :attr:`queried_weights_picture`.  These are computed by
        transforming the formal uncertainty on the right ascension, declination, and proper motion specified in the
        star catalog into the proper frame.

        In general this method is not called directly by the user and instead is called in the :meth:`id_stars` method.

        :param epoch: The epoch to get the star locations for
        :param compute_weights: A boolean specifying whether to compute the formal uncertainties for the unit vectors
                                and the pixel locations of the catalog stars.
        :param temperature: The temperature of the camera at the time of the image being processed
        :param image_number: The number of the image being processed
        """

        # # query the star catalog for predicted stars in the field of view
        # make a temporary OpNavImage.
        # TODO: probably we should rework this class to just operate on an OpNavImage
        (self.queried_catalog_star_records, 
         self.queried_catalog_unit_vectors, 
         catalog_unit_vectors_camera,
         self.queried_catalog_image_points) = self.catalog.get_stars_directions_and_pixels(image, 
                                                                                           self.model, 
                                                                                           self.max_magnitude, 
                                                                                           min_mag=self.min_magnitude, 
                                                                                           image_number=image_number)

        if self.compute_weights:
            # compute the covariance of the inertial catalog unit vectors
            dec_rad = np.deg2rad(self.queried_catalog_star_records['dec'].to_numpy())
            ra_rad = np.deg2rad(self.queried_catalog_star_records['ra'].to_numpy())
            cos_d = np.cos(dec_rad)
            cos_a = np.cos(ra_rad)
            sin_d = np.sin(dec_rad)
            sin_a = np.sin(ra_rad)
            zero = np.zeros(cos_d.shape)

            dv_da = np.array([-cos_d * sin_a, cos_d * sin_a, zero])
            dv_dd = np.array([-sin_d * cos_a, -sin_d * sin_a, cos_d])

            cov_v = (np.einsum('ij,jk->jik', dv_da, dv_da.T) *
                     (np.rad2deg(self.queried_catalog_star_records['ra_sigma'].to_numpy()) / cos_d).reshape(-1, 1, 1) ** 2
                     + np.einsum('ij,jk->jik', dv_dd, dv_dd.T) *
                     (self.queried_catalog_star_records['dec_sigma'].to_numpy().reshape(-1, 1, 1) / RAD2DEG) ** 2)

            # compute the covariance of the projected catalog points
            rot2camera = image.rotation_inertial_to_camera.matrix
            cov_xc = rot2camera @ cov_v @ rot2camera.T
            pj = self.model.compute_pixel_jacobian(catalog_unit_vectors_camera, temperature=image.temperature,
                                                   image=image_number)
            cov_xp = pj @ cov_xc @ pj.swapaxes(-1, -2)

            self.queried_weights_inertial = np.trace(cov_v, axis1=-2, axis2=-1)
            self.queried_weights_picture = np.diagonal(cov_xp, axis1=-2, axis2=-1)

    def id_stars(self, image: OpNavImage, extracted_image_points: DOUBLE_ARRAY, 
                 image_number: int = 0) -> Tuple[Optional[NDArray[np.bool]], Optional[NDArray[np.bool]]]:
        """
        This method attempts to match the image points of interest with catalog stars.

        The :meth:`id_stars` method is the primary interface of the :class:`StarID` class.  It performs all the tasks of
        querying the star catalog, performing the initial pairing using a nearest neighbor search, refining the
        initial pairings with the :attr:`second_closest_check` and :attr:`unique_check`, and passing the refined
        pairings to the RANSAC routines.  The matched and unmatched catalog stars and image points of interest are
        stored in the appropriate attributes.

        This method also returns a boolean index in the image points of interest vector, which extracts the image points
        that met the initial match criterion, and another boolean index into the image points of interest which
        extracts the image points of interest that were matched by the RANSAC algorithms.  This can be used to select
        the appropriate meta data about catalog stars or stars found in an image that isn't explicitly considered by
        this class (as is done in the :class:`.StellarOpNav` class), but if you do not have extra information you need
        to keep in sync, then you can ignore the output.

        If requested, the formal uncertainties for the catalog unit vectors and pixel locations are computed and
        stored in the :attr:`queried_weights_inertial` and :attr:`queried_weights_picture`.  These are computed by
        transforming the formal uncertainty on the right ascension, declination, and proper motion specified in the
        star catalog into the proper frame.

        :param image: The image being processed
        :param extracted_image_points: a 2xN array of image pixel locations (x first row, y second row) that are potential stars
        :param image_number: The number of the image being processed
        :return: The boolean index into the image points that met the original pairing criterion, and a second boolean
                 index into the the result from the previous boolean index that extracts the image points that were
                 successfully matched in the RANSAC algorithms.  If no stars are identified then returns a tuple of None, None
        """

        # first get the unit vectors and image locations for the stars in the field of view
        self.project_stars(image, image_number=image_number)

        # create a kdtree of the catalog image locations for faster searching
        assert self.queried_catalog_image_points is not None, "Need to have querried the catalog stors at this point"
        assert self.queried_catalog_star_records is not None, "Need to have querried the catalog stors at this point"
        assert self.queried_catalog_unit_vectors is not None, "Need to have querried the catalog stors at this point"
        catalog_image_locations_kdtree = spat.KDTree(self.queried_catalog_image_points.T)

        # query the kdtree to get the 2 closest catalog image locations to each image point of interest
        if not extracted_image_points.any():
            return None, None
        distance, inds = catalog_image_locations_kdtree.query(extracted_image_points.T, k=2)
        inds = np.asanyarray(inds) # for type hinting

        # check to see which pairs are less than the user specified matching tolerance
        dist_check = distance[:, 0] <= self.tolerance

        # throw out pairs where multiple catalog locations are < the tolerance to the image points of interest
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
            self.unmatched_catalog_image_points = self.queried_catalog_image_points
            self.unmatched_catalog_star_records = self.queried_catalog_star_records
            self.unmatched_catalog_unit_vectors = self.queried_catalog_unit_vectors
            self.unmatched_weights_inertial = self.queried_weights_inertial
            self.unmatched_weights_picture = self.queried_weights_picture
            self.unmatched_extracted_image_points = extracted_image_points
            return None, None
        # either return our current matches or further filter using ransac if desired
        if self.max_combos:
            self.matched_extracted_image_points, self.matched_catalog_unit_vectors, keep_inliers = self.ransac(
                extracted_image_points[:, keep_stars], self.queried_catalog_unit_vectors[:, inds[keep_stars, 0]],
                temperature=image.temperature, image_number=image_number
            )

            if keep_inliers is None:  # if none of the stars met the ransac criteria then throw everything out
                warnings.warn("no stars found for epoch {0}".format(image.observation_date))
                self.matched_catalog_star_records = None
                self.matched_catalog_image_points = None
                if self.compute_weights:
                    self.matched_weights_inertial = None
                    self.matched_weights_picture = None

            else:
                # update the matched catalog star records and image points
                self.matched_catalog_star_records = self.queried_catalog_star_records.iloc[
                    inds[keep_stars, 0]][keep_inliers]
                self.matched_catalog_image_points = self.queried_catalog_image_points[
                                                      :, inds[keep_stars, 0]][:, keep_inliers]

                if self.compute_weights:
                    assert self.queried_weights_inertial is not None, "Weights shouldn't be None at this point"
                    assert self.queried_weights_picture is not None, "Weights shouldn't be None at this point"
                    self.matched_weights_inertial = self.queried_weights_inertial[inds[keep_stars, 0]][keep_inliers]
                    self.matched_weights_picture = self.queried_weights_picture[inds[keep_stars, 0]][keep_inliers]

        else:
            # set the matches in the proper places
            self.matched_extracted_image_points = extracted_image_points[:, keep_stars]
            self.matched_catalog_image_points = self.queried_catalog_image_points[:, inds[keep_stars, 0]]
            self.matched_catalog_unit_vectors = self.queried_catalog_unit_vectors[:, inds[keep_stars, 0]]
            self.matched_catalog_star_records = self.queried_catalog_star_records.iloc[inds[keep_stars, 0]]
            if self.compute_weights:
                assert self.queried_weights_inertial is not None, "Weights shouldn't be None at this point"
                assert self.queried_weights_picture is not None, "Weights shouldn't be None at this point"
                self.matched_weights_inertial = self.queried_weights_inertial[inds[keep_stars, 0]]
                self.matched_weights_picture = self.queried_weights_picture[inds[keep_stars, 0]]

            keep_inliers = np.ones(self.matched_extracted_image_points.shape[1], dtype=bool)

        # use python set notation to determine the list of stars and image points that were never matched
        if keep_inliers is not None:

            # get the stars that weren't matched
            unmatched_inds = list({*np.arange(self.queried_catalog_image_points.shape[1])} -
                                  {*(inds[keep_stars, 0][keep_inliers])})

            self.unmatched_catalog_image_points = self.queried_catalog_image_points[:, unmatched_inds].copy()
            self.unmatched_catalog_star_records = self.queried_catalog_star_records.iloc[unmatched_inds].copy()
            self.unmatched_catalog_unit_vectors = self.queried_catalog_unit_vectors[:, unmatched_inds].copy()
            if self.compute_weights:
                assert self.queried_weights_inertial is not None, "Weights shouldn't be None at this point"
                assert self.queried_weights_picture is not None, "Weights shouldn't be None at this point"
                self.unmatched_weights_inertial = self.queried_weights_inertial[unmatched_inds].copy()
                self.unmatched_weights_picture = self.queried_weights_picture[unmatched_inds].copy()

            # get the points of interest that weren't matched
            camera_inds = np.arange(extracted_image_points.shape[1])
            unmatched_centroid_inds = list({*camera_inds} - {*camera_inds[keep_stars][keep_inliers]})
            self.unmatched_extracted_image_points = extracted_image_points[:, unmatched_centroid_inds].copy()

        else:  # nothing was matched
            self.unmatched_extracted_image_points = extracted_image_points.copy()
            self.unmatched_catalog_star_records = self.queried_catalog_star_records.copy()
            self.unmatched_catalog_unit_vectors = self.queried_catalog_unit_vectors.copy()
            self.unmatched_catalog_image_points = self.queried_catalog_image_points.copy()
            if self.compute_weights:
                assert self.queried_weights_inertial is not None, "Weights shouldn't be None at this point"
                assert self.queried_weights_picture is not None, "Weights shouldn't be None at this point"
                self.unmatched_weights_inertial = self.queried_weights_inertial.copy()
                self.unmatched_weights_picture = self.queried_weights_picture.copy()

        return keep_stars, keep_inliers

    def ransac(self, image_locs: DOUBLE_ARRAY, catalog_dirs: DOUBLE_ARRAY,
               temperature: float, image_number: int) -> tuple[DOUBLE_ARRAY, DOUBLE_ARRAY, NDArray[np.bool]] | tuple[None, None, None]:
        """
        This method performs RANSAC on the image poi-catalog location pairs.

        The RANSAC algorithm is described below

        #. The pairs are randomly sampled for 4 star pairs
        #. The sample is used to estimate a new attitude for the image using the :class:`.ESOQ2` routines.
        #. The new solved for attitude is used to re-rotate and project the catalog stars onto the image.
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

        In order to use this method, the ``image_locs`` input and the ``catalog_dirs`` input should represent the
        initial pairings between the image points found using image processing and the predicted catalog star unit
        vectors in the inertial frame. The columns in these 2 arrays should represent the matched pairs (that is column
        10 of ``image_locs`` should correspond to column 10 in ``catalog_dirs``).

        This method returns the paired image locations and catalog directions from the best RANSAC iteration
        and the boolean index into the input arrays that extract these values.

        In general this method is not used directly by the user and instead is called as part of the :meth:`id_stars`
        method.

        :param image_locs:  The image points of interest that met the initial matching criteria as a 2xn array
        :param catalog_dirs:  The catalog inertial unit vectors that met the initial matching criteria in the same
                                order as the ``image_locs`` input as a 3xn array.
        :param temperature: The temperature of the camera at the time of the image being processed
        :param image_number: The number of the image being processed
        :return: The matched image points of interest, the matched catalog unit vectors, and the boolean index that
                 represents these arrays
        """

        # initialize the maximum number of inliers and minimum sum of squares variables.
        max_inliers = 0
        max_rs: float = 2 * self.ransac_tolerance ** 2 * image_locs.shape[1]

        # get the maximum number of combinations that are available to sample
        n_comb = int(comb(image_locs.shape[1], min(image_locs.shape[1] - 1, 4)))

        # convert the image points of interest to unit vectors in the camera frame
        image_dirs = self.model.pixels_to_unit(image_locs, temperature=temperature, image=image_number)

        self._temp_image_locs = image_locs
        self._temp_image_dirs = image_dirs
        self._temp_catalog_dirs = catalog_dirs
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
        keep_catalog_dirs = None
        keep_inliers = None

        # find the best iteration and keep it
        for num_inliers, filtered_image_locs, filtered_catalog_dirs, inliers, rs in res:

            # check to see if this is the best ransac iteration yet
            if num_inliers > max_inliers:

                max_inliers = num_inliers

                keep_image_locs = filtered_image_locs

                keep_catalog_dirs = filtered_catalog_dirs

                keep_inliers = inliers

                assert rs is not None
                max_rs = rs

            elif num_inliers == max_inliers:
                assert rs is not None 
                if rs < max_rs:
                    max_inliers = num_inliers

                    keep_image_locs = filtered_image_locs

                    keep_catalog_dirs = filtered_catalog_dirs

                    keep_inliers = inliers

                    max_rs = rs

        # clear out the temp data
        self._temp_image_locs = None
        self._temp_image_dirs = None
        self._temp_catalog_dirs = None
        self._temp_temperature = 0
        self._temp_image_number = 0

        # return the matched results and the boolean index
        return keep_image_locs, keep_catalog_dirs, keep_inliers # type: ignore

    def ransac_iter_test(self, iter_num: int) -> Tuple[int, 
                                                       DOUBLE_ARRAY | None, 
                                                       DOUBLE_ARRAY | None, 
                                                       NDArray[np.bool] | None, 
                                                       float | None]:
        """
        This performs a single ransac iteration.

        See the :meth:`ransac` method for more details.

        :param iter_num: the iteration number for retrieving the combination to try
        :return: the number of inliers for this iteration, the image location inliers for this iteration, the
                 catalog direction inliers for this iteration, the boolean index for the inliers for this iteration,
                 and the sum of the squares of the residuals for this iteration
        """
        
        assert (self._temp_combinations is not None and
                self._temp_image_locs is not None and
                self._temp_att_est is not None and 
                self._temp_image_dirs is not None and
                self._temp_catalog_dirs is not None)
        image_locs = self._temp_image_locs
        image_dirs = self._temp_image_dirs
        catalog_dirs = self._temp_catalog_dirs

        # get a random combination of indices into the image_locs and catalog_dirs arrays
        # inds = random_combination(image_locs.shape[1], min(image_locs.shape[1] - 1, 4))
        inds = self._temp_combinations[iter_num]

        # extract the image directions to use for this ransac iteration
        image_dirs_use = image_dirs[:, inds]

        # extract the catalog directions ot use for this ransac iteration
        catalog_dirs_use = catalog_dirs[:, inds]

        # estimate an updated attitude
        new_rot = self._temp_att_est.estimate(image_dirs_use, catalog_dirs_use, None).matrix

        # rotate the catalog directions into the camera frame and project them onto the image using the new
        # attitude
        catalog_dirs_cam = np.matmul(new_rot, catalog_dirs)

        catalog_locs = self.model.project_onto_image(catalog_dirs_cam, temperature=self._temp_temperature,
                                                       image=self._temp_image_number)

        # compute the residual distance in all of the pairs
        resids: DOUBLE_ARRAY = np.linalg.norm(catalog_locs - image_locs, axis=0)

        # check to see which pairs meet the ransac tolerance
        inliers = resids < self.ransac_tolerance

        # get the sum of the squares of the residuals
        rs: float = np.sum(resids[inliers] * resids[inliers])

        if inliers.any():
            return inliers.sum(), image_locs[:, inliers], catalog_dirs[:, inliers], inliers, rs

        else:
            return -1, None, None, None, None
