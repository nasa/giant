


"""
This package provides the required routines and objects to identify stars in an image and to estimate attitude based on
those identified stars.

Description
___________
In GIANT, Stellar OpNav refers to the process of identifying stars in an image and then extracting the attitude 
information from those stars. There are many different sub-steps that need to be performed for all parts of Stellar
OpNav, which can lead to cluttered scripts and hard to maintain code when everything is thrown together.  Luckily, GIANT
has done most of the work for us, and created a simple single interface to perform all portions of Stellar OpNav through
the :class:`.StellarOpNav` class.

The :class:`.StellarOpNav` class is generally the only interface a user will require when performing Stellar OpNav, as
it provides easy access to every component you need.  It also abstracts away most of the nitty gritty details into 2
simple method calls :meth:`~.StellarOpNav.id_stars` and :meth:`~.StellarOpNav.estimate_attitude` which means you can
perform Stellar OpNav without an in depth understanding of what's going on in the background (though at least a basic
understanding certainly helps).  That being said, the substeps are also exposed throughout GIANT, so if you are doing
advanced design or analysis it is easy to get these components as well.

This package level documentation only focuses on the use of the class and some techniques for successfully performing
Stellar OpNav (in the upcoming tuning section).  To see behind the scenes of what's going on, refer to the submodule
documentations from this package
(:mod:`.stellar_class`, :mod:`.star_identification`, and :mod:`~.stellar_opnav.estimators`).

Tuning for Successful Stellar OpNav
___________________________________
The process of tuning the Stellar OpNav routines is both a science and an art.  In general, you should be able to find a
single set of tuning parameters that applies to a number of similar images (similar exposure times, similar scene, etc),
but this often takes a decent amount of experimentation.  With this section, we hope to introduce you to all the
different nobs you can turn when performing Stellar OpNav and give you some basic tips to getting a good tuning for a
wide range of images.

The primary goal with tuning in Stellar OpNav is to successfully identify a good number of stars in every image.  In
general, at least 2 stars are required for each image to be able to solve for an updated rotation matrix for the image,
but in practice it is best to get at least 4, and generally more is better.  In addition, it may frequently be useful to
perform other types of analysis on the images using identified stars, and in these cases more is almost always
better.  Therefore we will be shooting to get the tuning that provides the most correctly identified stars as possible.

Each of the parameters you will need is discussed briefly in the following table.  :class:`.PointOfInterestFinder` attributes
can easily be accessed through the :attr:`~.StellarOpNav.point_of_interest_finder` property, while :class:`.StarID`
attributes can easily be accessed through the :attr:`~.StellarOpNav.star_id` attribute.  For more detailed
descriptions of these attributes see the :class:`.PointOfInterestFinder` or :class:`.StarID` documentation.

==================================================== ======================================================================
 Attribute                                           Description
==================================================== ======================================================================
:attr:`.PointOfInterestFinder.point_spread_function` This function is used to compute the subpixel center of the points
                                                     of interest in the image. Changing this function can change where
                                                     your image points are identified.
:attr:`.PointOfInterestFinder.centroid_size`         The size of the sub-image to use when centroiding a point of
                                                     interest.  Changing the size of this parameter changes how big of a
                                                     widow is centeroided when identifying the subpixel centers of points
                                                     of interest in the image. The value for this depends on the
                                                     centroiding function you are using, but typical values range anywhere
                                                     from 1 (3x3 grid) to 10 (21x21 grid) depending on the centroid
                                                     function used and the expected PSF of the camera.  If you are using
                                                     a centroid function that doesn't estimate background then smaller is
                                                     usually better.
:attr:`.PointOfInterestFinder.threshold`             The threshold to use when identifying points of interest in
                                                     an image. This largely controls how many image points of interest are
                                                     extracted from the image. The higher this number is, the less points
                                                     of interest that will be extracted.  In general you want to set this
                                                     fairly high for an initial identification (in the range of 20-40) and
                                                     then lower for a subsequent identification (in the range of 8-20).
:attr:`.PointOfInterestFinder.max_size`              The maximum size for a blob before it is no longer considered a point
                                                     of interest. This can be used to exclude extended bodies, however, it
                                                     also can exclude really bright stars if set too low.  Generally a
                                                     setting of about 100 is good, which corresponds to a 10x10 window of
                                                     points above the threshold.
:attr:`.PointOfInterestFinder.min_size`              The minimum size for a blob before it is no longer considered a point
                                                     of interest. This can be used to exclude noise spikes from being
                                                     identified as points of interest, but it can also exclude very faint
                                                     stars. Generally a setting of 2-5 is good for an initial
                                                     identification, and a setting of 1-2 is good for a subsequent
                                                     identification trying to get dim stars.
:attr:`.PointOfInterestFinder.reject_saturation`     A flag specifying whether to include blobs which contain
                                                     saturated pixels as points of interest or not. In general you should
                                                     set this flag to True, as overexposed pixels can severely degrade
                                                     the centroiding accuracy.
:attr:`.StarID.max_magnitude`                        The maximum magnitude to retrieve when querying the star catalog.
                                                     The star magnitude is an inverse logarithmic scale of brightness,
                                                     which means the larger the magnitude, the dimmer the star.  In general
                                                     this should be set fairly low for an initial identification, and then
                                                     set higher for a subsequent identification of dimmer stars.
:attr:`.StarID.max_combos`                           The maximum number of combinations to try when performing the RANSAC
                                                     star identification. The higher this number, the more likely a
                                                     combination of correctly identified stars will be processed; however,
                                                     the amount of computation time will also increase.  In general values
                                                     between 500 and 1000 provide a good tradeoff between speed and
                                                     accuracy.
:attr:`.StarID.tolerance`                            The maximum initial separation between a catalog star and an image
                                                     star for a pair to be formed. Setting this parameter correctly is one
                                                     of the most crucial components of getting good star identification
                                                     results. For the first identification, this number must encompass the
                                                     errors in the projected star locations due to the error in the star
                                                     catalog (generally small), the error in the centroiding of the
                                                     image points of interest (generally small), the error caused by the
                                                     camera model (may be small or large), and the error caused by the
                                                     attitude error for the image (generally large).  After the initial
                                                     identification, the attitude error is largely removed and this value
                                                     can be set much smaller.  The correct value for this parameter can
                                                     vary widely from camera to camera and spacecraft to spacecraft so
                                                     there are no good rough estimates to provide for this setting.
:attr:`.StarID.ransac_tolerance`                     The maximum separation between a catalog star and an image star for
                                                     a pair to be formed after correcting the attitude in the RANSAC
                                                     routine. This value should generally be set fairly small since the
                                                     attitude error should be removed correctly for a correct RANSAC
                                                     iteration.  If there is a good camera model and good star catalog
                                                     being used then setting this between 1-5 is usually appropriate.
:attr:`.StarID.second_closest_check`                 Check if two catalog stars fall within the pair
                                                     tolerance for a single image point of interest. In general, you may
                                                     want to set this to False for an initial ID, and then set it to True
                                                     for a subsequent ID when trying to identify dimmer stars.
:attr:`.StarID.unique_check`                         Check if two image points of interest fall within the pair
                                                     tolerance for a single catalog star. In general, you may want to set
                                                     this to False for an initial ID, and then set it to True for a
                                                     subsequent ID when trying to identify dimmer stars.
================================================= ======================================================================

As you can see, there are 3 different processes that need tuned for a successful star identification, the image 
processing, the catalog query, and the identification routines themselves.  The following are a few suggestions for
attempting to find the correct tuning.

* Getting the initial identification is generally the most difficult; therefore, you should generally have 2 tunings
  for an image set.
* The first tuning should be fairly conservative in order to get a good refined attitude estimate for the image.  
  (Remember that we really only need 4 or 5 correctly identified stars to get a good attitude estimate.) 
  
  * :attr:`~.PointOfInterestFinder.threshold` set fairly high (around 20-40)
  * :attr:`~.StellarOpNav.denoising` set to something reasonable like :class:`.GaussianDenoising`
  * a large initial :attr:`~.StarID.tolerance`--typically greater than 10 pixels.  Note that this initial
    tolerance should include the errors in the star projections due to both the *a priori* attitude uncertainty and the
    camera model
  * a smaller but still relatively large :attr:`~.StarID.ransac_tolerance`--on the order of about 1-5 pixels. This
    tolerance should mostly reflect a very conservative estimate on the errors caused by the camera model as the 
    attitude errors should largely be removed
  * a small :attr:`~.StarID.max_magnitude`--only allowing bright stars.  Bright stars generally have more
    accurate catalog positions and are more likely to be picked up by the :class:`.ImageProcessing` algorithms
  * the :attr:`~.StarID.max_combos` set fairly large--on the order of 500-1000
  
* After getting the initial pairing and updating
  the attitude for the images (note that this is done external to the calls to :meth:`~.StellarOpNav.id_stars`), you can
  then attempt a larger identification with dimmer stars

  * decreasing :attr:`~.PointOfInterestFinder.threshold` (around 8-20)
  * decreasing the :attr:`~.StarID.tolerance` to be about the same as your previous
    :attr:`~.StarID.ransac_tolerance`
  * turning the RANSAC algorithm off by setting the :attr:`~.StarID.max_combos` to 0
  * increasing the :attr:`~.StarID.max_magnitude`.
  * potentially setting :attr:`~.StellarOpNav.denoising` to `None` if you're trying to extract as many stars as is possible
  
* If you are having problems getting the identification to work it can be useful to visually examine the results for a
  couple of images using the :func:`.show_id_results` function.

Example
_______

Below shows how stellar opnav can be used to id stars and estimate attitude corrections.  It assumes that
the :mod:`.generate_sample_data` script has been run already and that the ``sample_data`` directory is in the current
working directory.  For a more in depth example using real images, see the tutorial.

    >>> import pickle
    >>> from pathlib import Path
    >>> # use pathlib and pickle to get the data
    >>> data = Path.cwd() / "sample_data" / "camera.pickle"
    >>> with data.open('rb') as pfile:
    >>>     camera = pickle.load(pfile)
    >>> # import the stellar opnav class
    >>> from giant.stellar_opnav.stellar_class import StellarOpNav, StellarOpNavOptions
    >>> # setup options for setllar opnav
    >>> options = StellarOpNavOptions()
    >>> options.point_of_interest_finder_options.threshold = 10
    >>> options.point_of_interest_finder_options.centroid_size = 1
    >>> options.star_id_options.max_magnitude = 5
    >>> options.star_id_options.tolerance = 20
    >>> # form the stellaropnav object
    >>> sopnav = StellarOpNav(camera, options=options)
    >>> sopnav.id_stars()
    >>> sopnav.sid_summary()  # print a summary of the star id results
    >>> for _, image in camera: print(image.rotation_inertial_to_camera)  # print the attitude before
    >>> sopnav.estimate_attitude()
    >>> for _, image in camera: print(image.rotation_inertial_to_camera)  # print the attitude after
    >>> # import the visualizer to look at the results
    >>> from giant.stellar_opnav.visualizer import show_id_results
    >>> show_id_results(sopnav)
"""

from giant.stellar_opnav.stellar_class import StellarOpNav
from giant.stellar_opnav.estimators import DavenportQMethod
from giant.stellar_opnav.star_identification import StarID, StarIDOptions

# don't import visualizer here because matplotlib can take a long time to import and if we don't need it we don't
# want it

__all__ = ['StellarOpNav', 'DavenportQMethod', 'StarID', 'StarIDOptions']
