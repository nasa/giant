


r"""
This package provides the required routines and objects to identify stars in an image and then estimate attitude, camera
pointing alignment, and geometric camera model calibration using the observed stars.

Description
-----------
In GIANT, calibration refers primarily to the process of using identified stars in multiple images to estimate the
geometric camera calibration and camera frame alignment.  There are many different sub-steps that need to be performed
for this, particularly with respect to identifying the stars in the image, which can lead to cluttered scripts and hard
to maintain code when everything is thrown together manually.  Luckily, GIANT has done most of this nitty gritty work
for us by creating a simple, single interface in the :class:`.Calibration` class.

The :class:`.Calibration` class is a subclass of the :class:`.StellarOpNav` class, which provides the functionality for
identifying stars and estimating updated attitude information for single images.  In addition to the
:class:`.StellarOpNav` functionality, the :class:`.Calibration` class also provides the interfaces for using identified
stars in multiple images to estimate updates to geometric camera model (camera calibration,
:meth:`~.Calibration.estimate_geometric_calibration`) and the alignment between the camera frame and a base frame
(:meth:`~.Calibration.estimate_static_alignment` and :meth:`~.Calibration.estimate_temperature_dependent_alignment`).
While these methods make it easy to get everything packaged appropriately, GIANT also exposes all of the substeps to you
if you need them to do a more advanced analysis.

This package level documentation focuses specifically on using the :class:`.Calibration` class along with tips for
successfully doing camera calibration and alignment.  For more details about what exactly is happening, refer to the
documentation for the submodules from this package.

Tuning for Successful Calibration
---------------------------------
As with :mod:`.stellar_opnav`, tuning the :class:`.Calibration` class is both science and art.  Indeed, tuning for
calibration is nearly the same as tuning for stellar OpNav, therefore we urge you to start with the
:mod:`.stellar_opnav` documentation before proceeding with this documentation.  Once you are familiar with tuning for
stellar OpNav, then tuning for calibration will be fairly straight forward.

There are 2 main differences between tuning for calibration and tuning for stellar OpNav.  First, with calibration we
typically are considering many different view conditions across various temperatures and with various amounts of stray
light, which may make it difficult to find a single tuning that works to ID stars in all images under consideration.
The best way to work around this issue is to group the images into similar exposure times, temperatures, and stray light
patterns and figure out tuning for each of these groups independently using the recommended steps in
:mod:`.stellar_opnav`.  The :class:`.Calibration` class makes this process easy by providing the method
:meth:`~.Calibration.add_images` which, when coupled with calls to :meth:`.Camera.all_on` and :meth:`.Camera.all_off`
makes it easy to add/consider groups of images one by one, storing the results of images that have already been
processed.

The second main difference between calibration and stellar OpNav is that in calibration, particularly for the camera
model estimation, we typically want as many stars as possible extracted from each image instead of just finding the
brightest stars in the image.  The best way to handle this is to work iteratively, where you first tune for getting just
bright stars and estimate and update to the attitude (and possibly the camera model if it had a poor initial guess) and
then, when you have better a priori, turn off the RANSAC feature of the :class:`.StarID` class and identify dimmer
stars.  Once this has been done you can then re-estimate an update to the camera model (and maybe the pointing for each
image).

The only other real tuning that might need to be done is choosing which parameters are estimated as part of the
geometric camera model calibration, which is done through the :attr:`.CameraModel.estimation_parameters` attribute. For
many of the camera models, some of the parameters are highly correlated with each other and it is not recommended to
attempt to simultaneously estimate them unless you have a very large dataset to help break the correlation (for instance
misalignment and the principal point for the camera can be highly correlated unless you have a lot of images at
different viewing conditions).  Further details about good subsets of elements to estimate in calibration is included
with the documentation for the camera models provided with GIANT.

Example
-------
Below shows how calibration can be used to id stars, estimate attitude corrections, estimate a camera model, and
estimate alignment.  It assumes that the :mod:`.generate_sample_data` script has already be run and that the
``sample_data`` directory is in the current working directory.  For an in depth example using real images see the
tutorial.

    >>> import pickle
    >>> from pathlib import Path
    >>> # use pathlib and pickle to get the data
    >>> data = Path.cwd() / "sample_data" / "camera.pickle"
    >>> with data.open('rb') as pfile:
    >>>     camera = pickle.load(pfile)
    >>> # import the stellar opnav class
    >>> from giant.calibration.calibration_class import Calibration, CalibrationOptions
    >>> # import the default catalog
    >>> from giant.catalogs.giant_catalog import GIANTCatalog
    >>> # set the estimation parameters for the camera model
    >>> camera.model.estimation_parameters = ["fx", "fy", "px", "py", "k1"]
    >>> # form the calibration object
    >>> opts = CalibrationOptions()
    >>> opts.point_of_interest_finder_options.threshold = 10
    >>> opts.point_of_interest_finder_options.centroid_size = 1
    >>> opts.star_id_options.max_magnitude = 5
    >>> opts.star_id_options.tolerance = 20
    >>> cal = Calibration(camera, options=opts)
    >>> # identify stars
    >>> cal.id_stars()
    >>> cal.sid_summary()  # print a summary of the star id results
    >>> # estimate an update the attitude for each image
    >>> cal.estimate_attitude()
    >>> # update the star id settings
    >>> cal.star_id.max_magnitude = 5.5
    >>> cal.star_id.tolerance = 2
    >>> cal.star_id.max_combos = 0
    >>> # identify stars again to get dimmer stars
    >>> cal.id_stars()
    >>> # import the visualizer to look at the results
    >>> from giant.stellar_opnav.visualizer import show_id_results
    >>> show_id_results(cal)
    >>> # estimate the geometric camera model
    >>> cal.estimate_geometric_calibration()
    >>> cal.geometric_calibration_summary()
"""

import giant.calibration.estimators as estimators
import giant.calibration.calibration_class as calibration_class

from giant.calibration.calibration_class import Calibration, CalibrationOptions
from giant.calibration.estimators import (static_alignment_estimator, 
                                          temperature_dependent_alignment_estimator, 
                                          evaluate_temperature_dependent_alignment, TemperatureDependentResults,
                                          IterativeNonlinearLSTSQ, IterativeNonlinearLstSqOptions, 
                                          LMAEstimator, LMAEstimatorOptions)

__all__ = ['Calibration', 'CalibrationOptions', 
           'static_alignment_estimator', 
           'temperature_dependent_alignment_estimator', 'evaluate_temperature_dependent_alignment', 'TemperatureDependentResults',
           'IterativeNonlinearLSTSQ', 'IterativeNonlinearLstSqOptions', 
           'LMAEstimator', 'LMAEstimatorOptions']
