# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


r"""
This module provides the capability to locate surface features from a target in an image using 2D cross-correlation.

Description of the Technique
----------------------------

When a target grows to the point where we can begin to distinguish individual features on the surface in the images, we
typically consider switching to navigating using these features instead of just the center of figure of the target.
There are a number of reasons for this, including the fact that as the body grows in the field of view and errors in
your shape model can start contributing to larger and larger errors in your center-finding results and the fact that
having multiple observations instead of a single puts a stronger constraint on the location of the camera at each image
allowing us to more accurately estimate the trajectory of the camera through time.

One of the most common ways of extracting observations of each feature is through cross correlation, using a very
similar technique to that describe in :mod:`.cross_correlation`.  Essentially we render what we think each feature will
look like based on the current knowledge of the relative position and orientation of the camera with respect to each
feature (the features are stored in a special catalog called a :class:`.FeatureCatalog`).  We then take the rendered
template and use normalized cross correlation to identify the location of the feature in the image.  After we have
identified the features in the image, we optionally solve a PnP problem to refine our knowledge of the spacecraft state
and then repeat the process to correct any errors in the observations created by errors in our initial state estimates.

In more detail, GIANT implements this using the following steps

#. Identify which features we think should be visible in the image using the :attr:`.FeatureCatalog.feature_finder`
#. For each feature we predict should be visible, render the template based on the a priori relative state between the
   camera and the feature using a single bounce ray trace and the routines from :mod:`.ray_tracer`.
#. Perform 2D normalized cross correlation for every possible alignment between the center of the templates and the
   image.  We do this in a search region specified by the user and usually in the spatial domain so that we can include
   information about which pixels we want to consider when computing the correlation scores, as described in
   :func:`.sfn_correlator`.
#. Locate the peaks of the correlation surfaces (optionally locate the subpixel peak by fitting a 2D quadric to the
   correlation surface)
#. Correct the located peaks based on the location of the center-of-feature in the template to get the
   observed center-of-feature in the image.
#. Optionally solve the PnP problem for the best shift/rotation of the feature locations in the camera frame to minimize
   the residuals between the predicted feature locations and the observed feature locations.  Once complete, update the
   knowledge of the relative position/orientation of the camera with respect to the target and repeat all steps except
   this one to correct for errors introduced by a priori state knowledge errors.

Tuning
------

There are a few more tuning options in SFN verses normal cross correlation.  The first, and likely most important
tuning is for identifying potentially visible features in an image.  For this, you actually want to set the
:attr:`.FeatureCatalog.feature_finder` attribute to be something that will correctly determine which features are
possibly visible (typically to an instance of :class:`.VisibleFeatureFinder`).  We discuss the tuning for the
:class:`.VisibleFeatureFinder` here, though you could concievably use something else if you desired.

========================================================= ==============================================================
Parameter                                                 Description
========================================================= ==============================================================
:attr:`.VisibleFeatureFinder.off_boresight_angle_maximum` The maximum angle between the boresight of the camera and the
                                                          feature location in the camera frame in degrees.
:attr:`.VisibleFeatureFinder.gsd_scaling`                 The permissible ratio of the camera ground sample distance to
                                                          the feature ground sample distance
:attr:`.VisibleFeatureFinder.reflectance_angle_maximum`   The maximum angle between the viewing vector and the average
                                                          normal vector of the feature in degrees.
:attr:`.VisibleFeatureFinder.incident_angle_maximum`      The maximum angle between the incoming light vector and the
                                                          average feature normal vector in degrees.
:attr:`.VisibleFeatureFinder.percent_in_fov`              The percentage of the feature that is in the FOV
:attr:`.VisibleFeatureFinder.feature_list`                A list of feature names to consider
========================================================= ==============================================================

When tuning the feature finder you generally are looking to only get features that are likely to actually correlate well
in the image so that you don't waste time considering features that won't work for one reason or another.  All of these
parameters can contribute to this, but some of the most important are the ``gsd_scaling``, which should typically be
around 2 and the ``off_boresight_angle_maximum`` which should typically be just a little larger than the half diagonal
field of view of the detector to avoid possibly overflowing values in the projection computation and processing features
that are actually way outside the field of view.  Note that since you set the feature finder on each feature catalog,
this means that you can have different tunings for different feature catalogs (if you have multiple in a scene).

Next we have the parameters that control the actual rendering/correlation for each feature.  These are the same as for
:mod:`.cross_correlation`.

================================================ =======================================================================
Parameter                                        Description
================================================ =======================================================================
:attr:`~SurfaceFeatureNavigation.brdf`           The bidirectional reflectance distribution function used to compute the
                                                 expected illumination of a ray based on the geometry of the scene.
:attr:`~SurfaceFeatureNavigation.grid_size`      The size of the grid to use for subpixel sampling when rendering the
                                                 templates
:attr:`~SurfaceFeatureNavigation.peak_finder`    The function to use to detect the peaks of the correlation surfaces.
:attr:`~SurfaceFeatureNavigation.blur`           A flag specifying whether to blur the correlation surfaces to decrease
                                                 high frequency noise before identifying the peak.
:attr:`~SurfaceFeatureNavigation.search_region`  The search region in pixels to restrict the area the peak of
                                                 the correlation surfaces is searched for around the a priori predicted
                                                 centers for each feature
:attr:`~SurfaceFeatureNavigation.min_corr_score` The minimum correlation score to accept as a successful identification.
                                                 Correlation scores range from -1 to 1, with 1 indicating perfect
                                                 correlation.
================================================ =======================================================================

Of these options, most only make small changes to the results.  The 2 that can occasionally make large changes are
:attr:`~SurfaceFeatureNavigation.search_region` and :attr:`~SurfaceFeatureNavigation.blur`.  In general
:attr:`~SurfaceFeatureNavigation.search_region` should be set to a few pixels larger than the expected uncertainty in
the camera/feature relative state. Since we are doing spatial correlation here we typically want this number to be as
small as possible for efficiency while still capturing the actual peak. The :attr:`~SurfaceFeatureNavigation.blur`
attribute can also be used to help avoid mistaken correlation (perhaps were only empty space is aligned).  Finally, the
:attr:`~SurfaceFeatureNavigation.min_corr_score` can generally be left at the default, though if you have a poor a
priori knowledge of either the shape model or the relative position of the features then you may need to decrease this
some.

The last set of tuning parameters to consider are those for the PnP solver.  They are as follows:

=============================================================== ========================================================
Parameter                                                       Description
=============================================================== ========================================================
:attr:`~SurfaceFeatureNavigation.run_pnp_solver`                A flag to turn the PnP solver on
:attr:`~SurfaceFeatureNavigation.pnp_ransac_iterations`         The number of RANSAC iterations to attempt in the PnP
                                                                solver
:attr:`~SurfaceFeatureNavigation.second_search_region`          The search region in pixels to restrict the area the
                                                                peak of the correlation surfaces is searched for around
                                                                the a priori predicted centers for each feature after
                                                                a PnP solution has been done
:attr:`~SurfaceFeatureNavigation.measurement_sigma`             The uncertainty to assume for each measurement
:attr:`~SurfaceFeatureNavigation.position_sigma`                The uncertainty to assume in the a priori relative
                                                                position vector between the camera and the features in
                                                                kilometers
:attr:`~SurfaceFeatureNavigation.attitude_sigma`                The uncertainty to assume in the a priori relative
                                                                orientation between the camera and the features in
                                                                degrees
:attr:`~SurfaceFeatureNavigation.state_sigma`                   The uncertainty to assume in the relative position and
                                                                orientation between the camera and the features
                                                                (overrides the individual position and attitude sigmas)
:attr:`~SurfaceFeatureNavigation.max_lsq_iterations`            The maximum number of iterations to attempt to converge
                                                                in the linearized least squares solution of the PnP
                                                                problem
:attr:`~SurfaceFeatureNavigation.lsq_relative_error_tolerance`  The maximum change in the residuals from one iteration
                                                                to the next before we consider the PnP solution
                                                                converged
:attr:`~SurfaceFeatureNavigation.lsq_relative_update_tolerance` The maximum change in the update vector from one
                                                                iteration to the next before we consider the PnP
                                                                solution converged.
:attr:`~SurfaceFeatureNavigation.cf_results`                    A numpy array specifying the observed center of figure
                                                                for each target in the image (for instance from
                                                                :mod:`.cross_correlation`) to use to set the a priori
                                                                relative state information between the camera and the
                                                                feature catalog
:attr:`~SurfaceFeatureNavigation.cf_index`                      The mapping of feature catalog number to column of the
                                                                ``cf_result`` array.
=============================================================== ========================================================

All of these options can be important.  First, unless you have very good a priori error, you probably should turn the
PnP solver on.  Because of the way SFN works, errors in your a priori state can lead to somewhat significant errors in
the observed feature locations, and the PnP solver can correct a lot of these errors.  If you do turn the PnP solver on
then the rest of these options become important.  The ``pnp_ransac_iterations`` should typically be set to something
around 100-200, especially if you expect there to be outliers (which there usually are).  The ``second_search_distance``
should be set to capture the expected uncertainty after the PnP solution (typically mostly just the uncertainty in the
camera model and the uncertainty in the feature locations themselves).  Typically something around 5 works well.  The
``*_sigma`` attributes control the relative weighting between the a priori state and the observed locations, which can
be important to get a good PnP solution.  The ``max_lsq_iterations`, ``lsq_relative_error_tolerance``, and
``lsq_relative_update_tolerance`` can play an important role in getting the PnP solver to converge, though the defaults
are generally decent.  Finally, the ``cf_results`` and ``cf_index`` can help to decrease errors in the a priori relative
state knowledge, which in some cases can be critical to successfully identifying features.

Use
---

The class provided in this module is usually not used by the user directly, instead it is usually interfaced with
through the :class:`.RelativeOpNav` class using the identifier :attr:`~.RelativeOpNav.sfn`.  For more
details on using the :class:`.RelativeOpNav` interface, please refer to the :mod:`.relnav_class` documentation.  For
more details on using the technique class directly, as well as a description of the ``details`` dictionaries produced
by this technique, refer to the following class documentation.

One implementation detail we do want to note is that you should set your :attr:`.FeatureCatalog.feature_finder` on
your feature catalog before using this class. For instance, if your catalog is stored in a file called
``'features.pickle'``

    >>> import pickle
    >>> from giant.relative_opnav.estimators.sfn.surface_features import VisibleFeatureFinder
    >>> with open('features.pickle', 'rb') as in_file:
    >>>     fc: VisibleFeatureFinder = pickle.load(in_file) 
    >>> fc.feature_finder = VisibleFeatureFinder(fc, gsd_scaling=2.5)
"""

import gc
import time
from dataclasses import dataclass
from typing import Union, Optional, List, Callable, Tuple, cast

import numpy as np

from scipy.optimize import least_squares

import cv2

from matplotlib import pyplot as plt

import giant.rotations as rot
from giant.camera import Camera
from giant.rotations import Rotation, rotvec_to_rotmat
from giant.ray_tracer.rays import compute_rays
from giant.ray_tracer.scene import Scene, SceneObject
from giant.ray_tracer.rays import Rays
from giant.image_processing import otsu, cv2_correlator_2d, quadric_peak_finder_2d
from giant.image import OpNavImage
from giant.utilities.outlier_identifier import get_outliers
from giant.utilities.random_combination import RandomCombinations
from giant.utilities.mixin_classes import UserOptionConfigured
from giant.relative_opnav.estimators.sfn.surface_features import FeatureCatalog
from giant.relative_opnav.estimators.estimator_interface_abc import RelNavObservablesType, RelNavEstimator
from giant.relative_opnav.estimators._template_renderer import TemplateRendererOptions
from giant.relative_opnav.estimators.sfn.sfn_correlators import sfn_correlator

from giant._typing import SCALAR_OR_ARRAY, NONEARRAY, DOUBLE_ARRAY


@dataclass
class SurfaceFeatureNavigationOptions(TemplateRendererOptions):
    """
    This dataclass serves as one way to control the settings for the :class:`.SurfaceFeatureNavigation` class.

    You can set any of the options on an instance of this dataclass and pass it to the
    :class:`.SurfaceFeatureNavigation` class at initialization to set the settings on the class. 
    This class is the preferred way of setting options on the class due to ease of use in IDEs.
    
    :param min_corr_score: The minimum correlation score to accept for something to be considered found in an image.
                            The correlation score is the Pearson Product Moment Coefficient between the image and the
                            template. This should be a number between -1 and 1, and in nearly every cast a number
                            between 0 and 1.  Setting this to -1 essentially turns the minimum correlation score
                            check off.
    :param blur: A flag to perform a Gaussian blur on the correlation surface before locating the peak to remove
                    high frequency noise
    :param search_region: The number of pixels to search around the a priori predicted center for the peak of the
                            correlation surface.  If ``None`` then searches the entire correlation surface.
    :param run_pnp_solver: A flag specifying whether to use the PnP solver to correct errors in the initial
                            relative state between the camera and the target body
    :param pnp_ransac_iterations: The number of RANSAC iterations to attempt in the PnP solver.  Set to 0 to turn
                                    the RANSAC component of the PnP solver
    :param second_search_region: The distance around the nominal location to search for each feature in the image
                                    after correcting errors using the PnP solver.
    :param measurement_sigma: The uncertainty to assume for each measurement in pixels. This is used to set the
                                relative weight between the observed landmarks are the a priori knowledge in the PnP
                                problem. See the :attr:`measurement_sigma` documentation for a description of valid
                                inputs.
    :param position_sigma: The uncertainty to assume for the relative position vector in kilometers. This is used to
                            set the relative weight between the observed landmarks and the a priori knowledge in the
                            PnP problem. See the :attr:`position_sigma` documentation for a description of valid
                            inputs.  If the ``state_sigma`` input is not ``None`` then this is ignored.
    :param attitude_sigma: The uncertainty to assume for the relative orientation rotation vector in radians. This
                            is used to set the relative weight between the observed landmarks and the a priori
                            knowledge in the PnP problem. See the :attr:`attitude_sigma` documentation for a
                            description of valid inputs.  If the ``state_sigma`` input is not ``None`` then this is
                            ignored.
    :param state_sigma: The uncertainty to assume for the relative position vector and orientation rotation vector
                        in kilometers and radians respectively. This is used to set the relative weight between the
                        observed landmarks and the a priori knowledge in the PnP problem. See the
                        :attr:`state_sigma` documentation for a description of valid inputs.  If this input is not
                        ``None`` then the ``attitude_sigma`` and ``position_sigma`` inputs are ignored.
    :param max_lsq_iterations: The maximum number of iterations to make in the least squares solution to the PnP
                                problem.
    :param lsq_relative_error_tolerance: The relative tolerance in the residuals to signal convergence in the least
                                            squares solution to the PnP problem.
    :param lsq_relative_update_tolerance: The relative tolerance in the update vector to signal convergence in the
                                            least squares solution to the PnP problem
    :param cf_results: A numpy array containing the center finding residuals for the target that the feature
                        catalog is a part of.  If present this is used to correct errors in the a priori line of
                        sight to the target before searching for features in the image.
    :param cf_index: A list that maps the features catalogs contained in the ``scene`` (in order) to the
                        appropriate column of the ``cf_results`` matrix.  If left blank the mapping is assumed to be
                        in like order
    :param show_templates: A flag to show the rendered templates for each feature "live".  This is useful for
                            debugging but in general should not be used.
    """
    
    peak_finder:  Callable[[np.ndarray, bool], np.ndarray] = quadric_peak_finder_2d
    """
    The peak finder function to use. This should be a callable that takes in a 2D surface as a numpy array and returns 
    the (x,y) location of the peak of the surface.
    """

    min_corr_score: float = 0.3
    """
    The minimum correlation score to accept for something to be considered found in an image. The correlation score 
    is the Pearson Product Moment Coefficient between the image and the template. This should be a number between -1 
    and 1, and in nearly every cast a number between 0 and 1.  Setting this to -1 essentially turns the minimum 
    correlation score check off. 
    """

    blur: bool = True
    """
    A flag to perform a Gaussian blur on the correlation surface before locating the peak to remove high frequency noise
    """

    search_region: Optional[int] = None
    """
    The number of pixels to search around the a priori predicted center for the peak of the correlation surface.  If 
    ``None`` then searches the entire correlation surface.
    """

    run_pnp_solver: bool = False
    """
    A flag specifying whether to use the PnP solver to correct errors in the initial relative state between the camera 
    and the target body
    """

    pnp_ransac_iterations: int = 0
    """
    The number of RANSAC iterations to attempt in the PnP solver.  Set to 0 to turn the RANSAC component of the PnP 
    solver
    """

    second_search_region: Optional[int] = None
    """
    The distance around the nominal location to search for each feature in the image after correcting errors using the 
    PnP solver.
    """

    measurement_sigma: SCALAR_OR_ARRAY = 1
    """
    The uncertainty to assume for each measurement in pixels. This is used to set the relative weight between the 
    observed landmarks are the a priori knowledge in the PnP problem. See the :attr:`measurement_sigma` documentation 
    for a description of valid inputs.
    """

    position_sigma: Optional[SCALAR_OR_ARRAY] = None
    """
    The uncertainty to assume for the relative position vector in kilometers. This is used to set the relative weight 
    between the observed landmarks and the a priori knowledge in the PnP problem. See the :attr:`position_sigma` 
    documentation for a description of valid inputs.  If the ``state_sigma`` input is not ``None`` then this is 
    ignored.
    """

    attitude_sigma: Optional[SCALAR_OR_ARRAY] = None
    """
    The uncertainty to assume for the relative orientation rotation vector in radians. This is used to set the 
    relative weight between the observed landmarks and the a priori knowledge in the PnP problem. See the 
    :attr:`attitude_sigma` documentation for a description of valid inputs.  If the ``state_sigma`` input is not 
    ``None`` then this is ignored.
    """

    state_sigma: NONEARRAY = None
    """
    The uncertainty to assume for the relative position vector and orientation rotation vector in kilometers and 
    radians respectively. This is used to set the relative weight between the observed landmarks and the a priori 
    knowledge in the PnP problem. See the :attr:`state_sigma` documentation for a description of valid inputs.  If 
    this input is not ``None`` then the ``attitude_sigma`` and ``position_sigma`` inputs are ignored.
    """

    max_lsq_iterations: Optional[int] = None
    """
    The maximum number of iterations to make in the least squares solution to the PnP problem. 
    """

    lsq_relative_error_tolerance: float = 1e-8
    """
    The relative tolerance in the residuals to signal convergence in the least squares solution to the PnP problem.
    """

    lsq_relative_update_tolerance: float = 1e-8
    """
    The relative tolerance in the update vector to signal convergence in the least squares solution to the PnP problem
    """

    cf_results: Optional[np.ndarray] = None
    """ 
    A numpy array containing the center finding residuals for the target that the feature catalog is a part of.  If 
    present this is used to correct errors in the a priori line of sight to the target before searching for features 
    in the image.
    """

    cf_index: Optional[List[int]] = None
    """
    A list that maps the features catalogs contained in the ``scene`` (in order) to the appropriate column of the 
    ``cf_results`` matrix.  If left blank the mapping is assumed to be in like order
    """

    show_templates: bool = False
    """
    A flag to show the rendered templates for each feature "live". This is useful for debugging but in general should 
    not be used.
    """
    
    def override_options(self):

        if self.second_search_region is None:
            self.second_search_region = self.search_region


class SurfaceFeatureNavigation(UserOptionConfigured[SurfaceFeatureNavigationOptions], SurfaceFeatureNavigationOptions, RelNavEstimator):
    """
    This class implements surface feature navigation using normalized cross correlation template matching for GIANT.

    All of the steps required for performing surface feature navigation are handled by this class, including the
    identification of visible features in the image, the rendering of the templates for each feature, the actual cross
    correlation, the identification of the peaks of the correlation surfaces, and optionally the solution of a PnP
    problem based on the observed feature locations in the image.  This is all handled in the :meth:`estimate` method
    and is performed for each requested target.  Note that targets must have shapes of :class:`.FeatureCatalog` to use
    this class.

    When all of the required data has been successfully loaded into an instance of this class, the :meth:`estimate`
    method is used to perform the estimation for the requested image.  The results are stored into the
    :attr:`observed_bearings` attribute for the observed center of template locations. In addition, the predicted
    locations for the center of template for each template is stored in the :attr:`computed_bearings` attribute.
    Finally, the details about the fit are stored as a dictionary in the appropriate element in the :attr:`details`
    attribute.  Specifically, these dictionaries will contain the following keys.

    ================================= ==================================================================================
    Key                               Description
    ================================= ==================================================================================
    ``'Correlation Scores'``          The correlation score at the peak of the correlation surface for each feature as a
                                      list of floats. The corresponding element will be 0 for any features that were not
                                      found.  Each element of this list corresponds to the feature according to the
                                      corresponding element in the ``'Visible Features'`` list. If no potential visible
                                      features were expected in the image then this is not available.
    ``'Visible Features'``            The list of feature indices (into the :attr:`.FeatureCatalog.features` list)
                                      that were looked for in the image.  Each element of this list corresponds to the
                                      corresponding element in the :attr:`templates` list.  If no potential visible
                                      features were expected in the image then this is not available.
    ``'Correlation Peak Locations'``  The Location of the correlation peaks before correcting it to find the location of
                                      the location of the feature in the image as a list of size 2 numpy arrays. Each
                                      element of this list corresponds to the feature according to the corresponding
                                      element in the ``'Visible Features'`` list.  Any features that were not found in
                                      the image have ``np.nan`` for their values.  If no potential visible features were
                                      expected in the image then this is not available.
    ``'Correlation Surfaces'``        The raw correlation surfaces as 2D arrays of shape
                                      ``2*search_region+1 x 2*search_region+1``.  Each pixel in the correlation surface
                                      represents a shift between the predicted and expected location, according to
                                      :func:`.sfn_correlator`.  Each element of this list corresponds to the feature
                                      according to the corresponding element in the ``'Visible Features'`` list.  If no
                                      potential visible features were expected in the image then this is not available.
    ``'Target Template Coordinates'`` The location of the center of each feature in its corresponding template. Each
                                      element of this list corresponds to the feature according to the corresponding
                                      element in the ``'Visible Features'`` list.  If no potential visible features
                                      were expected in the image then this is not available.
    ``'Intersect Masks'``             The boolean arrays the shape shapes of each rendered template with ``True`` where
                                      a ray through that pixel struct the surface of the template and ``False``
                                      otherwise. Each element of this list corresponds to the feature according to the
                                      corresponding element in the ``'Visible Features'`` list.  If no potential
                                      visible features were expected in the image then this is not available.
    ``'Space Mask'``                  The boolean array the same shape as the image specifying which pixels of the image
                                      we thought were empty space with a ``True`` and which we though were on the body
                                      with a ``False``.  If no potential visible features were expected in the image
                                      then this is not available
    ``'PnP Solution'``                A boolean indicating whether the PnP solution was successful (``True``) or not.
                                      This is only available if a PnP solution was attempted.
    ``'PnP Translation'``             The solved for translation in the original camera frame that minimizes the
                                      residuals in the PnP solution as a length 3 array with units of kilometers.  This
                                      is only available if a PnP solution was attempted and the PnP solution was
                                      successful.
    ``'PnP Rotation'``                The solved for rotation of the original camera frame that minimizes the
                                      residuals in the PnP solution as a :class:`.Rotation`.  This
                                      is only available if a PnP solution was attempted and the PnP solution was
                                      successful.
    ``'PnP Position'``                The solved for relative position of the target in the camera frame after the PnP
                                      solution is applied as a length 3 numpy array in km.
    ``'PnP Orientation'``             The solved for relative orientation of the target frame with respect to the camera
                                      frame after the PnP solution is applied as a :class:`.Rotation`.
    ``'Failed'``                      A message indicating why the SFN failed.  This will only be present if the SFN fit
                                      failed (so you could do something like ``'Failed' in sfn.details[target_ind]`` to
                                      check if something failed.  The message should be a human readable description of
                                      what caused the failure.
    ================================= ==================================================================================

    .. warning::
        Before calling the :meth:`estimate` method be sure that the scene has been updated to correspond to the correct
        image time.  This class does not update the scene automatically.
    """

    observable_type = [RelNavObservablesType.LANDMARK]
    """
    This technique generates LANDMARK bearing observables to the center of landmarks in the image.
    """
    
    generates_templates = True
    """
    A flag specifying that this RelNav estimator generates and stores templates in the :attr:`templates` attribute.
    """

    technique = "sfn"
    """
    The name for the technique for registering with :class:`.RelativeOpNav`.  
    
    If None then the name will default to the name of the module where the class is defined.  
    
    This should typically be all lowercase and should not include any spaces or special characters except for ``_`` as 
    it will be used to make attribute/method names.  (That is ``MyEstimator.technique.isidentifier()`` should evaluate 
    ``True``).
    """

    def __init__(self, scene: Scene, camera: Camera, options: Optional[SurfaceFeatureNavigationOptions] = None):
        """
        :param scene: The scene describing the a priori locations of the targets and the light source.
        :param camera: The :class:`.Camera` object containing the camera model and images to be analyzed
        :param options: A dataclass specifying the options to set for this instance.
        """
        
        super().__init__(SurfaceFeatureNavigationOptions, scene, camera, options=options)
        
        if self.cf_index is None:
            self.cf_index = list(range(len(scene.target_objs)))
        else:
            self.cf_index = self.cf_index

        self.visible_features: List[Optional[List[int]]] = [None]*len(self.scene.target_objs)
        """
        This variable is used to notify which features are predicted to be visible in the image.
        
        Each visible feature is identified by its index in the :attr:`.FeatureCatalog.features` list.
        """


    def render(self, target_ind: int,
               target: SceneObject,
               temperature: float = 0) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        This method renders each visible feature for the current target according to the current estimate of the
        relative position/orientation between the target and the camera using single bounce ray tracing.

        The illumination values are computed by (a) determining the rays to trace through the scene (either user
        specified or by a call to :meth:`compute_rays`), (b) performing a single bounce ray trace through the scene
        using a call to :meth:`.Scene.get_illumination_inputs` with only the feature being processed "turned on",
        (c) converting the results of the ray trace into illumination values using :attr:`brdf`, and then summing the
        intensity of each ray into a 2D template array. The rendered templates are store in the :attr:`templates`
        attribute.

        Additionally, this method also produces a 2D boolean array specifying which pixels in the template had rays
        which actually struck the surface of the feature (``True``) and which hit empty space (``False``) for use in
        the correlation of the template with the image.  Finally, it also computes the :attr:`computed_bearings` for
        each visible template by projecting the template center onto the image using the camera model and storing it
        in a 2xn array.

        Note that before calling this method you must have set the visible feature list in the image to the
        :attr:`visible_features` attribute.

        Typically this method is not used directly by the user and instead is called by the :meth:`estimate` method
        automatically.

        :param target_ind: index into the :attr:`.Scene.target_objs` list of the target being rendering
        :param target: the :class:`.SceneObject` for the target being rendered
        :param temperature: The temperature of the camera at the time the scene is being rendered.
        :return: The intersect masks as a list of 2D boolean arrays and the location of the feature center in each
                 template as a 2xn array.
        """

        if (visible_features := self.visible_features[target_ind]) is None:
            raise ValueError('Cannot call render without having identified which features are visible')

        if not isinstance(target.shape, FeatureCatalog):
            raise ValueError('Cannot use SFN with non feature catalogs')

        # initialize the templates list to store the templates
        self.templates[target_ind] = cast(list[DOUBLE_ARRAY | None], [None] * len(visible_features))
        # initialize the intersects list and the template centers list
        intersects_list: List[np.ndarray] = []

        # initialize the array for the location of the feature center in each template as well as the computed bearings
        # for each template
        template_centers = np.empty((2, len(visible_features)), dtype=np.float64)
        self.computed_bearings[target_ind] = np.empty((2, len(visible_features)), dtype=np.float64)

        # loop through each possibly visible feature in the image
        for feature_number, feature_ind in enumerate(visible_features):

            start = time.time()
            
            # figure out what rays to trace.  Hopefully we are doing this ourselves because things might go wonky
            # otherwise
            if self.rays is None:
                (rays, locs), bounds = self.compute_rays(target.shape, feature_ind, temperature=temperature)
            elif isinstance(self.rays, Rays):
                rays = self.rays
                locs = self.camera.model.project_onto_image(rays.start + rays.direction, temperature=temperature)
                bounds = (locs.min(axis=1, initial=None).round(), locs.max(axis=1, initial=None).round()) # type: ignore
            elif self.rays[target_ind] is None:
                (rays, locs), bounds = self.compute_rays(target.shape, feature_ind, temperature=temperature)
            elif isinstance(self.rays[target_ind], list):
                rays = self.rays[target_ind][feature_ind] # type: ignore
                locs = self.camera.model.project_onto_image(rays.start + rays.direction, temperature=temperature)
                bounds = (locs.min(axis=1, initial=None).round(), locs.max(axis=1, initial=None).round()) # type: ignore
            else:
                rays: Rays = self.rays[target_ind]  # type: ignore
                locs = self.camera.model.project_onto_image(rays.start + rays.direction, temperature=temperature)
                bounds = (locs.min(axis=1, initial=None).round(), locs.max(axis=1, initial=None).round()) # type: ignore

            print('Tracing {} rays'.format(rays.num_rays), flush=True)

            target.shape.include_features = [feature_ind]

            # get the ray trace results along with the intersect array
            illum_inputs, intersects = self.scene.get_illumination_inputs(rays, return_intersects=True)

            # transform the ray trace results into relative intensity values
            illums = self.brdf(illum_inputs)

            template_size = (bounds[1] - bounds[0]) + 1

            # make the arrays for the template and the intersect mask
            intersects_out = np.ones(template_size[::-1].astype(int), dtype=bool)

            template = np.zeros(template_size[::-1].astype(int))

            # figure out the subscripts into the template/intersect mask arrays
            subs = (locs - bounds[0].reshape(2, 1)).round().astype(int)

            # make the template/intersect mask

            # logical and means only pixels where all rays hit the surface are included.  This is to ignore the edge
            # which will be darker because fewer rays hit it.

            # alternatively we could use a logical or and then divide the template by the count of rays that actually
            # hit the surface.  Something to consider for the future.
            np.logical_and.at(intersects_out, (subs[1], subs[0]), intersects['check'])

            np.add.at(template, (subs[1], subs[0]), illums.flatten())

            if self.camera.psf is not None:
                template = self.camera.psf(template)

            # compute the bearing to the center of the feature in the image/template.
            computed = self.camera.model.project_onto_image(target.shape.feature_locations[feature_ind], temperature=temperature) 

            self.computed_bearings[target_ind][:, feature_number] = computed # type: ignore 
            template_centers[:, feature_number] = (computed - bounds[0]) # type: ignore

            intersects_list.append(intersects_out)

            self.templates[target_ind][feature_number] = template # type: ignore
            print(f'Feature {target.shape.features[feature_ind].name} number {feature_number + 1} of '
                  f'{len(visible_features)} rendered in {time.time()-start:.3f} seconds', flush=True)

        # call the garbage collector to get rid of features that have been unloaded, because sometimes python doesn't do
        # this when we want it to
        gc.collect()

        return intersects_list, template_centers

    # noinspection PyMethodOverriding
    def compute_rays(self, feature_catalog: FeatureCatalog,
                     feature_ind: int,
                     temperature: float = 0) -> Tuple[Tuple[Rays, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        This method computes the required rays to render a given feature based on the current estimate of the location
        and orientation of the feature in the image.

        This method first determines which pixels to trace. If a circumscribing sphere is defined for the feature,
        the edges of the sphere are used to compute the required pixels; otherwise, the bounding box is used. The
        requested subsampling for each pixel is then applied, and the sub-pixels are then converted into rays
        originating at the camera origin using the :class:`.CameraModel`.

        :param feature_catalog: The feature catalog which contains the feature we are rendering
        :param feature_ind: The index of the feature in the feature catalog that we are rendering
        :param temperature: The temperature of the camera at the time the feature is being rendered
        :return: The rays to trace through the scene and the pixel coordinates for each ray as a tuple, plus
                 the bounds of the pixel coordinates
        """

        # the feature bounds are already in the camera frame at this point (at least they should be...)
        bounds = feature_catalog.feature_bounds[feature_ind]
        image_locs = self.camera.model.project_onto_image(bounds, temperature=temperature)

        local_min: np.ndarray = np.floor(image_locs.min(axis=1, initial=None))  # type: ignore
        local_max: np.ndarray = np.ceil(image_locs.max(axis=1, initial=None))  # type: ignore

        rays_pix = compute_rays(self.camera.model, (local_min[1], local_max[1]), (local_min[0], local_max[0]),
                                grid_size=self.grid_size, temperature=temperature)
        min_max = (local_min, local_max)
        return rays_pix, min_max

    def estimate(self, image: OpNavImage, include_targets: Optional[List[bool]] = None):
        """
        This method identifies the locations of surface features in the image through cross correlation of rendered
        templates with the image.

        This method first checks to ensure that the appropriate correlator is set for the :attr:`image_processing`
        instance (which should the :func:`.sfn_correlator`).  If it is not a warning is printed and we set the
        correlator to be the :func:`.sfn_correlator` (this is required for surface feature navigation).  Don't worry,
        we'll put things back the way they were when we're done :).

        This method also identifies the index into the :attr:`.Camera.images` list for the image being processed. This
        is done by first checking identity (to find the exact same image). If that doesn't work we then check based on
        equality (the pixel data is all exactly the same) however, this could lead to false pairing in some degenerate
        instances.  As long as you are using this method as intended (and not copying/modifying the image array from the
        camera before sending it to this method) the identity check should work.  We do this so that we can relocate
        each target to be along the line of sight vector found by center finding (if done/provided) before looking for
        features.  Therefore if you aren't seeding your SFN with center finding results you don't need to worry about
        this.

        Once the initial preparation is complete, for each requested target that is a :class:`.FeatureCatalog` we seek
        feature locations that are visible in the image. This is done by first predicting which features from the
        catalog should be visible in the image using the a priori relative state knowledge between the camera and the
        feature catalog and the :attr:`.FeatureCatalog.feature_finder` function which is usually an instance of
        :class:`.VisibleFeatureFinder`.  Once potentially visible features have been determined, we render a predicted
        template of each feature using a single bounce ray tracer.  We then do spatial cross correlation between the
        template and the image within a specified search region (if the search region is too large we attempt global
        frequency correlation first) to generate a correlation surface.  From this correlation surface, we identify the
        peak and use that to locate the center of the feature in the image.  Once this has been completed for all
        potentially visible features for a given target, we then optionally attempt to solve a PnP problem to refine the
        relative position and orientation of the camera with respect to the target based on the observed feature
        locations.  If we successfully solve the PnP problem, then we iterate 1 more time through the entire process
        (but not the PnP solver) and the results from that time become the observed locations stored in the
        :attr:`observed_bearings` attribute.

        More details about many of these steps can be seen in the :meth:`render` and :meth:`pnp_solver` methods.

        .. warning::
            Before calling this method be sure that the scene has been updated to correspond to the correct
            image time.  This method does not update the scene automatically.

        :param image: The image to locate the targets in
        :param include_targets: A list specifying whether to process the corresponding target in
                                :attr:`.Scene.target_objs` or ``None``.  If ``None`` then all targets are processed.
        """

        # If using PnP solver, iterate the rendering and registration twice:
        outer_iterations = 1

        if self.run_pnp_solver:
            outer_iterations = 2

        # figure out which image index the current image is in the camera
        # default to 0 in case we can't figure it out
        image_ind = None
        for ind, other_image in self.camera:
            # check for identity (first)
            if image is other_image:
                image_ind = ind
                break

        if image_ind is None:
            # if identity didn't work check for equality
            for ind, other_image in self.camera:
                if (image == other_image).all():
                    image_ind = ind
                    break

        if image_ind is None:
            print("We were unable to figure out which index in the camera corresponds to the current image. Are you "
                  "sure this image came from this camera?.  We'll assume that the index is 0 for now...")
            image_ind = 0

        # prepare the space mask by performing Otsu on a Gaussian blurred image and then flood fill
        # opencv only likes single precision
        image_32 = image.astype(np.float32)

        # GaussianBlur here helps close the figure.
        _, labeled = otsu(cv2.GaussianBlur(image_32, (3, 3), 0), 2)

        space_mask = np.zeros((labeled.shape[0] + 2, labeled.shape[1] + 2), np.uint8)
        # flood fill closes the shape
        cv2.floodFill(1 - labeled, space_mask, (0, 0), 1)

        # target number is a counter on the number of targets we've processed.
        target_number = 0
        for (target_ind, target) in self.target_generator(include_targets):
            # target_ind is the index into the target_objs list
            start = time.time()

            # initialize these here since PyCharm can't figure out they're set later
            correlation_surfaces = None
            correlation_peaks = None
            correlation_scores = None
            template_centers = None
            intersects = None

            processed = True
            for iteration in range(0, outer_iterations):
                # first time through use the original search distance.
                if iteration == 0:
                    search_dist_use = self.search_region
                else:
                    search_dist_use = self.second_search_region
                    
                assert search_dist_use is not None

                if not isinstance(target.shape, FeatureCatalog):
                    print(f"All targets in SFN must be feature catalogs. Skipping target {target_ind}", flush=True)
                    processed = False
                    break

                # only update the a priori target location the first time through
                if (self.cf_results is not None) and (iteration == 0):
                    if self.cf_index is None:
                        new_center = self.camera.model.pixels_to_unit(
                            self.cf_results[image_ind][target_number]['measured'][:2],
                            temperature=image.temperature, image=image_ind
                        ) * np.linalg.norm(target.position)
                    else:
                        new_center = self.camera.model.pixels_to_unit(
                            self.cf_results[image_ind][self.cf_index[target_number]]['measured'][:2],
                            temperature=image.temperature, image=image_ind
                        ) * np.linalg.norm(target.position)

                    if np.isfinite(new_center).all():
                        target.change_position(new_center)

                # use the feature finder to find the visible features in the image
                visible_features = target.shape.feature_finder(self.camera.model,
                                                               self.scene,
                                                               image.temperature)
                
                self.visible_features[target_ind] = visible_features

                if len(visible_features) == 0:
                    print(f'no visible features for target {target_ind} '
                          f'in image {image.observation_date}. Skipping', flush=True)
                    self.details[target_ind] = {"Failed": "No visible features in image"}
                    self.observed_bearings[target_ind] = None
                    self.visible_features[target_ind] = None
                    continue

                # render the visible features
                intersects, template_centers = self.render(target_ind, target, temperature=image.temperature)

                # initialize the lists to store things for the details
                correlation_surfaces = []
                correlation_peaks = []
                correlation_scores = []

                # initialize the observed bearings array
                self.observed_bearings[target_ind] = np.empty((2, len(visible_features)),
                                                              dtype=np.float64) + np.nan

                image_use = image_32

                print('Registering Templates with Image...', flush=True)
                # feature number is the number of the feature in the image
                # self.visible_features[target_ind][feature_number] is the index of the feature in the feature catalog
                templates = self.templates[target_ind]
                assert templates is not None
                for feature_number, template in enumerate(templates):  # type: ignore
                    
                    assert isinstance(template, np.ndarray)

                    if self.show_templates:
                        fig = plt.figure()
                        ax = fig.add_subplot(111)
                        ax.imshow(template, cmap='gray')
                        ax.set_title(target.shape.features[visible_features[feature_number]].name)

                    # this shift corrects from the center of the template to the center of the feature in the template.
                    # the peak of the correlation surface gives the center of the template in the image, but we want the
                    # center of the feature in the image
                    temp_middle = np.floor(np.flipud(np.asarray(template.shape, dtype=np.float64)) / 2)

                    delta = temp_middle - template_centers[:, feature_number]
                    
                    computed = self.computed_bearings[target_ind][:, feature_number] # type: ignore
                    assert isinstance(computed, np.ndarray)

                    # if the search distance is greater than 50, attempt to use the normal correlator first since
                    # it will be faster and then shrink to a smaller search distance
                    if search_dist_use > 50:
                        # get the correlation surface between the template and the full image
                        temp_surf = cv2_correlator_2d(image_use, template) # type: ignore

                        search_start = np.round(computed + delta) 
                        bounds = [np.maximum(search_start - search_dist_use, 0).astype(int),
                                  np.minimum(search_start + search_dist_use, image.shape[::-1]).astype(int)]

                        temp_peak = self.peak_finder(temp_surf[bounds[0][1]:bounds[1][1],
                                                               bounds[0][0]:bounds[1][0]], self.blur).ravel()

                        if not np.isnan(temp_peak).any():

                            search_center = temp_peak + bounds[0]
                            actual_search_distance = 20
                            adjustment = search_center - delta - computed

                        else:

                            # if that failed then fall back to the sfn correlator
                            search_center = computed + delta 
                            actual_search_distance = search_dist_use
                            adjustment = np.zeros((2,), dtype=np.float64)

                    else:
                        # if less than 50 then just proceed
                        search_center = computed + delta 
                        actual_search_distance = search_dist_use
                        adjustment = np.zeros((2,), dtype=np.float64)

                    # set the kwargs for the correlator
                    # make the correlation surface and find the peak of it
                    correlation_surfaces.append(sfn_correlator(image_use, template, 
                                                               space_mask=space_mask, 
                                                               intersects=intersects[feature_number], 
                                                               search_dist=actual_search_distance, 
                                                               center_predicted=search_center))

                    correlation_peaks.append(self.peak_finder(correlation_surfaces[feature_number], self.blur).ravel())

                    # check if we found a valid peak for this feature.  If so save the score and the location of the
                    # feature  in the image
                    if np.isfinite(correlation_peaks[feature_number]).all():
                        rounded_peak = np.round(correlation_peaks[feature_number])
                        correlation_scores.append(
                            correlation_surfaces[feature_number][int(rounded_peak[1]), int(rounded_peak[0])]
                        )

                        if correlation_scores[feature_number] < self.min_corr_score:
                            correlation_peaks[feature_number][:] = np.nan
                            # correlation_scores[feature_number] = 0
                    else:
                        correlation_peaks[feature_number][:] = np.nan
                        correlation_scores.append(0)

                    adjustment += correlation_peaks[feature_number] - actual_search_distance

                    observed = computed + adjustment
                    self.observed_bearings[target_ind][:, feature_number] =  observed  # type: ignore

                    # Print shift and correlation score:
                    feature_name = target.shape.features[visible_features[feature_number]].name
                    if not np.isfinite(self.observed_bearings[target_ind][:, feature_number]).all(): # type: ignore
                        print(f"{feature_name} :  Landmark location could be not be identified.")
                        print(f"\tPeak correlation score is {correlation_scores[feature_number]}.", flush=True)
                    else:
                        shift_to_print = np.round(observed - computed, 3) # type: ignore

                        print(f"{feature_name} : {shift_to_print[0]:>9.3f}, {shift_to_print[1]:>9.3f} | "
                              f"{int(np.round(correlation_scores[feature_number]*10)):d}/10",
                              flush=True)

                    if self.show_templates:
                        plt.show()

                all_observed = self.observed_bearings[target_ind]
                all_computed = self.computed_bearings[target_ind]
                if all_observed  is not None and all_computed is not None:
                    # compute statistics on the results and print them
                    diffs = all_observed - all_computed
                    valid = np.isfinite(diffs).all(axis=0)
                    diffed_mean = diffs[:, valid].mean(axis=-1)
                    diffed_std = diffs[:, valid].std(axis=-1)
                    print(f'{valid.sum()} of {valid.size} landmarks found.')
                    print(f'\tResidual mean: {diffed_mean[0]}, {diffed_mean[1]}')
                    print(f'\tResidual std: {diffed_std[0]}, {diffed_std[1]}', flush=True)

                # PnP Solver:
                if self.run_pnp_solver and (iteration == 0):
                    # check for which landmarks are valid
                    valid = np.isfinite(all_observed).all(axis=0)

                    # check that we have enough points to solve the PnP problem effectively
                    if valid.sum() >= 2:

                        print('\nSolving Perspective-n-Point problem to adjust camera Pose...', flush=True)
                        delta_pos, delta_quat = self.pnp_solver(target_ind, image, image_ind)

                        # if we got a valid shift report it and continue to the next iteration, otherwise break and just
                        # stick with what we've got
                        if delta_pos is not None and delta_quat is not None:

                            print(f'Shifting target position by: {delta_pos} km')
                            print(f'Rotating target orientation by: {np.rad2deg(delta_quat.vector)} degrees',
                                  flush=True)

                            # Update scene to reflect the change:
                            target.translate(delta_pos)
                            target.rotate(delta_quat)

                            print(f'\nRe-running SFN after PnP solve.\n'
                                  f'Decreasing search distance from {search_dist_use} to '
                                  f'{self.second_search_region}...', flush=True)

                            self.details[target_ind] = {"PnP Solution": True,
                                                        "PnP Translation": delta_pos,
                                                        "PnP Rotation": delta_quat,
                                                        "PnP Position": target.position.copy(),
                                                        "PnP Orientation": target.orientation.copy()}

                        else:
                            print('Could not compute PnP solution')
                            self.details[target_ind] = {"Correlation Scores": correlation_scores,
                                                        "Visible Features": self.visible_features[target_ind],
                                                        "Correlation Peak Locations": correlation_peaks,
                                                        "Correlation Surfaces": correlation_surfaces,
                                                        "Target Template Coordinates": template_centers,
                                                        "Intersect Masks": intersects,
                                                        "Space Mask": space_mask,
                                                        "PnP Solution": False}
                            break

                    else:
                        print('Insufficient number of landmarks identified to perform PnP.', flush=True)
                        self.details[target_ind] = {"Correlation Scores": correlation_scores,
                                                    "Visible Features": self.visible_features[target_ind],
                                                    "Correlation Peak Locations": correlation_peaks,
                                                    "Correlation Surfaces": correlation_surfaces,
                                                    "Target Template Coordinates": template_centers,
                                                    "Intersect Masks": intersects,
                                                    "Space Mask": space_mask,
                                                    "PnP Solution": False}
                        break

            if isinstance(self.details[target_ind], dict):
                self.details[target_ind].update({"Correlation Scores": correlation_scores, # type: ignore
                                                 "Visible Features": self.visible_features[target_ind],
                                                 "Correlation Peak Locations": correlation_peaks,
                                                 "Correlation Surfaces": correlation_surfaces,
                                                 "Target Template Coordinates": template_centers,
                                                 "Intersect Masks": intersects,
                                                 "Space Mask": space_mask}) 
            else:
                self.details[target_ind] = {"Correlation Scores": correlation_scores,
                                            "Visible Features": self.visible_features[target_ind],
                                            "Correlation Peak Locations": correlation_peaks,
                                            "Correlation Surfaces": correlation_surfaces,
                                            "Target Template Coordinates": template_centers,
                                            "Intersect Masks": intersects,
                                            "Space Mask": space_mask}

            if processed:
                target_number += 1
                print(f'Target {target_number} complete in {time.time()-start:.3f} seconds', flush=True)

    def pnp_solver(self, target_ind: int, image: OpNavImage, image_ind: int) -> Tuple[np.ndarray, Rotation] | tuple[None, None]:
        r"""
        This method attempts to solve for an update to the relative position/orientation of the target with respect to
        the image based on the observed feature locations in the image.

        We solve the PnP problem here looking for an update (ie not trying to solve the lost in space problem).  This is
        done for 2 reasons.  First, it reflects what we're doing.  In order to find the surface features in the image we
        need some reasonable a priori knowledge of the relative position/orientation between the camera and the target
        so we might as well use it (since it makes things simpler).  Second, because the PnP problem can be fickle,
        especially if the observations are noisy (as they can sometimes be if our a priori knowledge was not the
        greatest) by solving for an update we force it to stay near the a priori knowledge which helps to prevent
        outlandish solutions, even when there are very few points available.

        Because we are doing an update PnP solution we simply solve a nonlinear least squares problem for the
        translation and rotation (cast as a 3 element rotation vector) that minimizes the error between the predicted
        feature locations in the image and the observed feature locations.  This is generally robust and fast,
        especially since we can easily compute the analytic Jacobian for the least squares problem.  Specifically, the
        problem we are trying to minimize is

        .. math::

            \mathbf{r}_i = \mathbf{y}_i-f\left((\mathbf{I}-[\boldsymbol{\delta\theta}\times])
            (\mathbf{x}_i+\mathbf{d})\right) \\
            \min_{\mathbf{d},\boldsymbol{\delta\theta}}\left\{\sum_i \mathbf{r}_i^T\mathbf{r}_i\right\}

        where :math:`\mathbf{d}` is the shift, :math:`\boldsymbol{\delta\theta}` is the rotation vector,
        :math:`\mathbf{y}_i` is the observed location of the :math:`i^{th}` features, :math:`\mathbf{x}_i` is the
        location of the :math:`i^{th}` feature center in the current a priori camera frame, :math:`\mathbf{I}` is a
        3x3 identity matrix, and :math:`f(\bullet)` is the function which transforms points in the camera frame to
        points in the image (:meth:`.CameraModel.project_onto_image`).

        In addition, we provide the option to use RANSAC when solving the PnP problem to attempt to reject outliers in
        the solution.  This is very useful and should typically be used when possible (RANSAC will not be used when the
        number of observed features is less than 8).  We implement a typical RANSAC algorithm, where we choose a
        subsample of the available measurements, compute the PnP solution using the sample, and then compute the
        statistics/number of inliers from the full set of measurements given the estimated location from the sample,
        keeping the one with the most inliers and lowest statistics.

        To apply the results of this method to a scene object you should do ``object.translate(shift)`` and then
        ``object.rotate(rotation)`` where ``object`` is the scene object, ``shift`` is the shift vector from this
        method, and ``rotation`` is the rotation from this method (note translate first, then rotate).

        :param target_ind: The index of the target that we are currently doing the PnP problem with respect to
        :param image: The image that we are currently solving the PnP problem for
        :param image_ind: The index of the image that we are solving the PnP problem for. (This only really matters if
                          you still have per image misalignments in your camera model still for some reason)
        :return: The best fit shift as a length 3 numpy array in km and rotation as a :class:`.Rotation` to go from the
                 current camera frame to the new camera frame if successful or ``None``, ``None``
        """
        # first identify outliers
        observed = self.observed_bearings[target_ind]
        computed = self.computed_bearings[target_ind]
        assert isinstance(observed, np.ndarray) and isinstance(computed, np.ndarray)
        valid: np.ndarray = np.isfinite(observed).all(axis=0) # type: ignore

        a_priori_residuals = observed[:, valid] - computed[:, valid]

        a_priori_errors = np.linalg.norm(a_priori_residuals, axis=0)

        # check for outliers
        outliers = get_outliers(a_priori_errors, 3)

        valid[valid] = ~outliers

        # get the valid image points
        image_points = observed[:, valid]

        # get the location of the features in the current camera frame
        visible_features = self.visible_features[target_ind]
        assert visible_features is not None
        fc = self.scene.target_objs[target_ind].shape
        assert isinstance(fc, FeatureCatalog)
        valid_lmk_inds = [v for ind, v in enumerate(visible_features) if valid[ind]]
        world_points = fc.feature_locations[valid_lmk_inds].T

        if self.pnp_ransac_iterations:

            best_rotation = None
            best_translation = None

            best_inliers = np.zeros(image_points.shape[1], dtype=bool)

            best_standard_deviation = np.finfo(dtype=np.float64).max
            best_mean = np.finfo(dtype=np.float64).max

            # we can only do ransac if there are at least 10 points
            if world_points.shape[1] > 10:
                # choose at most 10 at a time
                for combination in RandomCombinations(world_points.shape[1], min(10, image_points.shape[1]-2),
                                                      self.pnp_ransac_iterations):

                    # solve the least squares problem given the current subset
                    shift, rotation = self._lls(world_points[:, combination], image_points[:, combination],
                                                image, image_ind)

                    # if we failed move on
                    if shift is None or rotation is None:
                        continue

                    # reproject the points and find inliers
                    diff = np.linalg.norm(
                        self.camera.model.project_onto_image(rotation.matrix @ (world_points + shift.reshape(3, 1)),
                                                             temperature=image.temperature, image=image_ind) -
                        image_points, axis=0
                    )

                    inliers = diff < self.second_search_region

                    standard_deviation = diff[inliers].std()
                    mean = diff[inliers].mean()

                    # if we got a good solution
                    if inliers.sum() > 8:
                        # if we have more inliers and our standard deviation is at least within 20% of the current best
                        if (inliers.sum() > best_inliers.sum()) and (standard_deviation < best_standard_deviation*1.2):
                            best_rotation = rotation
                            best_translation = shift
                            best_inliers = inliers
                            best_standard_deviation = standard_deviation
                            best_mean = mean
                        elif ((inliers.sum() >= best_inliers.sum()-2) and
                              (standard_deviation < best_standard_deviation*0.8)):
                            # if we are within 2 inliers of the other but our standard deviation is significantly better
                            best_rotation = rotation
                            best_translation = shift
                            best_inliers = inliers
                            best_standard_deviation = standard_deviation
                            best_mean = mean

                # at this point print out the best RANSAC solution
                if best_inliers.any():
                    assert best_rotation is not None and best_translation is not None
                    print('best inliers = {} of {}'.format(best_inliers.sum(), world_points.shape[1]))
                    print('resid std, mean = {}, {}'.format(best_standard_deviation, best_mean))
                    print('best rotation = {}, {}, {}'.format(*np.rad2deg(best_rotation.vector)))
                    print('best translation = {}, {}, {}'.format(*best_translation), flush=True)

                    # try to solve the LLS using all of the inliers from the best case
                    new_trans, new_rot = self._lls(world_points[:, best_inliers], image_points[:, best_inliers],
                                                   image, image_ind)

                    if new_trans is not None and new_rot is not None:
                        diff = np.linalg.norm(
                            self.camera.model.project_onto_image(new_rot.matrix @ (world_points + new_trans.reshape(3, 1)),
                                                                temperature=image.temperature, image=image_ind) -
                            image_points, axis=0
                        )

                        inliers = diff < self.second_search_region
                        standard_deviation = diff[inliers].std()
                        mean = diff[inliers].mean()

                        print('inlier rotation = {}, {}, {}'.format(*np.rad2deg(new_rot.vector)))
                        print('inlier translation = {}, {}, {}'.format(*new_trans))
                        print('inlier inliers, std, mean = {}, {}, {}'.format(inliers.sum(), standard_deviation, mean),
                            flush=True)

                        # keep either the original best or the fit of the inliers, depending on which is better
                        if (inliers.sum() >= best_inliers.sum() - 2) and (standard_deviation < best_standard_deviation):
                            print('keeping new stuff', flush=True)
                            best_rotation = new_rot
                            best_translation = new_trans
                        elif (inliers.sum() > best_inliers.sum()) and (standard_deviation < best_standard_deviation*1.2):
                            print('keeping new stuff', flush=True)
                            best_rotation = new_rot
                            best_translation = new_trans
                        elif ((inliers.sum() >= best_inliers.sum()) and
                            (mean < best_mean) and
                            (standard_deviation < best_standard_deviation*1.3)):
                            print('keeping new stuff', flush=True)
                            best_rotation = new_rot
                            best_translation = new_trans

            else:
                # just do lstsq on everything
                shift, rotation = self._lls(world_points, image_points, image, image_ind)
                best_rotation = rotation
                best_translation = shift

        else:
            # just do lstsq on everything
            shift, rotation = self._lls(world_points, image_points, image, image_ind)
            best_rotation = rotation
            best_translation = shift

        # return the best
        return best_translation, best_rotation  # type: ignore

    def _lls(self, world_points: np.ndarray, image_points: np.ndarray,
             image: OpNavImage, image_ind: int) -> Tuple[np.ndarray, Rotation] | tuple[None, None]:
        """
        Solves the PnP problem using Levenberg-Marquardt linearized least squares by a call to the
        :func:`least_squares` function from scipy.

        In this method we first determine the weighting matrix based on the provided sigmas for the state/measurements.
        We then create temporary functions which compute the residuals and the jacobian weighted by this weighting
        matrix.  We provide these to the :func:`scipy.optimize.least_squares` function along with an initial guess
        of 0 for both rotation and translation.

        Note that in the Jacobian and the residuals, we account for an update least squares problem by appending zeros
        and an identity matrix respectively.

        :param world_points: The feature centers in the original camera frame  as a 2xn array
        :param image_points: The observed pixel locations of each feature as a 2xn array
        :param image: The image we are solving the PnP for
        :param image_ind: The index of the image we are solving the PnP for
        :return: The best fit shift/rotation as a numpy array and a :class:`.Rotation` or ``None``s if the least squares
                 failed
        """

        # interpret the state sigma
        if self.state_sigma is None:
            if self.position_sigma is None:
                position_sigma = np.ones(3, dtype=np.float64)
            elif np.isscalar(self.position_sigma):
                position_sigma = np.zeros(3, dtype=np.float64) + cast(float, self.position_sigma) 
            else:
                position_sigma: np.ndarray = cast(np.ndarray, self.position_sigma)
            if self.attitude_sigma is None:
                attitude_sigma = np.ones(3, dtype=np.float64)*np.deg2rad(0.02)
            elif np.isscalar(self.attitude_sigma):
                attitude_sigma = np.zeros(3, dtype=np.float64) + cast(float, self.attitude_sigma) 
            else:
                attitude_sigma = cast(np.ndarray, self.attitude_sigma)

            if (position_sigma.ndim == 1) and (attitude_sigma.ndim == 1):
                state_sigma = np.concatenate([position_sigma, attitude_sigma])
            else:
                state_sigma = np.zeros((6, 6), dtype=np.float64)
                if position_sigma.ndim == 1:
                    state_sigma[:3, :3] = np.diag(position_sigma)
                else:
                    state_sigma[:3, :3] = position_sigma
                if attitude_sigma.ndim == 1:
                    state_sigma[3:, 3:] = np.diag(attitude_sigma)
                else:
                    state_sigma[3:, 3:] = attitude_sigma
        else:
            state_sigma = np.array(self.state_sigma)

        # interpret the measurement sigma
        if self.measurement_sigma is None:
            measurement_sigma = np.ones(image_points.size)
        elif np.isscalar(self.measurement_sigma):
            measurement_sigma = np.ones(image_points.size)*cast(float, self.measurement_sigma)
        elif np.ndim(self.measurement_sigma) == 1:
            measurement_sigma = (np.ones((1, image_points.shape[1]), dtype=np.float64) *
                                 np.reshape(self.measurement_sigma, (2, 1))).T.ravel()
        else:
            measurement_sigma = np.zeros((image_points.shape[1], image_points.shape[1]), dtype=np.float64)
            for measurement_number in range(image_points.shape[1]):
                measurement_sigma[2*measurement_number:2*(measurement_number+1),
                                  2*measurement_number:2*(measurement_number+1)] = self.measurement_sigma

        # form the sigma matrix
        if (measurement_sigma.ndim == 1) and (state_sigma.ndim == 1):
            sigma_matrix = np.concatenate([measurement_sigma, state_sigma])
        elif measurement_sigma.ndim == 1:
            sigma_matrix_shape = measurement_sigma.size + state_sigma.shape[0]
            sigma_matrix = np.zeros((sigma_matrix_shape, sigma_matrix_shape), dtype=np.float64)
            sigma_matrix[:measurement_sigma.size, :measurement_sigma.size] = np.diag(measurement_sigma)
            sigma_matrix[measurement_sigma.size:, measurement_sigma.size:] = state_sigma
        else:
            sigma_matrix_shape = measurement_sigma.shape[0] + state_sigma.size
            sigma_matrix = np.zeros((sigma_matrix_shape, sigma_matrix_shape), dtype=np.float64)
            sigma_matrix[:measurement_sigma.size, :measurement_sigma.size] = measurement_sigma
            sigma_matrix[measurement_sigma.size:, measurement_sigma.size:] = np.diag(state_sigma)

        del measurement_sigma, state_sigma

        # interpret the sigma matrix into the transform matrix (inspired by curve_fit from scipy)
        if sigma_matrix.ndim == 1:
            transform_matrix = 1/sigma_matrix
        else:
            # do the Cholesky of the sigma matrix
            transform_matrix = np.linalg.cholesky(sigma_matrix)

        # create the function we are trying to fit
        def pnp_residual_function(state_update: np.ndarray) -> np.ndarray:
            # get the updated world points
            updated_world_points = rotvec_to_rotmat(state_update[3:])@(world_points + state_update[:3].reshape(3, 1))

            # compute the observed - computed residuals.
            resids_omc = image_points - self.camera.model.project_onto_image(updated_world_points,
                                                                             image=image_ind,
                                                                             temperature=image.temperature)

            # append 0s at the end because we are doing an update to stay closer to the original
            out = np.concatenate([resids_omc.T.ravel(), np.zeros(6)])

            # apply the sigmas depending on the number of dimensions
            if transform_matrix.ndim == 1:
                return out * transform_matrix
            else:
                return np.linalg.solve(transform_matrix, out)

        # create the function that computes the jacobian matrix (2n+6 x 6) for the function we are trying to fit
        def pnp_jacobian_function(state_update: np.ndarray) -> np.ndarray:

            # get the updated world points
            shifted_world_points = world_points + state_update[:3].reshape(3, 1)
            rotation_matrix = rotvec_to_rotmat(state_update[3:])

            updated_world_points = rotation_matrix@shifted_world_points

            # build the jacobian matrix that predicts how a change in the world points changes the projected location of
            # the points
            # this is a n x 2 x 3 matrix
            jac_ip_wrt_cp = self.camera.model.compute_pixel_jacobian(updated_world_points,
                                                                     temperature=image.temperature,
                                                                     image=image_ind)

            # build the jacobian matrix that predicts how a change in the orientation changes the projected location of
            # the points
            # this is a 2n x 3 matrix
            jac_ip_wrt_or = np.vstack(jac_ip_wrt_cp @ rot.skew(shifted_world_points)) # type: ignore

            # build the jacobian matrix that predicts how a change in the shift changes the projected location of
            # the points
            # this is a 2n x 3 matrix
            jac_ip_wrt_shift = np.vstack(jac_ip_wrt_cp@rotation_matrix) # type: ignore

            # build the full jacobian matrix (the identity matrix at the bottom is because we are doing an update)
            out = np.vstack([np.hstack([jac_ip_wrt_shift, jac_ip_wrt_or]), np.eye(6)])

            # apply the transform matrix depending on the number of dimensions
            if transform_matrix.ndim == 1:
                return transform_matrix.reshape(-1, 1) * out
            else:
                return np.linalg.solve(transform_matrix, out)

        # now solve the least squares problem
        fit = least_squares(pnp_residual_function,  # function
                            np.zeros(6, dtype=np.float64),  # initial guess
                            jac=pnp_jacobian_function,  # type: ignore
                            method='lm',  # use Levenberg-Marquardt
                            ftol=self.lsq_relative_error_tolerance,  # the tolerance in the change in residuals
                            xtol=self.lsq_relative_update_tolerance,  # tolerance in the update
                            max_nfev=self.max_lsq_iterations,
                            x_scale='jac')
        
        if (fit.x == 0).all():
            # needed a better initial guess.  Do a single least squares fit
            c = np.zeros(6, dtype=np.float64)
            guess = np.linalg.lstsq(pnp_jacobian_function(c), pnp_residual_function(c))[0]
            
            fit2 = least_squares(pnp_residual_function,  # function
                                 guess,  # initial guess
                                 jac=pnp_jacobian_function,  # type: ignore
                                 method='lm',  # use Levenberg-Marquardt
                                 ftol=self.lsq_relative_error_tolerance,  # the tolerance in the change in residuals
                                 xtol=self.lsq_relative_update_tolerance,  # tolerance in the update
                                 max_nfev=self.max_lsq_iterations,
                                 x_scale='jac')
            if (fit2.cost <= fit.cost):
                fit = fit2

        if fit.success:
            shift = fit.x[:3]
            rotation = Rotation(fit.x[3:])
        else:
            print("The PnP problem didn't converge")
            print(fit.message)
            shift = None
            rotation = None

        return shift, rotation # type: ignore
