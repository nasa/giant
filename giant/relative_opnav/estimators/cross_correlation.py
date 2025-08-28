# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


r"""
This module provides the capability to locate the center-of-figure of a target in an image using 2D cross-correlation.

Description of the Technique
----------------------------

Center-finding is the process by which the bearing to the center-of-figure of a target is identified in an image.
The most popular way of performing center finding for OpNav is through a process known as 2D cross-correlation.  In
cross-correlation, a template is correlated with the image to locate where the template and image match the most.  The
template typically represents what we expect the object to look like and the correlation is performed for every possible
alignment between the image and the template.  The location where the correlation score is highest is then said to be
the location of the template in the image.

Cross-correlation based center finding is extremely accurate when the shape model of the target is well known, the range
to the target is pretty well known, and the orientation between the target and the camera is pretty well known.  The
results degrade and can even become biased when any of these conditions are not met, particularly the range to the
target.  If you expect that your knowledge will not be sufficient in one of these areas it may be better to attempt
to use a different technique such as :mod:`.sfn` or :mod:`.limb_matching`.  Alternatively, you can use this method
to get initial results, then refine your knowledge with OD and shape modelling, and then reprocess the images to get
more accurate results.

In GIANT, the cross correlation algorithm is implemented as follows.

#. Render the template based on the a priori relative state between the camera and the target using a single bounce
   ray trace and the routines from :mod:`.ray_tracer`.
#. Perform 2D normalized cross correlation for every possible alignment between the center of the template and the
   image.
#. Locate the peak of the correlation surface (optionally locate the subpixel peak by fitting a 2D quadric to the
   correlation surface)
#. Correct the located peak based on the location of the center-of-figure of the target in the template to get the
   observed center-of-figure in the image.

Tuning
------

There aren't too many parameters that need to be tuned for successful cross correlation as long as you have a decent
shape model and decent a priori state information.  Beyond that, parameters you can tune are

========================================== =============================================================================
Parameter                                  Description
========================================== =============================================================================
:attr:`~XCorrCenterFinding.brdf`           The bidirectional reflectance distribution function used to compute the
                                           expected illumination of a ray based on the geometry of the scene.
:attr:`~XCorrCenterFinding.grid_size`      The size of the grid to use for subpixel sampling when rendering the template
:attr:`~XCorrCenterFinding.peak_finder`    The function to use to detect the peak of the correlation surface.
:attr:`~XCorrCenterFinding.blur`           A flag specifying whether to blur the correlation surface to decrease high
                                           frequency noise before identifying the peak.
:attr:`~XCorrCenterFinding.search_region`  An optional search region in pixels to restrict the area the peak of the
                                           correlation surface is search for around the a priori predicted center
:attr:`~XCorrCenterFinding.min_corr_score` The minimum correlation score to accept as a successful identification.
                                           Correlation scores range from -1 to 1, with 1 indicating perfect correlation.
========================================== =============================================================================

Of these options, most only make small changes to the results.  The 2 that can occasionally make large changes are
:attr:`~XCorrCenterFinding.search_region` and :attr:`~XCorrCenterFinding.blur`.  In general
:attr:`~XCorrCenterFinding.search_region` should be left at ``None`` which searches the whole image.  However, if your
object is nearly unresolved (<10 pixels across or so) and your a priori knowledge is pretty good, then it may be
beneficial to set this to a smallish number to ensure that you don't mistakenly correlate with a noise spike or bright
star.  The :attr:`~XCorrCenterFinding.blur` attribute can also be used to help avoid correlating with a star but in
general should be left as ``False`` unless the object is small (<7-10 pixels).  Finally, the
:attr:`~XCorrCenterFinding.min_corr_score` can generally be left at the default, though if you have a poor a priori
knowledge of either the shape model or the relative position of the object then you may need to decrease this some.

Use
---

The class provided in this module is usually not used by the user directly, instead it is usually interfaced with
through the :class:`.RelativeOpNav` class using the identifier :attr:`~.RelativeOpNav.cross_correlation`.  For more
details on using the :class:`.RelativeOpNav` interface, please refer to the :mod:`.relnav_class` documentation.  For
more details on using the technique class directly, as well as a description of the ``details`` dictionaries produced
by this technique, refer to the following class documentation.
"""


import warnings

from dataclasses import dataclass

from typing import Optional, Callable, List

import numpy as np

from giant.relative_opnav.estimators.estimator_interface_abc import RelNavObservablesType
from giant.ray_tracer.scene import Scene
from giant.camera import Camera
from giant.image import OpNavImage
from giant.image_processing import quadric_peak_finder_2d
from giant.image_processing.correlators import CORRLATOR_SIGNATURE, cv2_correlator_2d

from giant.relative_opnav.estimators._template_renderer import TemplateRenderer, TemplateRendererOptions


@dataclass
class XCorrCenterFindingOptions(TemplateRendererOptions):
    """
    This dataclass serves as one way to control the settings for the :class:`.XCorrCenterFinding` class.

    You can set any of the options on an instance of this dataclass and pass it to the
    :class:`.XCorrCenterFinding` class at initialization (or through the method
    :meth:`.XCorrCenterFinding.apply_options`) to set the settings on the class. This class is the preferred way
    of setting options on the class due to ease of use in IDEs.
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
    
    correlator: CORRLATOR_SIGNATURE = cv2_correlator_2d
    """
    The normaized cross correlation routine to use
    """


class XCorrCenterFinding(TemplateRenderer, XCorrCenterFindingOptions):
    """
    This class implements normalized cross correlation center finding for GIANT.

    All of the steps required for performing cross correlation are handled by this class, including the rendering of the
    template, the actual cross correlation, and the identification of the peak of the correlation surface.  This is all
    handled in the :meth:`estimate` method and is performed for each requested target.

    When all of the required data has been successfully loaded into an instance of this class, the :meth:`estimate`
    method is used to perform the estimation for the requested image.  The results are stored into the
    :attr:`observed_bearings` attribute for the observed center of figure locations. In addition, the predicted location
    for the center of figure for each target is stored in the :attr:`computed_bearings` attribute. Finally, the details
    about the fit are stored as a dictionary in the appropriate element in the :attr:`details` attribute.  Specifically,
    these dictionaries will contain the following keys.

    ================================= ==================================================================================
    Key                               Description
    ================================= ==================================================================================
    ``'Correlation Score'``           The correlation score at the peak of the correlation surface.  This is only
                                      available if the fit was successful.
    ``'Correlation Surface'``         The raw correlation surface as a 2D array.  Each pixel in the correlation surface
                                      represents the correlation score when the center of the template is lined up with
                                      the corresponding image pixel.  This is only available if the fit was successful.
    ``'Correlation Peak Location'``   The Location of the correlation peak before correcting it to find the location of
                                      the target center of figure. This is only available if the fit was successful.
    ``'Target Template Coordinates'`` The location of the center of figure of the target in the template.  This is only
                                      available if the fit was successful.
    ``'Failed'``                      A message indicating why the fit failed.  This will only be present if the fit
                                      failed (so you could do something like
                                      ``'Failed' in cross_correlation.details[target_ind]`` to
                                      check if something failed.  The message should be a human readable description of
                                      what caused the failure.
    ``'Max Correlation'``             The peak value of the correlation surface.  This is only available if the fit
                                      failed due to too low of a correlation score.
    ================================= ==================================================================================

    .. warning::
        Before calling the :meth:`estimate` method be sure that the scene has been updated to correspond to the correct
        image time.  This class does not update the scene automatically, even if the :attr:`scene` attribute is an
        :class:`.Scene` instance.
    """

    observable_type = [RelNavObservablesType.CENTER_FINDING]
    """
    This technique generates CENTER-FINDING bearing observables to the center of figure of a target.
    """

    def __init__(self, scene: Scene, camera: Camera, 
                 options: Optional[XCorrCenterFindingOptions] = None):
        """
        :param scene: The scene describing the a priori locations of the targets and the light source.
        :param camera: The :class:`.Camera` object containing the camera model and images to be analyzed
        :param image_processing: An instance of :class:`.ImageProcessing`.  This is used for denoising the image and for
                                 generating the correlation surface using :meth:`.denoise_image` and :meth:`correlate`
                                 methods respectively
        :param options: A dataclass specifying the options to set for this instance.

        """

        super().__init__(scene, camera, options=options, options_type=XCorrCenterFindingOptions)

    def estimate(self, image: OpNavImage, include_targets: Optional[List[bool]] = None):
        """
        This method identifies the center of each target in the image using cross correlation

        This method first does ray tracing to render a template and then computes the expected location of the object
        within the image. The expected bounds and size of the template are computed in pixels. The psf of the camera
        is then applied to the template and the center of the body is determined within the template. Noise is
        removed from the image and the image and template are correlated. The peaks of the correlation surface are
        determined in addition to residuals between the solved for center in the template and center in the image.

        .. warning::
            Before calling this method be sure that the scene has been updated to correspond to the correct
            image time.  This method does not update the scene automatically, even if the :attr:`scene` attribute is an
            :class:`.Scene` instance.

        :param image: The image to locate the targets in
        :param include_targets: A list specifying whether to process the corresponding target in
                                :attr:`.Scene.target_objs` or ``None``.  If ``None`` then all targets are processed.
        """

        # Make sure the image is single precision for use with opencv
        image = image.astype(np.float32)  # type: ignore

        for target_ind, target in self.target_generator(include_targets):
            
            template, bounds = self.prepare_template(target_ind, target, image.temperature)
            
            self.templates[target_ind] = template

            # Compute the expected location of the object in the image
            computed = self.camera.model.project_onto_image(
                target.position,
                temperature=image.temperature
            ).ravel()

            self.computed_bearings[target_ind] = computed
            
            # Determine the center of the body in the template
            center_of_template = computed - bounds[0]

            # Perform the correlation between the image and the template
            correlation_surface = self.correlator(image, template)

            # Get the middle of the template
            temp_middle = np.floor(np.flipud(np.array(template.shape)) / 2)

            # Figure out the delta between the middle of the template and the center of the body in the template
            delta = temp_middle - center_of_template

            if self.search_region is not None:
                # If we want to restrict our search for the peak of the correlation surface around the a priori center

                # Compute the center of the search region
                search_start = np.round(computed + delta).astype(int)

                # Compute the beginning of the search region in pixels
                begin_roi = search_start - self.search_region
                begin_roi = np.maximum(begin_roi, 0)

                # Compute the end of the search region in pixels
                end_roi = search_start + self.search_region + 1
                end_roi = np.minimum(end_roi, correlation_surface.shape[::-1])

                # Extract the appropriate sub array from the correlation surface
                roi = correlation_surface[begin_roi[1]:end_roi[1], begin_roi[0]:end_roi[0]]

                # Find the peaks from the extracted correlation surface
                peaks = self.peak_finder(roi, self.blur).ravel()

                # Correct the peaks to correspond to the full image instead of the extracted surface and store them
                peaks += begin_roi

                peak_location = peaks

            else:
                # Find the peak of the correlation surface
                peak_location = self.peak_finder(correlation_surface, self.blur).ravel()

            # check if the correlation peak was found
            if not np.isfinite(peak_location).all():
                warnings.warn("Correlation peak not found.")
                self.details[target_ind] = {'Failed': 'Unable to find peak of correlation surface'}
                self.observed_bearings[target_ind] = np.nan*np.ones(2, dtype=np.float64)
                continue

            # check the peak score for the correlation surface
            correlation_peak_pix = np.round(peak_location).astype(int)
            correlation_score = correlation_surface[correlation_peak_pix[1], correlation_peak_pix[0]]

            # check if the correlation score was too low
            if correlation_score < self.min_corr_score:
                warnings.warn(f"Correlation peak too low. Peak of {correlation_score} is less "
                              f"than {self.min_corr_score}.")

                self.details[target_ind] = {'Failed': 'Correlation score too low',
                                            "Max Correlation": correlation_score}
                self.observed_bearings[target_ind] = np.nan * np.ones(2, dtype=np.float64)
                continue

            # Determine the solved for center of the body by correcting for the offset from the center of the template
            # to the center of the body
            self.observed_bearings[target_ind] = peak_location - delta

            # store the details of the fit
            self.details[target_ind] = {"Correlation Score": correlation_score,
                                        "Correlation Surface": correlation_surface,
                                        "Correlation Peak Location": peak_location,
                                        "Target Template Coordinates": center_of_template}
